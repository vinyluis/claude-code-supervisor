"""
Claude Code Supervisor Agent

An intelligent wrapper around Claude Code SDK that provides automated problem-solving
capabilities with session management, progress monitoring, and intelligent feedback loops.

This supervisor treats Claude Code as a coding assistant that can plan its own work using
todo lists, implement solutions, run tests, and iterate based on feedback. The supervisor
monitors Claude's progress, provides guidance when issues arise, and ensures solutions
meet quality standards through automated testing.

Key Features:
- Session-based continuity with Claude Code SDK
- Real-time progress monitoring via todo list tracking
- LLM-powered error analysis and guidance generation
- Configurable timeouts and iteration limits
- Comprehensive test execution and validation
- Support for multiple AI providers (Anthropic, Bedrock)

Architecture:
The supervisor uses LangGraph to orchestrate a workflow with these nodes:
1. initiate_claude: Start Claude Code session with problem description
2. execute_claude: Execute Claude session until completion or timeout
3. collect_results: Collect and analyze results from Claude's session
4. validate_solution: Run tests on generated files to verify correctness
5. provide_guidance: Analyze failures and generate actionable feedback
6. finalize: Complete the session and report results

Usage:
  agent = SupervisorAgent()
  result = agent.process('Create a sorting function', example_output='sort([3,1,2]) -> [1,2,3]')
"""

import os
import sys
import subprocess
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from langchain_core.language_models.base import BaseLanguageModel

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import (
  AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage,
  SystemMessage
)
from dotenv import load_dotenv
from .data_manager import DataManager
from .config import SupervisorConfig, development_config
from . import utils

# Suppress asyncio warnings from the Claude Code SDK
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited.*")
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
warnings.filterwarnings("ignore", message=".*cancel scope in a different task.*")


@dataclass
class WorkflowState:
  """Dynamic state that changes during workflow execution"""
  # Workflow progress
  current_iteration: int = 0
  is_solved: bool = False
  error_message: str = ""
  test_results: str = ""
  
  # Claude session state
  claude_session_id: str | None = None
  claude_session_active: bool = False
  claude_todos: list[dict] = field(default_factory=list)
  claude_output_log: list[str] = field(default_factory=list)
  should_terminate_early: bool = False
  latest_guidance: str = ""
  
  # Output data (dynamic - extracted during execution)
  output_data: Any = None

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)


class SupervisorAgent:
  """
  Intelligent supervisor for automated problem-solving using Claude Code SDK.

  This agent acts as a wrapper around Claude Code, providing session management,
  progress monitoring, and intelligent feedback loops to solve programming problems
  iteratively with minimal human intervention.

  The supervisor works by:
  1. Initiating Claude Code sessions with structured problem descriptions
  2. Monitoring Claude's real-time progress through SDK message streaming
  3. Tracking todo list updates to understand Claude's planning and execution
  4. Validating solutions by running automated tests
  5. Providing LLM-powered guidance when Claude encounters issues
  6. Managing session continuity across multiple iterations

  Configuration:
  - The agent is configured using the dataclasses available in `config.py`

  Key Features:
  - Session resumption for continuity across iterations
  - Configurable timeouts to prevent infinite loops
  - Real-time progress monitoring with timestamped logging
  - Intelligent error analysis using OpenAI models
  - Support for multiple AI providers (Anthropic, Bedrock, Vertex)
  - Comprehensive test execution with pytest integration
  - Automatic guidance generation for failed attempts

  Example:
      >>> # Basic usage
      >>> agent = SupervisorAgent()
      >>> result = agent.process(
      >>>     'Create a function to calculate fibonacci numbers',
      >>>     example_output='fib(8) should return 21'
      >>> )
      >>>
      >>> # With custom prompt
      >>> agent = SupervisorAgent(custom_prompt='Use object-oriented design')
      >>> result = agent.process('Create a calculator')
      >>>
      >>> # Bring Your Own Model (BYOM)
      >>> from langchain_openai import ChatOpenAI
      >>> my_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
      >>> agent = SupervisorAgent(llm=my_llm)
      >>> result = agent.process('Create a sorting function')
      >>>
      >>> # Check if solved
      >>> if result.is_solved:
      >>>     print(f'Solution: {result.solution_path}')
      >>>     print(f'Tests: {result.test_path}')
      >>> else:
      >>>     print(f'Error: {result.error_message}')

  Args:
    custom_prompt: Optional additional instructions for Claude Code
    llm: Optional LangChain LLM model for guidance (BYOM - Bring Your Own Model)

  Attributes:
    config: Configuration for the supervisor agent
    custom_prompt: Optional additional instructions for Claude Code
    llm: LLM instance for guidance generation (provided or configured)
    graph: LangGraph workflow compiled from node definitions
    base_claude_options: Pre-configured ClaudeCodeOptions for faster iterations
  """

  def __init__(
    self,
    custom_prompt: str | None = None,
    llm: BaseLanguageModel | None = None,
    config: SupervisorConfig | None = None,
    **kwargs,
  ) -> None:

    self._load_environment()

    if config is not None and isinstance(config, SupervisorConfig):
      self.config = config
    else:
      print(f"[{self._timestamp()}] 🔧 Loading default configuration...")
      self.config = development_config()

    if llm is not None:
      self.llm = llm
    else:
      self.llm = self._initialize_llm()

    self.custom_prompt = custom_prompt
    self._initialize_claude_code()
    self.graph = self._build_graph()
    
    # Static workflow configuration (set once per process call)
    self.problem_description: str = ""
    self.example_output: str | None = None
    self.solution_path: str = ""
    self.test_path: str = ""
    self.input_data: Any = None
    self.expected_output: Any = None
    self.data_format: str = 'auto'
    self.data_manager: DataManager | None = None
    self.integrate_into_codebase: bool = False

    # Check for credentials on the kwargs
    if 'aws_access_key_id' in kwargs:
      if not os.getenv('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = kwargs['aws_access_key_id']
    if 'aws_secret_access_key' in kwargs:
      if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = kwargs['aws_secret_access_key']
    if 'aws_region' in kwargs:
      if not os.getenv('AWS_REGION'):
        os.environ['AWS_REGION'] = kwargs['aws_region']
    if 'anthropic_api_key' in kwargs:
      if not os.getenv('ANTHROPIC_API_KEY'):
        os.environ['ANTHROPIC_API_KEY'] = kwargs['anthropic_api_key']
    if 'openai_api_key' in kwargs:
      if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = kwargs['openai_api_key']

  def _timestamp(self) -> str:
    """Get current timestamp for logging"""
    return utils.timestamp()



  def _load_environment(self) -> None:
    """
    Load environment variables from .env file
    """
    env_path = Path('.env')
    if env_path.exists():
      load_dotenv(env_path)
    else:
      print("⚠️Warning: .env file not found. Environment variables may need to be set manually.")

  def _initialize_claude_code(self) -> None:
    """
    Initialize Claude Code SDK configuration and prepare base options
    """
    claude_config = self.config.claude_code
    # Set environment variables based on provider choice
    if claude_config.use_bedrock:
      os.environ["CLAUDE_CODE_USE_BEDROCK"] = '1'
      print(f"[{self._timestamp()}] 🔧 Configured Claude Code to use Amazon Bedrock")
    else:
      # Default to Anthropic API
      if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"[{self._timestamp()}] Warning: ANTHROPIC_API_KEY not found in environment. Claude Code SDK may not work properly.")
        print(f"[{self._timestamp()}] Please set your API key in the .env file or environment variables.")
      else:
        print(f"[{self._timestamp()}] 🔧 Configured Claude Code to use Anthropic API")

    # Build system prompt with optional custom prompt
    base_system_prompt = 'You are an expert Python developer. Use the TodoWrite tool to plan and track your work. Always run tests to verify your solutions.'
    if self.custom_prompt:
      system_prompt = f"{base_system_prompt}\n\nAdditional instructions:\n{self.custom_prompt}"
    else:
      system_prompt = base_system_prompt

    # Pre-configure Claude Code options for faster reuse in iterations
    self.base_claude_options = ClaudeCodeOptions(
      cwd=os.getcwd(),
      permission_mode='acceptEdits',
      max_turns=claude_config.max_turns,
      system_prompt=system_prompt,
      max_thinking_tokens=claude_config.max_thinking_tokens
    )
    print(f"[{self._timestamp()}] 🔧 Pre-configured Claude Code options for faster iterations")

  def _build_graph(self):
    """Build the LangGraph workflow"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("initiate_claude", self._initiate_claude_code_session)
    workflow.add_node("execute_claude", self._execute_claude_session)
    workflow.add_node("collect_results", self._collect_session_results)
    workflow.add_node("validate_solution", self._validate_solution)
    workflow.add_node("provide_guidance", self._provide_guidance)
    workflow.add_node("finalize", self._finalize_solution)

    # Start with initiation
    workflow.add_edge(START, 'initiate_claude')
    
    # After initiation, execute Claude session
    workflow.add_edge("initiate_claude", 'execute_claude')
    
    # After execution, collect results
    workflow.add_edge("execute_claude", 'collect_results')

    # From collect_results, decide what to do next
    workflow.add_conditional_edges(
      'collect_results',
      self._decide_next_action,
      {
        'validate': 'validate_solution',
        'guide': 'provide_guidance', 
        'finish': 'finalize'
      }
    )

    # After validation, always finalize (success or failure)
    workflow.add_edge("validate_solution", 'finalize')
    
    # After guidance, execute Claude again (iteration)
    workflow.add_edge("provide_guidance", 'execute_claude')
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)

    return workflow.compile()

  def _initialize_llm(self):
    """Initialize the LLM for guidance analysis (OpenAI or AWS Bedrock)"""
    model_config = self.config.model
    provider = model_config.provider
    if provider is None:
      raise ValueError("Model provider not specified in configuration. Please set 'provider' in the model config.")
    elif provider == 'bedrock':
      if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ValueError("AWS credentials not found in environment variables. Please set 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET_ACCESS_KEY'.")
      aws_key = os.getenv('AWS_ACCESS_KEY_ID')
      aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY') 
      aws_region = os.getenv('AWS_REGION')
      if aws_key is None or aws_secret is None:
        raise ValueError("AWS credentials validation failed")
      
      return ChatBedrockConverse(
        model=model_config.name,
        temperature=model_config.temperature,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region,
      )
    elif provider == 'openai':
      return ChatOpenAI(
        model=model_config.name,
        temperature=model_config.temperature,
      )
    else:
      raise ValueError(f"Unsupported model provider: {provider}. Supported providers are 'openai' and 'bedrock'.")

  def _initiate_claude_code_session(self, state: WorkflowState) -> WorkflowState:
    """Prepare the initial setup for Claude Code session"""
    print(f"\n[{self._timestamp()}] 🚀 Setting up Claude Code session...")

    # Handle input data if provided
    input_data_info = ""
    if self.input_data is not None and self.data_manager is not None:
      try:
        print(f"[{self._timestamp()}] 📊 Processing input data...")

        # Get data description and format
        data_format = self.data_format if self.data_format != 'auto' else self.data_manager.infer_format(self.input_data)
        data_description = self.data_manager.get_data_description(self.input_data, data_format)
        data_context = self.data_manager.serialize_for_context(self.input_data, data_format)

        # Record the operation
        self.data_manager.record_operation(self.input_data, data_format, 'input')

        input_data_info = f"""
Input Data Available:
{data_context}

Data Description: {data_description}

The input data is available in your solution as a variable. You can access it directly in your code.
"""
        print(f"[{self._timestamp()}] 📊 Input data processed: {data_format} format")

      except Exception as e:
        print(f"[{self._timestamp()}] ❌ Failed to process input data: {e}")
        state.error_message = f"Failed to process input data: {e}"
        return state

    # Prepare the problem statement for Claude Code
    expected_output_info = ""
    if self.expected_output is not None:
      expected_output_info = f"\nExpected output format: {type(self.expected_output).__name__}"
      if hasattr(self.expected_output, '__len__') and len(self.expected_output) < 10:
        expected_output_info += f"\nExpected result example: {self.expected_output}"

    # Build the initial prompt
    if state.current_iteration == 0:
      problem_prompt = f"""\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works

Problem: {self.problem_description}
{f'Expected behavior: {self.example_output}' if self.example_output else ''}
{expected_output_info}
{input_data_info}

Requirements:
- Use Python
{f'- Save the solution as "{self.solution_path}"' if not self.integrate_into_codebase else '- Integrate your solution directly into the existing codebase by modifying the appropriate files'}
{f'- Save tests as "{self.test_path}"' if not self.integrate_into_codebase else '- Add tests to the existing test files, following the existing test structure and conventions'}
- Follow clean code practices with docstrings and type hints
- Ensure all tests pass before completing
{'- If input data is provided, make sure to read and process it correctly' if self.input_data is not None else ''}
{'- Return results in the same format as the expected output' if self.expected_output is not None else ''}

Please start by creating a todo list to plan your approach, then implement the solution.
"""
    else:
      # Subsequent iterations - provide guidance
      latest_guidance = ""
      if state.latest_guidance:
        latest_guidance = state.latest_guidance

      problem_prompt = f"""\
{latest_guidance if latest_guidance else 'Continue working on the problem based on the previous feedback.'}

Please update your todo list and continue with the implementation.
"""

    # Store the prompt for the execution phase
    state.claude_output_log = [f"PROMPT_ITERATION_{state.current_iteration}: {problem_prompt}"]
    state.claude_session_active = True
    
    print(f"[{self._timestamp()}] 📝 Prepared prompt for iteration {state.current_iteration}")
    print(f"[{self._timestamp()}] Working directory: {os.getcwd()}")

    return state

  async def _run_claude_session(self, problem_prompt: str, options: ClaudeCodeOptions, state: WorkflowState) -> None:
    """Simplified async session execution following SDK best practices"""
    try:
      # Simple async iteration as recommended in SDK docs
      async for message in query(prompt=problem_prompt, options=options):
            
        if isinstance(message, AssistantMessage):
          for block in message.content:
            if isinstance(block, TextBlock):
              state.claude_output_log.append(block.text)
              print(f"[{self._timestamp()}] 💬 {utils.blue('Claude:')} {utils.blue(block.text[:200] + ('...' if len(block.text) > 200 else ''))}")
            
            elif isinstance(block, ToolUseBlock):
              tool_info = utils.get_tool_info(block.name, block.input)
              print(f"[{self._timestamp()}] 🔧 {utils.blue('Tool:')} {utils.blue(block.name)} {tool_info}")
              
              # Track todo updates
              if block.name == 'TodoWrite':
                todos = block.input.get('todos', [])
                state.claude_todos = todos
                print(f"[{self._timestamp()}] 📋 {utils.blue('Todo list updated:')} {utils.blue(str(len(todos)) + ' items')}")
                for todo in todos:
                  status_emoji = {'pending': '⏳', 'in_progress': '🔄', 'completed': '✅'}.get(todo.get('status'), '❓')
                  print(f"[{self._timestamp()}]   {status_emoji} {utils.blue(todo.get('content', 'Unknown task'))}")
            
            elif isinstance(block, ToolResultBlock):
              if block.is_error:
                error_msg = f"Tool error: {block.content}"
                print(f"[{self._timestamp()}] ❌ {utils.red(error_msg)}")
                if not state.error_message:
                  state.error_message = error_msg
        
        elif isinstance(message, ResultMessage):
          # Session completed successfully
          state.claude_session_id = message.session_id
          print(f"[{self._timestamp()}] ✅ {utils.green('Claude session completed')} (ID: {message.session_id})")
          print(f"[{self._timestamp()}] {utils.cyan('Turns:')} {message.num_turns}, {utils.cyan('Duration:')} {message.duration_ms}ms")
          if message.total_cost_usd:
            print(f"[{self._timestamp()}] {utils.cyan('Cost:')} ${message.total_cost_usd:.4f}")
          break
        
        elif isinstance(message, SystemMessage):
          print(f"[{self._timestamp()}] ℹ️  System: {message.subtype}")
    
    except Exception as e:
      # Let SDK handle its own errors - only handle real failures
      error_msg = f"Error in Claude session: {e}"
      print(f"[{self._timestamp()}] ❌ {error_msg}")
      if not state.error_message:
        state.error_message = error_msg
      raise

  def _execute_claude_session(self, state: WorkflowState) -> WorkflowState:
    """Execute a Claude Code session until completion or timeout"""
    if not state.claude_session_active:
      state.error_message = "Claude session not active"
      return state

    print(f"\n[{self._timestamp()}] 🚀 Executing Claude Code session (iteration {state.current_iteration})...")

    # Get the prompt from the log
    problem_prompt = ""
    for log_entry in state.claude_output_log:
      if log_entry.startswith(f"PROMPT_ITERATION_{state.current_iteration}:"):
        problem_prompt = log_entry.split(":", 1)[1].strip()
        break

    if not problem_prompt:
      state.error_message = "No prompt found for current iteration"
      return state

    try:
      # Use pre-configured options and update session-specific parameters
      options = ClaudeCodeOptions(
        cwd=self.base_claude_options.cwd,
        permission_mode=self.base_claude_options.permission_mode,
        max_turns=self.base_claude_options.max_turns,
        system_prompt=self.base_claude_options.system_prompt,
        max_thinking_tokens=self.base_claude_options.max_thinking_tokens,
        # Session-specific parameters that change between iterations
        continue_conversation=state.current_iteration > 0,
        resume=state.claude_session_id if state.claude_session_id else None
      )

      # Execute Claude Code session - simplified approach
      try:
        import anyio
        anyio.run(self._run_claude_session, problem_prompt, options, state)
      except Exception as e:
        error_msg = f"Error in async session execution: {e}"
        print(f"[{self._timestamp()}] ❌ {error_msg}")
        # Let the SDK handle its own errors - only store real errors
        if "cancel scope" not in str(e).lower() and "task exception" not in str(e).lower():
          state.error_message = error_msg
          raise

      # Mark session as inactive after completion
      state.claude_session_active = False

      return state

    except Exception as e:
      state.error_message = f"Execution error: {e}"
      state.claude_session_active = False
      print(f"[{self._timestamp()}] ❌ Error executing Claude session: {e}")
      return state

  def _collect_session_results(self, state: WorkflowState) -> WorkflowState:
    """Collect and analyze the results from Claude's session"""
    print(f"\n[{self._timestamp()}] 📊 Collecting session results...")

    # Check if files were created
    solution_exists = os.path.exists(self.solution_path) if self.solution_path else False
    test_exists = os.path.exists(self.test_path) if self.test_path else False

    if solution_exists:
      print(f"[{self._timestamp()}] 📄 Solution file detected: {self.solution_path}")
    if test_exists:
      print(f"[{self._timestamp()}] 🧪 Test file detected: {self.test_path}")

    # Analyze todo completion status
    completed_todos = [todo for todo in state.claude_todos if todo.get('status') == 'completed']
    total_todos = len(state.claude_todos)
    
    if total_todos > 0:
      completion_rate = len(completed_todos) / total_todos
      print(f"[{self._timestamp()}] 📋 Todo completion: {len(completed_todos)}/{total_todos} ({completion_rate:.1%})")
    
    # Check for error patterns in the output using utility functions
    general_errors, credit_quota_errors = utils.detect_errors_in_output(state.claude_output_log)
    error_message, should_terminate_early = utils.format_error_message(general_errors, credit_quota_errors)
    
    # Update state with error information
    if error_message:
      state.error_message = error_message
      state.should_terminate_early = should_terminate_early
      
      if should_terminate_early:
        print(f"[{self._timestamp()}] 🚫 Credit/quota error detected - terminating early")
      else:
        print(f"[{self._timestamp()}] ⚠️  Detected error indicators in Claude's output")

    return state

  def _decide_next_action(self, state: WorkflowState) -> str:
    """Decide what to do next based on session results"""
    print(f"\n[{self._timestamp()}] 🤔 Deciding next action...")

    # Check for early termination due to credit/quota errors
    if state.should_terminate_early:
      print(f"[{self._timestamp()}] 🚫 Early termination requested due to API errors")
      return 'finish'

    # Check if we've exceeded maximum iterations
    if state.current_iteration >= self.config.agent.max_iterations:
      print(f"[{self._timestamp()}] 🔄 Maximum iterations ({self.config.agent.max_iterations}) reached")
      return 'finish'

    # If there are explicit errors, provide guidance
    if state.error_message:
      print(f"[{self._timestamp()}] ❌ Errors detected, providing guidance")
      return 'guide'

    # Check if solution appears complete
    if self.integrate_into_codebase:
      # In integration mode, check if Claude indicates completion
      if not state.claude_session_active:
        print(f"[{self._timestamp()}] ✅ Integration mode: session complete, validating")
        return 'validate'
    else:
      # Standard mode: check if both files exist
      if os.path.exists(self.solution_path) and os.path.exists(self.test_path):
        print(f"[{self._timestamp()}] ✅ Both solution and test files exist, validating")
        return 'validate'

    # Removed activity timeout check - let SDK handle its own timeouts

    # If Claude session is still active, there might be an issue
    if state.claude_session_active:
      state.error_message = "Claude session still active but no progress detected"
      print(f"[{self._timestamp()}] ⚠️  Session still active with no clear progress")
      return 'guide'

    # Default: provide guidance to continue
    print(f"[{self._timestamp()}] 🔄 No clear completion, providing guidance for next iteration")
    return 'guide'


  def _validate_solution(self, state: WorkflowState) -> WorkflowState:
    """Validate the solution created by Claude Code"""
    print(f"\n[{self._timestamp()}] 🔍 Validating Claude's solution...")

    if self.integrate_into_codebase:
      # In integration mode, we validate by running tests on the entire codebase
      print(f"[{self._timestamp()}] 🔧 Integration mode: validating codebase changes...")
      state = self._run_integration_tests(state)
    else:
      # Standard mode: check if both files exist
      if not os.path.exists(self.solution_path):
        state.error_message = f"Solution file {self.solution_path} not created"
        state.is_solved = False
        return state

      if not os.path.exists(self.test_path):
        state.error_message = f"Test file {self.test_path} not created"
        state.is_solved = False
        return state

      # Run the tests to validate
      state = self._run_tests(state)

    # If tests passed and we have input data, try to extract output data
    if state.is_solved and self.input_data is not None:
      state = self._extract_output_data(state)

    return state

  def _run_integration_tests(self, state: WorkflowState) -> WorkflowState:
    """Run tests in integration mode where solution is integrated into codebase"""
    print(f"\n[{self._timestamp()}] 🧪 Running integration tests...")
    
    try:
      # Look for test files in the codebase and run them
      # This is a simplified approach - in practice, you might want to
      # run specific test patterns or use project-specific test commands
      
      # First, try to find test files
      test_files = []
      for root, _, files in os.walk('.'):
        for file in files:
          if file.endswith('_test.py') or file.startswith('test_'):
            test_files.append(os.path.join(root, file))
      
      if not test_files:
        # If no test files found, assume integration was successful
        # This might be the case when tests are added to existing files
        state.is_solved = True
        state.test_results = "No separate test files found - assuming integration successful"
        print(f"[{self._timestamp()}] ✅ No separate test files found, assuming integration successful")
        return state
      
      # Run the tests
      timeout = self.config.agent.test_timeout
      result = subprocess.run(
        [sys.executable, '-m', 'pytest'] + test_files + ['-v', '--tb=short', '--no-header'],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.getcwd()
      )
      
      state.test_results = (f"Exit code: {result.returncode}\n"
                           f"STDOUT:\n{result.stdout}\n"
                           f"STDERR:\n{result.stderr}")
      state.is_solved = result.returncode == 0
      
      if state.is_solved:
        print(f"[{self._timestamp()}] ✅ All integration tests passed!")
        print(f"Test output:\n{result.stdout}")
      else:
        print(f"[{self._timestamp()}] ❌ Integration tests failed (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{result.stderr}")
        state.error_message = f"Integration test failures: {result.stdout}\n{result.stderr}"
        
    except subprocess.TimeoutExpired:
      timeout = self.config.agent.test_timeout
      state.test_results = f"Integration tests timed out after {timeout} seconds"
      state.is_solved = False
      state.error_message = f"Integration tests timed out after {timeout} seconds"
      print(f"[{self._timestamp()}] ⏰ Integration tests timed out after {timeout} seconds")
    except FileNotFoundError:
      state.test_results = "pytest not found. Please install pytest: pip install pytest"
      state.is_solved = False
      state.error_message = "pytest not found"
      print(f"[{self._timestamp()}] ❌ pytest not found. Install with: pip install pytest")
    except Exception as e:
      state.test_results = f"Error running integration tests: {str(e)}"
      state.is_solved = False
      state.error_message = f"Error running integration tests: {str(e)}"
      print(f"[{self._timestamp()}] 💥 Error running integration tests: {str(e)}")
    
    return state

  def _extract_output_data(self, state: WorkflowState) -> WorkflowState:
    """Extract output data by running the solution with input data"""
    print(f"\n[{self._timestamp()}] 📤 Extracting output data from solution...")

    try:
      # Import the solution module dynamically
      import importlib.util

      # Load the solution module
      spec = importlib.util.spec_from_file_location("solution_module", self.solution_path)
      if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {self.solution_path}")
      
      solution_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(solution_module)

      # Try to find a main function or process function
      possible_functions = ['main', 'process', 'solve', 'run', 'execute']
      main_function = None

      for func_name in possible_functions:
        if hasattr(solution_module, func_name):
          main_function = getattr(solution_module, func_name)
          break

      if main_function is None:
        # If no standard function found, try to find any callable function
        functions = [getattr(solution_module, name) for name in dir(solution_module)
                    if callable(getattr(solution_module, name)) and not name.startswith('_')]
        if functions:
          main_function = functions[0]  # Use the first callable function

      if main_function is not None:
        print(f"[{self._timestamp()}] 🔧 Found function: {main_function.__name__}")

        # Try to call the function with input data
        try:
          if self.input_data is not None:
            # Pass the input data directly as argument
            result = main_function(self.input_data)
          else:
            # Call without arguments
            result = main_function()

          state.output_data = result
          print(f"[{self._timestamp()}] ✅ Output data extracted: {type(result).__name__}")

          # Validate against expected output if provided
          if self.expected_output is not None:
            if self._validate_output(result, self.expected_output):
              print(f"[{self._timestamp()}] ✅ Output matches expected format!")
            else:
              print(f"[{self._timestamp()}] ⚠️ Output format doesn't match expected format")

        except Exception as e:
          print(f"[{self._timestamp()}] ❌ Failed to execute function: {e}")
          state.error_message = f"Failed to execute solution function: {e}"

      else:
        print(f"[{self._timestamp()}] ⚠️ No callable main function found in solution")

    except Exception as e:
      print(f"[{self._timestamp()}] ❌ Failed to extract output data: {e}")
      # Don't mark as unsolved, this is optional

    return state

  def _validate_output(self, actual: Any, expected: Any) -> bool:
    """Validate that actual output matches expected output format/type"""
    try:
      # Check type compatibility
      if type(actual) != type(expected):
        return False

      # For collections, check structure
      if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
          return False
        # Check if all elements have compatible types
        for a, e in zip(actual, expected):
          if type(a) != type(e):
            return False

      elif isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
          return False
        # Check value types
        for key in expected:
          if type(actual[key]) != type(expected[key]):
            return False

      return True

    except Exception:
      return False

  def _provide_guidance(self, state: WorkflowState) -> WorkflowState:
    """Provide guidance to Claude Code when it encounters issues"""
    print(f"\n[{self._timestamp()}] 🎯 Analyzing errors and providing guidance...")

    # Increment iteration counter
    state.current_iteration += 1
    print(f"[{self._timestamp()}] 🔄 Starting iteration {state.current_iteration}")

    # Analyze the current situation
    analysis_prompt = f"""\
Analyze this Claude Code implementation failure and provide specific guidance for the next iteration.

Problem: {self.problem_description}
{f'Expected behavior: {self.example_output}' if self.example_output else ''}

Current Issues:
- Error: {state.error_message}
- Test Results: {state.test_results if state.test_results else 'No tests run yet'}
- Current Iteration: {state.current_iteration}

Claude's Todo Progress:
{self._format_todos_for_analysis(state.claude_todos)}

Claude's Recent Output:
{self._format_output_log_for_analysis(state.claude_output_log)}

Provide specific, actionable guidance for Claude Code to fix these issues:
1. What went wrong?
2. What specific steps should Claude take next?
3. What should Claude focus on or avoid?

Keep your response concise and actionable (2-3 bullet points).
"""

    guidance = self._call_llm("error_analysis", analysis_prompt)
    print(f"[{self._timestamp()}] 📋 Guidance generated:")
    print(f"[{self._timestamp()}] {guidance}")

    # Store guidance in message buffer for next iteration
    guidance_message = f"""\
Based on the previous attempt, here's guidance for improvement:

{guidance}

Please update your todo list and continue working on the solution, addressing these specific points.
"""

    state.latest_guidance = guidance_message
    state.error_message = ""  # Clear the error after providing guidance
    state.test_results = ""  # Clear previous test results

    return state

  def _format_todos_for_analysis(self, todos: list[dict]) -> str:
    """Format todo list for LLM analysis"""
    if not todos:
      return 'No todos available'

    formatted = []
    for todo in todos[-5:]:  # Last 5 todos
      status = todo.get('status', 'unknown')
      content = todo.get('content', 'Unknown task')
      formatted.append(f"- [{status.upper()}] {content}")

    return '\n'.join(formatted)

  def _format_output_log_for_analysis(self, output_log: list[str]) -> str:
    """Format output log for LLM analysis"""
    if not output_log:
      return 'No output available'

    # Get last few entries, excluding the initial prompt
    relevant_entries = [entry for entry in output_log[-3:] if not entry.startswith('PROMPT:')]
    if not relevant_entries:
      return 'No relevant output available'

    # Truncate long entries
    formatted = []
    for entry in relevant_entries:
      if len(entry) > 300:
        formatted.append(entry[:300] + '...')
      else:
        formatted.append(entry)

    return '\n'.join(formatted)

  def _call_llm(self, operation: str, prompt: str) -> str:
    """Wrapper for LLM calls for guidance analysis"""
    try:
      print(f"[{self._timestamp()}] 🤖 Calling LLM for {operation}...")
      messages = [HumanMessage(content=prompt)]
      response = self.llm.invoke(messages)
      content = response.content
      print(f"[{self._timestamp()}] ✅ LLM response received for {operation}")

      # Ensure we have string content
      if not isinstance(content, str):
        content = str(content)

      # Check if response is empty or error-like
      if not content.strip():
        return f"Empty response received for {operation}"

      return content
    except Exception as e:
      error_msg = f"Error in {operation}: {str(e)}"
      print(f"[{self._timestamp()}] ❌ {error_msg}")
      return error_msg

  def _run_tests(self, state: WorkflowState) -> WorkflowState:
    """Execute the tests"""
    print(f"\n[{self._timestamp()}] ▶️  Running tests...")

    # Check if test file exists and has content
    if not os.path.exists(self.test_path):
      state.test_results = f"Test file {self.test_path} does not exist"
      state.is_solved = False
      print(f"[{self._timestamp()}] ❌ Test file not found: {self.test_path}\n")
      return state

    # Check if solution file exists
    if not os.path.exists(self.solution_path):
      state.test_results = (f"Solution file {self.solution_path} "
                            'does not exist')
      state.is_solved = False
      print(f"[{self._timestamp()}] ❌ Solution file not found: {self.solution_path}\n")
      return state

    try:
      timeout = self.config.agent.test_timeout
      # First try to run a syntax check on both files
      for file_path, file_type in [(self.solution_path, 'solution'),
                                   (self.test_path, 'test')]:
        try:
          with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
          error_msg = f"Syntax error in {file_type} file {file_path}: {e}"
          print(f"[{self._timestamp()}] ❌ {error_msg}")
          state.test_results = error_msg
          state.is_solved = False
          return state

      # Run the tests
      result = subprocess.run(
        [sys.executable, '-m', 'pytest', self.test_path, '-v',
         '--tb=short', '--no-header'],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.getcwd()  # Ensure we're in the right directory
      )

      state.test_results = (f"Exit code: {result.returncode}\n"
                            f"STDOUT:\n{result.stdout}\n"
                            f"STDERR:\n{result.stderr}")
      state.is_solved = result.returncode == 0

      if state.is_solved:
        print(f"[{self._timestamp()}] ✅ {utils.green('All tests passed!')}")
        print(f"Test output:\n{result.stdout}")
      else:
        print(f"[{self._timestamp()}] ❌ {utils.red('Tests failed')} (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{utils.red(result.stderr)}")
        # Store error for guidance
        state.error_message = f"Test failures: {result.stdout}\n{result.stderr}"
      print()

    except subprocess.TimeoutExpired:
      timeout = self.config.agent.test_timeout
      state.test_results = f"Tests timed out after {timeout} seconds"
      state.is_solved = False
      state.error_message = f"Tests timed out after {timeout} seconds"
      print(f"[{self._timestamp()}] ⏰ Tests timed out after {timeout} seconds\n")
    except FileNotFoundError:
      state.test_results = ("pytest not found. "
                            'Please install pytest: pip install pytest')
      state.is_solved = False
      state.error_message = 'pytest not found'
      print(f"[{self._timestamp()}] ❌ pytest not found. Install with: pip install pytest\n")
    except Exception as e:
      state.test_results = f"Error running tests: {str(e)}"
      state.is_solved = False
      state.error_message = f"Error running tests: {str(e)}"
      print(f"[{self._timestamp()}] 💥 Error running tests: {str(e)}\n")

    return state


  def _finalize_solution(self, state: WorkflowState) -> WorkflowState:
    """Finalize the solution"""
    if state.is_solved:
      print(f"\n[{self._timestamp()}] 🎉 {utils.green(utils.bold('Solution completed successfully'))} after "
            f"{state.current_iteration} iterations!")
      
      if self.integrate_into_codebase:
        print(f"[{self._timestamp()}] 🔧 {utils.green('Solution integrated into existing codebase')}")
        print(f"[{self._timestamp()}] 🧪 {utils.green('Tests integrated into existing test structure')}")
      else:
        print(f"[{self._timestamp()}] 💾 {utils.cyan('Code saved to:')} {self.solution_path}")
        print(f"[{self._timestamp()}] 💾 {utils.cyan('Tests saved to:')} {self.test_path}")

      if state.output_data is not None:
        print(f"[{self._timestamp()}] 📊 {utils.cyan('Output data:')} {type(state.output_data).__name__}")
        if hasattr(state.output_data, '__len__') and len(state.output_data) < 20:
          print(f"[{self._timestamp()}] 📊 {utils.cyan('Result:')} {state.output_data}")

      if not self.integrate_into_codebase:
        print(f"\n[{self._timestamp()}] 🚀 {utils.yellow('You can run the tests manually with:')} "
              f"pytest {self.test_path}")
    else:
      # Check if this is a credit/quota error that caused early termination
      if state.should_terminate_early and state.error_message and "Credit/Quota Error" in state.error_message:
        utils.display_credit_quota_error(
          error_message=state.error_message,
          use_bedrock=self.config.claude_code.use_bedrock,
          current_iteration=state.current_iteration,
          claude_todos=state.claude_todos,
          claude_output_log=state.claude_output_log
        )
      else:
        print(f"\n[{self._timestamp()}] ❌ {utils.red('Maximum iterations')} ({self.config.agent.max_iterations}) {utils.red('reached without solving the problem.')}")
        print(f"[{self._timestamp()}] 📝 {utils.yellow('Last error:')} {utils.red(state.error_message)}")
      
      if not self.integrate_into_codebase:
        print(f"\n[{self._timestamp()}] 📁 Files generated (may contain partial solutions):")
        if os.path.exists(self.solution_path):
          print(f"[{self._timestamp()}]   - {self.solution_path}")
        if os.path.exists(self.test_path):
          print(f"[{self._timestamp()}]   - {self.test_path}")

    # No cleanup needed - all data operations are in-memory only

    return state

  def process(self, problem_description: str, *,
              input_data: Any = None,
              expected_output: Any = None,
              data_format: str = 'auto',
              example_output: str | None = None,
              solution_path: str | None = None,
              test_path: str | None = None) -> WorkflowState:
    """
    Main method to process a problem with optional input/output data.

    Args:
        problem_description: Description of the problem to solve
        input_data: Input data for Claude Code to work with (optional)
        expected_output: Expected output for validation (optional)
        data_format: Format hint for data handling ('auto', 'csv', 'json', etc.)
        example_output: Text description of expected output (optional)
        solution_path: Path to save solution file (optional, if None integrates into codebase)
        test_path: Path to save test file (optional, if None integrates into codebase)

    Returns:
        WorkflowState with results and any output data
    """
    # Set static configuration as instance attributes
    self.problem_description = problem_description
    self.example_output = example_output
    self.input_data = input_data
    self.expected_output = expected_output
    self.data_format = data_format
    self.solution_path = solution_path or self.config.agent.solution_filename
    self.test_path = test_path or self.config.agent.test_filename
    self.integrate_into_codebase = solution_path is None and test_path is None
    
    # Only create DataManager if I/O data is provided
    if input_data is not None or expected_output is not None:
      self.data_manager = DataManager()
      print(f"[{self._timestamp()}] 📁 DataManager created for I/O operations")
    else:
      self.data_manager = None
    
    # Create simplified initial state with only dynamic fields
    initial_state = WorkflowState()

    print(f"[{self._timestamp()}] 🚀 Starting problem solving: {problem_description}")
    try:
      final_state = self.graph.invoke(initial_state)

      # Ensure we return an WorkflowState object
      if isinstance(final_state, dict):
        # Convert dict back to WorkflowState if needed
        final_state = WorkflowState(**final_state)

      return final_state

    except Exception as e:
      print(f"[{self._timestamp()}] 💥 Error during execution: {e}")
      initial_state.error_message = str(e)
      return initial_state

  @classmethod
  def cli_run(cls) -> None:
    """CLI interface"""
    if len(sys.argv) < 2:
      print("\n🤖 SupervisorAgent - Automated Code Generation and Testing")
      print("=" * 60)
      print("Usage: python supervisor.py '<problem_description>' [example_output] [--prompt='custom_prompt']")
      print("\nExamples:")
      print("  python supervisor.py 'Create a function to sort a list of numbers'")
      print("  python supervisor.py 'Calculate fibonacci numbers' 'fib(8) = 21'")
      print("  python supervisor.py 'Find the maximum element in a list' "
            "'max([1,5,3]) = 5'")
      print("  python supervisor.py 'Create a calculator' --prompt='Use object-oriented design'")
      print("\n📝 Configuration is loaded from supervisor_config.json")
      print("🔑 Make sure to set OPENAI_API_KEY environment variable")
      print("📦 Required: pip install pytest langchain langchain-openai "
            'langgraph')
      sys.exit(1)

    # Parse arguments
    problem_description = sys.argv[1]
    example_output = None
    custom_prompt = None

    # Parse remaining arguments
    for i in range(2, len(sys.argv)):
      arg = sys.argv[i]
      if arg.startswith('--prompt='):
        custom_prompt = arg.split('--prompt=', 1)[1]
      elif not example_output and not arg.startswith('--'):
        example_output = arg

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
      print("❌ Error: OPENAI_API_KEY environment variable not set")
      print("\nSet it with: export OPENAI_API_KEY='your-api-key-here'")
      sys.exit(1)

    # Check for pytest
    try:
      subprocess.run([sys.executable, '-m', 'pytest', '--version'],
                     capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
      print("⚠️ Warning: pytest not found. Installing pytest...")
      try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest'],
                       check=True)
        print("✅ pytest installed successfully")
      except subprocess.CalledProcessError:
        print("❌ Failed to install pytest. Please install manually: "
              'pip install pytest')
        sys.exit(1)

    try:
      print(f"\n🎯 Problem: {problem_description}")
      if example_output:
        print(f"📝 Example: {example_output}")
      if custom_prompt:
        print(f"💡 Custom prompt: {custom_prompt}")
      print("\n" + '=' * 60)

      agent = cls(custom_prompt=custom_prompt)
      final_state = agent.process(problem_description, example_output=example_output)

      print("\n" + '=' * 60)
      if final_state.is_solved:
        print("🎉 SUCCESS: Problem solved!")
      else:
        print("❌ INCOMPLETE: Problem not fully solved within iteration limit")
        print("\nYou can manually review and fix the files:")
        print(f"  - {agent.solution_path}")
        print(f"  - {agent.test_path}")

    except KeyboardInterrupt:
      print("\n\n⏹️ Interrupted by user")
      sys.exit(1)
    except Exception as e:
      print(f"\n💥 Unexpected error: {e}")
      import traceback
      traceback.print_exc()
      sys.exit(1)
