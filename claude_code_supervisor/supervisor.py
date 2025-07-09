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
from .config import SupervisorConfig, development_config
from . import utils, prompts

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
  messages: list["str"] = field(default_factory=list)
  solution_complete: bool = False
  validation_feedback: str = ""

  # Claude session state
  claude_session_id: str | None = None
  claude_session_active: bool = False
  claude_todos: list[dict] = field(default_factory=list)
  claude_log: list[str] = field(default_factory=list)
  should_terminate_early: bool = False

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
    llm: BaseLanguageModel | None = None,
    config: SupervisorConfig | None = None,
    append_system_prompt: str | None = None,
    **kwargs,
  ) -> None:

    self.load_environment()

    if config is not None and isinstance(config, SupervisorConfig):
      self.config = config
    else:
      utils.print_debug("Loading default configuration...")
      self.config = development_config()

    # Static workflow configuration (set once per process call)
    self.problem_description: str = ""
    self.example_output: str | None = None
    self.solution_path: str = ""
    self.test_path: str = ""
    self.input_data: utils.DataTypes | None = None
    self.output_data: utils.DataTypes | None = None
    self.data_format: str = 'auto'
    self.integrate_into_codebase: bool = False
    self.development_guidelines: str = prompts.development_guidelines()
    self.instruction_prompt: str = prompts.instruction_prompt()

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

    if llm is not None:
      self.llm = llm
    else:
      self.llm = self.initialize_llm()

    self.append_system_prompt = append_system_prompt
    self.initialize_claude_code()
    self.graph = self.build_graph()

  def load_environment(self) -> None:
    """
    Load environment variables from .env file
    """
    env_path = Path('.env')
    if env_path.exists():
      load_dotenv(env_path)
    else:
      utils.print_warning("Warning: .env file not found. Environment variables may need to be set manually.")

  def initialize_claude_code(self) -> None:
    """
    Initialize Claude Code SDK configuration and prepare base options
    """
    claude_config = self.config.claude_code
    # Set environment variables based on provider choice
    if claude_config.use_bedrock:
      os.environ["CLAUDE_CODE_USE_BEDROCK"] = '1'
      utils.print_debug("Configured Claude Code to use Amazon Bedrock")
    else:
      # Default to Anthropic API
      if not os.getenv("ANTHROPIC_API_KEY"):
        utils.print_warning("Warning: ANTHROPIC_API_KEY not found in environment. Claude Code SDK may not work properly.")
        utils.print_warning("Please set your API key in the .env file or environment variables.")
      else:
        utils.print_debug("Configured Claude Code to use Anthropic API")

    # Pre-configure Claude Code options for faster reuse in iterations
    self.base_claude_options = ClaudeCodeOptions(
      cwd=os.getcwd(),
      permission_mode='acceptEdits',
      max_turns=claude_config.max_turns,
      append_system_prompt=self.append_system_prompt,
      max_thinking_tokens=claude_config.max_thinking_tokens
    )
    utils.print_debug("Pre-configured Claude Code options")

  def build_graph(self):
    """Build the LangGraph workflow"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("initiate_claude", self.initiate_claude_code_session)
    workflow.add_node("execute_claude", self.execute_claude_session)
    workflow.add_node("collect_results", self.collect_session_results)
    workflow.add_node("validate_solution", self.validate_solution)
    workflow.add_node("provide_guidance", self._provide_guidance)
    workflow.add_node("finalize", self._finalize_solution)

    workflow.add_edge(START, 'initiate_claude')
    workflow.add_edge("initiate_claude", 'execute_claude')
    workflow.add_edge("execute_claude", 'collect_results')
    workflow.add_conditional_edges(
      'collect_results',
      self.decide_next_action,
      {
        'validate': 'validate_solution',
        'guide': 'provide_guidance',
        'finish': 'finalize'
      }
    )
    workflow.add_conditional_edges(
      ' validate_solution',
      self.should_iterate,
      {
        'continue': 'provide_guidance',
        ' finish': 'finalize',
      }
    )
    workflow.add_edge("provide_guidance", 'execute_claude')
    workflow.add_edge("finalize", END)

    return workflow.compile()

  def initialize_llm(self):
    """Initialize the LLM for guidance analysis (OpenAI or AWS Bedrock)"""
    model_config = self.config.model
    provider = model_config.provider

    if provider is None:
      raise ValueError("Model provider not specified in configuration. Please set 'provider' in the model config.")

    elif provider == 'bedrock':
      if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ValueError("AWS credentials not found in environment variables. Please set 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', and 'AWS_REGION'.")
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

  def initiate_claude_code_session(self, state: WorkflowState) -> WorkflowState:
    """Prepare the initial setup for Claude Code session"""
    utils.print_with_timestamp("\n🚀 Setting up Claude Code session...")
    utils.print_with_timestamp(f"Working directory: {os.getcwd()}")
    state.claude_session_active = True

    # Build the initial prompt and store on the message buffer
    claude_instructions = prompts.build_claude_instructions(
      instruction_prompt=self.instruction_prompt,
      problem_description=self.problem_description,
      development_guidelines=self.development_guidelines,
      solution_path=self.solution_path,
      test_path=self.test_path,
      input_data=self.input_data,
      output_data=self.output_data,
    )
    state.messages = [claude_instructions]
    return state

  async def _claude_run(self, claude_instructions: str, options: ClaudeCodeOptions,
                        state: WorkflowState) -> None:
    """Run claude code session asynchronously and retrieve the messages"""
    try:
      # Simple async iteration as recommended in SDK docs
      async for message in query(prompt=claude_instructions, options=options):

        if isinstance(message, AssistantMessage):
          for block in message.content:
            if isinstance(block, TextBlock):
              state.claude_log.append(block.text)
              utils.print_claude(block.text[:200] + ('...' if len(block.text) > 200 else ''))

            elif isinstance(block, ToolUseBlock):
              tool_info = utils.get_tool_info(block.name, block.input)
              utils.print_tool(f"{utils.blue('Tool:')} {utils.blue(block.name)} {tool_info}")

              # Track todo updates
              if block.name == 'TodoWrite':
                todos = block.input.get('todos', [])
                state.claude_todos = todos
                utils.print_todo(str(len(todos)) + ' items')
                for todo in todos:
                  status_emoji = {'pending': '⏳', 'in_progress': '🔄', 'completed': '✅'}.get(todo.get('status'), '❓')
                  utils.print_with_timestamp(f"  {status_emoji} {utils.blue(todo.get('content', 'Unknown task'))}")

            elif isinstance(block, ToolResultBlock):
              if block.is_error:
                error_msg = f"Tool error: {block.content}"
                utils.print_error(error_msg)
                if not state.error_message:
                  state.error_message = error_msg

        elif isinstance(message, ResultMessage):
          # Session completed successfully
          state.claude_session_id = message.session_id
          utils.print_success(f"Claude session completed (ID: {message.session_id})")
          utils.print_with_timestamp(f"{utils.cyan('Turns:')} {message.num_turns}, {utils.cyan('Duration:')} {message.duration_ms}ms")
          if message.total_cost_usd:
            utils.print_with_timestamp(f"{utils.cyan('Cost:')} ${message.total_cost_usd:.4f}")
          break

        elif isinstance(message, SystemMessage):
          utils.print_info(f"System: {message.subtype}")

    except Exception as e:
      # Let SDK handle its own errors - only handle real failures
      error_msg = f"Error in Claude session: {e}"
      utils.print_error(error_msg)
      if not state.error_message:
        state.error_message = error_msg
      raise

  def execute_claude_session(self, state: WorkflowState) -> WorkflowState:
    """Execute a Claude Code session until completion or timeout"""
    if not state.claude_session_active:
      state.error_message = "Claude session not active"
      return state

    utils.print_with_timestamp(f"\n🚀 Executing Claude Code session (iteration {state.current_iteration})...")

    # Get the prompt from the log
    claude_instructions = state.messages[-1]
    state.claude_log.append(f"PROMPT_ITERATION_{state.current_iteration}: {claude_instructions}")
    utils.print_with_timestamp(f"📝 Using prompt for iteration {state.current_iteration}:")
    utils.print_prompt(claude_instructions)

    # Prepare Claude Code options for this session
    options = ClaudeCodeOptions(
      cwd=self.base_claude_options.cwd,
      permission_mode=self.base_claude_options.permission_mode,
      max_turns=self.base_claude_options.max_turns,
      system_prompt=self.base_claude_options.system_prompt,
      max_thinking_tokens=self.base_claude_options.max_thinking_tokens,
      continue_conversation=state.current_iteration > 0,
      resume=state.claude_session_id if state.claude_session_id else None
    )

    # Execute Claude Code session - simplified approach
    try:
      import anyio
      anyio.run(self._claude_run, claude_instructions, options, state)
    except Exception as e:
      error_msg = f"Error in async session execution: {e}"
      utils.print_error(error_msg)
      # Let the SDK handle its own errors - only store real errors
      if "cancel scope" not in str(e).lower() and "task exception" not in str(e).lower():
        state.error_message = error_msg
        raise

    # Mark session as inactive after completion
    state.claude_session_active = False
    return state

  def collect_session_results(self, state: WorkflowState) -> WorkflowState:
    """Collect and analyze the results from Claude's session"""
    utils.print_with_timestamp("\n📊 Collecting session results...")

    # Check if files were created
    if self.solution_path is not None:
      if os.path.exists(self.solution_path):
        utils.print_with_timestamp(f"📄 Solution file detected: {self.solution_path}")
    if self.test_path is not None:
      if os.path.exists(self.test_path):
        utils.print_with_timestamp(f"🧪 Test file detected: {self.test_path}")

    # Analyze todo completion status
    completed_todos = [todo for todo in state.claude_todos if todo.get('status') == 'completed']
    todos_count = len(state.claude_todos)

    if todos_count > 0:
      completion_rate = len(completed_todos) / todos_count
      utils.print_with_timestamp(f"📋 Todo completion: {len(completed_todos)}/{todos_count} ({completion_rate:.1%})")

    # Check for error patterns in the output using utility functions
    general_errors, credit_quota_errors = utils.detect_errors_in_output(state.claude_log)
    error_message, should_terminate_early = utils.format_error_message(general_errors, credit_quota_errors)

    # Update state with error information
    if error_message:
      state.error_message = error_message
      state.should_terminate_early = should_terminate_early

      if should_terminate_early:
        utils.print_with_timestamp("🚫 Credit/quota error detected - terminating session")
      else:
        utils.print_warning("Detected error indicators in Claude's output")

    return state

  def decide_next_action(self, state: WorkflowState) -> str:
    """Decide what to do next based on session results"""
    utils.print_with_timestamp("\n🤔 Deciding next action...")

    # Check for early termination due to credit/quota errors
    if state.should_terminate_early:
      utils.print_with_timestamp("🚫 Early termination requested due to API errors")
      return 'finish'

    # Check if we've exceeded maximum iterations
    if state.current_iteration >= self.config.agent.max_iterations:
      utils.print_with_timestamp(f"🔄 Maximum iterations ({self.config.agent.max_iterations}) reached")
      return 'finish'

    # If there are explicit errors, provide guidance
    if state.error_message:
      utils.print_error("Errors detected, providing guidance")
      return 'guide'

    # Check if solution appears complete
    if self.integrate_into_codebase:
      # In integration mode, check if Claude indicates completion
      if not state.claude_session_active:
        utils.print_success("Integration mode: session complete, validating")
        return 'validate'
    else:
      # Standard mode: check if both files exist
      if os.path.exists(self.solution_path) and os.path.exists(self.test_path):
        utils.print_success("Both solution and test files exist, validating")
        return 'validate'

    # Default: validate if solution is already enough
    utils.print_with_timestamp("🤔 Solution unclear, starting validation step")
    return 'validate'

  def validate_solution(self, state: WorkflowState) -> WorkflowState:
    """Validate the solution created by Claude Code"""
    utils.print_with_timestamp("\n🔍 Validating Claude's solution...")

    if self.integrate_into_codebase:
      # In integration mode, we validate by running tests on the entire codebase
      utils.print_debug("Integration mode: validating codebase changes...")
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

  def should_iterate(self, state: WorkflowState) -> str:
    """If the solution is invalid, provide guidance and prepare for the next iteration"""
    if state.solution_complete:
      utils.print_with_timestamp("✅ Solution appears complete, finalizing")
      return 'finish'
    return 'continue'

  def _run_integration_tests(self, state: WorkflowState) -> WorkflowState:
    """Run tests in integration mode where solution is integrated into codebase"""
    utils.print_with_timestamp("\n🧪 Running integration tests...")

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
        utils.print_success("No separate test files found, assuming integration successful")
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
        utils.print_success("All integration tests passed!")
        print(f"Test output:\n{result.stdout}")
      else:
        utils.print_error(f"Integration tests failed (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{result.stderr}")
        state.error_message = f"Integration test failures: {result.stdout}\n{result.stderr}"
        
    except subprocess.TimeoutExpired:
      timeout = self.config.agent.test_timeout
      state.test_results = f"Integration tests timed out after {timeout} seconds"
      state.is_solved = False
      state.error_message = f"Integration tests timed out after {timeout} seconds"
      utils.print_with_timestamp(f"⏰ Integration tests timed out after {timeout} seconds")
    except FileNotFoundError:
      state.test_results = "pytest not found. Please install pytest: pip install pytest"
      state.is_solved = False
      state.error_message = "pytest not found"
      utils.print_error("pytest not found. Install with: pip install pytest")
    except Exception as e:
      state.test_results = f"Error running integration tests: {str(e)}"
      state.is_solved = False
      state.error_message = f"Error running integration tests: {str(e)}"
      utils.print_error(f"Error running integration tests: {str(e)}")
    
    return state

  def _extract_output_data(self, state: WorkflowState) -> WorkflowState:
    """Extract output data by running the solution with input data"""
    utils.print_with_timestamp("\n📤 Extracting output data from solution...")

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
        utils.print_debug(f"Found function: {main_function.__name__}")

        # Try to call the function with input data
        try:
          if self.input_data is not None:
            # Pass the input data directly as argument
            result = main_function(self.input_data)
          else:
            # Call without arguments
            result = main_function()

          state.output_data = result
          utils.print_success(f"Output data extracted: {type(result).__name__}")

          # Validate against expected output if provided
          if self.output_data is not None:
            if self._validate_output(result, self.output_data):
              utils.print_success("Output matches expected format!")
            else:
              utils.print_warning("Output format doesn't match expected format")

        except Exception as e:
          utils.print_error(f"Failed to execute function: {e}")
          state.error_message = f"Failed to execute solution function: {e}"

      else:
        utils.print_warning("No callable main function found in solution")

    except Exception as e:
      utils.print_error(f"Failed to extract output data: {e}")
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
    utils.print_with_timestamp("\n🎯 Analyzing errors and providing guidance...")

    # Increment iteration counter
    state.current_iteration += 1
    utils.print_with_timestamp(f"🔄 Starting iteration {state.current_iteration}")

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
{self._format_output_log_for_analysis(state.claude_log)}

Provide specific, actionable guidance for Claude Code to fix these issues:
1. What went wrong?
2. What specific steps should Claude take next?
3. What should Claude focus on or avoid?

Keep your response concise and actionable (2-3 bullet points).
"""

    guidance = self._call_llm("error_analysis", analysis_prompt)
    utils.print_with_timestamp("📋 Guidance generated:")
    utils.print_with_timestamp(guidance)

    # Store guidance in message buffer for next iteration
    guidance_message = f"""\
Based on the previous attempt, here's guidance for improvement:

{guidance}

Please update your todo list and continue working on the solution, addressing these specific points.
"""

    state.messages.append(guidance_message)
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
      utils.print_with_timestamp(f"🤖 Calling LLM for {operation}...")
      messages = [HumanMessage(content=prompt)]
      response = self.llm.invoke(messages)
      content = response.content
      utils.print_success(f"LLM response received for {operation}")

      # Ensure we have string content
      if not isinstance(content, str):
        content = str(content)

      # Check if response is empty or error-like
      if not content.strip():
        return f"Empty response received for {operation}"

      return content
    except Exception as e:
      error_msg = f"Error in {operation}: {str(e)}"
      utils.print_error(error_msg)
      return error_msg

  def _run_tests(self, state: WorkflowState) -> WorkflowState:
    """Execute the tests"""
    utils.print_with_timestamp("\n▶️  Running tests...")

    # Check if test file exists and has content
    if not os.path.exists(self.test_path):
      state.test_results = f"Test file {self.test_path} does not exist"
      state.is_solved = False
      utils.print_error(f"Test file not found: {self.test_path}\n")
      return state

    # Check if solution file exists
    if not os.path.exists(self.solution_path):
      state.test_results = (f"Solution file {self.solution_path} "
                            'does not exist')
      state.is_solved = False
      utils.print_error(f"Solution file not found: {self.solution_path}\n")
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
          utils.print_error(error_msg)
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
        utils.print_success(utils.green('All tests passed!'))
        print(f"Test output:\n{result.stdout}")
      else:
        utils.print_error(f"{utils.red('Tests failed')} (exit code: {result.returncode})")
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
      utils.print_with_timestamp(f"⏰ Tests timed out after {timeout} seconds\n")
    except FileNotFoundError:
      state.test_results = ("pytest not found. "
                            'Please install pytest: pip install pytest')
      state.is_solved = False
      state.error_message = 'pytest not found'
      utils.print_error("pytest not found. Install with: pip install pytest\n")
    except Exception as e:
      state.test_results = f"Error running tests: {str(e)}"
      state.is_solved = False
      state.error_message = f"Error running tests: {str(e)}"
      utils.print_error(f"Error running tests: {str(e)}\n")

    return state

  def _finalize_solution(self, state: WorkflowState) -> WorkflowState:
    """Finalize the solution"""
    if state.is_solved:
      utils.print_with_timestamp(f"\n🎉 {utils.green(utils.bold('Solution completed successfully'))} after "
            f"{state.current_iteration} iterations!")
      
      if self.integrate_into_codebase:
        utils.print_debug(utils.green('Solution integrated into existing codebase'))
        utils.print_debug(utils.green('Tests integrated into existing test structure'))
      else:
        utils.print_with_timestamp(f"💾 {utils.cyan('Code saved to:')} {self.solution_path}")
        utils.print_with_timestamp(f"💾 {utils.cyan('Tests saved to:')} {self.test_path}")

      if state.output_data is not None:
        utils.print_with_timestamp(f"📊 {utils.cyan('Output data:')} {type(state.output_data).__name__}")
        if hasattr(state.output_data, '__len__') and len(state.output_data) < 20:
          utils.print_with_timestamp(f"📊 {utils.cyan('Result:')} {state.output_data}")

      if not self.integrate_into_codebase:
        utils.print_with_timestamp(f"\n🚀 {utils.yellow('You can run the tests manually with:')} "
              f"pytest {self.test_path}")
    else:
      # Check if this is a credit/quota error that caused early termination
      if state.should_terminate_early and state.error_message and "Credit/Quota Error" in state.error_message:
        utils.display_credit_quota_error(
          error_message=state.error_message,
          use_bedrock=self.config.claude_code.use_bedrock,
          current_iteration=state.current_iteration,
          claude_todos=state.claude_todos,
          claude_log=state.claude_log
        )
      else:
        utils.print_with_timestamp(f"\n❌ {utils.red('Maximum iterations')} ({self.config.agent.max_iterations}) {utils.red('reached without solving the problem.')}")
        utils.print_with_timestamp(f"📝 {utils.yellow('Last error:')} {utils.red(state.error_message)}")
      
      if not self.integrate_into_codebase:
        utils.print_with_timestamp("\n📁 Files generated (may contain partial solutions):")
        if os.path.exists(self.solution_path):
          utils.print_with_timestamp(f"  - {self.solution_path}")
        if os.path.exists(self.test_path):
          utils.print_with_timestamp(f"  - {self.test_path}")

    # No cleanup needed - all data operations are in-memory only

    return state

  def process(
      self,
      problem_description: str, *,
      input_data: Any = None,
      output_data: Any = None,
      solution_path: str | None = None,
      test_path: str | None = None,
      **kwargs: Any,
    ) -> WorkflowState:
    """
    Main method to process a problem with optional input/output data.

    Args:
      problem_description: Description of the problem to solve
      input_data: Input data for Claude Code to work with (optional)
      output_data: Expected output for validation (optional)
      solution_path: Path to save solution file (optional, if None integrates into codebase)
      test_path: Path to save test file (optional, if None integrates into codebase)

    Kwargs:
      development_guidelines: Custom development guidelines for Claude Code
      instruction_prompt: Custom instruction prompt for Claude Code

    Returns:
      WorkflowState with results and any output data
    """
    # Set static configuration as instance attributes
    self.problem_description = problem_description
    self.input_data = input_data
    self.output_data = output_data
    self.solution_path = solution_path or self.config.agent.solution_filename
    self.test_path = test_path or self.config.agent.test_filename
    self.integrate_into_codebase = solution_path is None and test_path is None

    # Prompt overrides
    self.development_guidelines = kwargs.get('development_guidelines') or self.development_guidelines
    self.instruction_prompt = kwargs.get('instruction_prompt') or self.instruction_prompt

    # Create simplified initial state with only dynamic fields
    initial_state = WorkflowState()

    utils.print_with_timestamp(f"🚀 Starting problem solving: {problem_description}")
    try:
      final_state = self.graph.invoke(initial_state)

      # Ensure we return an WorkflowState object
      if isinstance(final_state, dict):
        # Convert dict back to WorkflowState if needed
        final_state = WorkflowState(**final_state)

      return final_state

    except Exception as e:
      utils.print_error(f"Error during execution: {e}")
      initial_state.error_message = str(e)
      return initial_state
