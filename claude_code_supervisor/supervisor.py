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
2. monitor_claude: Track progress via SDK message streaming and todo updates
3. validate_solution: Run tests on generated files to verify correctness
4. provide_guidance: Analyze failures and generate actionable feedback
5. finalize: Complete the session and report results

Usage:
  agent = SupervisorAgent('config.json')
  result = agent.process('Create a sorting function', example_output='sort([3,1,2]) -> [1,2,3]')
"""

import os
import sys
import json
import subprocess
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime
from typing import Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import (
  AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage,
  SystemMessage
)
from dotenv import load_dotenv
from .data_manager import DataManager


@dataclass
class AgentState:
  """State for the supervisor agent"""
  problem_description: str
  example_output: str | None = None
  current_iteration: int = 0
  max_iterations: int = 5
  test_results: str = ""
  solution_path: str = ""
  test_path: str = ""
  config: dict | None = None
  is_solved: bool = False
  error_message: str = ""
  guidance_messages: list = field(default_factory=list)
  # Session tracking fields
  claude_session_id: str | None = None
  claude_session_active: bool = False
  claude_todos: list[dict] = field(default_factory=list)
  claude_output_log: list[str] = field(default_factory=list)
  guidance_provided: bool = False
  last_activity_time: float = 0.0
  # Data I/O fields
  input_data: Any = None
  expected_output: Any = None
  data_format: str = "auto"
  output_data: Any = None
  data_manager: DataManager | None = None

  def __post_init__(self) -> None:
    self.last_activity_time = time.time()


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
  The agent is configured via JSON file containing:
  - model: LLM settings for guidance analysis (name, temperature)
  - agent: Iteration limits, file names, test timeouts
  - claude_code: SDK options, timeouts, provider settings

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
      >>> agent = SupervisorAgent('supervisor_config.json')
      >>> result = agent.process(
      >>>     'Create a function to calculate fibonacci numbers',
      >>>     example_output='fib(8) should return 21'
      >>> )
      >>>
      >>> # With custom prompt
      >>> agent = SupervisorAgent('supervisor_config.json',
      >>>                        custom_prompt='Use object-oriented design')
      >>> result = agent.process('Create a calculator')
      >>>
      >>> # Check if solved
      >>> if result.is_solved:
      >>>     print(f'Solution: {result.solution_path}')
      >>>     print(f'Tests: {result.test_path}')
      >>> else:
      >>>     print(f'Error: {result.error_message}')

  Attributes:
      config: Configuration dictionary loaded from JSON file
      custom_prompt: Optional additional instructions for Claude Code
      llm: OpenAI ChatLLM instance for guidance generation
      graph: LangGraph workflow compiled from node definitions
      base_claude_options: Pre-configured ClaudeCodeOptions for faster iterations
  """

  def __init__(self, config_path: str = "supervisor_config.json", custom_prompt: str | None = None) -> None:
    self._load_environment()
    self.config = self._load_config(config_path)
    self.custom_prompt = custom_prompt
    self.llm = self._initialize_llm()
    self._initialize_claude_code()
    self.graph = self._build_graph()

  def _timestamp(self) -> str:
    """Get current timestamp for logging"""
    return datetime.now().strftime('%H:%M:%S')

  def _load_environment(self) -> None:
    """
    Load environment variables from .env file
    """
    env_path = Path('.env')
    if env_path.exists():
      load_dotenv(env_path)
    else:
      print("⚠️Warning: .env file not found. Environment variables may need to be set manually.")

  def _load_config(self, config_path: str) -> dict:
    """
    Load configuration from JSON file.
    Useful for customizing model parameters and agent behavior without changing the .py file.
    """
    try:
      with open(config_path, 'r') as f:
        return json.load(f)
    except FileNotFoundError:
      print(f"⚠️Config file {config_path} not found. Using defaults.")
      return {
        "model": {
          "name": "gpt-4o",
          "temperature": 0.1
        },
        "agent": {
          "max_iterations": 3,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        },
        "claude_code": {
          "provider": "anthropic",
          "use_bedrock": False,
          "working_directory": None,
          "javascript_runtime": "node",
          "executable_args": [],
          "claude_code_path": None,
          "session_timeout_seconds": 300,
          "activity_timeout_seconds": 180,
          "max_turns": 20,
          "max_thinking_tokens": 8000
        }
      }
    except json.JSONDecodeError as e:
      print(f"⚠️Error parsing config file: {e}")
      sys.exit(1)

  def _initialize_claude_code(self) -> None:
    """
    Initialize Claude Code SDK configuration and prepare base options
    """
    claude_config = self.config.get("claude_code", {})

    # Set environment variables based on provider choice
    if claude_config.get("use_bedrock", False):
      os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
      print(f"[{self._timestamp()}] 🔧 Configured Claude Code to use Amazon Bedrock")
    else:
      # Default to Anthropic API
      if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"[{self._timestamp()}] Warning: ANTHROPIC_API_KEY not found in environment. Claude Code SDK may not work properly.")
        print(f"[{self._timestamp()}] Please set your API key in the .env file or environment variables.")
      else:
        print(f"[{self._timestamp()}] 🔧 Configured Claude Code to use Anthropic API")

    # Build system prompt with optional custom prompt
    base_system_prompt = "You are an expert Python developer. Use the TodoWrite tool to plan and track your work. Always run tests to verify your solutions."
    if self.custom_prompt:
      system_prompt = f"{base_system_prompt}\n\nAdditional instructions:\n{self.custom_prompt}"
    else:
      system_prompt = base_system_prompt

    # Pre-configure Claude Code options for faster reuse in iterations
    self.base_claude_options = ClaudeCodeOptions(
      cwd=os.getcwd(),
      permission_mode='acceptEdits',
      max_turns=claude_config.get('max_turns', 20),
      system_prompt=system_prompt,
      max_thinking_tokens=claude_config.get('max_thinking_tokens', 8000)
    )
    print(f"[{self._timestamp()}] 🔧 Pre-configured Claude Code options for faster iterations")

  def _build_graph(self):
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    workflow.add_node("initiate_claude", self._initiate_claude_session)
    workflow.add_node("monitor_claude", self._monitor_claude_progress)
    workflow.add_node("validate_solution", self._validate_solution)
    workflow.add_node("provide_guidance", self._provide_guidance)
    workflow.add_node("finalize", self._finalize_solution)

    workflow.add_edge(START, "initiate_claude")
    workflow.add_edge("initiate_claude", "monitor_claude")

    workflow.add_conditional_edges(
      "monitor_claude",
      self._should_continue_monitoring,
      {
        "continue": "monitor_claude",
        "validate": "validate_solution",
        "guide": "provide_guidance",
        "finish": "finalize"
      }
    )

    workflow.add_edge("validate_solution", "finalize")
    workflow.add_edge("provide_guidance", "monitor_claude")
    workflow.add_edge("finalize", END)

    return workflow.compile()

  def _initialize_llm(self):
    """Initialize the OpenAI LLM for guidance analysis"""
    return ChatOpenAI(
      model=self.config["model"]["name"],
      temperature=self.config["model"]["temperature"]
    )

  def _initiate_claude_session(self, state: AgentState) -> AgentState:
    """Initiate a Claude Code session for the problem"""
    print(f"\n[{self._timestamp()}] 🚀 Initiating Claude Code session...")

    # Handle input data if provided
    input_data_info = ""
    if state.input_data is not None and state.data_manager is not None:
      try:
        print(f"[{self._timestamp()}] 📊 Processing input data...")

        # Get data description and format
        data_format = state.data_format if state.data_format != 'auto' else state.data_manager.infer_format(state.input_data)
        data_description = state.data_manager.get_data_description(state.input_data, data_format)
        data_context = state.data_manager.serialize_for_context(state.input_data, data_format)

        # Record the operation
        state.data_manager.record_operation(state.input_data, data_format, 'input')

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
    if state.expected_output is not None:
      expected_output_info = f"\nExpected output format: {type(state.expected_output).__name__}"
      if hasattr(state.expected_output, '__len__') and len(state.expected_output) < 10:
        expected_output_info += f"\nExpected result example: {state.expected_output}"

    problem_prompt = f"""\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works

Problem: {state.problem_description}
{f'Expected behavior: {state.example_output}' if state.example_output else ''}
{expected_output_info}
{input_data_info}

Requirements:
- Use Python
- Save the solution as '{state.solution_path}'
- Save tests as '{state.test_path}'
- Follow clean code practices with docstrings and type hints
- Ensure all tests pass before completing
{'- If input data is provided, make sure to read and process it correctly' if state.input_data is not None else ''}
{'- Return results in the same format as the expected output' if state.expected_output is not None else ''}

Please start by creating a todo list to plan your approach, then implement the solution.
"""

    # Start the Claude Code session
    try:
      state.claude_session_active = True
      state.last_activity_time = time.time()
      print(f"[{self._timestamp()}] 📝 Sending problem to Claude Code...")
      print(f"[{self._timestamp()}] Working directory: {os.getcwd()}")

      # Store the initial prompt
      state.claude_output_log.append(f"PROMPT: {problem_prompt}")

      return state
    except Exception as e:
      print(f"[{self._timestamp()}] ❌ Failed to initiate Claude session: {e}")
      state.error_message = f"Failed to initiate Claude session: {e}"
      state.claude_session_active = False
      return state

  def _monitor_claude_progress(self, state: AgentState) -> AgentState:
    """Monitor Claude Code's progress and track its activities"""
    if not state.claude_session_active:
      return state

    print(f"\n[{self._timestamp()}] 👁️  Monitoring Claude Code progress...")

    try:
      # Build the prompt for continuation or initial request
      if state.current_iteration == 0:
        # First iteration - send the initial problem with data info
        input_data_info = ""
        if state.input_data and state.data_manager:
          input_data_info = f"""

Input Data Available:
{state.data_manager.serialize_for_context(state.input_data, state.data_format)}
- Format: {state.data_format}
- Description: {state.data_manager.get_data_description(state.input_data, state.data_format)}

Make sure to read and use this input data in your solution.
"""

        expected_output_info = ""
        if state.expected_output is not None:
          expected_output_info = f"\nExpected output format: {type(state.expected_output).__name__}"
          if hasattr(state.expected_output, '__len__') and len(state.expected_output) < 10:
            expected_output_info += f"\nExpected result example: {state.expected_output}"

        problem_prompt = f"""\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works

Problem: {state.problem_description}
{f'Expected behavior: {state.example_output}' if state.example_output else ''}
{expected_output_info}
{input_data_info}

Requirements:
- Use Python
- Save the solution as '{state.solution_path}'
- Save tests as '{state.test_path}'
- Follow clean code practices with docstrings and type hints
- Ensure all tests pass before completing
{'- If input data is provided, make sure to read and process it correctly' if state.input_data is not None else ''}
{'- Return results in the same format as the expected output' if state.expected_output is not None else ''}

Please start by creating a todo list to plan your approach, then implement the solution.
"""
      else:
        # Subsequent iterations - provide guidance or continue session
        latest_guidance = ""
        if state.guidance_messages:
          latest_guidance = state.guidance_messages[-1]['guidance']

        problem_prompt = f"""\
{latest_guidance if latest_guidance else 'Continue working on the problem.'}

Please update your todo list and continue with the implementation.
"""

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

      # Process Claude Code messages with timeout
      session_complete = False
      current_todos = []
      text_responses = []
      tool_calls = []

      async def process_claude_session():
        nonlocal session_complete, current_todos, text_responses, tool_calls

        try:
          # Add timeout to prevent infinite waiting
          claude_config = self.config.get('claude_code', {})
          timeout_seconds = claude_config.get('session_timeout_seconds', 300)
          start_time = time.time()
          print(f"[{self._timestamp()}] Session timeout set to {timeout_seconds} seconds")

          query_stream = query(prompt=problem_prompt, options=options)
          try:
            async for message in query_stream:
              # Check timeout
              if time.time() - start_time > timeout_seconds:
                print(f"[{self._timestamp()}] ⏰ Claude session timed out after {timeout_seconds} seconds")
                break

              state.last_activity_time = time.time()

              if isinstance(message, AssistantMessage):
                for block in message.content:
                  if isinstance(block, TextBlock):
                    text_responses.append(block.text)
                    print(f"[{self._timestamp()}] 💬 Claude: {block.text[:200]}{'...' if len(block.text) > 200 else ''}")
                  elif isinstance(block, ToolUseBlock):
                    tool_calls.append(f"{block.name}: {block.input}")
                    print(f"[{self._timestamp()}] 🔧 Tool used: {block.name}")

                    # Track todo updates
                    if block.name == 'TodoWrite':
                      todos = block.input.get('todos', [])
                      current_todos = todos
                      print(f"[{self._timestamp()}] 📋 Todo list updated: {len(todos)} items")
                      for todo in todos:
                        status_emoji = {'pending': '⏳', 'in_progress': '🔄', 'completed': '✅'}.get(todo.get('status'), '❓')
                        print(f"[{self._timestamp()}]   {status_emoji} {todo.get('content', 'Unknown task')}")

                  elif isinstance(block, ToolResultBlock):
                    if block.is_error:
                      print(f"[{self._timestamp()}] ❌ Tool error: {block.content}")

              elif isinstance(message, ResultMessage):
                # Session completed
                state.claude_session_id = message.session_id
                session_complete = True
                print(f"[{self._timestamp()}] ✅ Claude session completed (ID: {message.session_id})")
                print(f"[{self._timestamp()}] Turns: {message.num_turns}, Duration: {message.duration_ms}ms")
                if message.total_cost_usd:
                  print(f"[{self._timestamp()}] Cost: ${message.total_cost_usd:.4f}")
                break

              elif isinstance(message, SystemMessage):
                print(f"[{self._timestamp()}] ℹ️  System: {message.subtype}")

          except asyncio.CancelledError:
            # Handle cancellation gracefully
            print(f"[{self._timestamp()}] Claude session was cancelled")
            raise
          finally:
            # Ensure the async generator is properly closed
            if hasattr(query_stream, 'aclose'):
              try:
                await query_stream.aclose()
              except Exception:
                pass

        except Exception as e:
          print(f"[{self._timestamp()}] ❌ Error in Claude session: {e}")
          session_complete = True  # Mark as complete to exit
        finally:
          # Ensure proper cleanup
          session_complete = True

      # Run the async session
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(process_claude_session())
      finally:
        # Give tasks a moment to complete naturally before cleanup
        try:
          # Wait a brief moment for tasks to finish naturally
          loop.run_until_complete(asyncio.sleep(0.1))

          # Only cancel tasks that are still pending and not from external libraries
          pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
          if pending_tasks:
            # Cancel tasks more gently
            for task in pending_tasks:
              if not task.cancelled():
                task.cancel()

            # Give cancelled tasks time to handle cancellation
            try:
              loop.run_until_complete(asyncio.wait_for(
                asyncio.gather(*pending_tasks, return_exceptions=True),
                timeout=1.0
              ))
            except asyncio.TimeoutError:
              # If tasks don't respond to cancellation within 1 second, let them be
              pass
        except Exception:
          # If cleanup fails, continue anyway
          pass
        finally:
          loop.close()

      # Mark session as inactive after completion or timeout
      state.claude_session_active = not session_complete

      # Update state with results
      state.claude_todos = current_todos
      state.claude_output_log.extend(text_responses)

      # Check if files were created
      if os.path.exists(state.solution_path):
        print(f"[{self._timestamp()}] 📄 Solution file detected: {state.solution_path}")

      if os.path.exists(state.test_path):
        print(f"[{self._timestamp()}] 🧪 Test file detected: {state.test_path}")

      return state

    except Exception as e:
      print(f"[{self._timestamp()}] ❌ Error monitoring Claude session: {e}")
      state.error_message = f"Monitoring error: {e}"
      state.claude_session_active = False
      return state

  def _validate_solution(self, state: AgentState) -> AgentState:
    """Validate the solution created by Claude Code"""
    print(f"\n[{self._timestamp()}] 🔍 Validating Claude's solution...")

    # Check if both files exist
    if not os.path.exists(state.solution_path):
      state.error_message = f"Solution file {state.solution_path} not created"
      state.is_solved = False
      return state

    if not os.path.exists(state.test_path):
      state.error_message = f"Test file {state.test_path} not created"
      state.is_solved = False
      return state

    # Run the tests to validate
    state = self._run_tests(state)

    # If tests passed and we have input data, try to extract output data
    if state.is_solved and state.input_data is not None:
      state = self._extract_output_data(state)

    return state

  def _extract_output_data(self, state: AgentState) -> AgentState:
    """Extract output data by running the solution with input data"""
    print(f"\n[{self._timestamp()}] 📤 Extracting output data from solution...")

    try:
      # Import the solution module dynamically
      import importlib.util

      # Load the solution module
      spec = importlib.util.spec_from_file_location("solution_module", state.solution_path)
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
          if state.input_data is not None:
            # Pass the input data directly as argument
            result = main_function(state.input_data)
          else:
            # Call without arguments
            result = main_function()

          state.output_data = result
          print(f"[{self._timestamp()}] ✅ Output data extracted: {type(result).__name__}")

          # Validate against expected output if provided
          if state.expected_output is not None:
            if self._validate_output(result, state.expected_output):
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

  def _provide_guidance(self, state: AgentState) -> AgentState:
    """Provide guidance to Claude Code when it encounters issues"""
    print(f"\n[{self._timestamp()}] 🎯 Analyzing errors and providing guidance...")

    # Analyze the current situation
    analysis_prompt = f"""\
Analyze this Claude Code implementation failure and provide specific guidance for the next iteration.

Problem: {state.problem_description}
{f'Expected behavior: {state.example_output}' if state.example_output else ''}

Current Issues:
- Error: {state.error_message}
- Test Results: {state.test_results if state.test_results else 'No tests run yet'}
- Current Iteration: {state.current_iteration + 1}

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

    state.guidance_messages.append({
      'iteration': state.current_iteration,
      'guidance': guidance_message,
      'timestamp': self._timestamp()
    })

    state.guidance_provided = True
    state.error_message = ""  # Clear the error after providing guidance

    return state

  def _format_todos_for_analysis(self, todos: list[dict]) -> str:
    """Format todo list for LLM analysis"""
    if not todos:
      return "No todos available"

    formatted = []
    for todo in todos[-5:]:  # Last 5 todos
      status = todo.get('status', 'unknown')
      content = todo.get('content', 'Unknown task')
      formatted.append(f"- [{status.upper()}] {content}")

    return "\n".join(formatted)

  def _format_output_log_for_analysis(self, output_log: list[str]) -> str:
    """Format output log for LLM analysis"""
    if not output_log:
      return "No output available"

    # Get last few entries, excluding the initial prompt
    relevant_entries = [entry for entry in output_log[-3:] if not entry.startswith('PROMPT:')]
    if not relevant_entries:
      return "No relevant output available"

    # Truncate long entries
    formatted = []
    for entry in relevant_entries:
      if len(entry) > 300:
        formatted.append(entry[:300] + "...")
      else:
        formatted.append(entry)

    return "\n".join(formatted)

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

  def _run_tests(self, state: AgentState) -> AgentState:
    """Execute the tests"""
    print(f"\n[{self._timestamp()}] ▶️  Running tests...")

    # Check if test file exists and has content
    if not os.path.exists(state.test_path):
      state.test_results = f"Test file {state.test_path} does not exist"
      state.is_solved = False
      print(f"[{self._timestamp()}] ❌ Test file not found: {state.test_path}\n")
      return state

    # Check if solution file exists
    if not os.path.exists(state.solution_path):
      state.test_results = (f"Solution file {state.solution_path} "
                            "does not exist")
      state.is_solved = False
      print(f"[{self._timestamp()}] ❌ Solution file not found: {state.solution_path}\n")
      return state

    try:
      timeout = state.config["agent"]["test_timeout"] if state.config else 30

      # First try to run a syntax check on both files
      for file_path, file_type in [(state.solution_path, "solution"),
                                   (state.test_path, "test")]:
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
        [sys.executable, "-m", "pytest", state.test_path, "-v",
         "--tb=short", "--no-header"],
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
        print(f"[{self._timestamp()}] ✅ All tests passed!")
        print(f"Test output:\n{result.stdout}")
      else:
        print(f"[{self._timestamp()}] ❌ Tests failed (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{result.stderr}")
        # Store error for guidance
        state.error_message = f"Test failures: {result.stdout}\n{result.stderr}"
      print()

    except subprocess.TimeoutExpired:
      timeout = state.config["agent"]["test_timeout"] if state.config else 30
      state.test_results = f"Tests timed out after {timeout} seconds"
      state.is_solved = False
      state.error_message = f"Tests timed out after {timeout} seconds"
      print(f"[{self._timestamp()}] ⏰ Tests timed out after {timeout} seconds\n")
    except FileNotFoundError:
      state.test_results = ("pytest not found. "
                            "Please install pytest: pip install pytest")
      state.is_solved = False
      state.error_message = "pytest not found"
      print(f"[{self._timestamp()}] ❌ pytest not found. Install with: pip install pytest\n")
    except Exception as e:
      state.test_results = f"Error running tests: {str(e)}"
      state.is_solved = False
      state.error_message = f"Error running tests: {str(e)}"
      print(f"[{self._timestamp()}] 💥 Error running tests: {str(e)}\n")

    return state

  def _should_continue_monitoring(self, state: AgentState) -> str:
    """Decide what to do next based on Claude's progress"""
    # Check if we've exceeded maximum iterations
    if state.current_iteration >= state.max_iterations:
      return "finish"

    # If Claude session is not active, we need guidance
    if not state.claude_session_active and state.error_message:
      if not state.guidance_provided:
        state.current_iteration += 1
        return "guide"
      else:
        return "finish"

    # Check if solution files exist
    if os.path.exists(state.solution_path) and os.path.exists(state.test_path):
      return "validate"

    # Check for timeout (Claude has been working too long without progress)
    current_time = time.time()
    claude_config = self.config.get('claude_code', {})
    activity_timeout = claude_config.get('activity_timeout_seconds', 180)
    if current_time - state.last_activity_time > activity_timeout:
      state.error_message = f"Claude Code session timed out after {activity_timeout} seconds of inactivity"
      state.claude_session_active = False
      return "guide"

    # Continue monitoring if Claude is still active
    if state.claude_session_active:
      return "continue"

    # Default to finishing if we're in an unknown state
    return "finish"

  def _finalize_solution(self, state: AgentState) -> AgentState:
    """Finalize the solution"""
    if state.is_solved:
      print(f"\n[{self._timestamp()}] 🎉 Solution completed successfully after "
            f"{state.current_iteration} iterations!")
      print(f"[{self._timestamp()}] 💾 Code saved to: {state.solution_path}")
      print(f"[{self._timestamp()}] 💾 Tests saved to: {state.test_path}")

      if state.output_data is not None:
        print(f"[{self._timestamp()}] 📊 Output data: {type(state.output_data).__name__}")
        if hasattr(state.output_data, '__len__') and len(state.output_data) < 20:
          print(f"[{self._timestamp()}] 📊 Result: {state.output_data}")

      print(f"\n[{self._timestamp()}] 🚀 You can run the tests manually with: "
            f"pytest {state.test_path}")
    else:
      print(f"\n[{self._timestamp()}] ❌ Maximum iterations ({state.max_iterations}) reached "
            "without solving the problem.")
      print(f"[{self._timestamp()}] 📝 Last error: {state.error_message}")
      print(f"\n[{self._timestamp()}] 📁 Files generated (may contain partial solutions):")
      if os.path.exists(state.solution_path):
        print(f"[{self._timestamp()}]   - {state.solution_path}")
      if os.path.exists(state.test_path):
        print(f"[{self._timestamp()}]   - {state.test_path}")

    # No cleanup needed - all data operations are in-memory only

    return state

  def process(self, problem_description: str, *,
              input_data: Any = None,
              expected_output: Any = None,
              data_format: str = "auto",
              example_output: str | None = None) -> AgentState:
    """
    Main method to process a problem with optional input/output data.

    Args:
        problem_description: Description of the problem to solve
        input_data: Input data for Claude Code to work with (optional)
        expected_output: Expected output for validation (optional)
        data_format: Format hint for data handling ('auto', 'csv', 'json', etc.)
        example_output: Text description of expected output (optional)

    Returns:
        AgentState with results and any output data
    """
    # Only create DataManager if I/O data is provided
    data_manager = None
    if input_data is not None or expected_output is not None:
      data_manager = DataManager()
      print(f"[{self._timestamp()}] 📁 DataManager created for I/O operations")

    initial_state = AgentState(
      problem_description=problem_description,
      example_output=example_output,
      input_data=input_data,
      expected_output=expected_output,
      data_format=data_format,
      max_iterations=self.config["agent"]["max_iterations"],
      solution_path=self.config["agent"]["solution_filename"],
      test_path=self.config["agent"]["test_filename"],
      config=self.config,
      data_manager=data_manager
    )

    print(f"[{self._timestamp()}] 🚀 Starting problem solving: {problem_description}")
    try:
      final_state = self.graph.invoke(initial_state)

      # Ensure we return an AgentState object
      if isinstance(final_state, dict):
        # Convert dict back to AgentState if needed
        final_state = AgentState(**final_state)

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
            "langgraph")
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
      subprocess.run([sys.executable, "-m", "pytest", "--version"],
                     capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
      print("⚠️ Warning: pytest not found. Installing pytest...")
      try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"],
                       check=True)
        print("✅ pytest installed successfully")
      except subprocess.CalledProcessError:
        print("❌ Failed to install pytest. Please install manually: "
              "pip install pytest")
        sys.exit(1)

    try:
      print(f"\n🎯 Problem: {problem_description}")
      if example_output:
        print(f"📝 Example: {example_output}")
      if custom_prompt:
        print(f"💡 Custom prompt: {custom_prompt}")
      print("\n" + "=" * 60)

      agent = cls(custom_prompt=custom_prompt)
      final_state = agent.process(problem_description, example_output=example_output)

      print("\n" + "=" * 60)
      if final_state.is_solved:
        print("🎉 SUCCESS: Problem solved!")
      else:
        print("❌ INCOMPLETE: Problem not fully solved within iteration limit")
        print("\nYou can manually review and fix the files:")
        print(f"  - {final_state.solution_path}")
        print(f"  - {final_state.test_path}")

    except KeyboardInterrupt:
      print("\n\n⏹️ Interrupted by user")
      sys.exit(1)
    except Exception as e:
      print(f"\n💥 Unexpected error: {e}")
      import traceback
      traceback.print_exc()
      sys.exit(1)


if __name__ == "__main__":
  SupervisorAgent.cli_run()
