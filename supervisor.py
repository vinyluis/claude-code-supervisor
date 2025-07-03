#!/usr/bin/env python3
"""
Supervisor Agent for Code Generation and Testing
Uses LangGraph to orchestrate code generation, testing, and iteration.
"""

import os
import sys
import json
import subprocess
import asyncio
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime

try:
  from langgraph.graph import StateGraph, START, END
  from langchain_core.messages import HumanMessage
  from langchain_openai import ChatOpenAI
  from claude_code_sdk import query, ClaudeCodeOptions
  from claude_code_sdk.types import (
    AssistantMessage, 
    TextBlock, 
    ToolUseBlock, 
    ToolResultBlock, 
    ResultMessage, 
    SystemMessage
  )
  from dotenv import load_dotenv
except ImportError as e:
  print(f"Missing required dependencies: {e}")
  print("Install with: pip install langgraph langchain langchain-openai claude-code-sdk python-dotenv")
  sys.exit(1)


@dataclass
class AgentState:
  """State for the supervisor agent"""
  problem_description: str
  example_output: Optional[str] = None
  current_iteration: int = 0
  max_iterations: int = 5
  code_content: str = ""
  test_content: str = ""
  test_results: str = ""
  solution_path: str = ""
  test_path: str = ""
  config: Optional[Dict] = None
  is_solved: bool = False
  error_message: str = ""
  messages: Optional[List[Any]] = None
  # New fields for session tracking
  claude_session_id: Optional[str] = None
  claude_session_active: bool = False
  claude_todos: List[Dict] = field(default_factory=list)
  claude_output_log: List[str] = field(default_factory=list)
  guidance_provided: bool = False
  last_activity_time: float = 0.0

  def __post_init__(self):
    if self.messages is None:
      self.messages = []
    self.last_activity_time = time.time()


class SupervisorAgent:
  """Supervisor agent that orchestrates code generation and testing"""

  def __init__(self, config_path: str = "supervisor_config.json"):
    self._load_environment()
    self.config = self._load_config(config_path)
    self.llm = self._initialize_llm()
    self.claude_code_config = self._initialize_claude_code()
    self.graph = self._build_graph()
  
  def _timestamp(self) -> str:
    """Get current timestamp for logging"""
    return datetime.now().strftime('%H:%M:%S')

  def _load_environment(self):
    """
    Load environment variables from .env file
    """
    env_path = Path('.env')
    if env_path.exists():
      load_dotenv(env_path)
    else:
      print("Warning: .env file not found. Environment variables may need to be set manually.")

  def _load_config(self, config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    Useful for customizing model parameters and agent behavior without changing the .py file.
    """
    try:
      with open(config_path, 'r') as f:
        return json.load(f)
    except FileNotFoundError:
      print(f"Config file {config_path} not found. Using defaults.")
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
      print(f"Error parsing config file: {e}")
      sys.exit(1)

  def _initialize_llm(self):
    """Initialize the OpenAI LLM"""
    return ChatOpenAI(
      model=self.config["model"]["name"],
      temperature=self.config["model"]["temperature"]
    )

  def _initialize_claude_code(self) -> Dict:
    """
    Initialize Claude Code SDK configuration
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

    # Return configuration for potential future use
    return {
      "use_bedrock": claude_config.get("use_bedrock", False),
      "working_directory": claude_config.get("working_directory"),
      "javascript_runtime": claude_config.get("javascript_runtime", "node"),
      "executable_args": claude_config.get("executable_args", []),
      "claude_code_path": claude_config.get("claude_code_path")
    }

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

  def _initiate_claude_session(self, state: AgentState) -> AgentState:
    """Initiate a Claude Code session for the problem"""
    print(f"\n[{self._timestamp()}] 🚀 Initiating Claude Code session...")
    
    # Prepare the problem statement for Claude Code
    problem_prompt = f"""\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works

Problem: {state.problem_description}
{f'Expected behavior: {state.example_output}' if state.example_output else ''}

Requirements:
- Use Python
- Save the solution as '{state.solution_path}'
- Save tests as '{state.test_path}'
- Follow clean code practices with docstrings and type hints
- Ensure all tests pass before completing

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
        # First iteration - send the initial problem
        problem_prompt = f"""\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works

Problem: {state.problem_description}
{f'Expected behavior: {state.example_output}' if state.example_output else ''}

Requirements:
- Use Python
- Save the solution as '{state.solution_path}'
- Save tests as '{state.test_path}'
- Follow clean code practices with docstrings and type hints
- Ensure all tests pass before completing

Please start by creating a todo list to plan your approach, then implement the solution.
"""
      else:
        # Subsequent iterations - provide guidance or continue session
        problem_prompt = f"""\
Continue working on the problem. {state.error_message if state.error_message else ''}

Please update your todo list and continue with the implementation.
"""
      
      # Configure Claude Code options for this session
      claude_config = self.config.get('claude_code', {})
      options = ClaudeCodeOptions(
        cwd=os.getcwd(),
        permission_mode='acceptEdits',
        max_turns=claude_config.get('max_turns', 20),
        continue_conversation=state.current_iteration > 0,
        resume=state.claude_session_id if state.claude_session_id else None,
        system_prompt="You are an expert Python developer. Use the TodoWrite tool to plan and track your work. Always run tests to verify your solutions.",
        max_thinking_tokens=claude_config.get('max_thinking_tokens', 8000)
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
          
          async for message in query(prompt=problem_prompt, options=options):
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
        loop.close()
        
      # Mark session as inactive after completion or timeout
      state.claude_session_active = not session_complete
      
      # Update state with results
      state.claude_todos = current_todos
      state.claude_output_log.extend(text_responses)
      
      # Check if files were created
      if os.path.exists(state.solution_path):
        with open(state.solution_path, 'r') as f:
          state.code_content = f.read()
        print(f"[{self._timestamp()}] 📄 Solution file detected: {state.solution_path}")
        
      if os.path.exists(state.test_path):
        with open(state.test_path, 'r') as f:
          state.test_content = f.read()
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
    return self._run_tests(state)

  def _provide_guidance(self, state: AgentState) -> AgentState:
    """Provide guidance to Claude Code when it encounters issues"""
    print(f"\n[{self._timestamp()}] 🎯 Providing guidance to Claude Code...")
    
    # Analyze what went wrong
    # The guidance will be provided in the next monitoring cycle
    state.guidance_provided = True
    state.error_message = ""  # Clear the error after providing guidance
    
    return state
  
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

    return state


  def _clean_code_content(self, content: str) -> str:
    """Clean code content by removing markdown formatting and whitespace"""
    # Remove markdown code blocks
    if "```python" in content:
      content = content.split("```python", 1)[1]
      if "```" in content:
        content = content.split("```", 1)[0]
    elif "```" in content:
      # Handle generic code blocks
      parts = content.split("```")
      if len(parts) >= 3:
        content = parts[1]

    # Remove leading/trailing whitespace
    content = content.strip()

    # Ensure the content is not empty
    if not content:
      return "# Empty code generated - this is likely an error\npass"

    return content

  def _call_llm(self, operation: str, prompt: str) -> str:
    """
    Wrapper for LLM calls for intermediate processing steps
    Uses OpenAI via LangChain for faster responses
    """
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


  def solve_problem(self, problem_description: str,
                    example_output: Optional[str] = None) -> AgentState:
    """Main method to solve a problem"""
    initial_state = AgentState(
      problem_description=problem_description,
      example_output=example_output,
      max_iterations=self.config["agent"]["max_iterations"],
      solution_path=self.config["agent"]["solution_filename"],
      test_path=self.config["agent"]["test_filename"],
      config=self.config,
      messages=[]
    )

    print(f"[{self._timestamp()}] 🚀 Starting problem solving: {problem_description}")
    try:
      final_state = self.graph.invoke(initial_state)
      # Ensure we return an AgentState object
      if isinstance(final_state, dict):
        # Convert dict back to AgentState if needed
        return AgentState(**final_state)
      return final_state
    except Exception as e:
      print(f"[{self._timestamp()}] 💥 Error during execution: {e}")
      # Return the current state with error info
      initial_state.error_message = str(e)
      return initial_state


def run():
  """CLI interface"""
  if len(sys.argv) < 2:
    print("\n🤖 SupervisorAgent - Automated Code Generation and Testing")
    print("=" * 60)
    print("Usage: python agents.py '<problem_description>' [example_output]")
    print("\nExamples:")
    print("  python agents.py 'Create a function to sort a list of numbers'")
    print("  python agents.py 'Calculate fibonacci numbers' 'fib(8) = 21'")
    print("  python agents.py 'Find the maximum element in a list' "
          "'max([1,5,3]) = 5'")
    print("\n📝 Configuration is loaded from config.json")
    print("🔑 Make sure to set OPENAI_API_KEY environment variable")
    print("📦 Required: pip install pytest langchain langchain-openai "
          "langgraph")
    sys.exit(1)

  problem_description = sys.argv[1]
  example_output = sys.argv[2] if len(sys.argv) > 2 else None

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
    print("\n" + "=" * 60)

    agent = SupervisorAgent()
    final_state = agent.solve_problem(problem_description, example_output)

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
  run()
