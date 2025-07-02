#!/usr/bin/env python3
"""
Supervisor Agent for Code Generation and Testing
Uses LangGraph to orchestrate code generation, testing, and iteration.
"""

import os
import sys
import json
import subprocess
from typing import Optional, List, Any, Dict
from dataclasses import dataclass

try:
  from langgraph.graph import StateGraph, START, END
  from langchain_core.messages import HumanMessage, AIMessage
  from langchain_openai import ChatOpenAI
except ImportError as e:
  print(f"Missing required dependencies: {e}")
  print("Install with: pip install langgraph langchain langchain-openai")
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

  def __post_init__(self):
    if self.messages is None:
      self.messages = []


class SupervisorAgent:
  """Supervisor agent that orchestrates code generation and testing"""

  def __init__(self, config_path: str = "supervisor_config.json"):
    self.config = self._load_config(config_path)
    self.llm = self._initialize_llm()
    self.graph = self._build_graph()

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
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        },
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

  def _build_graph(self):
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    workflow.add_node("plan", self._plan_solution)
    workflow.add_node("generate_code", self._generate_code)
    workflow.add_node("generate_tests", self._generate_tests)
    workflow.add_node("run_tests", self._run_tests)
    workflow.add_node("evaluate", self._evaluate_results)
    workflow.add_node("iterate", self._iterate_solution)
    workflow.add_node("finalize", self._finalize_solution)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "generate_code")
    workflow.add_edge("generate_code", "generate_tests")
    workflow.add_edge("generate_tests", "run_tests")
    workflow.add_edge("run_tests", "evaluate")

    workflow.add_conditional_edges(
      "evaluate",
      self._should_continue,
      {
        "continue": "iterate",
        "finish": "finalize"
      }
    )

    workflow.add_edge("iterate", "generate_code")
    workflow.add_edge("finalize", END)

    return workflow.compile()

  def _plan_solution(self, state: AgentState) -> AgentState:
    """Plan the solution approach"""
    print("\n🎯 Planning solution...")
    planning_prompt = f"""
    Analyze this problem and create a plan for solving it:

    Problem: {state.problem_description}
    Example Output: {state.example_output or 'Not provided'}

    Provide a brief plan for the solution approach, including:
    1. Key components needed
    2. Main algorithm or logic
    3. Expected structure

    Keep the plan concise and focused.
    """

    response = self._call_claude_code_sdk("plan", planning_prompt)
    print("📋 Plan generated:")
    print(f"{response}\n")

    if state.messages is None:
      state.messages = []
    state.messages.append(HumanMessage(content=planning_prompt))
    state.messages.append(AIMessage(content=response))
    return state

  def _generate_code(self, state: AgentState) -> AgentState:
    """Generate code solution"""
    iteration_text = (f" (Iteration {state.current_iteration + 1})"
                      if state.current_iteration > 0 else "")
    print(f"\n💻 Generating code{iteration_text}...")

    code_prompt = f"""
    Generate Python code to solve this problem:

    Problem: {state.problem_description}
    Example Output: {state.example_output or 'Not provided'}
    Current Iteration: {state.current_iteration}

    {f"Previous test results: {state.test_results}"
     if state.test_results else ""}
    {f"Previous errors: {state.error_message}"
     if state.error_message else ""}

    Requirements:
    - Create a class-based solution with a clear main method
    - Include proper error handling and input validation
    - Add comprehensive docstrings
    - Make it production-ready
    - Ensure all imports are at the top
    - Handle edge cases gracefully

    Return only the Python code without explanation or markdown formatting.
    """

    code_content = self._call_claude_code_sdk("generate_code", code_prompt)

    # Clean the code content (remove markdown formatting if present)
    code_content = self._clean_code_content(code_content)

    print(f"\n📝 Code generated and saved to {state.solution_path}")
    print("Preview (first 10 lines):")
    lines = code_content.split('\n')[:10]
    for i, line in enumerate(lines, 1):
      print(f"{i:2d}: {line}")
    if len(code_content.split('\n')) > 10:
      print(f"... ({len(code_content.split('\n')) - 10} more lines)\n")
    else:
      print()

    try:
      with open(state.solution_path, 'w') as f:
        f.write(code_content)
    except Exception as e:
      print(f"❌ Error saving code to {state.solution_path}: {e}")
      state.error_message = f"Failed to save code: {e}"
      return state

    state.code_content = code_content
    return state

  def _generate_tests(self, state: AgentState) -> AgentState:
    """Generate test cases"""
    print("\n🧪 Generating tests...")

    # Extract the module name from solution path for proper imports
    module_name = state.solution_path.replace('.py', '')

    test_prompt = f"""
    Generate comprehensive test cases for this code:

    Problem: {state.problem_description}
    Example Output: {state.example_output or 'Not provided'}

    Code to test:
    {state.code_content}

    Requirements:
    - Use pytest framework
    - Import from {module_name} (the solution file)
    - Include edge cases and boundary conditions
    - Test both success and failure scenarios
    - Test with the exact example provided if available
    - Add clear test descriptions and assertions
    - Handle potential import errors gracefully
    - Include at least 3-5 test functions

    Return only the test code without explanation or markdown formatting.
    Start with necessary imports like: import pytest, from {module_name}
    import *
    """

    test_content = self._call_claude_code_sdk("generate_tests", test_prompt)

    # Clean the test content
    test_content = self._clean_code_content(test_content)

    print(f"\n🔬 Tests generated and saved to {state.test_path}")
    test_lines = test_content.split('\n')
    test_functions = [line for line in test_lines
                      if line.strip().startswith('def test_')]
    if test_functions:
      print(f"Generated {len(test_functions)} test functions:")
      for func in test_functions:
        print(f"  - {func.strip()}")
    else:
      print("⚠️ Warning: No test functions detected in generated tests")
    print()

    try:
      with open(state.test_path, 'w') as f:
        f.write(test_content)
    except Exception as e:
      print(f"❌ Error saving tests to {state.test_path}: {e}")
      state.error_message = f"Failed to save tests: {e}"
      return state

    state.test_content = test_content
    return state

  def _run_tests(self, state: AgentState) -> AgentState:
    """Execute the tests"""
    print("\n▶️  Running tests...")

    # Check if test file exists and has content
    if not os.path.exists(state.test_path):
      state.test_results = f"Test file {state.test_path} does not exist"
      state.is_solved = False
      print(f"❌ Test file not found: {state.test_path}\n")
      return state

    # Check if solution file exists
    if not os.path.exists(state.solution_path):
      state.test_results = (f"Solution file {state.solution_path} "
                            "does not exist")
      state.is_solved = False
      print(f"❌ Solution file not found: {state.solution_path}\n")
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
          print(f"❌ {error_msg}")
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
        print("✅ All tests passed!")
        print(f"Test output:\n{result.stdout}")
      else:
        print(f"❌ Tests failed (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{result.stderr}")
      print()

    except subprocess.TimeoutExpired:
      timeout = state.config["agent"]["test_timeout"] if state.config else 30
      state.test_results = f"Tests timed out after {timeout} seconds"
      state.is_solved = False
      print(f"⏰ Tests timed out after {timeout} seconds\n")
    except FileNotFoundError:
      state.test_results = ("pytest not found. "
                            "Please install pytest: pip install pytest")
      state.is_solved = False
      print("❌ pytest not found. Install with: pip install pytest\n")
    except Exception as e:
      state.test_results = f"Error running tests: {str(e)}"
      state.is_solved = False
      print(f"💥 Error running tests: {str(e)}\n")

    return state

  def _evaluate_results(self, state: AgentState) -> AgentState:
    """Evaluate test results and determine next steps"""
    if state.is_solved:
      print("🎉 Problem solved successfully!")
      return state

    print("\n🔍 Evaluating test failures...")

    # Truncate very long test results for better analysis
    test_results_summary = state.test_results
    if len(test_results_summary) > 2000:
      test_results_summary = (test_results_summary[:2000]
                              + "\n... (truncated for analysis)")

    evaluation_prompt = f"""
    Analyze these test results and provide specific, actionable feedback:

    Problem: {state.problem_description}
    Example Expected Output: {state.example_output or 'Not provided'}
    Test Results: {test_results_summary}
    Current Iteration: {state.current_iteration + 1}

    Focus on:
    1. The most critical errors preventing tests from passing
    2. Specific code changes needed (don't just describe problems)
    3. Import issues, syntax errors, or missing methods
    4. Logic errors in the implementation
    5. Edge cases that aren't handled

    Provide concise, actionable suggestions for fixing the code.
    """

    feedback = self._call_claude_code_sdk("evaluate", evaluation_prompt)
    print("\n🔧 Analysis and suggestions:")
    print(f"{feedback}\n")

    state.error_message = feedback
    return state

  def _iterate_solution(self, state: AgentState) -> AgentState:
    """Prepare for next iteration"""
    state.current_iteration += 1
    print(f"\n🔄 Starting iteration {state.current_iteration + 1} "
          f"of {state.max_iterations}...")
    return state

  def _finalize_solution(self, state: AgentState) -> AgentState:
    """Finalize the solution"""
    if state.is_solved:
      print(f"\n🎉 Solution completed successfully after "
            f"{state.current_iteration} iterations!")
      print(f"💾 Code saved to: {state.solution_path}")
      print(f"💾 Tests saved to: {state.test_path}")
      print(f"\n🚀 You can run the tests manually with: "
            f"pytest {state.test_path}")
    else:
      print(f"\n❌ Maximum iterations ({state.max_iterations}) reached "
            "without solving the problem.")
      print(f"📝 Last error: {state.error_message}")
      print("\n📁 Files generated (may contain partial solutions):")
      if os.path.exists(state.solution_path):
        print(f"  - {state.solution_path}")
      if os.path.exists(state.test_path):
        print(f"  - {state.test_path}")

    return state

  def _should_continue(self, state: AgentState) -> str:
    """Decide whether to continue iterating or finish"""
    if state.is_solved:
      return "finish"
    elif state.current_iteration >= state.max_iterations:
      return "finish"
    else:
      return "continue"

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

  def _call_claude_code_sdk(self, operation: str, prompt: str) -> str:
    """Simulate Claude Code SDK calls"""
    try:
      print(f"🤖 Calling Claude for {operation}...")
      messages = [HumanMessage(content=prompt)]
      response = self.llm.invoke(messages)
      content = response.content
      print(f"✅ Claude response received for {operation}")

      # Ensure we have string content
      if not isinstance(content, str):
        content = str(content)

      # Check if response is empty or error-like
      if not content.strip():
        return f"Empty response received for {operation}"

      return content
    except Exception as e:
      error_msg = f"Error in {operation}: {str(e)}"
      print(f"❌ {error_msg}")
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

    print(f"🚀 Starting problem solving: {problem_description}")
    try:
      final_state = self.graph.invoke(initial_state)
      # Ensure we return an AgentState object
      if isinstance(final_state, dict):
        # Convert dict back to AgentState if needed
        return AgentState(**final_state)
      return final_state
    except Exception as e:
      print(f"💥 Error during execution: {e}")
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
