"""Default prompts for Claude Code Supervisor."""
from .utils import DataTypes


def development_guidelines() -> str:
  """Provide development guidelines for the coding agent."""
  return """\
- Use Python
- The folders usually have README files that you can consult for more information
- Avoid magic numbers; use constants, variables or parameters instead
- Ensure that the code is well-documented, with docstrings one-liners for small functions, and more detailed docstrings for larger functions or classes.
- Use typehints whenever possible. Use modern syntax (list instead of List). Prefer using None instead of Optional when possible.
- If you create new code, ensure that it is covered by tests. Follow the same testing conventions as the existing code.
- If input data is provided, make sure to read and process it correctly
"""


def instruction_prompt() -> str:
  """Provide a default instruction prompt for the coding agent."""
  return """\
I need you to solve this programming problem step by step. Please:

1. Create your own plan using the TodoWrite tool to track your progress
2. Implement a complete solution with proper error handling
3. Create comprehensive tests for your solution
4. Run the tests to verify everything works
5. Iterate on your solution based on the test results and any feedback provided
"""


def test_instructions(test_path: str | None = None) -> str:
  if test_path is not None:
    return f"""\
- Create tests using pytest for the solution in {test_path}
- Run the tests with the command `pytest {test_path}` to ensure your solution works correctly.
"""
  else:
    return """\
- Write tests for your solution in the same style as the existing tests
- If you create new code, ensure that it is covered by tests.
"""


def build_claude_instructions(
    instruction_prompt: str,
    problem_description: str,
    development_guidelines: str,
    test_instructions: str,
    solution_path: str | None = None,
    test_path: str | None = None,
    input_data: DataTypes | None = None,
    output_data: DataTypes | None = None,
  ) -> str:
  """Prompt for the initial problem description on standalone mode."""
  requirements = development_guidelines
  if solution_path is None:
    requirements += "- Integrate your solution directly into the existing codebase by modifying the appropriate files\n"
  else:
    requirements += f"- Save the solution as \"{solution_path}\"\n"
  if test_path is None:
    requirements += "- Add tests to the existing test files, following the existing test structure and conventions"
  else:
    requirements += f"- Save tests as \"{test_path}\"\n- Tests should be written using pytest"
  if output_data is not None:
    requirements += "- Return results in the same format as the given output\n"

  return f"""\
{instruction_prompt}

### Requirements / Development Guidelines:
{requirements}

### Problem description:
{problem_description}

### Input Data:
{input_data if input_data is not None else 'No input data provided.'}

### Output Data:
{output_data if output_data is not None else 'No output data provided.'}

### Testing:
{test_instructions}

Please start by creating a todo list to plan your approach, then implement the solution.
When you're done, please tell me specifically which function/classes you created, the files you modified, and the tests you added. Also tell me how to run the tests and what the expected output is.
"""


def build_claude_guidance_prompt(latest_guidance: str) -> str:
  """Prompt for guiding Claude Code."""
  return f"""\
{latest_guidance if latest_guidance else 'Continue working on the problem based on the previous feedback.'}

Please update your todo list and continue with the implementation.
"""


def error_guidance_template() -> str:
  """Template for generating guidance when errors occur."""
  return """\
Analyze this Claude Code implementation failure and provide specific guidance for the next iteration.

Problem: {problem_description}
{example_output_section}

Current Issues:
- Error: {error_message}
- Test Results: {test_results}
- Current Iteration: {current_iteration}

Testing Instructions:
{test_instructions}

Claude's Todo Progress:
{todo_progress}

Claude's Recent Output:
{recent_output}

Provide specific, actionable guidance for Claude Code to fix these issues:
1. What went wrong?
2. What specific steps should Claude take next?
3. What should Claude focus on or avoid?
4. How should Claude run and interpret the tests based on the testing instructions?

Keep your response concise and actionable (2-3 bullet points).
"""


def feedback_guidance_template() -> str:
  """Template for generating guidance when providing feedback on working solution."""
  return """\
Analyze Claude Code's implementation and provide guidance for improvement.

Problem: {problem_description}
{example_output_section}

Validation Feedback:
{validation_feedback}

Testing Instructions:
{test_instructions}

Claude's Recent Messages:
{recent_messages}

Claude's Todo Progress:
{todo_progress}

Based on the validation feedback and Claude's current work, provide specific guidance to help Claude improve their solution:
1. What aspects of the solution need improvement?
2. What specific changes should Claude make?
3. How can the implementation be enhanced?
4. How should Claude run and interpret the tests based on the testing instructions?

Focus on actionable improvements that will address the validation feedback.
Keep your response concise and specific (2-3 bullet points).
"""


def test_analysis_template() -> str:
  """Template for analyzing Claude's output to determine test strategy."""
  return """\
Analyze Claude Code's output to determine the best testing strategy.

Claude's Recent Output:
{claude_output}

Claude mentioned these files/functions/classes:
{mentioned_items}

Testing Instructions:
{test_instructions}

Available test files in project:
{available_test_files}

Determine the best approach for testing:
1. Should we run specific test files that Claude mentioned?
2. Should we run integration tests on the entire project?
3. What specific test commands should be executed based on the testing instructions?

Provide a clear testing strategy:
- Test type: (specific/integration)
- Test files/patterns to run: (list specific files or patterns)
- Expected test command: (exact command to execute, following the testing instructions)

Be specific and actionable. Focus on what Claude actually implemented and follow the testing instructions provided.
"""
