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
"""


def build_claude_instructions(
    instruction_prompt: str,
    problem_description: str,
    development_guidelines: str,
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

Please start by creating a todo list to plan your approach, then implement the solution.
On the end, please specifically state the function/classes you created, the files you modified, and the tests you added. Also tell us how to run the tests and what the expected output is.
"""


def build_claude_guidance_prompt(latest_guidance: str) -> str:
  """Prompt for guiding Claude Code."""
  return f"""\
{latest_guidance if latest_guidance else 'Continue working on the problem based on the previous feedback.'}

Please update your todo list and continue with the implementation.
"""
