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


def plan_mode_instruction_prompt() -> str:
  """Provide an instruction prompt specifically for plan mode (high-level planning only)."""
  return """\
I need you to create a comprehensive execution plan for this programming problem. Focus on:

1. Analyzing the problem requirements and constraints thoroughly
2. Designing the overall architecture and approach
3. Identifying key components and their responsibilities
4. Planning the implementation sequence and dependencies
5. Considering error handling, testing strategy, and edge cases

Please create a detailed, structured plan that covers all aspects of the solution. Use the ExitPlanMode tool to present your final plan when ready.

Do NOT start implementing code or creating detailed task lists - focus on high-level planning and architectural decisions.
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
    approved_plan: str | None = None,
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

  # Build approved plan section if available
  approved_plan_section = ""
  if approved_plan:
    approved_plan_section = f"""

## ✅ APPROVED EXECUTION PLAN:
The following execution plan has been reviewed and approved. Please follow this plan closely:

{approved_plan}

"""

  return f"""\
{instruction_prompt}

## Requirements / Development Guidelines:
{requirements}

## Problem description:
{problem_description}{approved_plan_section}

## Input Data:
{input_data if input_data is not None else 'No input data provided.'}

## Output Data:
{output_data if output_data is not None else 'No output data provided.'}

## Testing:
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


def plan_review_template() -> str:
  """Template for LLM-powered plan review, inspired by OCR validation patterns."""
  return """\
<role>
You are an expert software engineering plan reviewer with deep knowledge of software development best practices, system architecture, and implementation feasibility. Your job is to analyze execution plans generated by Claude Code and provide structured feedback on their quality and viability.
</role>

<plan_evaluation_approach>
Follow this systematic 3-step evaluation process:

1. **COMPLETENESS ASSESSMENT** - Does the plan include all necessary steps?
   - Are all required components identified and addressed?
   - Does it account for dependencies between different parts?
   - Are testing, validation, and integration steps included?

2. **FEASIBILITY ANALYSIS** - Can this plan realistically be executed?
   - Are the proposed approaches technically sound?
   - Are the complexity estimates reasonable?
   - Does it account for potential edge cases and error scenarios?

3. **QUALITY EVALUATION** - Is this an optimal approach?
   - Are there more efficient or elegant solutions?
   - Does it follow software engineering best practices?
   - Is the plan clear and actionable for implementation?
</plan_evaluation_approach>

<scoring_guidelines>
Use this structured scoring framework (0.0-1.0):

**0.0-0.3: Poor Plan Quality**
- Missing critical components or steps
- Technically infeasible or fundamentally flawed approaches
- Lacks essential testing or validation considerations
- Unclear or contradictory instructions

**0.4-0.6: Adequate Plan Quality**
- Covers basic requirements but missing important details
- Generally feasible but may have some technical issues
- Basic testing approach but limited validation strategy
- Somewhat clear but could benefit from more specificity

**0.7-0.8: Good Plan Quality**
- Comprehensive coverage of requirements and dependencies
- Technically sound and well-structured approach
- Includes proper testing and validation strategy
- Clear, actionable steps with good organization

**0.9-1.0: Excellent Plan Quality**
- Complete, thorough coverage of all aspects
- Optimal technical approach with best practices
- Comprehensive testing and error handling strategy
- Exceptionally clear, detailed, and well-organized
</scoring_guidelines>

<problem_context>
Problem: {problem_description}

Input Data: {input_data}

Output Requirements: {output_data}

Development Context: {development_context}
</problem_context>

<plan_to_review>
{claude_plan}
</plan_to_review>

<review_instructions>
Provide a structured analysis with:

1. **Strengths**: What does this plan do well?
2. **Weaknesses**: What are the main issues or gaps?
3. **Specific Improvements**: Concrete suggestions for enhancement
4. **Risk Assessment**: Potential problems and mitigation strategies
5. **Overall Score**: Numeric score (0.0-1.0) based on scoring guidelines

For scores below 0.8, provide specific, actionable feedback for plan improvement:
- What key components are missing?
- Which approaches should be reconsidered?
- What additional steps or considerations are needed?
- How can clarity and organization be improved?

Be constructive and specific - focus on actionable improvements that will help Claude Code generate a better plan in the next iteration.
</review_instructions>

<output_format>
Return your analysis as a JSON object with this structure:
{{
  "overall_score": 0.0-1.0,
  "strengths": ["strength1", "strength2", ...],
  "weaknesses": ["weakness1", "weakness2", ...],
  "specific_improvements": ["improvement1", "improvement2", ...],
  "risk_assessment": ["risk1", "risk2", ...],
  "recommendation": "approve" | "refine",
  "feedback_summary": "Concise summary of key points for Claude Code"
}}

Be decisive and thorough in your evaluation. Focus on helping Claude Code create the most effective implementation plan possible.
</output_format>"""


def plan_refinement_guidance_template() -> str:
  """Template for generating plan refinement guidance based on review feedback."""
  return """\
Based on the plan review feedback, provide specific guidance for Claude Code to improve the execution plan.

<current_context>
Problem: {problem_description}
Plan Iteration: {plan_iteration}
Previous Plan Score: {plan_review_score}
</current_context>

<review_feedback>
{plan_feedback}
</review_feedback>

<previous_plan>
{previous_plan}
</previous_plan>

<refinement_instructions>
Help Claude Code address the specific issues identified in the review feedback:

1. **Priority Fixes**: What are the most critical issues that must be addressed?
2. **Specific Changes**: What concrete modifications should be made to the plan?
3. **Additional Considerations**: What new aspects should be included?
4. **Structure Improvements**: How can the plan organization be enhanced?

Focus on the specific weaknesses and improvement suggestions from the review feedback.
Provide clear, actionable guidance that directly addresses the identified issues.

Be constructive and specific - help Claude Code understand exactly what needs to change and why.
</refinement_instructions>

Generate a refined plan that addresses these concerns while maintaining the strengths of the original approach.
"""


def plan_display_template() -> str:
  """Template for displaying plan review results to users."""
  return """\
📋 **Plan Review Results** (Iteration {plan_iteration})

**Overall Score:** {plan_review_score:.2f}/1.0
**Status:** {approval_status}

**Plan Summary:**
{claude_plan}

**Review Analysis:**
✅ **Strengths:** {strengths}
⚠️  **Areas for Improvement:** {improvements}
🔍 **Next Steps:** {next_action}

{detailed_feedback}
"""
