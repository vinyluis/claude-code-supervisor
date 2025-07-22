"""
FeedbackSupervisorAgent Usage Example

This example demonstrates how to use the FeedbackSupervisorAgent for iterative
problem solving with intelligent feedback loops and error analysis.

Shows the agent's ability to:
- Iterate on failed attempts
- Provide intelligent guidance
- Handle complex problems requiring multiple refinements
- Learn from test failures and improve solutions
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import development_config
from claude_code_supervisor import utils


def main():
  """Demonstrate FeedbackSupervisorAgent usage with complex problems"""
  print('=== FeedbackSupervisorAgent Usage Example ===')
  print()

  # Create feedback supervisor agent
  try:
    agent = FeedbackSupervisorAgent()
    print('✓ FeedbackSupervisorAgent initialized successfully')
    print(f'✓ Max iterations configured: {agent.config.agent.max_iterations}')
  except Exception as e:
    print(f'✗ Failed to initialize FeedbackSupervisorAgent: {e}')
    return

  # Example 1: Complex problem that might require iteration
  print('\n🧠 Example 1: Complex algorithm implementation')
  print('-' * 50)
  
  problem1 = '''Implement a binary search algorithm that works on a sorted list.

Requirements:
- Function should be called 'binary_search'
- Takes a sorted list and target value as parameters
- Returns the index of the target if found, -1 if not found
- Must use iterative approach (not recursive)
- Handle edge cases: empty list, target not in list
- Include comprehensive type hints and docstring
- Must be efficient with O(log n) time complexity

Example: binary_search([1, 3, 5, 7, 9, 11], 7) should return 3'''

  print(f'Problem: Complex binary search implementation')
  print('This problem often requires multiple iterations to get all edge cases right...')
  
  result1 = agent.process(
    problem1,
    solution_path='binary_search.py',
    test_path='test_binary_search.py'
  )

  print(f'\nResult after {result1.current_iteration} iterations:')
  print(f'- Solved: {"✓ Yes" if result1.is_solved else "✗ No"}')
  if result1.error_message:
    print(f'- Final error: {result1.error_message}')
  if result1.validation_feedback:
    print(f'- Last validation feedback: {result1.validation_feedback[:100]}...')

  # Example 2: Problem with specific custom guidance
  print('\n🎨 Example 2: With custom system prompt for specific style')
  print('-' * 50)
  
  styled_agent = FeedbackSupervisorAgent(
    append_system_prompt='''Focus on object-oriented design principles. 
Create classes with proper encapsulation, use descriptive method names, 
and follow PEP 8 style guidelines strictly.'''
  )
  
  problem2 = '''Create a simple inventory management system.

Requirements:
- Create an Item class with name, price, and quantity attributes
- Create an Inventory class that can add items, remove items, and calculate total value
- Include proper error handling for invalid operations
- Use object-oriented best practices
- Include comprehensive tests for all methods

The system should handle basic inventory operations efficiently.'''

  print('Problem: OOP inventory system with custom style requirements')
  print('Custom prompt: Focus on object-oriented design principles')
  
  result2 = styled_agent.process(
    problem2,
    solution_path='inventory_system.py',
    test_path='test_inventory_system.py'
  )

  print(f'\nResult after {result2.current_iteration} iterations:')
  print(f'- Solved: {"✓ Yes" if result2.is_solved else "✗ No"}')
  if result2.error_message:
    print(f'- Final error: {result2.error_message}')

  # Example 3: Show iteration process with a challenging problem
  print('\n🔄 Example 3: Challenging problem to demonstrate iteration')
  print('-' * 50)
  
  problem3 = '''Implement a function that validates and parses complex email addresses.

Requirements:
- Function name: validate_email
- Must handle various email formats including international domains
- Support multiple validation rules (proper @ placement, domain validation, etc.)
- Return tuple: (is_valid: bool, parsed_parts: dict)
- Comprehensive test coverage for edge cases
- Handle malformed inputs gracefully
- Support both strict and lenient validation modes

This is intentionally complex to demonstrate iterative refinement.'''

  print('Problem: Complex email validation (designed to require multiple iterations)')
  
  # Configure for more iterations to handle complexity
  complex_config = development_config()
  complex_config.agent.max_iterations = 5
  
  complex_agent = FeedbackSupervisorAgent(config=complex_config)
  
  result3 = complex_agent.process(
    problem3,
    solution_path='email_validator.py',
    test_path='test_email_validator.py'
  )

  print(f'\nResult after {result3.current_iteration} iterations:')
  print(f'- Solved: {"✓ Yes" if result3.is_solved else "✗ No"}')
  print(f'- Total iterations used: {result3.current_iteration}/{complex_config.agent.max_iterations}')

  # Summary and insights
  print('\n' + '=' * 60)
  print('📊 FEEDBACK SUPERVISOR ANALYSIS')
  print('=' * 60)
  
  results = [result1, result2, result3]
  solved_count = sum(1 for r in results if r.is_solved)
  total_iterations = sum(r.current_iteration for r in results)
  avg_iterations = total_iterations / len(results) if results else 0
  
  print(f'Problems attempted: {len(results)}')
  print(f'Problems solved: {solved_count}')
  print(f'Success rate: {(solved_count / len(results)) * 100:.1f}%')
  print(f'Average iterations per problem: {avg_iterations:.1f}')
  print(f'Total iterations across all problems: {total_iterations}')
  
  print('\n🔄 Iterative Benefits Observed:')
  if any(r.current_iteration > 0 for r in results):
    print('✓ Agent used multiple iterations to refine solutions')
    print('✓ Intelligent error analysis guided improvements')
    print('✓ Test failures were addressed systematically')
  else:
    print('• All problems solved in single iteration (simpler than expected)')
    
  print('\n💡 When FeedbackSupervisorAgent excels:')
  print('• Complex algorithms requiring edge case handling')
  print('• Problems with multiple interacting components')
  print('• When first attempts commonly fail')
  print('• Code that benefits from iterative refinement')
  print('• Maximum solution quality is the priority')
  
  print('\n🎯 Key Features Demonstrated:')
  print('• Automatic retry with intelligent guidance')
  print('• LLM-powered error analysis and improvement suggestions')
  print('• Test-driven iteration (failures guide next attempts)')
  print('• Context-aware problem solving')
  print('• Session continuity across iterations')

  if solved_count > 0:
    print(f'\n📁 Generated files in current directory:')
    if result1.is_solved:
      print('• binary_search.py, test_binary_search.py')
    if result2.is_solved:
      print('• inventory_system.py, test_inventory_system.py')  
    if result3.is_solved:
      print('• email_validator.py, test_email_validator.py')
    print('\n🚀 Test the solutions: pytest <test_file> -v')


if __name__ == '__main__':
  main()