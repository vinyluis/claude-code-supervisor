"""
Basic FeedbackSupervisorAgent Usage Example

This example demonstrates the simplest way to use the FeedbackSupervisorAgent
to solve a programming problem without any input/output data.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor import utils


def main():
  """Run a basic example without input/output data"""
  print('=== Basic FeedbackSupervisorAgent Usage Example ===')
  print()

  # Create supervisor agent with default config
  try:
    agent = FeedbackSupervisorAgent()
    print('✓ FeedbackSupervisorAgent initialized successfully')
  except Exception as e:
    print(f'✗ Failed to initialize FeedbackSupervisorAgent: {e}')
    return

  # Define a simple programming problem
  problem = '''Create a function called 'fibonacci' that calculates the nth Fibonacci number using recursion.

Example: fibonacci(8) should return 21'''

  print(f'Problem: {problem}')
  print()

  # Process the problem
  print('Processing... (this may take a few minutes)')
  result = agent.process(
    problem,
    solution_path='solution.py',
    test_path='test_solution.py',
  )

  # Display results
  print()
  print('=' * 50)
  print('RESULTS:')
  print('=' * 50)

  if result.is_solved:
    print('✓ Problem solved successfully!')
    print(f'Solution file: {agent.solution_path}')
    print(f'Test file: {agent.test_path}')
    print(f'Iterations: {result.current_iteration}')

    if result.test_results:
      print(f'\nTest Results:\n{result.test_results}')

  else:
    print('✗ Problem not solved')
    if result.error_message:
      print(f'Error: {result.error_message}')
    print(f'Completed iterations: {result.current_iteration}/{agent.config.agent.max_iterations}')

  # Show results
  if result.validation_feedback:
    print(f'\n{utils.red("Validation Feedback :")}\n{result.validation_feedback}')
  print(f"\n{utils.blue(f"Last Claude Message:\n{result.claude_log[-1]}")}")


if __name__ == '__main__':
  main()
