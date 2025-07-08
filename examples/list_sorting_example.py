#!/usr/bin/env python3
"""
List Sorting Example

This example demonstrates using the SupervisorAgent with input/output data
to solve a simple list sorting problem.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SupervisorAgent


def main():
  """Run a list sorting example with input/output data"""
  print('=== List Sorting Example ===')
  print()

  # Create supervisor agent
  try:
    agent = SupervisorAgent()
    print('✓ SupervisorAgent initialized successfully')
  except Exception as e:
    print(f'✗ Failed to initialize SupervisorAgent: {e}')
    return

  # Example: List sorting
  print('\\n--- List Sorting ---')

  input_data = [64, 34, 25, 12, 22, 11, 90, 5]
  expected_output = [5, 11, 12, 22, 25, 34, 64, 90]

  problem = '''Create a function called 'sort_list' that takes a list of numbers
and returns a new list with the numbers sorted in ascending order.
Use the provided input data for testing.'''

  print(f'Input data: {input_data}')
  print(f'Expected output: {expected_output}')
  print(f'Problem: {problem}')

  result = agent.process(
    problem_description=problem,
    input_data=input_data,
    expected_output=expected_output,
    data_format='list',
    solution_path='solution.py',
    test_path='test_solution.py',
  )

  print_results('List Sorting', result, agent)


def print_results(example_name: str, result, agent):
  """Helper function to print results in a consistent format"""
  print(f"\\n{example_name} Results:")
  print("-" * (len(example_name) + 9))

  if result.is_solved:
    print('✓ Problem solved successfully!')
    print(f'Solution file: {agent.solution_path}')
    print(f'Test file: {agent.test_path}')
    print(f'Iterations: {result.current_iteration}')

    if result.output_data is not None:
      print(f'Generated output: {result.output_data}')

    if agent.data_manager:
      summary = agent.data_manager.get_summary()
      print(f'Data operations: {summary["total_operations"]}')
      print(f'Formats processed: {summary["formats_processed"]}')

  else:
    print('✗ Problem not solved')
    if result.error_message:
      print(f'Error: {result.error_message}')
    print(f'Completed iterations: {result.current_iteration}/{agent.config.agent.max_iterations}')


if __name__ == '__main__':
  main()
