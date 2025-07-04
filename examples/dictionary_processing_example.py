#!/usr/bin/env python3
"""
Dictionary Processing Example

This example demonstrates using the SupervisorAgent with dictionary input/output data
to solve employee sorting problems.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SupervisorAgent


def main():
  """Run a dictionary processing example with employee data"""
  print('=== Dictionary Processing Example ===')
  print()

  # Create supervisor agent
  try:
    agent = SupervisorAgent()
    print('✓ SupervisorAgent initialized successfully')
  except Exception as e:
    print(f'✗ Failed to initialize SupervisorAgent: {e}')
    return

  # Example: Dictionary processing
  print('\\n--- Employee Sorting by Salary ---')

  input_data = [
    {'name': 'Alice', 'age': 30, 'salary': 50000},
    {'name': 'Bob', 'age': 25, 'salary': 45000},
    {'name': 'Charlie', 'age': 35, 'salary': 60000},
    {'name': 'Diana', 'age': 28, 'salary': 52000}
  ]

  expected_output = [
    {'name': 'Charlie', 'age': 35, 'salary': 60000},
    {'name': 'Diana', 'age': 28, 'salary': 52000},
    {'name': 'Alice', 'age': 30, 'salary': 50000},
    {'name': 'Bob', 'age': 25, 'salary': 45000}
  ]

  problem = 'Create a function called \'sort_employees\' that takes a list of employee '\
            'dictionaries and returns them sorted by salary in descending order. '\
            'Each dictionary has \'name\', \'age\', and \'salary\' keys.'

  print(f'Input data: {input_data}')
  print(f'Expected output: {expected_output}')
  print(f'Problem: {problem}')

  result = agent.process(
    problem_description=problem,
    input_data=input_data,
    expected_output=expected_output,
    data_format='auto'  # Let the system detect the format
  )

  print_results('Dictionary Processing', result)


def print_results(example_name: str, result):
  """Helper function to print results in a consistent format"""
  print(f"\\n{example_name} Results:")
  print("-" * (len(example_name) + 9))

  if result.is_solved:
    print('✓ Problem solved successfully!')
    print(f'Solution file: {result.solution_path}')
    print(f'Test file: {result.test_path}')
    print(f'Iterations: {result.current_iteration}')

    if result.output_data is not None:
      print(f'Generated output: {result.output_data}')

    if result.data_manager:
      summary = result.data_manager.get_summary()
      print(f'Data operations: {summary["total_operations"]}')
      print(f'Formats processed: {summary["formats_processed"]}')

  else:
    print('✗ Problem not solved')
    if result.error_message:
      print(f'Error: {result.error_message}')
    print(f'Completed iterations: {result.current_iteration}/{result.max_iterations}')


if __name__ == '__main__':
  main()
