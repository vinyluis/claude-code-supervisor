#!/usr/bin/env python3
"""
Data Science Prompt Example

This example demonstrates how to use custom prompts to guide the SupervisorAgent
toward data science best practices using pandas and numpy.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SupervisorAgent


def main():
  """Run an example with data science-focused prompt"""
  print('=== Data Science Prompt Example ===')
  print()

  # Define data science-focused custom prompt
  data_science_prompt = 'Use data science best practices. Prefer pandas for data '\
                        'manipulation, numpy for numerical operations, and include data validation, '\
                        'descriptive statistics, and proper error handling for missing or invalid data.'

  try:
    agent = SupervisorAgent(custom_prompt=data_science_prompt)
    print('✓ SupervisorAgent initialized with data science prompt')

    input_data = [
      {'name': 'Alice', 'age': 25, 'score': 85.5, 'department': 'Engineering'},
      {'name': 'Bob', 'age': 30, 'score': 92.0, 'department': 'Marketing'},
      {'name': 'Charlie', 'age': 35, 'score': 78.5, 'department': 'Engineering'},
      {'name': 'Diana', 'age': 28, 'score': 95.0, 'department': 'Sales'},
      {'name': 'Eve', 'age': 32, 'score': 88.0, 'department': 'Marketing'}
    ]

    problem = 'Create a function called "analyze_employee_data" that takes employee data '\
              'and returns comprehensive statistics including: mean score by department, '\
              'age distribution, top performers, and data quality checks for missing values.'

    expected_output = {
      'department_stats': {'Engineering': 82.0, 'Marketing': 90.0, 'Sales': 95.0},
      'age_stats': {'mean': 30.0, 'min': 25, 'max': 35},
      'top_performers': ['Diana', 'Bob'],
      'data_quality': {'missing_values': 0, 'total_records': 5}
    }

    print(f'Problem: {problem}')
    print(f'Input: Employee data with {len(input_data)} records')
    print(f'Custom prompt: {data_science_prompt}')
    print('Processing...')

    result = agent.process(
      problem_description=problem,
      input_data=input_data,
      expected_output=expected_output,
      data_format='list'
    )

    print_results(result)

  except Exception as e:
    print(f'Error in data science example: {e}')


def print_results(result):
  """Print results in a consistent format"""
  print()
  print('=' * 60)
  print('RESULTS:')
  print('=' * 60)

  if result.is_solved:
    print('✓ Problem solved successfully!')
    print(f'Solution: {result.solution_path}')
    print(f'Tests: {result.test_path}')
    print(f'Iterations: {result.current_iteration}')

    if result.output_data:
      print('Output data generated: Yes')

  else:
    print('✗ Problem not solved')
    if result.error_message:
      print(f'Error: {result.error_message}')
    print(f'Iterations: {result.current_iteration}/{result.max_iterations}')

  if result.guidance_messages:
    print(f'Guidance provided: {len(result.guidance_messages)} times')


if __name__ == '__main__':
  main()
