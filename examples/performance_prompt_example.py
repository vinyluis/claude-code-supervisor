#!/usr/bin/env python3
"""
Performance-Focused Prompt Example

This example demonstrates how to use custom prompts to guide the SupervisorAgent
toward performance-optimized implementations with Big O analysis.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SupervisorAgent


def main():
  """Run an example with performance-focused prompt"""
  print('=== Performance-Focused Prompt Example ===')
  print()

  # Define performance-focused custom prompt
  performance_prompt = 'Focus on performance and efficiency. Use appropriate data '\
                       'structures, consider time and space complexity, and optimize for speed. Include '\
                       'performance benchmarks in your tests and document the Big O complexity.'

  try:
    agent = SupervisorAgent(custom_prompt=performance_prompt)
    print('✓ SupervisorAgent initialized with performance prompt')

    problem = 'Implement a function to find all prime numbers up to a given number N. '\
              'The function should be optimized for large values of N.'

    input_data = 1000
    expected_output = 'List of all prime numbers from 2 to 1000'

    print(f'Problem: {problem}')
    print(f'Input: N = {input_data}')
    print(f'Custom prompt: {performance_prompt}')
    print('Processing...')

    result = agent.process(
      problem_description=problem,
      input_data=input_data,
      expected_output=expected_output,
      data_format='auto',
      solution_path='solution.py',
      test_path='test_solution.py',
    )

    print_results(result)

  except Exception as e:
    print(f'Error in performance example: {e}')


def print_results(result):
  """Print results in a consistent format"""
  print()
  print('=' * 60)
  print('RESULTS:')
  print('=' * 60)

  if result.is_solved:
    print('✓ Problem solved successfully!')
    print(f'Solution: {agent.solution_path}')
    print(f'Tests: {agent.test_path}')
    print(f'Iterations: {result.current_iteration}')

    if result.output_data:
      print('Output data generated: Yes')

  else:
    print('✗ Problem not solved')
    if result.error_message:
      print(f'Error: {result.error_message}')
    print(f'Iterations: {result.current_iteration}/{agent.config.agent.max_iterations}')

  if result.latest_guidance:
    print(f'Guidance provided: {len(result.latest_guidance)} times')


if __name__ == '__main__':
  main()
