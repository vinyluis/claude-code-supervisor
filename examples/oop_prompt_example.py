#!/usr/bin/env python3
"""
Object-Oriented Programming Prompt Example

This example demonstrates how to use custom prompts to guide the SupervisorAgent
toward object-oriented programming patterns and SOLID principles.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SupervisorAgent


def main():
  """Run an example with object-oriented programming prompt"""
  print('=== Object-Oriented Programming Prompt Example ===')
  print()

  # Define OOP-focused custom prompt
  oop_prompt = 'Use object-oriented programming principles. Create classes with proper '\
               'encapsulation, methods, and if applicable, inheritance. '\
               'Follow SOLID principles and include proper docstrings for all classes and methods.'

  try:
    agent = SupervisorAgent(custom_prompt=oop_prompt)
    print('✓ SupervisorAgent initialized with OOP prompt')

    problem = 'Create a calculator that can perform basic arithmetic operations '\
              '(addition, subtraction, multiplication, division). It should handle errors '\
              'gracefully and maintain a history of operations.'

    example_output = '''calc = Calculator()
calc.add(5, 3) -> 8
calc.multiply(4, 2) -> 8
calc.get_history() -> ['5 + 3 = 8', '4 * 2 = 8']'''

    print(f'Problem: {problem}')
    print(f'Custom prompt: {oop_prompt}')
    print(f'Expected behavior: {example_output}')
    print()
    print('Processing... (this may take a few minutes)')

    result = agent.process(
      problem_description=problem,
      example_output=example_output,
      solution_path='solution.py',
      test_path='test_solution.py',
    )

    print_results(result)

  except Exception as e:
    print(f'Error in OOP example: {e}')


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
