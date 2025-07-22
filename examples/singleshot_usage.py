"""
SingleShotSupervisorAgent Usage Example

This example demonstrates how to use the SingleShotSupervisorAgent for fast,
single-execution problem solving without iterative feedback loops.

Ideal for:
- Simple problems that don't require iteration
- Fast code generation and testing
- Situations where iteration is handled externally
- Benchmarking Claude Code capabilities
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import SingleShotSupervisorAgent


def main():
  """Demonstrate SingleShotSupervisorAgent usage"""
  print('=== SingleShotSupervisorAgent Usage Example ===')
  print()

  # Create single-shot supervisor agent
  try:
    agent = SingleShotSupervisorAgent()
    print('✓ SingleShotSupervisorAgent initialized successfully')
  except Exception as e:
    print(f'✗ Failed to initialize SingleShotSupervisorAgent: {e}')
    return

  # Example 1: Simple function creation
  print('\n📝 Example 1: Simple function creation')
  print('-' * 40)

  problem1 = '''Create a function called 'reverse_string' that takes a string and returns it reversed.

Requirements:
- Handle empty strings gracefully
- Use Python's built-in string slicing
- Include type hints and docstring

Example: reverse_string("hello") should return "olleh"'''

  print(f'Problem: {problem1[:100]}...')

  result1 = agent.process(
    problem1,
    solution_path='reverse_string.py',
    test_path='test_reverse_string.py'
  )

  print(f'Result: {"✓ Solved" if result1.is_solved else "✗ Failed"}')
  if result1.error_message:
    print(f'Error: {result1.error_message}')


if __name__ == '__main__':
  main()
