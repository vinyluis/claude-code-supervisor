#!/usr/bin/env python3
"""
Bring Your Own Model (BYOM) Example for SupervisorAgent

This example demonstrates how to use SupervisorAgent with a custom LangChain LLM
for guidance, rather than relying on the configuration file. This is useful when
you want to integrate SupervisorAgent into existing pipelines that already have
configured LLM instances.
"""

import sys
import os
sys.path.insert(0, '.')


def byom_example():
  """Demonstrate BYOM usage with different LLM providers"""

  print('=== Bring Your Own Model (BYOM) Example ===\n')

  # Check if required environment variables are set
  if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
    print('⚠️  This example requires either OPENAI_API_KEY or ANTHROPIC_API_KEY')
    print('   Set one of these environment variables to run the example.')
    return

  try:
    from claude_code_supervisor import SupervisorAgent
    from langchain_openai import ChatOpenAI

    # Example 1: Use a custom OpenAI model for guidance
    print('📱 Example 1: Custom OpenAI Model for Guidance')
    print('-' * 50)

    # Create a custom LLM with specific parameters
    custom_guidance_llm = ChatOpenAI(
      model='gpt-4o-mini',  # Smaller, faster model for guidance
      temperature=0.1       # Low temperature for consistent guidance
    )

    # Initialize SupervisorAgent with the custom LLM
    agent = SupervisorAgent(llm=custom_guidance_llm)

    print('✅ SupervisorAgent initialized with custom LLM')
    print(f'   LLM Type: {type(agent.llm).__name__}')
    print(f'   Model: {custom_guidance_llm.model_name}')
    print(f'   Temperature: {custom_guidance_llm.temperature}')

    # Test with a simple problem
    print('\n🔍 Testing with a simple problem...')

    result = agent.process(
      'Create a function that finds the maximum value in a list',
      example_output='max_value([1, 5, 3, 9, 2]) should return 9',
      solution_path='solution.py',
      test_path='test_solution.py',
    )

    print(f'Problem solved: {result.is_solved}')
    if result.is_solved:
      print(f'✅ Solution created: {agent.solution_path}')
      print(f'✅ Tests created: {agent.test_path}')
    else:
      print(f'❌ Error: {result.error_message}')

  except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('   Install required packages: pip install langchain-openai')
  except Exception as e:
    print(f'❌ Error during example: {e}')


def main():
  """Run all BYOM examples"""
  byom_example()

  print('\n' + '=' * 60)
  print('🎯 BYOM (Bring Your Own Model) Summary:')
  print('   SupervisorAgent now accepts any LangChain LLM via the \'llm\' parameter')
  print('   This enables easy integration into existing ML pipelines')
  print('   and provides flexibility in choosing guidance models')


if __name__ == '__main__':
  main()
