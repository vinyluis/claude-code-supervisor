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
      example_output='max_value([1, 5, 3, 9, 2]) should return 9'
    )

    print(f'Problem solved: {result.is_solved}')
    if result.is_solved:
      print(f'✅ Solution created: {result.solution_path}')
      print(f'✅ Tests created: {result.test_path}')
    else:
      print(f'❌ Error: {result.error_message}')

  except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('   Install required packages: pip install langchain-openai')
  except Exception as e:
    print(f'❌ Error during example: {e}')


def byom_pipeline_integration_example():
  """Example showing how to integrate BYOM into existing ML pipelines"""

  print('\n=== Pipeline Integration Example ===')
  print('This shows how to reuse an existing LLM in your pipeline\n')

  try:
    from langchain_openai import ChatOpenAI
    from claude_code_supervisor import SupervisorAgent

    # Simulate an existing ML pipeline with a configured LLM
    print('🔧 Simulating existing ML pipeline...')

    # This could be your existing LLM used elsewhere in your pipeline
    pipeline_llm = ChatOpenAI(
      model='gpt-4o',
      temperature=0.0
    )

    print(f'   Pipeline LLM: {pipeline_llm.model_name}')

    # Reuse the same LLM for SupervisorAgent guidance
    print('\n♻️  Reusing pipeline LLM for SupervisorAgent...')

    SupervisorAgent(
      custom_prompt='Write efficient, well-documented code with comprehensive tests',
      llm=pipeline_llm  # Reuse existing LLM
    )

    print('✅ SupervisorAgent configured to reuse pipeline LLM')
    print('   This avoids creating duplicate LLM instances and')
    print('   ensures consistent model behavior across your pipeline')

    # Benefits of BYOM approach
    print('\n💡 Benefits of BYOM approach:')
    print('   • Reuse existing LLM configurations')
    print('   • Consistent model behavior across pipeline')
    print('   • Avoid duplicate API connections')
    print('   • Centralized LLM management')
    print('   • Cost optimization through shared instances')

  except ImportError as e:
    print(f'❌ Missing dependency: {e}')
  except Exception as e:
    print(f'❌ Error: {e}')


def main():
  """Run all BYOM examples"""
  byom_example()
  byom_pipeline_integration_example()

  print('\n' + '=' * 60)
  print('🎯 BYOM (Bring Your Own Model) Summary:')
  print('   SupervisorAgent now accepts any LangChain LLM via the \'llm\' parameter')
  print('   This enables easy integration into existing ML pipelines')
  print('   and provides flexibility in choosing guidance models')


if __name__ == '__main__':
  main()
