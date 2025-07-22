"""
CSV Data Processing Example

This example demonstrates using the FeedbackSupervisorAgent to process CSV data,
showing how to work with file-based input and output.
"""

import sys
import os

# Add parent directory to path to import supervisor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor import utils


def create_sample_data():
  """Create sample inventory data (as would be loaded from CSV)"""
  return [
    {'product': 'Laptop', 'category': 'Electronics', 'price': 999.99, 'quantity': 10},
    {'product': 'Mouse', 'category': 'Electronics', 'price': 25.50, 'quantity': 50},
    {'product': 'Desk', 'category': 'Furniture', 'price': 299.00, 'quantity': 5},
    {'product': 'Chair', 'category': 'Furniture', 'price': 150.75, 'quantity': 8},
    {'product': 'Phone', 'category': 'Electronics', 'price': 699.99, 'quantity': 15},
  ]


def main():
  """Run inventory data processing example"""
  print("=== Inventory Data Processing Example ===")
  print()

  # Create supervisor agent
  try:
    agent = FeedbackSupervisorAgent()
    print("✓ FeedbackSupervisorAgent initialized successfully")
  except Exception as e:
    print(f"✗ Failed to initialize FeedbackSupervisorAgent: {e}")
    return

  # Create sample inventory data (as if loaded from CSV)
  inventory_data = create_sample_data()
  print(f"✓ Sample inventory data created: {len(inventory_data)} products")

  try:
    # Example: Calculate total inventory value
    problem = """Create a function called 'calculate_inventory_value' that:
  1. Takes a list of product dictionaries as input
  2. Calculates the total value for each product (price * quantity)
  3. Returns a list of dictionaries with product info and total value
  4. Sorts results by total value in descending order

Each input dictionary has keys: product, category, price, quantity
Return format: [{'product': 'name', 'category': 'cat', 'total_value': float}, ...]"""

    expected_output = [
      {'product': 'Laptop', 'category': 'Electronics', 'total_value': 9999.0},
      {'product': 'Phone', 'category': 'Electronics', 'total_value': 10499.85},
      {'product': 'Desk', 'category': 'Furniture', 'total_value': 1495.0},
      {'product': 'Mouse', 'category': 'Electronics', 'total_value': 1275.0},
      {'product': 'Chair', 'category': 'Furniture', 'total_value': 1206.0},
    ]

    print(f"Problem: {problem}")
    print(f"Input: Inventory data with {len(inventory_data)} products")
    print("Expected output format: List of dictionaries with total values")
    print()

    # Process with inventory data as input
    result = agent.process(
      problem,
      input_data=inventory_data,
      output_data=expected_output,
      solution_path='solution.py',
      test_path='test_solution.py',
    )

    # Display results
    print()
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if result.is_solved:
      print("✓ Inventory processing problem solved successfully!")
      print(f"Solution file: {agent.solution_path}")
      print(f"Test file: {agent.test_path}")
      print(f"Iterations: {result.current_iteration}")

      if result.output_data:
        print("\nGenerated output preview:")
        for item in result.output_data[:3]:  # Show first 3 items
          print(f"  {item}")
        if len(result.output_data) > 3:
          print(f"  ... and {len(result.output_data) - 3} more items")

      if agent.data_manager:
        summary = agent.data_manager.get_summary()
        print("\nData processing summary:")
        print(f'  Operations performed: {summary["total_operations"]}')
        print(f'  Formats processed: {summary["formats_processed"]}')
        print(f'  Total data size: {summary["total_data_size"]} bytes')

    else:
      print("✗ Inventory processing problem not solved")
      if result.error_message:
        print(f"Error: {result.error_message}")
      print(f"Completed iterations: {result.current_iteration}/{agent.config.agent.max_iterations}")

    # Show test results if available
    if result.test_results:
      print(f"\nTest execution details:\n{result.test_results}")

    # Show validation feedback and last Claude message
    if result.validation_feedback:
      print(f'\n{utils.red("Validation Feedback :")}\n{result.validation_feedback}')
    print(f"\n{utils.blue(f"Last Claude Message:\n{result.claude_log[-1]}")}")

  except Exception as e:
    print(f"Error processing inventory data: {e}")


if __name__ == '__main__':
  main()
