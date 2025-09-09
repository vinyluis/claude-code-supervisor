#!/usr/bin/env python3
"""
Plan Mode Example - Claude Code Supervisor with Intelligent Planning

This example demonstrates the new plan mode functionality that generates and reviews
execution plans before implementation, inspired by validation patterns from OCR agents.

Features demonstrated:
- Plan generation using Claude Code's plan mode
- LLM-powered plan review and scoring
- Iterative plan refinement based on feedback
- Auto-approval after maximum iterations
- Comprehensive logging of the planning process

Usage:
    python examples/plan_mode_example.py
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import claude_code_supervisor
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import plan_mode_development_config


def main():
    """Run plan mode example with iterative planning and review."""
    print("🚀 Claude Code Supervisor - Plan Mode Example")
    print("=" * 60)
    print()
    
    # Configure plan mode with development-friendly settings
    config = plan_mode_development_config(
        max_plan_iterations=3,  # Allow up to 3 plan refinement cycles
        plan_auto_approval_threshold=0.7  # More permissive threshold for development
    )
    
    print("📋 Plan Mode Configuration:")
    print(f"   • Max Plan Iterations: {config.claude_code.max_plan_iterations}")
    print(f"   • Plan Review Enabled: {config.claude_code.plan_review_enabled}")
    print(f"   • Auto-approval Threshold: {config.claude_code.plan_auto_approval_threshold}")
    print(f"   • LLM Model: {config.agent.model_name}")
    print()
    
    # Create supervisor with plan mode enabled
    agent = FeedbackSupervisorAgent(config=config)
    
    # Problem description that should benefit from planning
    problem_description = """
    Create a comprehensive Python calculator module that can:
    1. Perform basic arithmetic operations (add, subtract, multiply, divide)
    2. Handle advanced operations (power, square root, logarithm)
    3. Support expression parsing (e.g., "2 + 3 * 4")
    4. Include proper error handling for edge cases
    5. Have comprehensive unit tests using pytest
    6. Include docstrings and type hints
    
    The calculator should be designed with extensibility in mind for adding new operations.
    """
    
    print("🎯 Problem Description:")
    print(problem_description.strip())
    print()
    
    print("▶️ Starting Claude Code Supervisor with Plan Mode...")
    print("   This will demonstrate:")
    print("   1. Plan generation using Claude Code's plan mode")
    print("   2. LLM-powered plan review and scoring")
    print("   3. Iterative plan refinement (if needed)")
    print("   4. Plan approval and execution")
    print()
    
    try:
        # Execute with plan mode - this will show the full planning workflow
        result = agent.process(
            problem_description=problem_description,
            solution_path="calculator.py",
            test_path="test_calculator.py",
            enable_plan_mode=True
        )
        
        print()
        print("🎉 Execution Complete!")
        print("=" * 60)
        
        # Display results - plan mode was executed before execution mode
        print("📋 Plan Mode: ✅ Completed (plan generated, reviewed, and approved)")
        print(f"   Approved plan was used to guide implementation")
        
        print(f"✅ Solution Status: {'✅ Solved' if result.is_solved else '❌ Not Solved'}")
        print(f"🔄 Total Iterations: {result.current_iteration}")
        
        if result.is_solved:
            print(f"💾 Solution saved to: calculator.py")
            print(f"🧪 Tests saved to: test_calculator.py")
            print()
            print("💡 Next steps:")
            print("   • Review the generated files")
            print("   • Run: pytest test_calculator.py")
            print("   • Compare the implementation with the approved plan")
        else:
            print(f"❌ Error: {result.error_message}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()