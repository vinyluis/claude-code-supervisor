"""
Claude Code Supervisor

An intelligent wrapper around Claude Code SDK that provides automated problem-solving
capabilities with session management, progress monitoring, and intelligent feedback loops.

This package provides two main supervisor types:
- FeedbackSupervisorAgent: Iterative supervisor with intelligent feedback loops
- SingleShotSupervisorAgent: Single-execution supervisor without iteration

Both supervisors treat Claude Code as a coding assistant that can plan its own work using
todo lists, implement solutions, run tests, and provide results.

Example:
    Iterative feedback with refinement:
    >>> from claude_code_supervisor import FeedbackSupervisorAgent
    >>> agent = FeedbackSupervisorAgent()
    >>> result = agent.process('Create a function to calculate fibonacci numbers')
    >>> if result.is_solved:
    ...     print(f'Solved after {result.current_iteration} iterations')

    Single-shot execution:
    >>> from claude_code_supervisor import SingleShotSupervisorAgent
    >>> agent = SingleShotSupervisorAgent()
    >>> result = agent.process('Create a simple function')
    >>> if result.is_solved:
    ...     print('Solved in single attempt!')

    With input/output data:
    >>> from claude_code_supervisor import FeedbackSupervisorAgent
    >>> agent = FeedbackSupervisorAgent()
    >>> result = agent.process(
    ...     'Sort this list in ascending order',
    ...     input_data=[64, 34, 25, 12],
    ...     output_data=[12, 25, 34, 64]
    ... )

    With custom configuration:
    >>> from claude_code_supervisor.config import openai_config
    >>> from claude_code_supervisor import FeedbackSupervisorAgent
    >>> config = openai_config(model='gpt-4o-mini', temperature=0.2)
    >>> agent = FeedbackSupervisorAgent(config=config, append_system_prompt='Use object-oriented design')
    >>> result = agent.process('Create a calculator')
"""

from .__version__ import __version__, __author__, __email__, __description__
from .supervisor import (
    BaseSupervisorAgent,
    FeedbackSupervisorAgent,
    SingleShotSupervisorAgent,
    WorkflowState
)

__all__ = [
  '__version__',
  '__author__',
  '__email__',
  '__description__',
  'BaseSupervisorAgent',
  'FeedbackSupervisorAgent',
  'SingleShotSupervisorAgent',
  'WorkflowState',
]
