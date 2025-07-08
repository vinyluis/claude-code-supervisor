"""
Claude Code Supervisor

An intelligent wrapper around Claude Code SDK that provides automated problem-solving
capabilities with session management, progress monitoring, and intelligent feedback loops.

This package treats Claude Code as a coding assistant that can plan its own work using
todo lists, implement solutions, run tests, and iterate based on feedback.

Example:
    Basic usage:
    >>> from claude_code_supervisor import SupervisorAgent
    >>> agent = SupervisorAgent()
    >>> result = agent.process('Create a function to calculate fibonacci numbers')
    >>> print(f'Solution: {result.solution_path}')

    With input/output data:
    >>> result = agent.process(
    ...     'Sort this list in ascending order',
    ...     input_data=[64, 34, 25, 12],
    ...     expected_output=[12, 25, 34, 64]
    ... )

    With custom configuration:
    >>> from claude_code_supervisor.config import openai_config
    >>> config = openai_config(model='gpt-4o-mini', temperature=0.2)
    >>> agent = SupervisorAgent(config=config, custom_prompt='Use object-oriented design')
    >>> result = agent.process('Create a calculator')
"""

from .__version__ import __version__, __author__, __email__, __description__
from .supervisor import SupervisorAgent, WorkflowState
from .data_manager import DataManager, DataInfo

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'SupervisorAgent',
    'WorkflowState',
    'DataManager',
    'DataInfo',
]
