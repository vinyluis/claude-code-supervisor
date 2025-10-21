"""
Configuration dataclasses for SupervisorAgent.

This module provides a simplified, type-safe configuration system using dataclasses.
All model configuration (name, temperature, provider) is consolidated into AgentConfig,
while Claude Code SDK specific settings are in ClaudeCodeConfig.

Example:
    >>> from claude_code_supervisor.config import openai_config, bedrock_config
    >>>
    >>> # Use convenience functions
    >>> config = openai_config(model_name='gpt-4o-mini', temperature=0.2)
    >>>
    >>> # Or build from scratch
    >>> from claude_code_supervisor.config import SupervisorConfig, AgentConfig
    >>> config = SupervisorConfig(
    ...     agent=AgentConfig(
    ...         model_name='gpt-4o',
    ...         temperature=0.1,
    ...         provider='openai',
    ...         max_iterations=3
    ...     )
    ... )
"""

from dataclasses import dataclass, field
from .utils import ToolsEnum


@dataclass
class AgentConfig:
  """
  Configuration for the supervisor agent behavior and LLM settings.

  This dataclass consolidates both agent behavior (iterations, timeouts) and
  LLM configuration (model, temperature, provider) in one place.

  Attributes:
    max_iterations: Maximum number of feedback iterations (default: 3)
    test_timeout: Timeout for test execution in seconds (default: 30)
    model_name: LLM model name for guidance generation (default: 'gpt-4o')
    temperature: LLM temperature for guidance generation (default: 0.1)
    provider: LLM provider ('openai' or 'bedrock', default: 'openai')
  """
  max_iterations: int = 3
  test_timeout: int = 30
  model_name: str = 'gpt-4o'
  temperature: float = 0.1
  provider: str = 'openai'


@dataclass
class ClaudeCodeConfig:
  """
  Configuration for Claude Code SDK integration.

  These settings control how the supervisor interacts with the Claude Code SDK,
  including which Claude provider to use and SDK-specific parameters.

  Attributes:
    use_bedrock: Use AWS Bedrock for Claude Code instead of Anthropic API (default: False)
    working_directory: Working directory for Claude Code execution (default: current dir)
    javascript_runtime: JavaScript runtime for Claude Code (default: 'node')
    executable_args: Additional arguments for Claude Code executable (default: [])
    claude_code_path: Custom path to Claude Code executable (default: auto-detect)
    max_turns: Maximum turns per Claude Code session (default: None = unlimited)
    tools: List of tools available to Claude Code (default: all tools)
    max_plan_iterations: Maximum plan review cycles before auto-approval (default: 3)
    plan_review_enabled: Enable LLM-powered plan review and refinement (default: True)
    plan_auto_approval_threshold: LLM confidence score threshold for auto-approval (default: 0.8)
    suppress_cli_warnings: Suppress known harmless warnings from Claude Code CLI (default: True)
    cli_warning_patterns: Patterns of CLI warnings to suppress (default: pre-flight warnings)
    enable_prompt_preview_truncation: Enable truncation of long prompts in refinement (default: False)
    prompt_preview_head_lines: Number of lines to show from start of prompt when truncated (default: 100)
    prompt_preview_tail_lines: Number of lines to show from end of prompt when truncated (default: 100)
  """
  use_bedrock: bool = False
  working_directory: str | None = None
  javascript_runtime: str = 'node'
  executable_args: list[str] = field(default_factory=list)
  claude_code_path: str | None = None
  max_turns: int | None = None
  tools: list[str] = field(default_factory=lambda: ToolsEnum.all())
  max_plan_iterations: int = 3
  plan_review_enabled: bool = True
  plan_auto_approval_threshold: float = 0.8
  suppress_cli_warnings: bool = True
  cli_warning_patterns: list[str] = field(default_factory=lambda: [
    'Pre-flight check is taking longer than expected',
    'Pre-flight check',  # Broader match if needed
  ])
  enable_prompt_preview_truncation: bool = False
  prompt_preview_head_lines: int = 100
  prompt_preview_tail_lines: int = 100


@dataclass
class SupervisorConfig:
  """
  Complete configuration for SupervisorAgent.

  This is the main configuration dataclass that combines agent behavior settings
  and Claude Code SDK settings into a single, type-safe configuration object.

  Attributes:
    agent: Agent behavior and LLM configuration (AgentConfig)
    claude_code: Claude Code SDK integration settings (ClaudeCodeConfig)

  Example:
    >>> config = SupervisorConfig(
    ...     agent=AgentConfig(max_iterations=5, model_name='gpt-4o-mini'),
    ...     claude_code=ClaudeCodeConfig(
    ...         max_turns=30,
    ...         use_bedrock=True,
    ...         tools=['Read', 'Write', 'Edit', 'Bash', 'TodoWrite']
    ...     )
    ... )
  """
  agent: AgentConfig = field(default_factory=AgentConfig)
  claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)


# Convenience functions for common configurations
def openai_config(model_name: str = 'gpt-4o', temperature: float = 0.1) -> SupervisorConfig:
  """
  Create a configuration for OpenAI models.

  Args:
    model_name: OpenAI model name (default: 'gpt-4o')
    temperature: Temperature for guidance generation (default: 0.1)

  Returns:
    SupervisorConfig configured for OpenAI with specified model and temperature

  Example:
    >>> config = openai_config(model_name='gpt-4o-mini', temperature=0.2)
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    agent=AgentConfig(
      model_name=model_name,
      temperature=temperature,
      provider='openai'
    )
  )


def bedrock_config(
  model_name: str = 'anthropic.claude-3-5-sonnet-20241022-v2:0',
  temperature: float = 0.1,
) -> SupervisorConfig:
  """
  Create a configuration for AWS Bedrock models.

  This configuration sets up both the guidance LLM and Claude Code SDK to use AWS Bedrock.

  Args:
    model_name: Bedrock model name (default: 'anthropic.claude-3-5-sonnet-20241022-v2:0')
    temperature: Temperature for guidance generation (default: 0.1)

  Returns:
    SupervisorConfig configured for AWS Bedrock with specified model and temperature

  Example:
    >>> config = bedrock_config(model_name='anthropic.claude-3-haiku-20240307-v1:0')
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    agent=AgentConfig(
      model_name=model_name,
      temperature=temperature,
      provider='bedrock',
    ),
    claude_code=ClaudeCodeConfig(use_bedrock=True),
  )


def development_config() -> SupervisorConfig:
  """
  Create a configuration optimized for development.

  Uses gpt-4o-mini for cost efficiency, higher temperature for creativity,
  more iterations for exploration, and longer test timeout for complex scenarios.

  Returns:
    SupervisorConfig optimized for development use

  Example:
    >>> config = development_config()
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    agent=AgentConfig(
      model_name='gpt-4o-mini',
      temperature=0.2,
      provider='openai',
      max_iterations=5,
      test_timeout=60,
    ),
  )


def production_config() -> SupervisorConfig:
  """
  Create a configuration optimized for production.

  Uses gpt-4o for higher quality, lower temperature for consistency,
  and optimized iteration/timeout settings for reliability.

  Returns:
    SupervisorConfig optimized for production use

  Example:
    >>> config = production_config()
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    agent=AgentConfig(
      model_name='gpt-4o',
      temperature=0.1,
      provider='openai',
      max_iterations=5,
      test_timeout=60,
    ),
  )


def plan_mode_config(
  max_plan_iterations: int = 3,
  plan_review_enabled: bool = True,
  plan_auto_approval_threshold: float = 0.8
) -> SupervisorConfig:
  """
  Create a configuration with plan mode enabled.

  This configuration enables Claude Code's plan mode functionality with intelligent
  LLM-powered plan review and iterative refinement.

  Args:
    max_plan_iterations: Maximum plan review cycles before auto-approval (default: 3)
    plan_review_enabled: Enable LLM-powered plan review and refinement (default: True)
    plan_auto_approval_threshold: LLM confidence score threshold for auto-approval (default: 0.8)

  Returns:
    SupervisorConfig with plan mode enabled and specified review settings

  Example:
    >>> # Thorough plan review
    >>> config = plan_mode_config(max_plan_iterations=5, plan_auto_approval_threshold=0.9)
    >>> agent = SupervisorAgent(config=config)

    >>> # Quick plan mode
    >>> config = plan_mode_config(max_plan_iterations=1, plan_review_enabled=False)
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    claude_code=ClaudeCodeConfig(
      max_plan_iterations=max_plan_iterations,
      plan_review_enabled=plan_review_enabled,
      plan_auto_approval_threshold=plan_auto_approval_threshold
    )
  )


def plan_mode_development_config(
  max_plan_iterations: int = 5,
  plan_auto_approval_threshold: float = 0.7
) -> SupervisorConfig:
  """
  Create a plan mode configuration optimized for development.

  Uses more permissive settings for exploration and learning, with more iterations
  and lower approval threshold to encourage experimentation.

  Args:
    max_plan_iterations: Maximum plan review cycles (default: 5 for more exploration)
    plan_auto_approval_threshold: Approval threshold (default: 0.7 for more permissive)

  Returns:
    SupervisorConfig optimized for development with plan mode enabled

  Example:
    >>> config = plan_mode_development_config()
    >>> agent = SupervisorAgent(config=config)
  """
  return SupervisorConfig(
    agent=AgentConfig(
      model_name='gpt-4o-mini',
      temperature=0.2,
      provider='openai',
      max_iterations=5,
      test_timeout=60,
    ),
    claude_code=ClaudeCodeConfig(
      max_plan_iterations=max_plan_iterations,
      plan_review_enabled=True,
      plan_auto_approval_threshold=plan_auto_approval_threshold
    )
  )
