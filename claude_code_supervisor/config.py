"""Configuration dataclasses for SupervisorAgent"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
  """Configuration for the supervisor agent behavior"""
  max_iterations: int = 3
  test_timeout: int = 30
  model_name: str = 'gpt-4o'
  temperature: float = 0.1
  provider: str = 'openai'


@dataclass
class ClaudeCodeConfig:
  """Configuration for Claude Code SDK integration"""
  use_bedrock: bool = False
  working_directory: str | None = None
  javascript_runtime: str = 'node'
  executable_args: list[str] = field(default_factory=list)
  claude_code_path: str | None = None
  max_turns: int | None = None
  max_thinking_tokens: int = 8000


@dataclass
class SupervisorConfig:
  """Complete configuration for SupervisorAgent"""
  agent: AgentConfig = field(default_factory=AgentConfig)
  claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)


# Convenience functions for common configurations
def openai_config(model_name: str = 'gpt-4o', temperature: float = 0.1) -> SupervisorConfig:
  """Create a configuration for OpenAI models"""
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
  """Create a configuration for AWS Bedrock models"""
  return SupervisorConfig(
    agent=AgentConfig(
      model_name=model_name,
      temperature=temperature,
      provider='bedrock',
    ),
    claude_code=ClaudeCodeConfig(use_bedrock=True),
  )


def development_config() -> SupervisorConfig:
  """Create a configuration optimized for development"""
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
  """Create a configuration optimized for production"""
  return SupervisorConfig(
    agent=AgentConfig(
      model_name='gpt-4o',
      temperature=0.1,
      provider='openai',
      max_iterations=5,
      test_timeout=60,
    ),
  )
