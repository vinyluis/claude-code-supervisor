"""Configuration dataclasses for SupervisorAgent"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
  """Configuration for the guidance LLM model"""
  name: str = 'gpt-4o'
  temperature: float = 0.1
  provider: str = 'openai'


@dataclass
class AgentConfig:
  """Configuration for the supervisor agent behavior"""
  max_iterations: int = 3
  solution_filename: str = 'solution.py'
  test_filename: str = 'test_solution.py'
  test_timeout: int = 30


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

  model: ModelConfig = field(default_factory=ModelConfig)
  agent: AgentConfig = field(default_factory=AgentConfig)
  claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)


# Convenience functions for common configurations
def openai_config(model: str = 'gpt-4o', temperature: float = 0.1) -> SupervisorConfig:
  """Create a configuration for OpenAI models"""
  return SupervisorConfig(
    model=ModelConfig(
      name=model,
      temperature=temperature,
      provider='openai'
    )
  )


def bedrock_config(
  model: str = 'anthropic.claude-3-5-sonnet-20241022-v2:0',
  temperature: float = 0.1,
) -> SupervisorConfig:
  """Create a configuration for AWS Bedrock models"""
  return SupervisorConfig(
    model=ModelConfig(
      name=model,
      temperature=temperature,
      provider='bedrock',
    ),
    claude_code=ClaudeCodeConfig(use_bedrock=True),
  )


def development_config() -> SupervisorConfig:
  """Create a configuration optimized for development"""
  return SupervisorConfig(
    model=ModelConfig(name='gpt-4o-mini', temperature=0.2, provider='openai'),
    agent=AgentConfig(max_iterations=5, test_timeout=60),
    claude_code=ClaudeCodeConfig(max_turns=30),
  )


def production_config() -> SupervisorConfig:
  """Create a configuration optimized for production"""
  return SupervisorConfig(
    model=ModelConfig(name='gpt-4o', temperature=0.1, provider='openai'),
    agent=AgentConfig(max_iterations=3, test_timeout=45),
    claude_code=ClaudeCodeConfig(max_turns=20),
  )
