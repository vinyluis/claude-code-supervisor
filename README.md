# Claude Code Supervisor

[![PyPI version](https://badge.fury.io/py/claude-code-supervisor.svg)](https://badge.fury.io/py/claude-code-supervisor)
[![Python Support](https://img.shields.io/pypi/pyversions/claude-code-supervisor.svg)](https://pypi.org/project/claude-code-supervisor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent wrapper around Claude Code SDK that provides automated problem-solving capabilities with session management, progress monitoring, and intelligent feedback loops.

## 🚀 Features

- **Automated Problem Solving**: Describes problems to Claude Code and gets complete solutions
- **Session Management**: Maintains context across multiple iterations with intelligent workflow orchestration
- **Progress Monitoring**: Real-time tracking of Claude's progress via todo list updates and output analysis
- **Intelligent Feedback Loop**: LLM-powered guidance generation that analyzes Claude's work and provides specific, actionable feedback when issues arise
- **Data I/O Support**: Handles various data formats (lists, dicts, CSV, DataFrames, etc.)
- **Custom Prompts**: Guide implementation toward specific patterns or requirements
- **Test Automation**: Automatically generates and runs tests for solutions
- **Multiple Providers**: Support for Anthropic, AWS Bedrock, and OpenAI

## 📦 Installation

### From PyPI (recommended)

```bash
pip install claude-code-supervisor
```

### From Source

```bash
git clone https://github.com/vinyluis/claude-code-supervisor.git
cd claude-code-supervisor
pip install -e .
```

## 🛠️ Prerequisites

1. **Claude Code CLI**:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **API Key** (choose one):
   ```bash
   # Anthropic (default)
   export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
   
   # AWS Bedrock
   export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
   export AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
   export AWS_REGION=<AWS_REGION>

   ```

3. **LLM API Key** (for guidance, choose one):
   ```bash
   # OpenAI (default)
   export OPENAI_API_KEY="your-openai-api-key"
   # Configure supervisor_config.json with "provider": "openai"
   
   # AWS Bedrock (for guidance LLM)
   # Will use the access keys above
   # Configure supervisor_config.json with "provider": "bedrock"
   ```

## 🚀 Quick Start

### Basic Usage

```python
from claude_code_supervisor import SingleShotSupervisorAgent

# Initialize the agent
agent = SingleShotSupervisorAgent()

# Solve a problem
result = agent.process(
    "Create a function to calculate fibonacci numbers",
    solution_path='solution.py',
    test_path='test_solution.py'
)

if result.is_solved:
    print(f"Solution: {agent.solution_path}")
    print(f"Tests: {agent.test_path}")
```

## 🎯 Supervisor Types

Claude Code Supervisor provides two main supervisor types for different use cases:

### FeedbackSupervisorAgent
Iterative supervisor with intelligent feedback loops - continues refining solutions until success or max iterations:

```python
from claude_code_supervisor import FeedbackSupervisorAgent

agent = FeedbackSupervisorAgent()
result = agent.process("Create a complex sorting algorithm")
# Will iterate with intelligent feedback until solved
```

**Best for:**
- Complex problems requiring multiple iterations
- Maximum solution quality with automated improvement
- Problems where first attempts commonly fail
- When you want intelligent error analysis and guidance

### SingleShotSupervisorAgent
Single-execution supervisor without iteration - fast, deterministic results:

```python
from claude_code_supervisor import SingleShotSupervisorAgent

agent = SingleShotSupervisorAgent()
result = agent.process("Create a simple utility function")
# Executes once and reports results
```

**Best for:**
- Simple problems that don't require iteration
- Fast code generation and testing
- When iteration is handled externally
- Benchmarking Claude Code capabilities

### With Input/Output Data

```python
# Process data with input/output examples
result = agent.process(
    "Sort this list in ascending order",
    input_data=[64, 34, 25, 12, 22, 11, 90, 5],
    output_data=[5, 11, 12, 22, 25, 34, 64, 90]
)
```

### With Custom Prompts

```python
# Guide implementation style
agent = FeedbackSupervisorAgent(
    append_system_prompt="Use object-oriented programming with SOLID principles"
)

result = agent.process("Create a calculator with basic operations")
```

### Bring Your Own Model (BYOM)

```python
# Use your own LangChain LLM for guidance
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
agent = FeedbackSupervisorAgent(llm=custom_llm)
result = agent.process("Create a data processing function")
```

### With Custom Configuration

```python
# Pass configuration as type-safe dataclass
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import openai_config

config = openai_config(model_name='gpt-4o-mini', temperature=0.1)
config.agent.max_iterations = 3
config.claude_code.max_turns = 25

agent = FeedbackSupervisorAgent(config=config)
result = agent.process(
    "Create a web scraper function",
    solution_path='scraper.py',
    test_path='test_scraper.py'
)
```

### Advanced Configuration Examples

```python
# Use structured, type-safe configuration with dataclasses
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import (
    SupervisorConfig, AgentConfig, ClaudeCodeConfig,
    development_config, openai_config, bedrock_config
)

# Method 1: Use convenience functions
config = development_config()  # Pre-configured for development
agent = FeedbackSupervisorAgent(config=config)

# Method 2: Use builder functions with customization
config = openai_config(model_name='gpt-4o-mini', temperature=0.2)
config.agent.max_iterations = 5
agent = FeedbackSupervisorAgent(config=config)

# Method 3: Build from scratch with type safety
config = SupervisorConfig(
    agent=AgentConfig(
        model_name='gpt-4o',
        temperature=0.1,
        provider='openai',
        max_iterations=3,
        test_timeout=60
    ),
    claude_code=ClaudeCodeConfig(
        max_turns=20,
        use_bedrock=False,
        tools=['Read', 'Write', 'Edit', 'Bash', 'TodoWrite']  # Custom tool set
    )
)
agent = FeedbackSupervisorAgent(config=config)
result = agent.process(
    "Create a validation function",
    solution_path='validator.py',
    test_path='test_validator.py'
)
```

### Combining Configuration with Custom LLM

```python
# Use dataclass config + custom LLM together
from langchain_aws import ChatBedrockConverse
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, AgentConfig

# Custom LLM for guidance
guidance_llm = ChatBedrockConverse(
    model='anthropic.claude-3-haiku-20240307-v1:0',
    temperature=0.1,
)

# Type-safe configuration (model settings in custom LLM are ignored when llm is provided)
config = SupervisorConfig(
    agent=AgentConfig(max_iterations=2, test_timeout=45)
)

agent = FeedbackSupervisorAgent(config=config, llm=guidance_llm)
result = agent.process(
    "Create a file parser",
    solution_path='parser.py',
    test_path='test_parser.py'
)
```


## 📊 Data Format Support

The supervisor supports various data formats:

- **Lists**: `[1, 2, 3, 4]`
- **Dictionaries**: `{"name": "Alice", "age": 30}`
- **Pandas DataFrames**: For data analysis tasks
- **NumPy Arrays**: For numerical computations
- **Strings**: Text processing tasks
- **CSV Data**: Business logic and data processing

## 🎯 Examples

Check out the [examples directory](examples/) for detailed usage examples:

- **Basic Usage** (`basic_usage.py`): Simple problem solving without I/O
- **Data Processing**: 
  - `list_sorting_example.py`: Working with lists and numbers
  - `dictionary_processing_example.py`: Processing employee dictionaries 
  - `csv_processing_example.py`: Complex inventory data processing
- **Custom Prompts**:
  - `oop_prompt_example.py`: Object-oriented programming patterns
  - `performance_prompt_example.py`: Performance-optimized implementations
  - `data_science_prompt_example.py`: Data science best practices with pandas

## 🔧 Configuration

SupervisorAgent uses type-safe dataclass configuration for better IDE support and validation:

### Quick Setup with Convenience Functions

```python
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import openai_config, bedrock_config

# OpenAI configuration
config = openai_config(model_name='gpt-4o-mini', temperature=0.2)
agent = FeedbackSupervisorAgent(config=config)

# AWS Bedrock configuration
config = bedrock_config(
  model_name='anthropic.claude-3-haiku-20240307-v1:0',
)
agent = FeedbackSupervisorAgent(config=config)
```

### Custom Configuration from Scratch

```python
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, AgentConfig, ClaudeCodeConfig

# Build custom configuration
config = SupervisorConfig(
  agent=AgentConfig(
    model_name='gpt-4o',
    temperature=0.1,
    provider='openai',
    max_iterations=5,
    test_timeout=60
  ),
  claude_code=ClaudeCodeConfig(
    max_turns=25,
    use_bedrock=False,
    max_thinking_tokens=8000
  )
)

agent = FeedbackSupervisorAgent(config=config)
```

### Environment-Specific Configurations

```python
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import development_config, production_config

# Development environment (uses gpt-4o-mini, higher iterations)
dev_config = development_config()
dev_agent = FeedbackSupervisorAgent(config=dev_config)

# Production environment (uses gpt-4o, optimized settings)
prod_config = production_config()
prod_agent = FeedbackSupervisorAgent(config=prod_config)
```

### Tool Configuration

Claude Code has access to various tools. By default, all tools are enabled, but you can customize which tools are available:

```python
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, ClaudeCodeConfig
from claude_code_supervisor.utils import ToolsEnum

# All tools (default)
config = SupervisorConfig(
    claude_code=ClaudeCodeConfig(tools=ToolsEnum.all())
)

# Custom tool set
config = SupervisorConfig(
    claude_code=ClaudeCodeConfig(
        tools=['Read', 'Write', 'Edit', 'Bash', 'TodoWrite', 'LS', 'Grep']
    )
)

# Minimal tools for simple tasks
from claude_code_supervisor.config import minimal_tools_config
config = minimal_tools_config()

# Notebook-focused tools
from claude_code_supervisor.config import notebook_config
config = notebook_config()
```

**Available Tools:**
- `Read`, `Write`, `Edit`, `MultiEdit` - File operations
- `Bash` - Command execution
- `LS`, `Glob`, `Grep` - File system navigation and search
- `TodoWrite` - Task management
- `NotebookRead`, `NotebookEdit` - Jupyter notebook support
- `WebFetch`, `WebSearch` - Web access
- `Agent` - Delegate tasks to other agents

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=claude_code_supervisor

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Claude Code SDK](https://github.com/anthropics/claude-code-sdk-python) for the core Claude Code integration
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) for LLM integrations

## 📚 Documentation

For detailed usage examples, see the [examples directory](examples/) and the configuration examples above.

## 🐛 Issues

Found a bug? Have a feature request? Please [open an issue](https://github.com/vinyluis/claude-code-supervisor/issues).

---

**Made with ❤️ by [Vinícius Trevisan](mailto:vinicius@viniciustrevisan.com)**