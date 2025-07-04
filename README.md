# Claude Code Supervisor

[![PyPI version](https://badge.fury.io/py/claude-code-supervisor.svg)](https://badge.fury.io/py/claude-code-supervisor)
[![Python Support](https://img.shields.io/pypi/pyversions/claude-code-supervisor.svg)](https://pypi.org/project/claude-code-supervisor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent wrapper around Claude Code SDK that provides automated problem-solving capabilities with session management, progress monitoring, and intelligent feedback loops.

## 🚀 Features

- **Automated Problem Solving**: Describes problems to Claude Code and gets complete solutions
- **Session Management**: Maintains context across multiple iterations
- **Progress Monitoring**: Tracks Claude's progress via todo list updates and output analysis
- **Intelligent Feedback**: Provides guidance when Claude encounters issues
- **Data I/O Support**: Handles various data formats (lists, dicts, CSV, DataFrames, etc.)
- **Custom Prompts**: Guide implementation toward specific patterns or requirements
- **Test Automation**: Automatically generates and runs tests for solutions
- **Multiple Providers**: Support for Anthropic, AWS Bedrock, and Google Vertex AI

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
from claude_code_supervisor import SupervisorAgent

# Initialize the agent
agent = SupervisorAgent()

# Solve a problem
result = agent.process(
    "Create a function to calculate fibonacci numbers",
    example_output="fib(8) should return 21"
)

if result.is_solved:
    print(f"Solution: {result.solution_path}")
    print(f"Tests: {result.test_path}")
```

### With Input/Output Data

```python
# Process data with input/output examples
result = agent.process(
    "Sort this list in ascending order",
    input_data=[64, 34, 25, 12, 22, 11, 90, 5],
    expected_output=[5, 11, 12, 22, 25, 34, 64, 90],
    data_format="list"
)
```

### With Custom Prompts

```python
# Guide implementation style
agent = SupervisorAgent(
    custom_prompt="Use object-oriented programming with SOLID principles"
)

result = agent.process("Create a calculator with basic operations")
```

### Bring Your Own Model (BYOM)

```python
# Use your own LangChain LLM for guidance
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
agent = SupervisorAgent(llm=custom_llm)
result = agent.process("Create a data processing function")
```

### With Custom Configuration

```python
# Pass configuration as type-safe dataclass
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import openai_config

config = openai_config(model='gpt-4o-mini', temperature=0.1)
config.agent.max_iterations = 3
config.agent.solution_filename = 'my_solution.py'
config.claude_code.max_turns = 25

agent = SupervisorAgent(config=config)
result = agent.process("Create a web scraper function")
```

### Advanced Configuration Examples

```python
# Use structured, type-safe configuration with dataclasses
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import (
    SupervisorConfig, ModelConfig, AgentConfig,
    development_config, openai_config, bedrock_config
)

# Method 1: Use convenience functions
config = development_config()  # Pre-configured for development
agent = SupervisorAgent(config=config)

# Method 2: Use builder functions with customization
config = openai_config(model='gpt-4o-mini', temperature=0.2)
config.agent.max_iterations = 5
config.agent.solution_filename = 'custom_solution.py'
agent = SupervisorAgent(config=config)

# Method 3: Build from scratch with type safety
config = SupervisorConfig(
    model=ModelConfig(name='gpt-4o', temperature=0.1, provider='openai'),
    agent=AgentConfig(max_iterations=3, test_timeout=60)
)
agent = SupervisorAgent(config=config)
result = agent.process("Create a validation function")
```

### Combining Configuration with Custom LLM

```python
# Use dataclass config + custom LLM together
from langchain_aws import ChatBedrockConverse
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, AgentConfig

# Custom LLM for guidance
guidance_llm = ChatBedrockConverse(
    model='anthropic.claude-3-haiku-20240307-v1:0',
    temperature=0.1,
)

# Type-safe configuration (no model config needed since we provide LLM)
config = SupervisorConfig(
    agent=AgentConfig(max_iterations=2, solution_filename='solution.py')
)

agent = SupervisorAgent(config=config, llm=guidance_llm)
result = agent.process("Create a file parser")
```

### Command Line Interface

```bash
# Basic usage
claude-supervisor "Create a hello world function"

# With custom prompt
claude-supervisor "Create a web scraper" --prompt="Use requests and BeautifulSoup"
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

- **Basic Usage**: Simple problem solving without I/O
- **Data Processing**: Working with lists, dictionaries, and complex data
- **Custom Prompts**: Guiding implementation toward specific patterns
- **Advanced Scenarios**: Real-world data processing examples

## 🔧 Configuration

SupervisorAgent uses type-safe dataclass configuration for better IDE support and validation:

### Quick Setup with Convenience Functions

```python
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import openai_config, bedrock_config

# OpenAI configuration
config = openai_config(model='gpt-4o-mini', temperature=0.2)
agent = SupervisorAgent(config=config)

# AWS Bedrock configuration
config = bedrock_config(
  model='anthropic.claude-3-haiku-20240307-v1:0',
)
agent = SupervisorAgent(config=config)
```

### Custom Configuration from Scratch

```python
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, ModelConfig, AgentConfig

# Build custom configuration
config = SupervisorConfig(
  model=ModelConfig(
    name='gpt-4o',
    temperature=0.1,
    provider='openai'
  ),
  agent=AgentConfig(
    max_iterations=5,
    solution_filename='solution.py',
    test_filename='test_solution.py'
  )
)

agent = SupervisorAgent(config=config)
```

### Environment-Specific Configurations

```python
from claude_code_supervisor import SupervisorAgent
from claude_code_supervisor.config import development_config, production_config

# Development environment
dev_config = development_config()
dev_agent = SupervisorAgent(config=dev_config)

# Production environment  
prod_config = production_config()
prod_agent = SupervisorAgent(config=prod_config)
```

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

- [Claude Code SDK](https://github.com/anthropics/claude-code-sdk) for the core Claude Code integration
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) for LLM integrations

## 📚 Documentation

For detailed documentation, visit our [docs](docs/) directory or check out the [API Reference](docs/api.md).

## 🐛 Issues

Found a bug? Have a feature request? Please [open an issue](https://github.com/vinyluis/claude-code-supervisor/issues).

---

**Made with ❤️ by [Vinícius Trevisan](mailto:vinicius@viniciustrevisan.com)**