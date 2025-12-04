# Claude Code Supervisor

[![PyPI version](https://badge.fury.io/py/claude-code-supervisor.svg)](https://badge.fury.io/py/claude-code-supervisor)
[![Python Support](https://img.shields.io/pypi/pyversions/claude-code-supervisor.svg)](https://pypi.org/project/claude-code-supervisor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent wrapper around Claude Agent SDK that provides automated problem-solving capabilities with session management, progress monitoring, and intelligent feedback loops.

## 🚀 Features

- **Automated Problem Solving**: Describes problems to Claude Code and gets complete solutions
- **Session Management**: Maintains context across multiple iterations with intelligent workflow orchestration
- **Progress Monitoring**: Real-time tracking of Claude's progress via todo list updates and output analysis
- **Intelligent Feedback Loop**: LLM-powered guidance generation that analyzes Claude's work and provides specific, actionable feedback when issues arise
- **🆕 Custom Tools Support** (v1.1.0): Extend capabilities by providing custom tools to coding agent, supervisor, or both
- **Plan Mode**: Intelligent plan generation, review, and iterative refinement before execution (inspired by validation patterns)
- **Custom Prompt Overrides**: Separate customizable prompts for execution and plan mode instructions
- **Dual Graph Architecture**: Independent plan graph and execution graph for clean separation of concerns
- **Data I/O Support**: Handles various data formats (lists, dicts, CSV, DataFrames, etc.)
- **Custom Prompts**: Guide implementation toward specific patterns or requirements
- **Test Automation**: Automatically generates and runs tests for solutions
- **Multiple Providers**: Support for Anthropic, AWS Bedrock, and OpenAI

## 🧠 How It Works

Claude Code Supervisor uses a **two-agent architecture** where agents work together to solve coding problems:

### The Two Agents

1. **Coding Agent (Claude Code)**
   - Autonomous coding assistant that writes, edits, and tests code
   - Has access to tools: Read, Write, Edit, Bash, Grep, etc.
   - Can plan its own work using todo lists
   - Executes the actual implementation

2. **Supervisor Agent (LLM)**
   - Orchestrates the coding process
   - Analyzes the coding agent's work
   - Generates intelligent feedback when issues arise
   - Decides when solution is complete or needs refinement

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT (LLM)                   │
│  • Receives problem description from user                   │
│  • Generates instructions for coding agent                  │
│  • Reviews results and provides feedback                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Instructions
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 CODING AGENT (Claude Code)                  │
│  • Reads the instructions                                   │
│  • Plans work with todo lists                               │
│  • Writes/edits code using tools                            │
│  • Runs tests to verify solution                            │
│  • Reports results back                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │ Results (code, tests, logs)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT (LLM)                   │
│  • Analyzes results and test outcomes                       │
│  • If successful: Done! ✓                                   │
│  • If failed: Generate specific feedback → repeat cycle     │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits of This Architecture

- **Separation of Concerns**: Coding agent focuses on implementation, supervisor focuses on strategy
- **Intelligent Feedback**: Supervisor provides targeted guidance instead of generic retries
- **Autonomous Operation**: No human intervention needed during iterations
- **Cost Effective**: Use expensive models (GPT-4) for supervisor, efficient models for coding
- **Extensible**: Add custom tools to either agent independently

### Example Flow

```python
# User provides problem
agent = FeedbackSupervisorAgent()
result = agent.process("Create a sorting algorithm with tests")

# Behind the scenes:
# 1. Supervisor → Coding Agent: "Create a sorting function with comprehensive tests"
# 2. Coding Agent: Writes sort.py and test_sort.py, runs tests
# 3. Coding Agent → Supervisor: "Tests failing: IndexError on line 15"
# 4. Supervisor analyzes: "Array bounds issue detected"
# 5. Supervisor → Coding Agent: "Fix the index bounds check in line 15"
# 6. Coding Agent: Fixes the bug, re-runs tests
# 7. Coding Agent → Supervisor: "All tests passing!"
# 8. Supervisor: "Solution complete!" ✓
```

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

# 🆕 Custom prompts for both execution and plan mode
result = agent.process(
    "Create a data processing pipeline",
    instruction_prompt="Use functional programming with immutable data structures",
    plan_mode_instruction_prompt="Focus on scalability and performance optimization",
    enable_plan_mode=True
)
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

### 🆕 With Plan Mode (Intelligent Planning)

```python
# Enable plan mode with intelligent review and refinement
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import plan_mode_config

config = plan_mode_config(
    max_plan_iterations=3,
    plan_auto_approval_threshold=0.8,
    plan_review_enabled=True
)

agent = FeedbackSupervisorAgent(config=config)
result = agent.process(
    "Create a comprehensive calculator module with advanced operations",
    solution_path='calculator.py',
    test_path='test_calculator.py',
    enable_plan_mode=True  # Enable plan mode for this specific task
)

# Alternative: Pass enable_plan_mode directly without config changes
agent = FeedbackSupervisorAgent()
result = agent.process(
    "Create a web scraper with error handling",
    enable_plan_mode=True,
    plan_mode_instruction_prompt="Focus on robustness and rate limiting"
)

# Plan mode workflow:
# 1. Generates execution plan using Claude Code's plan mode
# 2. LLM reviews plan and scores quality (0.0-1.0)
# 3. Iteratively refines plan based on feedback (if needed)
# 4. Executes approved plan with full implementation
```

### 🆕 With Custom Tools (v1.1.0+)

Extend supervisor capabilities by providing custom tools to either the coding agent (Claude Code) or the supervisor LLM, or both:

```python
from claude_agent_sdk import tool
from claude_code_supervisor import FeedbackSupervisorAgent
from claude_code_supervisor.config import SupervisorConfig, ClaudeCodeConfig

# Define a custom tool using @tool decorator
@tool(
    name='validate_code_style',
    description='Validates code follows project style guidelines',
    input_schema={'code': str, 'language': str}
)
async def validate_code_style(args: dict) -> dict:
    '''Custom validation logic for your project'''
    code = args['code']
    language = args.get('language', 'python')

    # Your validation logic here
    is_valid = True  # ... validation code ...

    return {
        'content': [{
            'type': 'text',
            'text': f'Code style is {"valid" if is_valid else "invalid"}'
        }]
    }

# Scenario 1: Tools for coding agent only (extends Claude Code capabilities)
config = SupervisorConfig(
    claude_code=ClaudeCodeConfig(
        coding_agent_tools=[validate_code_style]
    )
)
agent = FeedbackSupervisorAgent(config=config)
result = agent.process('Create a Python module with proper style')

# Scenario 2: Tools for both agents (shared capabilities)
@tool('analyze_complexity', 'Analyze code complexity', {'code': str})
async def analyze_complexity(args: dict) -> dict:
    # Complexity analysis logic
    return {'content': [{'type': 'text', 'text': 'Complexity: O(n)'}]}

config = SupervisorConfig(
    claude_code=ClaudeCodeConfig(
        shared_tools=[analyze_complexity]  # Available to both supervisor and coding agent
    )
)
agent = FeedbackSupervisorAgent(config=config)

# Scenario 3: Separate tools for each agent
from langchain_core.tools import tool as langchain_tool

# Define a LangChain tool for supervisor
@langchain_tool
def review_architecture(code: str) -> str:
    '''Reviews code architecture and provides feedback'''
    # Supervisor-specific analysis
    return 'Architecture looks good'

config = SupervisorConfig(
    claude_code=ClaudeCodeConfig(
        coding_agent_tools=[validate_code_style],   # SDK tools for Claude Code
        supervisor_agent_tools=[review_architecture], # LangChain tools for supervisor LLM
        shared_tools=[analyze_complexity]            # SDK tools for Claude Code (both agents coming soon)
    )
)
```

**Tool Types:**
- **`coding_agent_tools`**: SDK tools (using `@tool` decorator from `claude_agent_sdk`) for Claude Code agent
- **`supervisor_agent_tools`**: LangChain tools (using `@tool` from `langchain_core.tools`) for supervisor LLM
- **`shared_tools`**: SDK tools for Claude Code agent (supervisor support coming in future release)

**Benefits of Custom Tools:**
- Extend capabilities per project without modifying supervisor code
- Reusable tools across different projects
- Type-safe tool definitions with appropriate decorators
- Clean separation: tools can target specific agents
- Full access to SDK's MCP features for coding agent
- LangChain's tool ecosystem for supervisor LLM

**Use Cases:**
- Project-specific validation (coding style, architecture patterns)
- Domain-specific analysis (database queries, API contracts)
- Integration with external services (linters, formatters, CI/CD)
- Custom test frameworks or validation logic
- Data processing and transformation utilities

See [SDK_UPGRADE_PLAN.md](SDK_UPGRADE_PLAN.md) Appendix for more detailed examples.

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
- **🆕 Plan Mode** (`plan_mode_example.py`): Intelligent planning with review and refinement
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
from claude_code_supervisor.config import (
    development_config, production_config,
    plan_mode_config, plan_mode_development_config
)

# Development environment (uses gpt-4o-mini, higher iterations)
dev_config = development_config()
dev_agent = FeedbackSupervisorAgent(config=dev_config)

# Production environment (uses gpt-4o, optimized settings)
prod_config = production_config()
prod_agent = FeedbackSupervisorAgent(config=prod_config)

# 🆕 Plan mode configurations
# Thorough plan review for complex tasks
plan_config = plan_mode_config(
    max_plan_iterations=5,
    plan_auto_approval_threshold=0.9
)
plan_agent = FeedbackSupervisorAgent(config=plan_config)

# Plan mode optimized for development
dev_plan_config = plan_mode_development_config()
dev_plan_agent = FeedbackSupervisorAgent(config=dev_plan_config)
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

## 🏗️ Architecture Improvements

### Dual Graph Architecture
The supervisor now uses two independent LangGraph workflows:

- **Plan Graph**: Handles plan generation, review, and refinement (when plan mode is enabled)
- **Execution Graph**: Handles the main implementation workflow

This separation provides cleaner architecture, better error isolation, and more focused workflows.

### Unified Claude Integration
The `_claude_run` method now supports both `PlanState` and `WorkflowState` directly, eliminating unnecessary state conversions and improving performance.

### Utility Functions
Core utility functions are organized in `claude_code_supervisor.utils`:

```python
from claude_code_supervisor.utils import is_quota_error, node_encountered_quota_error

# Check for API quota/credit errors in text
if is_quota_error(error_message):
    print("API quota exceeded")

# Check if workflow state indicates quota errors  
if node_encountered_quota_error(workflow_state):
    print("Node encountered quota error")
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

- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) for the core Claude Code integration
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) for LLM integrations

## 📚 Documentation

For detailed usage examples, see the [examples directory](examples/) and the configuration examples above.

## 🐛 Issues

Found a bug? Have a feature request? Please [open an issue](https://github.com/vinyluis/claude-code-supervisor/issues).

---

**Made with ❤️ by [Vinícius Trevisan](mailto:vinicius@viniciustrevisan.com)**