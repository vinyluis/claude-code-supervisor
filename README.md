# Claude Code Supervisor

An intelligent supervisor agent that wraps Claude Code SDK to provide automated problem-solving with session management, progress monitoring, and intelligent feedback loops.

## Overview

This project implements a supervisor that works as an intelligent wrapper around Claude Code, enabling automated programming problem solving by:

1. **Initiating Claude Code sessions** with structured problem descriptions
2. **Monitoring real-time progress** via SDK message streaming and todo tracking
3. **Validating solutions** through automated test execution
4. **Providing intelligent guidance** when Claude encounters issues
5. **Managing session continuity** across multiple iterations

The key innovation is treating Claude Code as a collaborative coding assistant that can plan its own work using todo lists, rather than a simple LLM that needs pre-planning.

## Architecture

The supervisor uses LangGraph to orchestrate this workflow:

- **initiate_claude**: Start Claude Code session with problem description
- **monitor_claude**: Track progress via SDK message streaming and todo updates  
- **validate_solution**: Run tests on generated files to verify correctness
- **provide_guidance**: Analyze failures and generate actionable feedback using LLM
- **finalize**: Complete the session and report results

## Features

- **Claude Code SDK Integration** - Direct integration with Claude Code as subprocess
- **Session-based Continuity** - Resume sessions across iterations for complex problems
- **Real-time Progress Monitoring** - Track Claude's todo lists and tool usage
- **LLM-powered Guidance** - Intelligent error analysis and feedback generation
- **Configurable Timeouts** - Prevent infinite loops with session and activity timeouts
- **Multi-provider Support** - Anthropic API, Amazon Bedrock, Google Vertex AI
- **Comprehensive Testing** - Automated pytest execution with detailed reporting
- **Timestamped Logging** - Detailed progress tracking with timestamps

## Prerequisites

- Python 3.8+
- Node.js (for Claude Code CLI)
- Anthropic API key (or Bedrock/Vertex access)
- Required Python dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude-code-supervisor
```

2. Install Claude Code CLI:
```bash
npm install -g @anthropic-ai/claude-code
```

3. Install Python dependencies:
```bash
pip install langgraph langchain langchain-openai claude-code-sdk python-dotenv pytest
```

4. Set up environment variables:
```bash
# For Anthropic API (default)
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'  # For guidance analysis

# For Amazon Bedrock (optional)
export CLAUDE_CODE_USE_BEDROCK=1

# For Google Vertex AI (optional)  
export CLAUDE_CODE_USE_VERTEX=1
```

## Configuration

Create `supervisor_config.json` to customize behavior:

```json
{
  "model": {
    "name": "gpt-4o",
    "temperature": 0.1
  },
  "agent": {
    "max_iterations": 5,
    "solution_filename": "solution.py",
    "test_filename": "test_solution.py",
    "test_timeout": 30
  },
  "claude_code": {
    "provider": "anthropic",
    "use_bedrock": false,
    "session_timeout_seconds": 300,
    "activity_timeout_seconds": 180,
    "max_turns": 20,
    "max_thinking_tokens": 8000
  }
}
```

## Usage

### Basic Usage
```bash
python supervisor.py "Create a function to sort a list of numbers"
```

### With Example Output
```bash
python supervisor.py "Calculate fibonacci numbers" "fib(8) = 21"
```

### Programmatic Usage
```python
from supervisor import SupervisorAgent

agent = SupervisorAgent('supervisor_config.json')
result = agent.solve_problem(
    'Create a function to calculate fibonacci numbers',
    'fib(8) should return 21'
)

if result.is_solved:
    print(f'Solution: {result.solution_path}')
    print(f'Tests: {result.test_path}')
    print(f'Completed in {result.current_iteration} iterations')
else:
    print(f'Failed: {result.error_message}')
```

## Workflow Details

### 1. Session Initiation
- Claude Code receives structured problem description
- Supervisor monitors SDK message stream in real-time
- Session ID tracked for continuity

### 2. Progress Monitoring  
- Real-time tracking of Claude's todo list updates
- Tool usage monitoring (Read, Write, Bash, etc.)
- Activity timeout detection to prevent stalls

### 3. Solution Validation
- Automatic detection of generated solution and test files
- Pytest execution with comprehensive error reporting
- Syntax validation before test execution

### 4. Intelligent Guidance
- LLM analysis of Claude's errors and outputs
- Context-aware feedback generation
- Guidance stored in message buffer for next iteration

### 5. Session Continuity
- Resume capability using Claude Code SDK session IDs
- Iteration tracking across multiple attempts
- Progressive guidance accumulation

## Output Files

The supervisor generates:
- `solution.py` - Claude's generated code solution
- `test_solution.py` - Claude's comprehensive test cases
- Console logs with timestamped progress tracking

## Key Components

### AgentState
Tracks supervisor state including:
- Problem description and requirements
- Claude session information and activity
- Todo list progress and output logs
- Guidance messages and error tracking
- Iteration count and solution status

### SupervisorAgent
Main orchestrator providing:
- Configuration management from JSON
- LangGraph workflow coordination
- Claude Code SDK integration
- LLM-powered guidance analysis
- Session and timeout management

## Error Handling

The supervisor handles:
- Claude Code session failures and timeouts
- Syntax errors in generated code
- Test execution failures and timeouts
- API communication issues
- File I/O and path resolution errors
- SDK integration problems

## Testing

Run the test suite:
```bash
python -m pytest test_supervisor.py -v
```

Tests cover:
- AgentState dataclass functionality
- Configuration loading and validation
- LLM integration and guidance generation
- Claude Code SDK integration
- Workflow orchestration
- Error handling scenarios

## Development

### Dependencies
- **Core**: langgraph, langchain-openai, claude-code-sdk
- **Environment**: python-dotenv
- **Testing**: pytest

### Key Files
- `supervisor.py` - Main supervisor implementation
- `supervisor_config.json` - Configuration file
- `test_supervisor.py` - Comprehensive test suite
- `CLAUDE.md` - Claude Code SDK usage instructions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new functionality
4. Ensure all tests pass: `pytest test_supervisor.py -v`
5. Follow code style guidelines in `CLAUDE.md`
6. Submit a pull request

## License

[Add your license information here]