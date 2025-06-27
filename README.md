# Claude Code Supervisor

An intelligent supervisor agent that uses LangGraph to orchestrate automated code generation, testing, and iterative improvement.

## Overview

This project implements a supervisor agent that can solve programming problems by:
1. Planning the solution approach
2. Generating Python code
3. Creating comprehensive tests
4. Running tests and analyzing failures
5. Iteratively improving the solution until tests pass

## Features

- **Automated code generation** using OpenAI's GPT models
- **Test-driven development** with pytest integration
- **Iterative improvement** based on test failures
- **Configurable workflow** via JSON configuration
- **Multi-framework support** (NumPy, PyTorch, scikit-learn, OpenCV)
- **Comprehensive error handling** and timeout management

## Prerequisites

- Python 3.7+
- OpenAI API key
- Required dependencies (see installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude-code-supervisor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
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

### Complex Problems
```bash
python supervisor.py "Find the maximum element in a list" "max([1,5,3]) = 5"
```

## Configuration

Customize the agent behavior through `config.json`:

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
  }
}
```

## Output Files

The supervisor generates:
- `solution.py` - The generated code solution
- `test_solution.py` - Comprehensive test cases

## Workflow

1. **Plan** - Analyzes the problem and creates a solution plan
2. **Generate Code** - Creates Python code based on the plan
3. **Generate Tests** - Creates comprehensive pytest test cases
4. **Run Tests** - Executes tests and captures results
5. **Evaluate** - Analyzes failures and provides feedback
6. **Iterate** - Improves code based on test failures (up to max iterations)
7. **Finalize** - Completes when tests pass or max iterations reached

## Dependencies

- **Core**: langchain, langgraph, langchain-openai
- **Testing**: pytest
- **Data Science**: numpy, pandas, matplotlib, seaborn, scikit-learn
- **Computer Vision**: opencv-python
- **Deep Learning**: torch, torchvision
- **Claude Integration**: claude-code-sdk

## Error Handling

The supervisor handles:
- Syntax errors in generated code
- Import errors and missing dependencies
- Test timeouts and failures
- API communication issues
- File I/O errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]