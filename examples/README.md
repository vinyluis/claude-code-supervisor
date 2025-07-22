# Claude Code Supervisor Examples

This directory contains practical examples demonstrating how to use the Claude Code Supervisor in various scenarios. Each example is focused on a specific use case and can be run independently.

## Available Examples

### Basic Usage
- **`basic_usage.py`** - Simplest way to use SupervisorAgent without input/output data

### Supervisor Types
- **`feedback_usage.py`** - FeedbackSupervisorAgent with complex problems and iteration
- **`singleshot_usage.py`** - SingleShotSupervisorAgent for fast, single-execution solutions

### Data Processing Examples
- **`list_sorting_example.py`** - Sort a list of numbers with input/output data
- **`dictionary_processing_example.py`** - Process employee dictionaries, sort by salary
- **`csv_processing_example.py`** - Work with inventory data (simulated CSV processing)

### Custom Prompt Examples
- **`oop_prompt_example.py`** - Object-oriented programming patterns and SOLID principles (shows import options)
- **`performance_prompt_example.py`** - Performance-optimized implementations with Big O analysis
- **`data_science_prompt_example.py`** - Data science best practices using pandas and numpy

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install -g @anthropic-ai/claude-code
   pip install -r requirements.txt
   ```

2. **Set API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key"
   export OPENAI_API_KEY="your-openai-key"  # For LLM guidance
   ```

3. **Run any example:**
   ```bash
   python examples/basic_usage.py
   python examples/list_sorting_example.py
   python examples/oop_prompt_example.py
   ```

## Example Categories

### 🚀 **Basic Usage**
Perfect for getting started:
```bash
python examples/basic_usage.py
```
- No input/output data required
- Simple function creation
- Basic workflow demonstration

### 📊 **Data Processing**
Working with different data types:
```bash
python examples/list_sorting_example.py         # Lists and numbers
python examples/dictionary_processing_example.py # Dictionaries and objects
python examples/csv_processing_example.py        # Complex data structures
```

### 🎯 **Custom Prompts**
Guiding implementation style:
```bash
python examples/oop_prompt_example.py           # Object-oriented design
python examples/performance_prompt_example.py   # Performance optimization
python examples/data_science_prompt_example.py  # Data science practices
```

## What Each Example Demonstrates

| Example | Input Data | Custom Prompt | Key Features |
|---------|------------|---------------|--------------|
| `basic_usage.py` | None | None | Basic workflow, error handling |
| `list_sorting_example.py` | List of numbers | None | Simple I/O, list processing |
| `dictionary_processing_example.py` | Employee dicts | None | Complex objects, sorting |
| `csv_processing_example.py` | Inventory data | None | Business logic, calculations |
| `oop_prompt_example.py` | None | OOP focus | Classes, encapsulation, SOLID |
| `performance_prompt_example.py` | Number (N) | Performance | Algorithms, Big O analysis |
| `data_science_prompt_example.py` | Employee stats | Data science | Pandas, numpy, statistics |

## Expected Behavior

Each example will:
1. ✅ Initialize SupervisorAgent
2. 🎯 Present a programming problem  
3. 🤖 Process using Claude Code
4. 📋 Display results:
   - Success/failure status
   - Generated files (`solution.py`, `test_solution.py`)
   - Number of iterations
   - Test results
   - Guidance provided (if any)

## File Structure

```
examples/
├── README.md                           # This file
├── basic_usage.py                      # 🚀 Start here
├── list_sorting_example.py             # 📊 Simple data processing
├── dictionary_processing_example.py    # 📊 Complex data structures  
├── csv_processing_example.py           # 📊 Business logic example
├── oop_prompt_example.py               # 🎯 Object-oriented patterns
├── performance_prompt_example.py       # 🎯 Algorithm optimization
└── data_science_prompt_example.py      # 🎯 Data science practices
```

## Troubleshooting

**Common issues:**

1. **"Claude Code CLI not found"**
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **"API key not configured"**
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **"Import error for supervisor"**
   - Run from project root or check the examples' path setup

4. **"Tests failing"**
   - Normal during development - supervisor will iterate and provide guidance

## Customization

You can modify these examples to:
- 🔄 Change the problems being solved
- 📝 Adjust input/output data formats  
- 🎨 Modify custom prompts
- ➕ Add your own examples

Each example is self-contained and easy to customize for your specific needs.

---

For more advanced usage and configuration options, see the main project documentation.