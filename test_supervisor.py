#!/usr/bin/env python3
"""
Unit tests for supervisor.py
Tests the SupervisorAgent class and AgentState dataclass functionality.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, mock_open

# Import the classes to test
from supervisor import SupervisorAgent, AgentState


class TestAgentState:
  """Test cases for the AgentState dataclass"""

  def test_agent_state_initialization_defaults(self):
    """Test AgentState with minimal required parameters"""
    state = AgentState(problem_description="Test problem")

    assert state.problem_description == "Test problem"
    assert state.example_output is None
    assert state.current_iteration == 0
    assert state.max_iterations == 5
    assert state.code_content == ""
    assert state.test_content == ""
    assert state.test_results == ""
    assert state.solution_path == ""
    assert state.test_path == ""
    assert state.config is None
    assert state.is_solved is False
    assert state.error_message == ""
    assert state.messages == []

  def test_agent_state_initialization_custom(self):
    """Test AgentState with custom parameters"""
    config = {"test": "config"}
    state = AgentState(
      problem_description="Custom problem",
      example_output="Expected output",
      current_iteration=2,
      max_iterations=10,
      code_content="print('hello')",
      test_content="def test_hello(): pass",
      test_results="All tests passed",
      solution_path="/path/to/solution.py",
      test_path="/path/to/test.py",
      config=config,
      is_solved=True,
      error_message="No errors"
    )

    assert state.problem_description == "Custom problem"
    assert state.example_output == "Expected output"
    assert state.current_iteration == 2
    assert state.max_iterations == 10
    assert state.code_content == "print('hello')"
    assert state.test_content == "def test_hello(): pass"
    assert state.test_results == "All tests passed"
    assert state.solution_path == "/path/to/solution.py"
    assert state.test_path == "/path/to/test.py"
    assert state.config == config
    assert state.is_solved is True
    assert state.error_message == "No errors"
    assert state.messages == []

  def test_agent_state_post_init_messages(self):
    """Test that messages list is initialized in __post_init__"""
    state = AgentState(problem_description="Test", messages=None)
    assert state.messages == []

    custom_messages = ["message1", "message2"]
    state = AgentState(problem_description="Test", messages=custom_messages)
    assert state.messages == custom_messages


class TestSupervisorAgent:
  """Test cases for the SupervisorAgent class"""

  @pytest.fixture
  def mock_config(self):
    """Fixture providing a mock configuration"""
    return {
      "model": {"name": "gpt-4o", "temperature": 0.1},
      "agent": {
        "max_iterations": 5,
        "solution_filename": "solution.py",
        "test_filename": "test_solution.py",
        "test_timeout": 30
      }
    }

  @pytest.fixture
  def temp_config_file(self, mock_config):
    """Fixture creating a temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as f:
      json.dump(mock_config, f)
      temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

  def test_load_config_success(self, temp_config_file, mock_config):
    """Test successful config loading from file"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    config = agent._load_config(temp_config_file)
    assert config == mock_config

  def test_load_config_file_not_found(self):
    """Test config loading when file doesn't exist (uses defaults)"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    config = agent._load_config("nonexistent_config.json")

    expected_defaults = {
      "model": {"name": "gpt-4o", "temperature": 0.1},
      "agent": {
        "max_iterations": 5,
        "solution_filename": "solution.py",
        "test_filename": "test_solution.py",
        "test_timeout": 30
      }
    }
    assert config == expected_defaults

  @patch('builtins.open', mock_open(read_data='invalid json'))
  def test_load_config_invalid_json(self):
    """Test config loading with invalid JSON (should exit)"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    with pytest.raises(SystemExit):
      agent._load_config("config.json")

  @patch('supervisor.ChatOpenAI')
  def test_initialize_llm(self, mock_chat_openai):
    """Test LLM initialization"""
    mock_config = {
      "model": {"name": "gpt-4o", "temperature": 0.1}
    }

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = mock_config

    agent._initialize_llm()

    mock_chat_openai.assert_called_once_with(
      model="gpt-4o",
      temperature=0.1
    )

  def test_clean_code_content_python_blocks(self):
    """Test cleaning code content with Python markdown blocks"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    # Test with ```python blocks
    input_code = "```python\nprint('hello')\nprint('world')\n```"
    expected = "print('hello')\nprint('world')"
    assert agent._clean_code_content(input_code) == expected

    # Test with generic ``` blocks
    input_code = "```\nprint('hello')\nprint('world')\n```"
    expected = "print('hello')\nprint('world')"
    assert agent._clean_code_content(input_code) == expected

  def test_clean_code_content_no_blocks(self):
    """Test cleaning code content without markdown blocks"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    input_code = "  print('hello')\nprint('world')  "
    expected = "print('hello')\nprint('world')"
    assert agent._clean_code_content(input_code) == expected

  def test_clean_code_content_empty(self):
    """Test cleaning empty code content"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    result = agent._clean_code_content("")
    assert "Empty code generated" in result
    assert "pass" in result

  def test_should_continue_solved(self):
    """Test _should_continue when problem is solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(problem_description="test", is_solved=True)

    result = agent._should_continue(state)
    assert result == "finish"

  def test_should_continue_max_iterations(self):
    """Test _should_continue when max iterations reached"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      is_solved=False,
      current_iteration=5,
      max_iterations=5
    )

    result = agent._should_continue(state)
    assert result == "finish"

  def test_should_continue_can_continue(self):
    """Test _should_continue when can continue iterating"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      is_solved=False,
      current_iteration=2,
      max_iterations=5
    )

    result = agent._should_continue(state)
    assert result == "continue"

  @patch('supervisor.ChatOpenAI')
  def test_call_claude_code_sdk_success(self, mock_chat_openai):
    """Test successful Claude Code SDK call"""
    # Setup mock response
    mock_response = Mock()
    mock_response.content = "Generated code content"
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.llm = mock_llm

    result = agent._call_claude_code_sdk("test_operation", "test prompt")

    assert result == "Generated code content"
    mock_llm.invoke.assert_called_once()

  @patch('supervisor.ChatOpenAI')
  def test_call_claude_code_sdk_exception(self, mock_chat_openai):
    """Test Claude Code SDK call with exception"""
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("API Error")

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.llm = mock_llm

    result = agent._call_claude_code_sdk("test_operation", "test prompt")

    assert "Error in test_operation: API Error" in result

  def test_iterate_solution(self):
    """Test _iterate_solution increments iteration counter"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(problem_description="test", current_iteration=1)

    result_state = agent._iterate_solution(state)

    assert result_state.current_iteration == 2

  @patch('builtins.open', new_callable=mock_open)
  @patch('os.path.exists')
  def test_generate_code_success(self, mock_exists, mock_file):
    """Test successful code generation and saving"""
    mock_exists.return_value = True

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._call_claude_code_sdk = Mock(return_value="print('hello world')")

    state = AgentState(
      problem_description="Create hello world",
      solution_path="solution.py"
    )

    result_state = agent._generate_code(state)

    assert result_state.code_content == "print('hello world')"
    mock_file.assert_called_once_with("solution.py", 'w')
    mock_file().write.assert_called_once_with("print('hello world')")

  @patch('builtins.open', side_effect=IOError("Permission denied"))
  def test_generate_code_file_error(self, mock_file):
    """Test code generation with file writing error"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._call_claude_code_sdk = Mock(return_value="print('hello world')")

    state = AgentState(
      problem_description="Create hello world",
      solution_path="solution.py"
    )

    result_state = agent._generate_code(state)

    assert ("Failed to save code: Permission denied" in
            result_state.error_message)

  @patch('subprocess.run')
  @patch('os.path.exists')
  @patch('builtins.open', new_callable=mock_open, read_data="print('test')")
  def test_run_tests_success(self, mock_file, mock_exists, mock_subprocess):
    """Test successful test execution"""
    mock_exists.return_value = True
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "2 passed"
    mock_result.stderr = ""
    mock_subprocess.return_value = mock_result

    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py",
      config={"agent": {"test_timeout": 30}}
    )

    result_state = agent._run_tests(state)

    assert result_state.is_solved is True
    assert "Exit code: 0" in result_state.test_results
    assert "2 passed" in result_state.test_results

  @patch('subprocess.run')
  @patch('os.path.exists')
  @patch('builtins.open', new_callable=mock_open, read_data="print('test')")
  def test_run_tests_failure(self, mock_file, mock_exists, mock_subprocess):
    """Test failed test execution"""
    mock_exists.return_value = True
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = "1 failed"
    mock_result.stderr = "AssertionError"
    mock_subprocess.return_value = mock_result

    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py",
      config={"agent": {"test_timeout": 30}}
    )

    result_state = agent._run_tests(state)

    assert result_state.is_solved is False
    assert "Exit code: 1" in result_state.test_results
    assert "1 failed" in result_state.test_results

  @patch('os.path.exists')
  def test_run_tests_missing_files(self, mock_exists):
    """Test test execution with missing files"""
    mock_exists.return_value = False

    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    result_state = agent._run_tests(state)

    assert result_state.is_solved is False
    assert "does not exist" in result_state.test_results

  @patch('builtins.open', new_callable=mock_open,
         read_data="invalid syntax $$")
  @patch('os.path.exists')
  def test_run_tests_syntax_error(self, mock_exists, mock_file):
    """Test test execution with syntax errors"""
    mock_exists.return_value = True

    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    result_state = agent._run_tests(state)

    assert result_state.is_solved is False
    assert "Syntax error" in result_state.test_results

  def test_evaluate_results_solved(self):
    """Test evaluation when problem is solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(problem_description="test", is_solved=True)

    result_state = agent._evaluate_results(state)

    assert result_state.is_solved is True

  def test_evaluate_results_not_solved(self):
    """Test evaluation when problem is not solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._call_claude_code_sdk = Mock(return_value="Fix the import statement")

    state = AgentState(
      problem_description="test problem",
      is_solved=False,
      test_results="ImportError: No module named 'missing_module'"
    )

    result_state = agent._evaluate_results(state)

    assert result_state.error_message == "Fix the import statement"
    agent._call_claude_code_sdk.assert_called_once()

  def test_finalize_solution_success(self):
    """Test solution finalization when solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      is_solved=True,
      current_iteration=2,
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    result_state = agent._finalize_solution(state)

    assert result_state.is_solved is True

  def test_finalize_solution_failure(self):
    """Test solution finalization when not solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(
      problem_description="test",
      is_solved=False,
      max_iterations=5,
      error_message="Could not solve"
    )

    result_state = agent._finalize_solution(state)

    assert result_state.is_solved is False


class TestSupervisorAgentIntegration:
  """Integration tests for SupervisorAgent workflow"""

  @patch('supervisor.ChatOpenAI')
  @patch('supervisor.StateGraph')
  def test_solve_problem_success_workflow(self, mock_state_graph,
                                          mock_chat_openai):
    """Test complete successful problem-solving workflow"""
    # Mock LLM
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm

    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result
    final_state = AgentState(
      problem_description="Create a hello function",
      is_solved=True,
      current_iteration=1,
      solution_path="solution.py",
      test_path="test_solution.py"
    )
    mock_graph_instance.invoke.return_value = final_state

    # Create agent and run
    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, "config.json")
      config_data = {
        "model": {"name": "gpt-4o", "temperature": 0.1},
        "agent": {
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      result = agent.solve_problem("Create a hello function",
                                   "hello() = 'world'")

      assert result.is_solved is True
      assert result.current_iteration >= 0

  @patch('supervisor.ChatOpenAI')
  @patch('supervisor.StateGraph')
  def test_solve_problem_exception_handling(self, mock_state_graph,
                                            mock_chat_openai):
    """Test exception handling in solve_problem"""
    # Mock LLM
    mock_llm = Mock()
    mock_chat_openai.return_value = mock_llm

    # Mock graph workflow that raises exception
    mock_graph_instance = Mock()
    mock_graph_instance.invoke.side_effect = Exception("Graph execution error")
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, "config.json")
      config_data = {
        "model": {"name": "gpt-4o", "temperature": 0.1},
        "agent": {
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      result = agent.solve_problem("Test problem")

      assert "Graph execution error" in result.error_message


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
