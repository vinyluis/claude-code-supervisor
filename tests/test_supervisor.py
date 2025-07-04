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
from claude_code_supervisor import SupervisorAgent, AgentState
from claude_code_supervisor.data_manager import DataManager


class TestAgentState:
  """Test cases for the AgentState dataclass"""

  def test_agent_state_initialization_defaults(self) -> None:
    """Test AgentState with minimal required parameters"""
    state = AgentState(problem_description="Test problem")

    assert state.problem_description == "Test problem"
    assert state.example_output is None
    assert state.current_iteration == 0
    assert state.max_iterations == 5
    assert state.test_results == ""
    assert state.solution_path == ""
    assert state.test_path == ""
    assert state.config is None
    assert state.is_solved is False
    assert state.error_message == ""
    assert state.guidance_messages == []
    # Session tracking fields
    assert state.claude_session_id is None
    assert state.claude_session_active is False
    assert state.claude_todos == []
    assert state.claude_output_log == []
    assert state.guidance_provided is False
    assert state.last_activity_time > 0  # Should be set in __post_init__
    # Data I/O fields
    assert state.input_data is None
    assert state.expected_output is None
    assert state.data_format == "auto"
    # Removed input_data_files field in new in-memory approach
    assert state.output_data is None
    assert state.data_manager is None

  def test_agent_state_initialization_custom(self) -> None:
    """Test AgentState with custom parameters"""
    config = {"test": "config"}
    state = AgentState(
      problem_description="Custom problem",
      example_output="Expected output",
      current_iteration=2,
      max_iterations=10,
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
    assert state.test_results == "All tests passed"
    assert state.solution_path == "/path/to/solution.py"
    assert state.test_path == "/path/to/test.py"
    assert state.config == config
    assert state.is_solved is True
    assert state.error_message == "No errors"
    assert state.guidance_messages == []

  def test_agent_state_post_init_time(self) -> None:
    """Test that last_activity_time is initialized in __post_init__"""
    state = AgentState(problem_description="Test")
    assert state.last_activity_time > 0


class TestSupervisorAgent:
  """Test cases for the SupervisorAgent class"""

  @pytest.fixture
  def mock_config(self) -> dict:
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
  def temp_config_file(self, mock_config: dict):
    """Fixture creating a temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as f:
      json.dump(mock_config, f)
      temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

  def test_load_config_success(self, temp_config_file, mock_config) -> None:
    """Test successful config loading from file"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    config = agent._load_config(temp_config_file)
    assert config == mock_config

  def test_load_config_file_not_found(self) -> None:
    """Test config loading when file doesn't exist (uses defaults)"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    config = agent._load_config("nonexistent_config.json")

    expected_defaults = {
      "model": {"name": "gpt-4o", "temperature": 0.1},
      "agent": {
        "max_iterations": 3,
        "solution_filename": "solution.py",
        "test_filename": "test_solution.py",
        "test_timeout": 30
      },
      "claude_code": {
        "provider": "anthropic",
        "use_bedrock": False,
        "working_directory": None,
        "javascript_runtime": "node",
        "executable_args": [],
        "claude_code_path": None,
        "session_timeout_seconds": 300,
        "activity_timeout_seconds": 180,
        "max_turns": 20,
        "max_thinking_tokens": 8000
      }
    }
    assert config == expected_defaults

  @patch('builtins.open', mock_open(read_data='invalid json'))
  def test_load_config_invalid_json(self) -> None:
    """Test config loading with invalid JSON (should exit)"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    with pytest.raises(SystemExit):
      agent._load_config("config.json")

  def test_timestamp_format(self) -> None:
    """Test timestamp method returns correct format"""
    agent = SupervisorAgent.__new__(SupervisorAgent)

    timestamp = agent._timestamp()
    # Should be in HH:MM:SS format
    import re
    assert re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp)

  @patch('claude_code_supervisor.supervisor.ChatOpenAI')
  def test_initialize_llm(self, mock_chat_openai) -> None:
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

  @patch('claude_code_supervisor.supervisor.ChatOpenAI')
  def test_call_llm_success(self, mock_chat_openai) -> None:
    """Test successful LLM call"""
    # Setup mock response
    mock_response = Mock()
    mock_response.content = "Generated guidance content"
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.llm = mock_llm
    agent._timestamp = Mock(return_value="12:00:00")

    result = agent._call_llm("test_operation", "test prompt")

    assert result == "Generated guidance content"
    mock_llm.invoke.assert_called_once()

  @patch('claude_code_supervisor.supervisor.ChatOpenAI')
  def test_call_llm_exception(self, mock_chat_openai) -> None:
    """Test LLM call with exception"""
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("API Error")

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.llm = mock_llm
    agent._timestamp = Mock(return_value="12:00:00")

    result = agent._call_llm("test_operation", "test prompt")

    assert "Error in test_operation: API Error" in result

  @patch('claude_code_supervisor.supervisor.os.getenv')
  def test_initialize_claude_code_anthropic(self, mock_getenv) -> None:
    """Test Claude Code initialization with Anthropic provider"""
    mock_getenv.return_value = "test-api-key"
    
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {
      "claude_code": {
        "provider": "anthropic",
        "use_bedrock": False,
        "working_directory": "/test/path",
        "javascript_runtime": "node",
        "executable_args": ["--verbose"],
        "claude_code_path": "/usr/local/bin/claude-code",
        "max_turns": 20,
        "max_thinking_tokens": 8000
      }
    }
    agent.custom_prompt = None
    agent._timestamp = Mock(return_value="12:00:00")

    result = agent._initialize_claude_code()
    
    # Method now returns None, just verify it runs without error
    assert result is None
    # Verify that base_claude_options was created
    assert hasattr(agent, 'base_claude_options')
    assert agent.base_claude_options.permission_mode == 'acceptEdits'
    assert agent.base_claude_options.max_turns == 20
    assert agent.base_claude_options.max_thinking_tokens == 8000
    # Verify default system prompt
    assert "You are an expert Python developer" in agent.base_claude_options.system_prompt

  @patch('claude_code_supervisor.supervisor.os.environ')
  def test_initialize_claude_code_bedrock(self, mock_environ) -> None:
    """Test Claude Code initialization with Bedrock provider"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {
      "claude_code": {
        "provider": "anthropic",
        "use_bedrock": True
      }
    }
    agent.custom_prompt = None

    agent._initialize_claude_code()

    mock_environ.__setitem__.assert_called_with("CLAUDE_CODE_USE_BEDROCK", "1")

  @patch('claude_code_supervisor.supervisor.os.getenv')
  def test_initialize_claude_code_with_custom_prompt(self, mock_getenv) -> None:
    """Test Claude Code initialization with custom prompt"""
    mock_getenv.return_value = "test-api-key"
    
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {
      "claude_code": {
        "provider": "anthropic",
        "use_bedrock": False,
        "max_turns": 20,
        "max_thinking_tokens": 8000
      }
    }
    agent.custom_prompt = "Always use type hints and add comprehensive docstrings"
    agent._timestamp = Mock(return_value="12:00:00")

    result = agent._initialize_claude_code()
    
    # Verify that base_claude_options was created with custom prompt
    assert result is None
    assert hasattr(agent, 'base_claude_options')
    assert "You are an expert Python developer" in agent.base_claude_options.system_prompt
    assert "Always use type hints and add comprehensive docstrings" in agent.base_claude_options.system_prompt
    assert "Additional instructions:" in agent.base_claude_options.system_prompt


  @patch('claude_code_supervisor.supervisor.load_dotenv')
  @patch('claude_code_supervisor.supervisor.Path')
  def test_load_environment_file_exists(self, mock_path, mock_load_dotenv) -> None:
    """Test loading environment when .env file exists"""
    mock_env_path = Mock()
    mock_env_path.exists.return_value = True
    mock_path.return_value = mock_env_path

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._load_environment()

    mock_load_dotenv.assert_called_once_with(mock_env_path)

  @patch('claude_code_supervisor.supervisor.load_dotenv')
  @patch('claude_code_supervisor.supervisor.Path')
  def test_load_environment_file_missing(self, mock_path, mock_load_dotenv) -> None:
    """Test loading environment when .env file is missing"""
    mock_env_path = Mock()
    mock_env_path.exists.return_value = False
    mock_path.return_value = mock_env_path

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._load_environment()

    mock_load_dotenv.assert_not_called()


  def test_should_continue_monitoring_solved(self) -> None:
    """Test _should_continue_monitoring when problem is solved and files exist"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {"claude_code": {}}
    state = AgentState(problem_description="test", is_solved=True)
    
    with patch('os.path.exists', return_value=True):
      result = agent._should_continue_monitoring(state)
      assert result == "validate"

  def test_should_continue_monitoring_max_iterations(self) -> None:
    """Test _should_continue_monitoring when max iterations reached"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {"claude_code": {}}
    state = AgentState(
      problem_description="test",
      is_solved=False,
      current_iteration=5,
      max_iterations=5
    )

    result = agent._should_continue_monitoring(state)
    assert result == "finish"

  def test_should_continue_monitoring_need_guidance(self) -> None:
    """Test _should_continue_monitoring when guidance is needed"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = {"claude_code": {}}
    state = AgentState(
      problem_description="test",
      is_solved=False,
      current_iteration=2,
      max_iterations=5,
      claude_session_active=False,
      error_message="Some error",
      guidance_provided=False
    )

    result = agent._should_continue_monitoring(state)
    assert result == "guide"


  def test_monitor_claude_progress_session_inactive(self) -> None:
    """Test monitoring when Claude session is inactive"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = AgentState(problem_description="test", claude_session_active=False)
    
    result = agent._monitor_claude_progress(state)
    assert result == state  # Should return unchanged

  def test_provide_guidance(self) -> None:
    """Test providing guidance to Claude Code"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    agent._call_llm = Mock(return_value="Fix the import errors and ensure proper syntax")
    agent._format_todos_for_analysis = Mock(return_value="- [PENDING] Task 1")
    agent._format_output_log_for_analysis = Mock(return_value="Error: ImportError")
    
    state = AgentState(
      problem_description="test",
      error_message="Some error",
      guidance_provided=False,
      current_iteration=1
    )

    result = agent._provide_guidance(state)
    assert result.guidance_provided is True
    assert result.error_message == ""  # Should be cleared
    assert len(result.guidance_messages) == 1
    assert "Fix the import errors" in result.guidance_messages[0]['guidance']
    agent._call_llm.assert_called_once()

  def test_validate_solution_files_exist(self) -> None:
    """Test validation when both solution and test files exist"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._run_tests = Mock(return_value=AgentState(problem_description="test", is_solved=True))
    agent._timestamp = Mock(return_value="12:00:00")
    
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    with patch('os.path.exists', return_value=True):
      result = agent._validate_solution(state)
      agent._run_tests.assert_called_once_with(state)

  def test_initiate_claude_session_success(self) -> None:
    """Test successful Claude session initiation"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    
    state = AgentState(
      problem_description="Create hello world",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    result = agent._initiate_claude_session(state)

    assert result.claude_session_active is True
    assert result.last_activity_time > 0
    assert "PROMPT:" in result.claude_output_log[0]

  def test_initiate_claude_session_exception(self) -> None:
    """Test Claude session initiation with exception"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    
    state = AgentState(
      problem_description="Create hello world",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    with patch('time.time', side_effect=Exception("Time error")):
      result = agent._initiate_claude_session(state)
      assert "Failed to initiate Claude session" in result.error_message
      assert result.claude_session_active is False

  @patch('subprocess.run')
  @patch('os.path.exists')
  @patch('builtins.open', new_callable=mock_open, read_data="print('test')")
  def test_run_tests_success(self, mock_file, mock_exists, mock_subprocess) -> None:
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
  def test_run_tests_failure(self, mock_file, mock_exists, mock_subprocess) -> None:
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
  def test_run_tests_missing_files(self, mock_exists) -> None:
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
  def test_run_tests_syntax_error(self, mock_exists, mock_file) -> None:
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

  def test_validate_solution_missing_files(self) -> None:
    """Test validation when files are missing"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    
    state = AgentState(
      problem_description="test",
      solution_path="solution.py",
      test_path="test_solution.py"
    )

    with patch('os.path.exists', return_value=False):
      result = agent._validate_solution(state)
      assert "not created" in result.error_message
      assert result.is_solved is False

  def test_timestamp_method(self) -> None:
    """Test timestamp method returns correct format"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    
    timestamp = agent._timestamp()
    # Should be in HH:MM:SS format
    import re
    assert re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp)

  def test_finalize_solution_success(self) -> None:
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

  def test_finalize_solution_failure(self) -> None:
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

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_success_workflow(self, mock_state_graph) -> None:
    """Test complete successful problem-solving workflow"""
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
        },
        "claude_code": {
          "use_bedrock": False
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      result = agent.process("Create a hello function",
                             example_output="hello() = 'world'")

      assert result.is_solved is True
      assert result.current_iteration >= 0

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_exception_handling(self, mock_state_graph) -> None:
    """Test exception handling in process"""
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
        },
        "claude_code": {
          "use_bedrock": False
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      result = agent.process("Test problem")

      assert "Graph execution error" in result.error_message

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_with_custom_prompt(self, mock_state_graph) -> None:
    """Test process method with custom prompt"""
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

    # Create agent with custom prompt
    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, "config.json")
      config_data = {
        "model": {"name": "gpt-4o", "temperature": 0.1},
        "agent": {
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        },
        "claude_code": {
          "use_bedrock": False
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path, custom_prompt="Use object-oriented design")
      result = agent.process("Create a hello function")

      assert result.is_solved is True
      assert result.current_iteration >= 0
      # Verify custom prompt was integrated
      assert "object-oriented design" in agent.base_claude_options.system_prompt

  def test_agent_state_with_io_data(self) -> None:
    """Test AgentState initialization with I/O data"""
    input_data = [1, 2, 3, 4]
    expected_output = [4, 3, 2, 1]
    data_manager = DataManager()
    
    state = AgentState(
      problem_description="Sort this list in reverse",
      input_data=input_data,
      expected_output=expected_output,
      data_format="list",
      data_manager=data_manager
    )
    
    assert state.input_data == input_data
    assert state.expected_output == expected_output
    assert state.data_format == "list"
    assert state.data_manager == data_manager

  def test_supervisor_with_data_manager(self) -> None:
    """Test SupervisorAgent initialization includes DataManager"""
    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, "config.json")
      config_data = {
        "model": {"name": "gpt-4o", "temperature": 0.1},
        "agent": {
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        },
        "claude_code": {
          "use_bedrock": False
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      
      assert hasattr(agent, 'data_manager')
      assert isinstance(agent.data_manager, DataManager)

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_with_input_data(self, mock_state_graph) -> None:
    """Test process method with input data"""
    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result with output data
    final_state = AgentState(
      problem_description="Sort this list",
      input_data=[3, 1, 4, 1, 5],
      expected_output=[1, 1, 3, 4, 5],
      data_format="list",
      output_data=[1, 1, 3, 4, 5],
      is_solved=True,
      current_iteration=1,
      solution_path="solution.py",
      test_path="test_solution.py"
    )
    mock_graph_instance.invoke.return_value = final_state

    # Create agent and test
    with tempfile.TemporaryDirectory() as temp_dir:
      config_path = os.path.join(temp_dir, "config.json")
      config_data = {
        "model": {"name": "gpt-4o", "temperature": 0.1},
        "agent": {
          "max_iterations": 5,
          "solution_filename": "solution.py",
          "test_filename": "test_solution.py",
          "test_timeout": 30
        },
        "claude_code": {
          "use_bedrock": False
        }
      }
      with open(config_path, 'w') as f:
        json.dump(config_data, f)

      agent = SupervisorAgent(config_path)
      result = agent.process(
        "Sort this list in ascending order",
        input_data=[3, 1, 4, 1, 5],
        expected_output=[1, 1, 3, 4, 5],
        data_format="list"
      )

      assert result.is_solved is True
      assert result.input_data == [3, 1, 4, 1, 5]
      assert result.expected_output == [1, 1, 3, 4, 5]
      assert result.data_format == "list"
      assert result.output_data == [1, 1, 3, 4, 5]


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
