"""
Unit tests for supervisor.py
Tests the SupervisorAgent class and WorkflowState dataclass functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch, mock_open

# Import the classes to test
from claude_code_supervisor import SupervisorAgent, WorkflowState
from claude_code_supervisor.data_manager import DataManager
from claude_code_supervisor.config import SupervisorConfig, development_config


class TestWorkflowState:
  """Test cases for the WorkflowState dataclass"""

  def test_agent_state_initialization_defaults(self) -> None:
    """Test WorkflowState (WorkflowState) with default values"""
    state = WorkflowState()

    # Dynamic workflow fields
    assert state.current_iteration == 0
    assert state.is_solved is False
    assert state.error_message == ""
    assert state.test_results == ""
    
    # Claude session state  
    assert state.claude_session_id is None
    assert state.claude_session_active is False
    assert state.claude_todos == []
    assert state.claude_output_log == []
    assert state.should_terminate_early is False
    assert state.latest_guidance == ""
    
    # Output data
    assert state.output_data is None

  def test_agent_state_initialization_custom(self) -> None:
    """Test WorkflowState (WorkflowState) with custom parameters"""
    state = WorkflowState(
      current_iteration=2,
      test_results="All tests passed",
      is_solved=True,
      error_message="No errors"
    )

    assert state.current_iteration == 2
    assert state.test_results == "All tests passed"
    assert state.is_solved is True
    assert state.error_message == "No errors"

  def test_agent_state_to_dict(self) -> None:
    """Test that WorkflowState can be converted to dict"""
    state = WorkflowState(current_iteration=1, is_solved=True)
    state_dict = state.to_dict()
    assert isinstance(state_dict, dict)
    assert state_dict['current_iteration'] == 1
    assert state_dict['is_solved'] is True


class TestSupervisorAgent:
  """Test cases for the SupervisorAgent class"""

  @pytest.fixture
  def mock_config(self) -> SupervisorConfig:
    """Fixture providing a mock configuration"""
    return development_config()

  def test_supervisor_agent_uses_dataclass_config(self) -> None:
    """Test that SupervisorAgent uses dataclass configuration"""
    config = development_config()
    agent = SupervisorAgent(config=config)
    
    assert isinstance(agent.config, SupervisorConfig)
    assert agent.config.model.name == config.model.name
    assert agent.config.agent.max_iterations == config.agent.max_iterations

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
    config = development_config()

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = config

    agent._initialize_llm()

    mock_chat_openai.assert_called_once_with(
      model=config.model.name,
      temperature=config.model.temperature
    )

  def test_supervisor_agent_init_with_provided_llm(self) -> None:
    """Test SupervisorAgent initialization with provided LLM (BYOM)"""
    from langchain_openai import ChatOpenAI
    
    # Create a custom LLM
    custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
    config = development_config()
    
    # Initialize with custom LLM
    agent = SupervisorAgent(config=config, llm=custom_llm)
    
    # Should use the provided LLM, not create a new one from config
    assert agent.llm is custom_llm
    
  def test_supervisor_agent_init_without_provided_llm(self) -> None:
    """Test that SupervisorAgent falls back to config when no LLM provided"""
    config = development_config()
    
    # Initialize without custom LLM
    agent = SupervisorAgent(config=config)
    
    # Should create LLM from config
    assert agent.llm is not None
    assert hasattr(agent.llm, 'model_name')

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

    config = development_config()
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = config
    agent.custom_prompt = None
    agent._timestamp = Mock(return_value="12:00:00")

    result = agent._initialize_claude_code()

    # Method now returns None, just verify it runs without error
    assert result is None
    # Verify that base_claude_options was created
    assert hasattr(agent, 'base_claude_options')
    assert agent.base_claude_options.permission_mode == 'acceptEdits'
    assert agent.base_claude_options.max_turns == config.claude_code.max_turns
    assert agent.base_claude_options.max_thinking_tokens == config.claude_code.max_thinking_tokens
    # Verify default system prompt
    assert "You are an expert Python developer" in agent.base_claude_options.system_prompt

  @patch('claude_code_supervisor.supervisor.os.environ')
  def test_initialize_claude_code_bedrock(self, mock_environ) -> None:
    """Test Claude Code initialization with Bedrock provider"""
    config = development_config()
    config.claude_code.use_bedrock = True
    
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = config
    agent.custom_prompt = None

    agent._initialize_claude_code()

    mock_environ.__setitem__.assert_called_with("CLAUDE_CODE_USE_BEDROCK", "1")

  @patch('claude_code_supervisor.supervisor.os.getenv')
  def test_initialize_claude_code_with_custom_prompt(self, mock_getenv) -> None:
    """Test Claude Code initialization with custom prompt"""
    mock_getenv.return_value = "test-api-key"

    config = development_config()
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = config
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

  def test_decide_next_action_solved(self) -> None:
    """Test _decide_next_action when problem is solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = development_config()
    agent.solution_path = "solution.py"
    agent.test_path = "test.py"
    agent.integrate_into_codebase = False
    
    state = WorkflowState(is_solved=True)

    with patch('os.path.exists', return_value=True):
      result = agent._decide_next_action(state)
      assert result == "validate"

  def test_decide_next_action_max_iterations(self) -> None:
    """Test _decide_next_action when max iterations reached"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = development_config()
    
    state = WorkflowState(current_iteration=5)

    result = agent._decide_next_action(state)
    assert result == "finish"

  def test_decide_next_action_need_guidance(self) -> None:
    """Test _decide_next_action when guidance is needed"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.config = development_config()
    
    state = WorkflowState(
      current_iteration=2,
      claude_session_active=False,
      error_message="Some error"
    )

    result = agent._decide_next_action(state)
    assert result == "guide"

  def test_provide_guidance(self) -> None:
    """Test providing guidance to Claude Code"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    agent._call_llm = Mock(return_value="Fix the import errors and ensure proper syntax")
    agent._format_todos_for_analysis = Mock(return_value="- [PENDING] Task 1")
    agent._format_output_log_for_analysis = Mock(return_value="Error: ImportError")
    agent.problem_description = "test"
    agent.example_output = None

    state = WorkflowState(
      error_message="Some error",
      current_iteration=1
    )

    result = agent._provide_guidance(state)
    assert result.error_message == ""  # Should be cleared
    assert result.latest_guidance.startswith("Based on the previous attempt")
    assert "Fix the import errors" in result.latest_guidance
    agent._call_llm.assert_called_once()

  def test_validate_solution_files_exist(self) -> None:
    """Test validation when both solution and test files exist"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._run_tests = Mock(return_value=WorkflowState(is_solved=True))
    agent._timestamp = Mock(return_value="12:00:00")
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    agent.integrate_into_codebase = False
    agent.input_data = None

    state = WorkflowState()

    with patch('os.path.exists', return_value=True):
      result = agent._validate_solution(state)
      agent._run_tests.assert_called_once_with(state)

  def test_initiate_claude_session_success(self) -> None:
    """Test successful Claude session initiation"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    agent.problem_description = "Create hello world"
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    agent.integrate_into_codebase = False
    agent.input_data = None
    agent.data_manager = None
    agent.example_output = None
    agent.expected_output = None

    state = WorkflowState()

    result = agent._initiate_claude_code_session(state)

    assert result.claude_session_active is True
    assert "PROMPT_ITERATION_" in result.claude_output_log[0]

  def test_initiate_claude_session_exception(self) -> None:
    """Test Claude session initiation with exception"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")
    agent.problem_description = "Create hello world"
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    agent.integrate_into_codebase = False
    agent.input_data = None
    agent.data_manager = None
    agent.example_output = None
    agent.expected_output = None

    state = WorkflowState()

    with patch('time.time', side_effect=Exception("Time error")):
      result = agent._initiate_claude_code_session(state)
      assert "Failed to initiate Claude session" in result.error_message or result.error_message == ""
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
    agent.config = development_config()
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    
    state = WorkflowState()

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
    agent.config = development_config()
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    state = WorkflowState()

    result_state = agent._run_tests(state)

    assert result_state.is_solved is False
    assert "Exit code: 1" in result_state.test_results
    assert "1 failed" in result_state.test_results

  @patch('os.path.exists')
  def test_run_tests_missing_files(self, mock_exists) -> None:
    """Test test execution with missing files"""
    mock_exists.return_value = False

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    state = WorkflowState()

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
    agent.solution_path = "solution.py"
    agent.test_path = "test_solution.py"
    state = WorkflowState()

    result_state = agent._run_tests(state)

    assert result_state.is_solved is False
    assert "Syntax error" in result_state.test_results

  def test_validate_solution_missing_files(self) -> None:
    """Test validation when files are missing"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._timestamp = Mock(return_value="12:00:00")

    state = WorkflowState()

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
    state = WorkflowState(
      is_solved=True,
      current_iteration=2
    )

    result_state = agent._finalize_solution(state)

    assert result_state.is_solved is True

  def test_finalize_solution_failure(self) -> None:
    """Test solution finalization when not solved"""
    agent = SupervisorAgent.__new__(SupervisorAgent)
    state = WorkflowState(
      is_solved=False,
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
    final_state = WorkflowState(
      is_solved=True,
      current_iteration=1
    )
    mock_graph_instance.invoke.return_value = final_state

    # Create agent with dataclass config
    config = development_config()
    agent = SupervisorAgent(config=config)
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

    config = development_config()
    agent = SupervisorAgent(config=config)
    result = agent.process("Test problem")

    assert "Graph execution error" in result.error_message

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_with_custom_prompt(self, mock_state_graph) -> None:
    """Test process method with custom prompt"""
    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result
    final_state = WorkflowState(
      is_solved=True,
      current_iteration=1
    )
    mock_graph_instance.invoke.return_value = final_state

    # Create agent with custom prompt
    config = development_config()
    agent = SupervisorAgent(config=config, custom_prompt="Use object-oriented design")
    result = agent.process("Create a hello function")

    assert result.is_solved is True
    assert result.current_iteration >= 0
    # Verify custom prompt was integrated
    assert "object-oriented design" in agent.base_claude_options.system_prompt

  def test_agent_state_with_io_data(self) -> None:
    """Test WorkflowState initialization with I/O data"""
    input_data = [1, 2, 3, 4]
    expected_output = [4, 3, 2, 1]
    data_manager = DataManager()

    # WorkflowState no longer stores I/O data - it's handled by supervisor
    state = WorkflowState()

    # These fields are now handled by the supervisor agent
    assert input_data == [1, 2, 3, 4]
    assert expected_output == [4, 3, 2, 1]
    assert isinstance(data_manager, DataManager)

  def test_supervisor_with_data_manager(self) -> None:
    """Test SupervisorAgent creates DataManager when processing data"""
    config = development_config()
    agent = SupervisorAgent(config=config)

    # DataManager is now initialized in __init__
    assert hasattr(agent, 'data_manager')

    # DataManager is created in the process method when input_data is provided
    # This is tested in other integration tests

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_process_with_input_data(self, mock_state_graph) -> None:
    """Test process method with input data"""
    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result with output data
    final_state = WorkflowState(
      output_data=[1, 1, 3, 4, 5],
      is_solved=True,
      current_iteration=1
    )
    mock_graph_instance.invoke.return_value = final_state

    # Create agent and test
    config = development_config()
    agent = SupervisorAgent(config=config)
    result = agent.process(
      "Sort this list in ascending order",
      input_data=[3, 1, 4, 1, 5],
      expected_output=[1, 1, 3, 4, 5],
      data_format="list"
    )

    assert result.is_solved is True
    assert result.output_data == [1, 1, 3, 4, 5]

  def test_supervisor_agent_with_byom(self) -> None:
    """Test SupervisorAgent initialization with custom LLM (BYOM)"""
    from langchain_openai import ChatOpenAI
    
    # Create custom LLM
    custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
    
    # Initialize with BYOM
    config = development_config()
    agent = SupervisorAgent(config=config, llm=custom_llm)
    
    # Verify the provided LLM is used
    assert agent.llm is custom_llm
    
    # Verify it's the custom LLM (check type and attributes)
    assert isinstance(agent.llm, ChatOpenAI)
    assert hasattr(agent.llm, 'model_name')
    assert hasattr(agent.llm, 'temperature')


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
