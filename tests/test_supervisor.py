"""
Updated tests for supervisor.py to match the new architecture.
Tests focus on the core functionality and current API.
"""

import pytest
from unittest.mock import Mock, patch

# Import the classes to test
from claude_code_supervisor import SupervisorAgent, WorkflowState
from claude_code_supervisor.config import SupervisorConfig, development_config


class TestWorkflowState:
  """Test cases for the WorkflowState dataclass"""

  def test_workflow_state_initialization_defaults(self) -> None:
    """Test WorkflowState with default values"""
    state = WorkflowState()

    # Dynamic workflow fields
    assert state.current_iteration == 0
    assert state.is_solved is False
    assert state.error_message == ""
    assert state.test_results == ""
    assert state.messages == []
    assert state.validation_feedback == ""

    # Claude session state
    assert state.claude_session_id is None
    assert state.claude_session_active is False
    assert state.claude_todos == []
    assert state.claude_log == []
    assert state.should_terminate_early is False

  def test_workflow_state_initialization_custom(self) -> None:
    """Test WorkflowState with custom parameters"""
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

  def test_workflow_state_to_dict(self) -> None:
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

    # Mock the graph building to avoid complex initialization
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=config)

      assert isinstance(agent.config, SupervisorConfig)
      assert agent.config.model.name == config.model.name
      assert agent.config.agent.max_iterations == config.agent.max_iterations

  def test_supervisor_agent_init_with_provided_llm(self) -> None:
    """Test SupervisorAgent initialization with provided LLM (BYOM)"""
    from langchain_openai import ChatOpenAI

    # Create a custom LLM
    custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
    config = development_config()

    # Mock the graph building to avoid complex initialization
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      # Initialize with custom LLM
      agent = SupervisorAgent(config=config, llm=custom_llm)

      # Should use the provided LLM, not create a new one from config
      assert agent.llm is custom_llm

  def test_supervisor_agent_init_without_provided_llm(self) -> None:
    """Test that SupervisorAgent falls back to config when no LLM provided"""
    config = development_config()

    # Mock the graph building and LLM initialization
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()), \
         patch.object(SupervisorAgent, 'initialize_llm'):
      # Initialize without custom LLM
      agent = SupervisorAgent(config=config)

      # Should have an llm attribute after initialization
      assert hasattr(agent, 'llm')

  @patch('claude_code_supervisor.supervisor.ChatOpenAI')
  def test_call_llm_success(self, mock_chat_openai) -> None:
    """Test successful LLM call"""
    # Setup mock response
    mock_response = Mock()
    mock_response.content = "Generated guidance content"
    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

    # Create agent instance with mocked graph
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=development_config())
      agent.llm = mock_llm
      result = agent._call_llm("test_operation", "test prompt")
      assert result == "Generated guidance content"
      mock_llm.invoke.assert_called_once()

  @patch('claude_code_supervisor.supervisor.ChatOpenAI')
  def test_call_llm_exception(self, mock_chat_openai) -> None:
    """Test LLM call with exception"""
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("API Error")
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=development_config())
      agent.llm = mock_llm
      result = agent._call_llm("test_operation", "test prompt")
      assert "Error in test_operation: API Error" in result

  def test_provide_guidance(self) -> None:
    """Test providing guidance to Claude Code"""
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=development_config())
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
      agent._call_llm.assert_called_once()

  def test_format_todos_for_analysis(self) -> None:
    """Test todo formatting for analysis"""
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=development_config())
      todos = [
          {"status": "completed", "content": "Task 1"},
          {"status": "in_progress", "content": "Task 2"},
          {"status": "pending", "content": "Task 3"}
      ]
      result = agent._format_todos_for_analysis(todos)
      assert "- [COMPLETED] Task 1" in result
      assert "- [IN_PROGRESS] Task 2" in result
      assert "- [PENDING] Task 3" in result


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
      with patch.object(SupervisorAgent, 'build_graph', return_value=mock_graph_instance):
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
    with patch.object(SupervisorAgent, 'build_graph', return_value=mock_graph_instance):
      agent = SupervisorAgent(config=config)
      result = agent.process("Test problem")
      assert "Graph execution error" in result.error_message

  def test_supervisor_with_data_handling(self) -> None:
    """Test SupervisorAgent handles data processing directly"""
    config = development_config()
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=config)
      # Data handling is done directly by the supervisor when needed
      assert hasattr(agent, 'input_data')
      assert hasattr(agent, 'output_data')
      # Input data starts as None
      assert agent.input_data is None
      assert agent.output_data is None

  def test_supervisor_agent_with_byom(self) -> None:
    """Test SupervisorAgent initialization with custom LLM (BYOM)"""
    from langchain_openai import ChatOpenAI
    # Create custom LLM
    custom_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
    # Initialize with BYOM
    config = development_config()
    with patch.object(SupervisorAgent, 'build_graph', return_value=Mock()):
      agent = SupervisorAgent(config=config, llm=custom_llm)
      # Verify the provided LLM is used
      assert agent.llm is custom_llm
      # Verify it's the custom LLM (check type and attributes)
      assert isinstance(agent.llm, ChatOpenAI)
      assert hasattr(agent.llm, 'model_name')
      assert hasattr(agent.llm, 'temperature')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
