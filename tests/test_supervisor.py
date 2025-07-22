"""
Updated tests for supervisor.py to match the new architecture.
Tests focus on the core functionality and current API.
"""

from unittest.mock import Mock, patch

# Import the classes to test
from claude_code_supervisor import (
    BaseSupervisorAgent,
    FeedbackSupervisorAgent,
    SingleShotSupervisorAgent,
    WorkflowState
)
from claude_code_supervisor.config import development_config


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


class TestBaseSupervisorAgent:
  """Test cases for the BaseSupervisorAgent base class"""

  def test_base_supervisor_agent_shared_functionality(self) -> None:
    """Test that BaseSupervisorAgent has the shared functionality through concrete implementation"""
    config = development_config()
    # Use a concrete implementation to test shared functionality
    agent = FeedbackSupervisorAgent(config=config)

    # Should have shared attributes
    assert hasattr(agent, 'config')
    assert hasattr(agent, 'llm')
    assert hasattr(agent, 'base_claude_options')
    assert hasattr(agent, 'problem_description')
    assert hasattr(agent, 'input_data')
    assert hasattr(agent, 'output_data')

    # Should have shared methods
    assert hasattr(agent, 'load_environment')
    assert hasattr(agent, 'initialize_claude_code')
    assert hasattr(agent, 'initialize_llm')
    assert hasattr(agent, '_call_llm')
    assert hasattr(agent, '_extract_output_data')


class TestFeedbackSupervisorAgent:
  """Test cases for the FeedbackSupervisorAgent class"""

  def test_feedback_supervisor_agent_initialization(self) -> None:
    """Test FeedbackSupervisorAgent initialization"""
    config = development_config()
    agent = FeedbackSupervisorAgent(config=config)

    # Should inherit from base
    assert isinstance(agent, BaseSupervisorAgent)
    assert isinstance(agent, FeedbackSupervisorAgent)

    # Should have a graph for feedback workflow
    assert hasattr(agent, 'graph')
    assert agent.graph is not None

  def test_feedback_supervisor_agent_workflow_methods(self) -> None:
    """Test that FeedbackSupervisorAgent has feedback-specific methods"""
    config = development_config()
    agent = FeedbackSupervisorAgent(config=config)

    # Should have feedback-specific methods
    assert hasattr(agent, 'decide_next_action')
    assert hasattr(agent, 'test_and_analyze')
    assert hasattr(agent, 'should_iterate')
    assert hasattr(agent, 'generate_guidance')
    assert hasattr(agent, 'reduce_message_and_retry')

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_feedback_supervisor_agent_process_workflow(self, mock_state_graph) -> None:
    """Test FeedbackSupervisorAgent process method"""
    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result
    final_state = WorkflowState(is_solved=True, current_iteration=2)
    mock_graph_instance.invoke.return_value = final_state

    config = development_config()
    agent = FeedbackSupervisorAgent(config=config)
    agent.graph = mock_graph_instance

    result = agent.process("Create a hello function", solution_path="test.py", test_path="test_test.py")

    assert result.is_solved is True
    assert result.current_iteration == 2
    mock_graph_instance.invoke.assert_called_once()


class TestSingleShotSupervisorAgent:
  """Test cases for the SingleShotSupervisorAgent class"""

  def test_singleshot_supervisor_agent_initialization(self) -> None:
    """Test SingleShotSupervisorAgent initialization"""
    config = development_config()
    agent = SingleShotSupervisorAgent(config=config)

    # Should inherit from base
    assert isinstance(agent, BaseSupervisorAgent)
    assert isinstance(agent, SingleShotSupervisorAgent)

    # Should have a graph for single-shot workflow
    assert hasattr(agent, 'graph')
    assert agent.graph is not None

  def test_singleshot_supervisor_agent_workflow_methods(self) -> None:
    """Test that SingleShotSupervisorAgent uses simplified workflow with base class methods"""
    config = development_config()
    agent = SingleShotSupervisorAgent(config=config)

    # Should have base class methods (not single-shot specific ones)
    assert hasattr(agent, 'finalize_solution')
    assert hasattr(agent, 'build_graph')
    assert hasattr(agent, 'process')

    # Should NOT have single-shot specific methods (simplified architecture)
    assert not hasattr(agent, '_validate_solution_singleshot')
    assert not hasattr(agent, '_run_tests_singleshot')
    assert not hasattr(agent, '_run_integration_tests_singleshot')
    assert not hasattr(agent, '_finalize_solution_singleshot')

    # Should have overridden finalize_solution method for single-shot behavior
    assert agent.finalize_solution.__qualname__ == 'SingleShotSupervisorAgent.finalize_solution'
    
    # Should have single-shot specific test_solution method
    assert hasattr(agent, 'test_solution')
    assert agent.test_solution.__qualname__ == 'SingleShotSupervisorAgent.test_solution'

  @patch('claude_code_supervisor.supervisor.StateGraph')
  def test_singleshot_supervisor_agent_process_workflow(self, mock_state_graph) -> None:
    """Test SingleShotSupervisorAgent process method"""
    # Mock graph workflow
    mock_graph_instance = Mock()
    mock_state_graph.return_value.compile.return_value = mock_graph_instance

    # Mock successful workflow result
    final_state = WorkflowState(is_solved=True, current_iteration=0)
    mock_graph_instance.invoke.return_value = final_state

    config = development_config()
    agent = SingleShotSupervisorAgent(config=config)
    agent.graph = mock_graph_instance

    result = agent.process("Create a hello function", solution_path="test.py", test_path="test_test.py")

    assert result.is_solved is True
    assert result.current_iteration == 0  # Single-shot should not iterate
    mock_graph_instance.invoke.assert_called_once()
