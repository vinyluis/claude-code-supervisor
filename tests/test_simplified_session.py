"""
Tests for simplified session handling without custom timeouts.
"""

import pytest
from unittest.mock import Mock, patch

from claude_code_supervisor import SupervisorAgent, WorkflowState
from claude_code_supervisor.config import development_config


class TestSimplifiedSession:
    """Test cases for simplified session handling"""

    def test_config_no_timeout_parameters(self):
        """Test that config no longer has timeout parameters"""
        config = development_config()
        
        # Should not have timeout parameters anymore
        assert not hasattr(config.claude_code, 'session_timeout_seconds')
        assert not hasattr(config.claude_code, 'activity_timeout_seconds')
        
        # Should have the remaining parameters
        assert hasattr(config.claude_code, 'max_turns')
        assert hasattr(config.claude_code, 'max_thinking_tokens')
        assert hasattr(config.claude_code, 'use_bedrock')

    def test_max_turns_default_is_none(self):
        """Test that max_turns defaults to None (unlimited)"""
        config = development_config()
        
        # Development config should have turns limit, but default should be None
        assert config.claude_code.max_turns == 30  # Development override
        
        # Default config should be None
        from claude_code_supervisor.config import SupervisorConfig
        default_config = SupervisorConfig()
        assert default_config.claude_code.max_turns is None

    @patch('anyio.run')
    @patch('claude_code_supervisor.supervisor.query')
    def test_execute_claude_session_simplified(self, mock_query, mock_anyio_run):
        """Test that execute_claude_session uses simplified anyio.run"""
        mock_anyio_run.return_value = None
        
        # Mock the async iterator
        mock_query.return_value = iter([])
        
        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent.config = development_config()
        agent._timestamp = Mock(return_value="12:00:00")
        agent.solution_path = "solution.py"
        agent.test_path = "test_solution.py"
        
        # Mock the required attributes
        from claude_code_sdk import ClaudeCodeOptions
        agent.base_claude_options = ClaudeCodeOptions()
        
        state = WorkflowState(
            claude_session_active=True
        )
        
        # Mock the prompt from log
        state.claude_output_log = ["PROMPT_ITERATION_0: Test prompt"]
        
        # This should not raise any timeout-related errors
        result = agent._execute_claude_session(state)
        
        # Should have called anyio.run (simplified version)
        mock_anyio_run.assert_called_once()
        
        # Result should be valid
        assert isinstance(result, WorkflowState)

    def test_decision_logic_no_activity_timeout(self):
        """Test that decision logic no longer checks activity timeout"""
        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent.config = development_config()
        agent._timestamp = Mock(return_value="12:00:00")
        agent.integrate_into_codebase = False
        agent.solution_path = "solution.py"
        agent.test_path = "test_solution.py"
        
        state = WorkflowState(
            current_iteration=1,
            claude_session_active=False
        )
        
        # Should not timeout based on activity - this check was removed  
        with patch('os.path.exists', return_value=False):
            result = agent._decide_next_action(state)
        
        # Should provide guidance instead of timing out
        assert result in ['guide', 'finish']

    @patch('claude_code_supervisor.supervisor.query')
    def test_error_handling_simplified(self, mock_query):
        """Test that error handling is simplified"""
        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent.config = development_config()
        agent._timestamp = Mock(return_value="12:00:00")
        agent.solution_path = "solution.py"
        agent.test_path = "test_solution.py"
        
        # Mock the required attributes
        from claude_code_sdk import ClaudeCodeOptions
        agent.base_claude_options = ClaudeCodeOptions()
        
        state = WorkflowState(
            claude_session_active=True
        )
        
        # Mock the prompt from log
        state.claude_output_log = ["PROMPT_ITERATION_0: Test prompt"]
        
        # Test with a mock that raises a real error (not cancel scope)
        with patch('anyio.run') as mock_run:
            mock_run.side_effect = Exception("real error")
            
            result = agent._execute_claude_session(state)
            
            # Should handle the error gracefully
            assert result.error_message  # Should have error message
            assert "real error" in result.error_message

if __name__ == "__main__":
    pytest.main([__file__, "-v"])