"""
Claude Code Supervisor Agent

An intelligent wrapper around Claude Code SDK that provides automated problem-solving
capabilities with session management, progress monitoring, and intelligent feedback loops.

This supervisor treats Claude Code as a coding assistant that can plan its own work using
todo lists, implement solutions, run tests, and iterate based on feedback. The supervisor
monitors Claude's progress, provides guidance when issues arise, and ensures solutions
meet quality standards through automated testing.

Key Features:
- Session-based continuity with Claude Code SDK
- Real-time progress monitoring via todo list tracking
- Intelligent feedback loops with LLM-powered guidance generation
- Configurable timeouts and iteration limits
- Comprehensive test execution with detailed validation feedback
- Support for multiple AI providers (Anthropic, AWS Bedrock, OpenAI)
- Separation of static configuration from dynamic workflow state
- Bring Your Own Model (BYOM) support for custom LLM integration

Architecture:
The supervisor uses LangGraph to orchestrate a workflow with these nodes:
1. initiate_claude: Start Claude Code session with problem description
2. execute_claude: Execute Claude session until completion or timeout
3. collect_results: Collect and analyze results from Claude's session
4. validate_solution: Run tests on generated files to verify correctness
5. provide_guidance: Analyze failures and generate actionable feedback
6. finalize: Complete the session and report results

Usage:
  agent = SupervisorAgent()
  result = agent.process('Create a function that sorts a list in ascending order')
"""

import os
import sys
import anyio
import json
import subprocess
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from enum import StrEnum
from langchain_core.language_models.base import BaseLanguageModel

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import (
  AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage,
  SystemMessage
)
from dotenv import load_dotenv
from .config import SupervisorConfig, development_config
from . import utils, prompts

# Suppress asyncio warnings from the Claude Code SDK
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited.*")
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
warnings.filterwarnings("ignore", message=".*cancel scope in a different task.*")
warnings.filterwarnings("ignore", message=".*Attempted to exit cancel scope in a different task.*")


class AsyncioErrorFilter:
    """
    A stderr filter that suppresses specific asyncio-related error messages from the Claude Code SDK.

    The Claude Code SDK uses anyio for async task management, which can generate harmless but
    confusing error messages when tasks are cancelled or when cancel scopes exit in different
    tasks. These errors don't affect functionality but create noise in the output.

    This filter intercepts stderr output and suppresses the following specific error patterns:
    - "Task exception was never retrieved"
    - "cancel scope in a different task"
    - "RuntimeError: Attempted to exit cancel scope in a different task"

    All other stderr output is passed through unchanged, preserving legitimate error messages.

    The filter is installed at module import time to ensure it catches errors that occur
    during the cleanup phase of async operations.
    """

    def __init__(self, original_stderr):
      self.original_stderr = original_stderr
      self.buffer = []

    def write(self, text):
      if ("Task exception was never retrieved" in text
          or "cancel scope in a different task" in text
          or "RuntimeError: Attempted to exit cancel scope" in text):
        return  # Suppress these specific errors
      self.original_stderr.write(text)

    def flush(self):
      self.original_stderr.flush()

    def __getattr__(self, name):
      return getattr(self.original_stderr, name)


# Install the filter at module import time
_original_stderr = sys.stderr
sys.stderr = AsyncioErrorFilter(_original_stderr)


class PlanModeNodes(StrEnum):
  """Plan mode specific node names to avoid string literals"""
  GENERATE_PLAN = 'generate_plan'
  REVIEW_PLAN = 'review_plan'
  REFINE_PLAN = 'refine_plan'
  APPROVE_PLAN = 'approve_plan'


class ExecutionModeNodes(StrEnum):
  """Execution mode specific node names to avoid string literals"""
  INITIATE_CLAUDE = 'initiate_claude'
  EXECUTE_CLAUDE = 'execute_claude'
  REVIEW_SESSION = 'review_session'
  TEST_AND_ANALYZE = 'test_and_analyze'
  TEST_SOLUTION = 'test_solution'
  GENERATE_GUIDANCE = 'generate_guidance'
  REDUCE_MESSAGE = 'reduce_message'
  FINALIZE = 'finalize'


@dataclass
class PlanState:
  """State management for Plan Mode Subgraph operations"""
  # Plan generation and content
  plan_content: str = ""
  plan_iteration: int = 0
  plan_history: list[str] = field(default_factory=list)

  # Plan review and scoring
  plan_approved: bool = False
  plan_review_score: float = 0.0
  plan_feedback: str = ""
  plan_strengths: list[str] = field(default_factory=list)
  plan_improvements: list[str] = field(default_factory=list)
  plan_risks: list[str] = field(default_factory=list)

  # Claude session management
  claude_session_id: str | None = None
  claude_log: list[str] = field(default_factory=list)
  claude_todos: list[dict] = field(default_factory=list)
  messages: list[str] = field(default_factory=list)

  # Error handling
  error_message: str = ""
  should_terminate_early: bool = False

  def should_auto_approve_plan(self, threshold: float = 0.8) -> bool:
    """OCR-inspired decision method for plan auto-approval"""
    return (self.plan_iteration >= 3
            or self.plan_review_score >= threshold)

  def should_retry_plan(self, max_iterations: int = 3) -> bool:
    """OCR-inspired retry logic for plan refinement"""
    return (self.plan_iteration < max_iterations
            and not self.plan_approved
            and self.plan_review_score < 0.8)

  def should_end_for_max_plan_iterations(self, max_iterations: int = 3) -> bool:
    """Check if we've reached max plan iterations and should proceed"""
    return self.plan_iteration >= max_iterations


@dataclass
class WorkflowState:
  """
  Dynamic state that changes during workflow execution.

  This dataclass contains only the state that changes as the workflow progresses,
  while static configuration is stored as attributes on the SupervisorAgent.

  Fields:
    Workflow Progress:
      current_iteration: Number of iterations completed (0-based)
      is_solved: Whether the problem has been successfully solved
      error_message: Any error that occurred during execution
      test_results: Output from running tests
      messages: List of messages sent to Claude Code
      validation_feedback: Detailed feedback from validation phase
      approved_plan: Approved plan from plan mode (if any)

    Claude Session State:
      claude_session_id: Unique ID of the current Claude Code session
      claude_session_active: Whether a Claude session is currently running
      claude_todos: List of todo items from Claude's TodoWrite tool
      claude_log: Log of Claude's output and tool usage
      should_terminate_early: Flag for early termination (e.g., quota exceeded)
  """
  # Workflow progress
  current_iteration: int = 0
  is_solved: bool = False
  error_message: str = ""
  test_results: str = ""
  messages: list["str"] = field(default_factory=list)
  validation_feedback: str = ""
  approved_plan: str | None = None

  # Claude session state
  claude_session_id: str | None = None
  claude_session_active: bool = False
  claude_todos: list[dict] = field(default_factory=list)
  claude_log: list[str] = field(default_factory=list)
  should_terminate_early: bool = False
  should_reduce_message: bool = False

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)


class BaseSupervisorAgent(ABC):
  """
  Base class for intelligent supervisors that manage Claude Code SDK sessions.

  This class provides all shared functionality for supervising Claude Code sessions,
  including configuration management, LLM initialization, session execution,
  and common utilities. Specialized supervisor classes inherit from this base
  to implement different execution strategies.

  Key Features:
  - Configuration management with multiple AI providers
  - Claude Code SDK session management and message processing
  - LLM-powered guidance generation for error analysis
  - Test execution and validation utilities
  - Output data extraction for data science workflows
  - Comprehensive error handling and logging

  Architecture:
  - Separates static configuration (agent attributes) from dynamic state (WorkflowState)
  - Supports both standalone file creation and codebase integration modes
  - Integrates multiple LLM providers for guidance generation
  - Provides extensible hooks for specialized supervisor implementations

  Args:
    llm: Optional LangChain LLM model for guidance (BYOM - Bring Your Own Model)
    config: Optional SupervisorConfig instance for agent configuration
    append_system_prompt: Optional additional instructions appended to Claude's system prompt

  Attributes:
    config: Configuration for the supervisor agent (SupervisorConfig)
    llm: LLM instance for guidance generation (provided or auto-configured)
    base_claude_options: Pre-configured ClaudeAgentOptions for faster iterations
    problem_description: Current problem being solved
    input_data: Input data provided for the problem (if any)
    output_data: Expected output data for validation (if any)
    solution_path: Path where solution file will be saved
    test_path: Path where test file will be saved
  """

  def __init__(
    self,
    llm: BaseLanguageModel | None = None,
    config: SupervisorConfig | None = None,
    append_system_prompt: str | None = None,
    **kwargs,
  ) -> None:

    self.load_environment()

    if config is not None and isinstance(config, SupervisorConfig):
      self.config = config
    else:
      utils.print_debug("Loading default configuration...")
      self.config = development_config()

    # Static workflow configuration (set once per process call)
    self.problem_description: str = ""
    self.example_output: str | None = None
    self.solution_path: str | None = None
    self.test_path: str | None = None
    self.input_data: utils.DataTypes | None = None
    self.output_data: utils.DataTypes | None = None
    self.data_format: str = 'auto'
    self.integrate_into_codebase: bool = False
    self.development_guidelines: str = prompts.development_guidelines()
    self.instruction_prompt: str = prompts.instruction_prompt()
    self.plan_mode_instruction_prompt: str = prompts.plan_mode_instruction_prompt()
    self.test_instructions: str = prompts.test_instructions(self.solution_path)

    # Check for credentials on the kwargs
    if 'aws_access_key_id' in kwargs:
      if not os.getenv('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = kwargs['aws_access_key_id']
    if 'aws_secret_access_key' in kwargs:
      if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = kwargs['aws_secret_access_key']
    if 'aws_region' in kwargs:
      if not os.getenv('AWS_REGION'):
        os.environ['AWS_REGION'] = kwargs['aws_region']
    if 'anthropic_api_key' in kwargs:
      if not os.getenv('ANTHROPIC_API_KEY'):
        os.environ['ANTHROPIC_API_KEY'] = kwargs['anthropic_api_key']
    if 'openai_api_key' in kwargs:
      if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = kwargs['openai_api_key']

    if llm is not None:
      self.llm = llm
    else:
      self.llm = self.initialize_llm()

    self.append_system_prompt = append_system_prompt
    self.enable_plan_mode = False
    self.initialize_claude_code()

    # Build both graphs using the subclass implementations
    self.plan_graph = self.build_plan_graph()
    self.execution_graph = self.build_execution_graph()

  def load_environment(self) -> None:
    """
    Load environment variables from .env file
    """
    env_path = Path('.env')
    if env_path.exists():
      load_dotenv(env_path)
    else:
      utils.print_warning("Warning: .env file not found. Environment variables may need to be set manually.")

  def initialize_claude_code(self) -> None:
    """
    Initialize Claude Code SDK configuration and prepare base options
    """
    claude_config = self.config.claude_code
    # Set environment variables based on provider choice
    if claude_config.use_bedrock:
      os.environ["CLAUDE_CODE_USE_BEDROCK"] = '1'
      utils.print_debug("Configured Claude Code to use Amazon Bedrock")
    else:
      # Default to Anthropic API
      if not os.getenv("ANTHROPIC_API_KEY"):
        utils.print_warning("Warning: ANTHROPIC_API_KEY not found in environment. Claude Code SDK may not work properly.")
        utils.print_warning("Please set your API key in the .env file or environment variables.")
      else:
        utils.print_debug("Configured Claude Code to use Anthropic API")

    # Pre-configure Claude Code options for faster reuse in iterations
    self.base_claude_options = ClaudeAgentOptions(
      cwd=os.getcwd(),
      permission_mode='acceptEdits',
      max_turns=claude_config.max_turns,
      system_prompt=self.append_system_prompt,
      allowed_tools=claude_config.tools
    )
    utils.print_debug("Pre-configured Claude Code options")

  def initialize_llm(self):
    """Initialize the LLM for guidance analysis (OpenAI or AWS Bedrock)"""
    agent_config = self.config.agent
    provider = agent_config.provider

    if provider is None:
      raise ValueError("Model provider not specified in configuration. Please set 'provider' in the model config.")

    elif provider == 'bedrock':
      if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ValueError("AWS credentials not found in environment variables. Please set 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', and 'AWS_REGION'.")
      aws_key = os.getenv('AWS_ACCESS_KEY_ID')
      aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
      aws_region = os.getenv('AWS_REGION')
      if aws_key is None or aws_secret is None:
        raise ValueError("AWS credentials validation failed")
      return ChatBedrockConverse(
        model=agent_config.model_name,
        temperature=agent_config.temperature,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region,
      )

    elif provider == 'openai':
      if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API key not found in environment variables. Please set 'OPENAI_API_KEY'.")
      return ChatOpenAI(
        model=agent_config.model_name,
        temperature=agent_config.temperature,
      )

    else:
      raise ValueError(f"Unsupported model provider: {provider}. Supported providers are 'openai' and 'bedrock'.")

  def initiate_claude_code_session(self, state: WorkflowState) -> WorkflowState:
    """
    Prepare the initial setup for Claude Code session.

    This method builds the initial prompt with problem description, requirements,
    and any input/output data, then stores it in the state for execution.

    Args:
      state: Current workflow state

    Returns:
      Updated state initial message
    """
    utils.print_with_timestamp("\n🚀 Setting up Claude Code session...")
    utils.print_with_timestamp(f"Working directory: {os.getcwd()}")

    # Build the initial prompt with approved plan if available
    claude_instructions = prompts.build_claude_instructions(
      instruction_prompt=self.instruction_prompt,
      problem_description=self.problem_description,
      development_guidelines=self.development_guidelines,
      test_instructions=self.test_instructions,
      solution_path=self.solution_path,
      test_path=self.test_path,
      input_data=self.input_data,
      output_data=self.output_data,
      approved_plan=state.approved_plan,
    )
    state.messages = [claude_instructions]
    return state

  async def _claude_run(self, claude_instructions: str, options: ClaudeAgentOptions,
                        state: PlanState | WorkflowState) -> None:
    """Run claude code session asynchronously and retrieve the messages"""
    try:
      # Simple async iteration as recommended in SDK docs
      async for message in query(prompt=claude_instructions, options=options):

        if isinstance(message, AssistantMessage):
          for block in message.content:
            if isinstance(block, TextBlock):
              state.claude_log.append(block.text)

              # Distinguish plan mode messages in logs
              if options.permission_mode == 'plan':
                utils.print_claude(f"📋 Plan: {block.text[:200] + ('...' if len(block.text) > 200 else '')}")
              else:
                utils.print_claude(block.text[:200] + ('...' if len(block.text) > 200 else ''))

              # Pure async execution - just flag potential issues for node to handle
              if utils.is_quota_error(block.text):
                state.should_terminate_early = True
                state.error_message = f"API Credit/Quota Error - Early Termination: {block.text}"
                return  # Exit cleanly, letting node handle workflow decisions

            elif isinstance(block, ToolUseBlock):
              tool_info = utils.get_tool_info(block.name, block.input)
              utils.print_tool(f"{utils.blue('Tool:')} {utils.blue(block.name)} {tool_info}")

              # Track todo updates
              if block.name == 'TodoWrite':
                todos = block.input.get('todos', [])
                state.claude_todos = todos
                utils.print_todo(str(len(todos)) + ' items')
                for todo in todos:
                  status_emoji = {'pending': '⏳', 'in_progress': '🔄', 'completed': '✅'}.get(todo.get('status'), '❓')
                  utils.print_with_timestamp(f"  {status_emoji} {utils.blue(todo.get('content', 'Unknown task'))}")

              # Capture plan content from ExitPlanMode tool calls
              elif block.name == 'ExitPlanMode' and options.permission_mode == 'plan':
                plan_content = block.input.get('plan', '')
                if plan_content:
                  # Store plan content directly in PlanState if available, otherwise use temp attribute
                  if isinstance(state, PlanState):
                    state.plan_content = plan_content
                  else:
                    setattr(state, 'claude_plan', plan_content)
                  utils.print_with_timestamp("📋 Plan captured from ExitPlanMode")
                  utils.print_plan_content(plan_content)

            elif isinstance(block, ToolResultBlock):
              if block.is_error:
                error_msg = f"Tool error: {block.content}"
                utils.print_error(error_msg)
                if not state.error_message:
                  state.error_message = error_msg

        elif isinstance(message, ResultMessage):
          # Session completed successfully
          state.claude_session_id = message.session_id
          utils.print_success(f"Claude session completed (ID: {message.session_id})")
          utils.print_with_timestamp(f"{utils.cyan('Turns:')} {message.num_turns}, {utils.cyan('Duration:')} {message.duration_ms}ms")
          if message.total_cost_usd:
            utils.print_with_timestamp(f"{utils.cyan('Cost:')} ${message.total_cost_usd:.4f}")
          break

        elif isinstance(message, SystemMessage):
          # Filter CLI warnings if configured
          message_text = str(message.subtype)
          should_suppress = False

          if self.config.claude_code.suppress_cli_warnings:
            for pattern in self.config.claude_code.cli_warning_patterns:
              if pattern in message_text:
                should_suppress = True
                utils.print_debug(f"Suppressed CLI warning: {message_text[:100]}")
                break

          if not should_suppress:
            utils.print_info(f"System: {message.subtype}")

    except Exception as e:
      # Let SDK handle its own errors - only handle real failures
      error_msg = f"Error in Claude session: {e}"
      utils.print_error(error_msg)
      if not state.error_message:
        state.error_message = error_msg
      raise

  def execute_claude_session(self, state: WorkflowState) -> WorkflowState:
    """Execute a Claude Code session - fully self-contained with complete error handling"""
    try:
      # Activate session at the start of actual execution
      state.claude_session_active = True

      utils.print_with_timestamp(f"\n🚀 Executing Claude Code session (iteration {state.current_iteration})...")

      # Get the prompt from the log
      claude_instructions = state.messages[-1]
      state.claude_log.append(f"PROMPT_ITERATION_{state.current_iteration}: {claude_instructions}")
      utils.print_with_timestamp(f"📝 Using prompt for iteration {state.current_iteration}:")
      utils.print_prompt(claude_instructions)

      # Prepare Claude Code options for this session
      options = ClaudeAgentOptions(
        cwd=self.base_claude_options.cwd,
        permission_mode=self.base_claude_options.permission_mode,
        max_turns=self.base_claude_options.max_turns,
        system_prompt=self.base_claude_options.system_prompt,
        allowed_tools=self.base_claude_options.allowed_tools,
        continue_conversation=state.current_iteration > 0,
        resume=state.claude_session_id if state.claude_session_id else None
      )

      # Execute async operation within this node's boundary
      anyio.run(self._claude_run, claude_instructions, options, state)

      # Node processes its own results completely - check for quota errors
      if utils.node_encountered_quota_error(state):
        state.should_terminate_early = True
        utils.print_with_timestamp("🚫 Claude session terminated: quota exhausted")
        return state  # Node completes with clear error state

      utils.print_with_timestamp("✅ Claude session completed successfully")
      return state  # Node completes with clear success state

    except Exception as e:
      # Node handles its own exceptions completely
      error_msg = f"Error in async session execution: {e}"
      utils.print_error(error_msg)
      # Let the SDK handle its own errors - only store real errors
      if "cancel scope" not in str(e).lower() and "task exception" not in str(e).lower():
        state.error_message = error_msg
        state.should_terminate_early = True
      return state  # Node completes with clear error state

    finally:
      # Always mark session as inactive after completion
      state.claude_session_active = False

  def build_plan_graph(self):
    """Build the plan mode graph workflow"""
    graph = StateGraph(PlanState)

    # Plan mode nodes
    graph.add_node(PlanModeNodes.GENERATE_PLAN, self._generate_plan_node)
    graph.add_node(PlanModeNodes.REVIEW_PLAN, self._review_plan_node)
    graph.add_node(PlanModeNodes.REFINE_PLAN, self._refine_plan_node)
    graph.add_node(PlanModeNodes.APPROVE_PLAN, self._approve_plan_node)

    # Plan workflow edges
    graph.add_edge(START, PlanModeNodes.GENERATE_PLAN)
    graph.add_conditional_edges(PlanModeNodes.GENERATE_PLAN, self._plan_generation_decision)
    graph.add_conditional_edges(PlanModeNodes.REVIEW_PLAN, self._plan_review_decision)
    graph.add_edge(PlanModeNodes.REFINE_PLAN, PlanModeNodes.GENERATE_PLAN)
    graph.add_edge(PlanModeNodes.APPROVE_PLAN, END)

    return graph.compile()

  def _generate_plan_node(self, state: PlanState) -> PlanState:
    """Generate execution plan using Claude Code in plan mode"""
    try:
      utils.print_with_timestamp("📋 Generating execution plan...")

      # Build plan mode prompt
      claude_instructions = prompts.build_claude_instructions(
        instruction_prompt=self.plan_mode_instruction_prompt,
        problem_description=self.problem_description,
        development_guidelines=self.development_guidelines,
        test_instructions=self.test_instructions,
        solution_path=self.solution_path,
        test_path=self.test_path,
        input_data=self.input_data,
        output_data=self.output_data,
      )

      # If this is a refinement iteration, the refinement feedback is already in state.messages
      # Only replace messages on the first iteration to preserve refinement guidance
      if state.plan_iteration == 0:
        state.messages = [claude_instructions]
        prompt_to_send = claude_instructions
      else:
        # Refinement iteration: format full conversation with role markers
        # state.messages contains: [original_instructions, refinement_feedback_1, refinement_feedback_2, ...]
        # state.plan_history contains: [plan_1, plan_2, ...]

        # Prepare custom labels for clarity
        user_labels = ["Original Plan Instructions"]
        user_labels.extend([f"Plan Refinement Guidance {i}" for i in range(1, len(state.messages))])

        assistant_labels = [f"Previous Plan (Iteration {i+1})" for i in range(len(state.plan_history))]

        # Format conversation with roles
        prompt_to_send = utils.format_conversation_with_roles(
          user_messages=state.messages,
          assistant_messages=state.plan_history,
          user_labels=user_labels,
          assistant_labels=assistant_labels
        )

      # Display the prompt
      utils.print_with_timestamp("💬 Plan mode prompt:")
      if state.plan_iteration == 0:
        utils.print_prompt(claude_instructions)
      else:
        utils.print_with_timestamp(
          f"📝 Sending full conversation history "
          f"({len(state.messages)} user messages, {len(state.plan_history)} assistant messages)"
        )
        utils.print_prompt(
          prompt_to_send,
          truncate=self.config.claude_code.enable_prompt_preview_truncation,
          head_lines=self.config.claude_code.prompt_preview_head_lines,
          tail_lines=self.config.claude_code.prompt_preview_tail_lines
        )

      # Execute Claude in plan mode
      self._execute_claude_plan_mode(state, prompt_to_send)

      # Check for errors
      if self._plan_node_encountered_error(state):
        state.should_terminate_early = True
        utils.print_with_timestamp("🚫 Plan generation terminated due to error")
        return state

      # Extract and store plan
      if state.plan_content:
        state.plan_iteration += 1
        state.plan_history.append(state.plan_content)
        utils.print_with_timestamp(f"✅ Plan generated successfully (iteration {state.plan_iteration})")
        utils.print_plan_content(state.plan_content)
      else:
        state.error_message = "No plan content captured from Claude"
        state.should_terminate_early = True
      return state

    except Exception as e:
      state.error_message = f"Plan generation failed: {e}"
      state.should_terminate_early = True
      utils.print_error(f"Plan generation error: {e}")
      return state

  def _review_plan_node(self, state: PlanState) -> PlanState:
    """LLM-powered plan analysis and scoring"""
    try:
      utils.print_with_timestamp("🔍 Reviewing execution plan with supervisor LLM...")

      # Skip review if disabled or auto-approve conditions met
      config = self.config.claude_code
      if (not config.plan_review_enabled
          or state.should_auto_approve_plan(config.plan_auto_approval_threshold)):
        state.plan_approved = True
        state.plan_review_score = 1.0
        state.plan_feedback = "Auto-approved (review disabled or high confidence)"
        utils.print_with_timestamp("✅ Plan auto-approved")
        return state

      # Generate and execute review
      review_prompt = self._get_plan_review_prompt(state)
      review_result = self._call_llm("plan_review", review_prompt)

      # Parse review results
      review_data = self._parse_plan_review_result(review_result)
      state.plan_review_score = review_data.get('overall_score', 0.0)
      state.plan_feedback = review_data.get('feedback_summary', '')
      state.plan_strengths = review_data.get('strengths', [])
      state.plan_improvements = review_data.get('specific_improvements', [])
      state.plan_risks = review_data.get('risk_assessment', [])

      # Display detailed review results
      self._display_plan_review_results(state)

      # Auto-approve logic
      if (state.should_auto_approve_plan(config.plan_auto_approval_threshold)
          or state.should_end_for_max_plan_iterations(config.max_plan_iterations)):
        state.plan_approved = True
        if state.should_end_for_max_plan_iterations(config.max_plan_iterations):
          state.plan_feedback = f"Auto-approved after {config.max_plan_iterations} iterations"
      else:
        state.plan_approved = review_data.get('recommendation') == 'approve'

      return state

    except Exception as e:
      # Fallback to auto-approval on review failure
      utils.print_error(f"Plan review failed, auto-approving: {e}")
      state.plan_approved = True
      state.plan_review_score = 1.0
      state.error_message = f"Plan review failed, auto-approving: {e}"
      return state

  def _refine_plan_node(self, state: PlanState) -> PlanState:
    """Generate plan refinement guidance"""
    try:
      utils.print_with_timestamp("🔄 Refining execution plan based on feedback...")

      # Generate refinement guidance
      refinement_prompt = self._get_plan_refinement_prompt(state)
      claude_instructions = prompts.build_claude_guidance_prompt(refinement_prompt)
      state.messages.append(claude_instructions)

      # Clear error state for retry
      state.error_message = ""

      utils.print_with_timestamp(f"📝 Refinement guidance provided for iteration {state.plan_iteration}")
      return state

    except Exception as e:
      utils.print_error(f"Plan refinement failed: {e}")
      state.error_message = f"Plan refinement failed: {e}"
      return state

  def _approve_plan_node(self, state: PlanState) -> PlanState:
    """Final plan approval and preparation for execution"""
    try:
      utils.print_with_timestamp("✅ Plan approved for execution!")

      # Display final approved plan
      utils.print_plan_approved(state.plan_content)

      # Set approval status
      state.plan_approved = True

      utils.print_with_timestamp("🚀 Plan ready for implementation phase...")
      return state

    except Exception as e:
      utils.print_error(f"Plan approval failed: {e}")
      state.error_message = f"Plan approval failed: {e}"
      return state

  def _plan_generation_decision(self, state: PlanState) -> Literal[PlanModeNodes.REVIEW_PLAN, '__end__']:
    """Route after plan generation"""
    if state.should_terminate_early:
      utils.print_with_timestamp("🚫 Terminating due to error")
      return END
    utils.print_with_timestamp("✅ Routing to plan review")
    return PlanModeNodes.REVIEW_PLAN

  def _plan_review_decision(self, state: PlanState) -> Literal[PlanModeNodes.REFINE_PLAN, PlanModeNodes.APPROVE_PLAN, '__end__']:
    """Route after plan review"""
    if state.should_terminate_early:
      utils.print_with_timestamp("🚫 Terminating due to error")
      return END

    config = self.config.claude_code
    if (state.plan_approved
        or not state.should_retry_plan(config.max_plan_iterations)):
      utils.print_with_timestamp("✅ Routing to approve plan")
      return PlanModeNodes.APPROVE_PLAN

    utils.print_with_timestamp("🔄 Routing to refine plan")
    return PlanModeNodes.REFINE_PLAN

  # Plan mode helper methods
  def _execute_claude_plan_mode(self, state: PlanState, instructions: str) -> None:
    """Execute Claude Code in plan mode and capture plan content"""
    # Create plan-specific options
    options = ClaudeAgentOptions(
      cwd=self.base_claude_options.cwd,
      permission_mode='plan',
      max_turns=self.base_claude_options.max_turns,
      system_prompt=self.base_claude_options.system_prompt,
      allowed_tools=self.base_claude_options.allowed_tools,
    )
    anyio.run(self._claude_run, instructions, options, state)

  def _plan_node_encountered_error(self, state: PlanState) -> bool:
    """Check if plan node encountered errors"""
    return bool(
      state.should_terminate_early
      or (state.error_message and any(error in state.error_message.lower()
                                  for error in ['credit balance', 'quota exceeded', 'rate limit']))
    )

  def _get_plan_review_prompt(self, state: PlanState) -> str:
    """Generate plan review prompt"""
    template = prompts.plan_review_template()
    return template.format(
      problem_description=self.problem_description,
      input_data=str(self.input_data) if self.input_data is not None else "None provided",
      output_data=str(self.output_data) if self.output_data is not None else "None provided",
      development_context=self.development_guidelines,
      claude_plan=state.plan_content
    )

  def _get_plan_refinement_prompt(self, state: PlanState) -> str:
    """Generate plan refinement prompt"""
    template = prompts.plan_refinement_guidance_template()
    return template.format(
      problem_description=self.problem_description,
      plan_iteration=state.plan_iteration,
      plan_review_score=state.plan_review_score,
      plan_feedback=state.plan_feedback,
      previous_plan=state.plan_content
    )

  def _parse_plan_review_result(self, review_result: str) -> dict:
    """Parse structured JSON response from plan review"""
    try:
      # Try to extract JSON from the response
      start_idx = review_result.find('{')
      end_idx = review_result.rfind('}') + 1
      if start_idx != -1 and end_idx > start_idx:
        json_str = review_result[start_idx:end_idx]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
      pass

    # Fallback parsing for non-JSON responses
    return {
      'overall_score': 0.5,
      'strengths': ["Plan structure provided"],
      'specific_improvements': ["Review response was not in expected JSON format"],
      'risk_assessment': ["Unable to properly parse review results"],
      'recommendation': 'refine',
      'feedback_summary': review_result[:200] + ('...' if len(review_result) > 200 else '')
    }

  def _display_plan_review_results(self, state: PlanState) -> None:
    """Display comprehensive plan review results"""
    score = state.plan_review_score
    status = "✅ Approved" if state.plan_approved else ("🔄 Needs Refinement" if score < 0.8 else "⚠️ Conditional")

    print(f"\n{utils.orange('=' * 60)}")
    print(f"{utils.orange('📋 PLAN REVIEW RESULTS')} (Iteration {state.plan_iteration})")
    print(f"{utils.orange('=' * 60)}")

    # Score and status
    score_color = utils.green if score >= 0.8 else utils.yellow if score >= 0.6 else utils.red
    print(f"   📊 Score: {score_color(f'{score:.2f}/1.0')}")
    print(f"   📋 Status: {status}")
    print()

    # Strengths
    if state.plan_strengths:
      print(f"   ✅ {utils.green('Strengths:')}")
      for strength in state.plan_strengths:
        print(f"      • {strength}")
      print()

    # Improvements needed
    if state.plan_improvements:
      print(f"   ⚠️  {utils.yellow('Areas for Improvement:')}")
      for improvement in state.plan_improvements:
        print(f"      • {improvement}")
      print()

    # Risk assessment
    if state.plan_risks:
      print(f"   🔍 {utils.red('Risk Assessment:')}")
      for risk in state.plan_risks:
        print(f"      • {risk}")
      print()

    # Feedback summary
    if state.plan_feedback:
      print(f"   💬 {utils.cyan('Feedback Summary:')}")
      print(f"      {state.plan_feedback}")
      print()

    print(f"{utils.orange('=' * 60)}\n")

  def review_session(self, state: WorkflowState) -> WorkflowState:
    """Review and analyze the results from Claude's session"""
    utils.print_with_timestamp("\n📊 Reviewing session results...")

    # Check if files were created
    if self.solution_path is not None:
      if os.path.exists(self.solution_path):
        utils.print_with_timestamp(f"📄 Solution file detected: {self.solution_path}")
    if self.test_path is not None:
      if os.path.exists(self.test_path):
        utils.print_with_timestamp(f"🧪 Test file detected: {self.test_path}")

    # Analyze todo completion status
    completed_todos = [todo for todo in state.claude_todos if todo.get('status') == 'completed']
    todos_count = len(state.claude_todos)

    if todos_count > 0:
      completion_rate = len(completed_todos) / todos_count
      utils.print_with_timestamp(f"📋 Todo completion: {len(completed_todos)}/{todos_count} ({completion_rate:.1%})")

    # Check for error patterns in the output using utility functions
    general_errors, credit_quota_errors, context_length_errors = utils.detect_errors_in_output(state.claude_log)
    error_message, should_terminate_early, should_reduce_message = utils.format_error_message(
        general_errors, credit_quota_errors, context_length_errors)

    # Update state with error information
    if error_message:
      state.error_message = error_message
      state.should_terminate_early = should_terminate_early
      state.should_reduce_message = should_reduce_message

      if should_terminate_early:
        utils.print_with_timestamp("🚫 Credit/quota error detected - terminating session")
      elif should_reduce_message:
        utils.print_with_timestamp("⚠️  Context length error detected - will reduce message and retry")
      else:
        utils.print_warning("Detected error indicators in Claude's output")

    return state

  def decide_next_action(self, state: WorkflowState) -> str:
    """Decide what to do next based on session results"""
    utils.print_with_timestamp("\n🤔 Deciding next action...")

    # Check for early termination due to credit/quota errors
    if state.should_terminate_early:
      utils.print_with_timestamp("🚫 Early termination requested due to API errors")
      return 'finish'

    # Check if we've exceeded maximum iterations
    if state.current_iteration >= self.config.agent.max_iterations:
      utils.print_with_timestamp(f"🔄 Maximum iterations ({self.config.agent.max_iterations}) reached")
      return 'finish'

    # Check if we need to reduce message due to context length errors
    if state.should_reduce_message:
      utils.print_with_timestamp("📏 Context length error detected - reducing message")
      return 'reduce'

    # If there are explicit errors, provide guidance
    if state.error_message:
      utils.print_error("Errors detected, providing guidance")
      return 'guide'

    # Check if solution appears complete
    if self.integrate_into_codebase:
      # In integration mode, check if Claude indicates completion
      if not state.claude_session_active:
        utils.print_success("Integration mode: session complete, validating")
        return 'validate'
    else:
      # Standard mode: check if both files exist
      if (self.solution_path and os.path.exists(self.solution_path)
          and self.test_path and os.path.exists(self.test_path)):
        utils.print_success("Both solution and test files exist, validating")
        return 'validate'

    # Default: validate if solution is already enough
    utils.print_with_timestamp("🤔 Solution unclear, starting validation step")
    return 'validate'

  def test_and_analyze(self, state: WorkflowState) -> WorkflowState:
    """
    Validate the solution created by Claude Code and provide detailed feedback.

    This method performs comprehensive validation including:
    1. Checking if solution and test files exist
    2. Running tests to verify correctness
    3. Extracting output data if input data was provided
    4. Generating detailed validation feedback for failed cases

    Unlike passive test running, this generates specific feedback that guides
    Claude's next iteration rather than just reporting pass/fail status.

    Args:
      state: Current workflow state

    Returns:
      Updated state with is_solved status and validation_feedback
    """
    utils.print_with_timestamp("\n🔍 Validating Claude's solution...")

    # Reset validation state
    state.is_solved = False
    state.validation_feedback = ""

    if self.integrate_into_codebase:
      # In integration mode, we validate by running tests on the entire codebase
      utils.print_debug("Integration mode: validating codebase changes...")
      state = self._run_integration_tests(state)
    else:
      # Standard mode: check if both files exist first
      missing_files = []
      if self.solution_path is not None and not os.path.exists(self.solution_path):
        missing_files.append(f"Solution file {self.solution_path}")
      if self.test_path is not None and not os.path.exists(self.test_path):
        missing_files.append(f"Test file {self.test_path}")

      if missing_files:
        state.validation_feedback = f"Missing required files: {', '.join(missing_files)}. Please create these files before proceeding."
        utils.print_error(f"Missing files: {', '.join(missing_files)}")
        return state

      # Run the tests to validate
      state = self._run_tests(state)

    # Determine final state based on validation results
    if state.is_solved and not state.validation_feedback:
      # Tests passed and no feedback - solution is complete
      utils.print_success("✅ Solution validation successful - marking as complete")

      # Extract output data if needed (for data science workflows)
      # Only extract if we have a solution_path (skip in integration mode)
      if self.input_data is not None and self.solution_path is not None:
        state = self._extract_output_data(state)
        # If output extraction fails, it doesn't invalidate the solution
        # since tests already passed
    elif state.validation_feedback:
      # We have specific feedback - continue iteration
      utils.print_info("📋 Validation feedback generated - will continue iteration")
      state.is_solved = False  # Ensure we continue iterating
    else:
      # Tests failed but no specific feedback yet - need to analyze
      if state.error_message:
        state.validation_feedback = f"Implementation has errors that need to be fixed: {state.error_message}"
      else:
        state.validation_feedback = "Tests are not passing. Please review and fix the implementation."
      state.is_solved = False

    return state

  def should_iterate(self, state: WorkflowState) -> str:
    """If the solution is valid, finish; otherwise provide guidance and continue iteration"""
    if state.is_solved:
      utils.print_with_timestamp("✅ Solution appears complete, finalizing")
      return 'finish'
    return 'continue'

  def _run_integration_tests(self, state: WorkflowState) -> WorkflowState:
    """Validate integration using Claude's own assessment and todo completion"""
    utils.print_with_timestamp("\n🔍 Validating solution based on Claude's assessment...")

    try:
      # Validate based on Claude's final messages
      claude_success = self._validate_by_claude_messages(state)

      # Validate based on todo completion
      todo_success = self._validate_by_todo_completion(state)

      # Check for error indicators in recent messages
      has_errors = self._detect_recent_errors(state)

      # Determine validation result
      if claude_success and todo_success and not has_errors:
        state.is_solved = True
        utils.print_success("✅ Solution validated - Claude reported success and all todos completed")
        state.test_results = self._generate_success_summary(state)

      elif claude_success or todo_success:
        if has_errors:
          state.validation_feedback = "Claude reported partial success but errors were detected. Please review and address any remaining issues."
          utils.print_warning("⚠️ Partial success with errors detected")
        else:
          state.is_solved = True
          utils.print_success("✅ Solution validated - partial indicators suggest success")
          state.test_results = self._generate_success_summary(state)

      else:
        state.validation_feedback = "Claude did not clearly indicate successful completion. Please review the implementation and ensure all requirements are met."
        utils.print_warning("❌ No clear success indicators found")

      # Always provide guidance for running tests manually if test instructions exist
      if hasattr(self, 'test_instructions') and self.test_instructions:
        manual_test_guidance = self._generate_manual_test_guidance()
        if state.validation_feedback:
          state.validation_feedback += f"\n\n{manual_test_guidance}"
        else:
          state.test_results += f"\n\n{manual_test_guidance}"

      return state

    except Exception as e:
      error_msg = f"Error during validation: {str(e)}"
      utils.print_error(error_msg)
      state.validation_feedback = error_msg
      return state

  def _validate_by_claude_messages(self, state: WorkflowState) -> bool:
    """Validate based on Claude's own success indicators in recent messages"""
    if not state.claude_log:
      return False

    recent_messages = state.claude_log[-3:]  # Last 3 messages

    success_indicators = [
      "tests are passing", "all tests pass", "tests passed", "test passed",
      "successfully implemented", "task completed", "implementation complete",
      "fix worked", "solution works", "working correctly", "working properly",
      "all the tests are passing", "perfect! all", "great! all", "excellent! all",
      "tests pass", "solution is complete", "implementation is working"
    ]

    for message in recent_messages:
      message_lower = message.lower()
      if any(indicator in message_lower for indicator in success_indicators):
        return True

    return False

  def _validate_by_todo_completion(self, state: WorkflowState) -> bool:
    """Check if todos indicate successful completion"""
    if not state.claude_todos:
      return False

    completed = sum(1 for todo in state.claude_todos if todo.get('status') == 'completed')
    total = len(state.claude_todos)

    # Consider success if 100% completion or very high completion (90%+)
    if total == 0:
      return False

    completion_rate = completed / total
    return completion_rate >= 0.9  # 90% or higher completion

  def _detect_recent_errors(self, state: WorkflowState) -> bool:
    """Detect error indicators in recent Claude messages"""
    if not state.claude_log:
      return False

    recent_messages = state.claude_log[-2:]  # Last 2 messages

    error_indicators = [
      "error occurred", "failed to", "exception", "traceback",
      "cannot", "unable to", "not working", "didn't work",
      "tests are failing", "test failed", "tests failed",
      "something went wrong", "issue with", "problem with"
    ]

    for message in recent_messages:
      message_lower = message.lower()
      if any(indicator in message_lower for indicator in error_indicators):
        return True

    return False

  def _generate_success_summary(self, state: WorkflowState) -> str:
    """Generate a summary of the successful validation"""
    summary_parts = []

    # Todo completion summary
    if state.claude_todos:
      completed = sum(1 for todo in state.claude_todos if todo.get('status') == 'completed')
      total = len(state.claude_todos)
      summary_parts.append(f"✅ Todo completion: {completed}/{total} ({completed / total * 100:.1f}%)")

    # Claude's own assessment
    summary_parts.append("✅ Claude reported successful implementation")

    # Add final guidance
    summary_parts.append("\n📋 Validation Summary:")
    summary_parts.append("- Solution implemented successfully based on Claude's assessment")
    summary_parts.append("- All or most todos completed")
    summary_parts.append("- No critical errors detected in recent output")

    return "\n".join(summary_parts)

  def _generate_manual_test_guidance(self) -> str:
    """Generate guidance for running tests manually"""
    if not hasattr(self, 'test_instructions') or not self.test_instructions:
      return ""

    guidance = ["🧪 Manual Test Guidance:",
                "To verify the implementation manually, you can run the tests as specified:",
                ""]

    # Extract test command from test_instructions if available
    test_instructions = getattr(self, 'test_instructions', '')
    if '```bash' in test_instructions:
      # Extract bash command from test instructions
      lines = test_instructions.split('\n')
      in_bash_block = False
      for line in lines:
        if '```bash' in line:
          in_bash_block = True
          continue
        elif '```' in line and in_bash_block:
          break
        elif in_bash_block and line.strip():
          guidance.append(f"  {line.strip()}")

    if len(guidance) <= 3:  # No specific commands found
      guidance.extend([
        "Please refer to the test instructions provided to Claude for the exact commands.",
        "The test instructions contain project-specific commands for running tests."
      ])

    return "\n".join(guidance)

  def _analyze_claude_output_for_testing(self, state: WorkflowState) -> dict:
    """Use LLM to analyze Claude's output and determine test strategy"""
    # Get recent Claude output to analyze
    recent_output = "\n".join(state.claude_log[-5:]) if state.claude_log else "No recent output"

    # Find available test files
    test_files = []
    for root, _, files in os.walk('.'):
      for file in files:
        if file.endswith('_test.py') or file.startswith('test_'):
          test_files.append(os.path.join(root, file))

    # Extract mentioned items from Claude's output
    mentioned_items = self._extract_mentioned_items(recent_output)

    # Use LLM to analyze and determine strategy
    analysis_prompt = prompts.test_analysis_template().format(
      claude_output=recent_output,
      mentioned_items=mentioned_items,
      test_instructions=self.test_instructions,
      available_test_files=test_files
    )

    analysis = self._call_llm("test_analysis", analysis_prompt)

    # Parse the analysis to determine strategy
    strategy = self._parse_test_strategy(analysis, test_files)

    utils.print_debug(f"Test strategy determined: {strategy}")
    return strategy

  def _extract_mentioned_items(self, output: str) -> str:
    """Extract files, functions, and classes mentioned in Claude's output"""
    import re

    # Look for common patterns in Claude's output
    patterns = [
      r'function[s]?\s+(\w+)',
      r'class[es]?\s+(\w+)',
      r'file[s]?\s+(\w+\.py)',
      r'test[s]?\s+(\w+)',
      r'created?\s+(\w+\.py)',
      r'modified?\s+(\w+\.py)'
    ]

    mentioned = []
    for pattern in patterns:
      matches = re.findall(pattern, output, re.IGNORECASE)
      mentioned.extend(matches)

    return ", ".join(set(mentioned)) if mentioned else "No specific items mentioned"

  def _parse_test_strategy(self, analysis: str, available_files: list) -> dict:
    """Parse LLM analysis into actionable test strategy"""
    analysis_lower = analysis.lower()

    # Default strategy
    strategy = {
      'test_type': 'integration',
      'test_files': available_files,
      'command': [sys.executable, '-m', 'pytest'] + available_files + ['-v', '--tb=short', '--no-header']
    }

    # Look for specific instructions in the analysis
    if 'no test' in analysis_lower or 'no separate test' in analysis_lower:
      strategy['test_type'] = 'no_tests'
    elif 'specific' in analysis_lower and any(f in analysis for f in available_files):
      # Extract specific files mentioned
      specific_files = [f for f in available_files if f in analysis]
      if specific_files:
        strategy['test_type'] = 'specific'
        strategy['test_files'] = specific_files
        strategy['command'] = [sys.executable, '-m', 'pytest'] + specific_files + ['-v', '--tb=short', '--no-header']

    return strategy

  def _run_specific_tests(self, test_files: list) -> subprocess.CompletedProcess:
    """Run specific test files"""
    timeout = self.config.agent.test_timeout
    return subprocess.run(
      [sys.executable, '-m', 'pytest'] + test_files + ['-v', '--tb=short', '--no-header'],
      capture_output=True,
      text=True,
      timeout=timeout,
      cwd=os.getcwd()
    )

  def _run_all_integration_tests(self) -> subprocess.CompletedProcess:
    """Run all integration tests"""
    # Find all test files
    test_files = []
    for root, _, files in os.walk('.'):
      for file in files:
        if file.endswith('_test.py') or file.startswith('test_'):
          test_files.append(os.path.join(root, file))

    if not test_files:
      # Create a dummy result if no tests found
      return subprocess.CompletedProcess(
        args=['pytest'],
        returncode=0,
        stdout="No test files found",
        stderr=""
      )

    timeout = self.config.agent.test_timeout
    return subprocess.run(
      [sys.executable, '-m', 'pytest'] + test_files + ['-v', '--tb=short', '--no-header'],
      capture_output=True,
      text=True,
      timeout=timeout,
      cwd=os.getcwd()
    )

  def _generate_test_failure_feedback(self, result: subprocess.CompletedProcess, strategy: dict) -> str:
    """Generate detailed feedback for test failures"""
    feedback_parts = []

    if result.returncode != 0:
      feedback_parts.append(f"Tests failed with exit code {result.returncode}.")

      # Analyze common failure patterns
      if "FAILED" in result.stdout:
        failures = [line for line in result.stdout.split('\n') if 'FAILED' in line]
        feedback_parts.append(f"Failed tests: {', '.join(failures[:3])}")

      if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
        feedback_parts.append("Import errors detected - check your module imports and file structure.")

      if "AssertionError" in result.stdout:
        feedback_parts.append("Assertion errors detected - your implementation may not be producing expected results.")

      if "SyntaxError" in result.stderr:
        feedback_parts.append("Syntax errors detected - check your code for syntax issues.")

      # Add strategy-specific feedback
      if strategy['test_type'] == 'specific':
        feedback_parts.append(f"Focus on fixing the specific test files: {', '.join(strategy['test_files'])}")

      feedback_parts.append("Please review the test output above and fix the identified issues.")

    return " ".join(feedback_parts)

  def _extract_output_data(self, state: WorkflowState) -> WorkflowState:
    """Extract output data by running the solution with input data"""
    utils.print_with_timestamp("\n📤 Extracting output data from solution...")

    try:
      # Import the solution module dynamically
      import importlib.util

      # Load the solution module
      spec = importlib.util.spec_from_file_location("solution_module", self.solution_path)
      if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {self.solution_path}")

      solution_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(solution_module)

      # Try to find a main function or process function
      possible_functions = ['main', 'process', 'solve', 'run', 'execute']
      main_function = None

      for func_name in possible_functions:
        if hasattr(solution_module, func_name):
          main_function = getattr(solution_module, func_name)
          break

      if main_function is None:
        # If no standard function found, try to find any callable function
        # Filter out types, built-in functions, and imports
        functions = []
        for name in dir(solution_module):
          if not name.startswith('_'):
            attr = getattr(solution_module, name)
            if (callable(attr)
                and hasattr(attr, '__name__')
                and not isinstance(attr, type)
                and hasattr(attr, '__module__')
                and attr.__module__ == solution_module.__name__):
              functions.append(attr)
        if functions:
          main_function = functions[0]  # Use the first callable function

      if main_function is not None:
        func_name = getattr(main_function, '__name__', str(main_function))
        utils.print_debug(f"Found function: {func_name}")

        # Try to call the function with input data
        try:
          if self.input_data is not None:
            # Pass the input data directly as argument
            result = main_function(self.input_data)
          else:
            # Call without arguments
            result = main_function()

          self.output_data = result
          utils.print_success(f"Output data extracted: {type(result).__name__}")

          # Validate against expected output if provided
          if self.output_data is not None:
            if self._validate_output(result, self.output_data):
              utils.print_success("Output matches expected format!")
            else:
              utils.print_warning("Output format doesn't match expected format")

        except Exception as e:
          utils.print_error(f"Failed to execute function: {e}")
          state.error_message = f"Failed to execute solution function: {e}"

      else:
        utils.print_warning("No callable main function found in solution")

    except Exception as e:
      utils.print_error(f"Failed to extract output data: {e}")
      # Don't mark as unsolved, this is optional

    return state

  def _validate_output(self, actual: Any, expected: Any) -> bool:
    """Validate that actual output matches expected output format/type"""
    try:
      # Check type compatibility
      if type(actual) is not type(expected):
        return False

      # For collections, check structure
      if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
          return False
        # Check if all elements have compatible types
        for a, e in zip(actual, expected):
          if type(a) is not type(e):
            return False

      elif isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
          return False
        # Check value types
        for key in expected:
          if type(actual[key]) is not type(expected[key]):
            return False

      return True

    except Exception:
      return False

  def generate_guidance(self, state: WorkflowState) -> WorkflowState:
    """
    Provide intelligent guidance to Claude Code based on current situation.

    This is the core of the intelligent feedback loop. It analyzes the current state
    and generates specific, actionable guidance using an LLM. Handles two modes:
    1. Error mode: When explicit errors occurred during execution
    2. Validation mode: When tests failed or validation feedback is available

    Args:
      state: Current workflow state with error_message or validation_feedback

    Returns:
      Updated state with guidance message for Claude and cleared error state
    """
    utils.print_with_timestamp("\n🎯 Analyzing situation and providing guidance...")

    # Increment iteration counter
    state.current_iteration += 1
    utils.print_with_timestamp(f"🔄 Starting iteration {state.current_iteration}")

    # Determine guidance mode based on state
    if state.error_message:
      # Error mode: Handle explicit errors
      guidance = self._generate_error_guidance(state)
      utils.print_with_timestamp("📋 Error guidance generated:")
    elif state.validation_feedback:
      # Feedback mode: Handle validation feedback
      guidance = self._generate_feedback_guidance(state)
      utils.print_with_timestamp("📋 Validation feedback guidance generated:")
    else:
      # Fallback mode: General guidance
      guidance = "Continue working on the problem. Review your current progress and ensure all requirements are met."
      utils.print_with_timestamp("📋 General guidance generated:")

    utils.print_with_timestamp(guidance)

    # Store guidance in message buffer for next iteration
    guidance_message = f"""\
Based on the previous attempt, here's guidance for improvement:

{guidance}

Please update your todo list and continue working on the solution, addressing these specific points.
"""

    state.messages.append(guidance_message)

    # Clear state for next iteration
    state.error_message = ""
    state.test_results = ""
    state.validation_feedback = ""

    return state

  def reduce_message_and_retry(self, state: WorkflowState) -> WorkflowState:
    """
    Reduce the message length when context length errors occur and retry.

    This method is triggered when Claude outputs "API Error: 400 Input is too long for requested model."
    It reduces the last message in the queue by removing non-essential content while preserving
    the core requirements, then prepares for a retry.

    Args:
      state: Current workflow state

    Returns:
      Updated state with reduced message for retry
    """
    utils.print_with_timestamp("\n📏 Reducing message length due to context limit...")

    if not state.messages:
      utils.print_error("No messages to reduce")
      state.error_message = "Context length error but no messages available to reduce"
      return state

    # Get the last message (current prompt)
    original_message = state.messages[-1]
    utils.print_with_timestamp(f"Original message length: {len(original_message)} characters")

    # Reduce the message using the utility function
    reduced_message = utils.reduce_message_length(original_message)
    utils.print_with_timestamp(f"Reduced message length: {len(reduced_message)} characters")

    # Append reduced version to the message buffer
    state.messages.append(reduced_message)

    # Clear the context length error flags
    state.should_reduce_message = False
    state.error_message = ""

    # Add a note to the log about the reduction
    state.claude_log.append(f"MESSAGE_REDUCED: Original length {len(original_message)}, reduced to {len(reduced_message)}")

    utils.print_success("Message reduced successfully - ready for retry")

    state.current_iteration += 1
    return state

  def _generate_error_guidance(self, state: WorkflowState) -> str:
    """Generate guidance for error cases using LLM"""
    example_output_section = f'Expected behavior: {self.example_output}' if self.example_output else ''

    analysis_prompt = prompts.error_guidance_template().format(
      problem_description=self.problem_description,
      example_output_section=example_output_section,
      error_message=state.error_message,
      test_results=state.test_results if state.test_results else 'No tests run yet',
      current_iteration=state.current_iteration,
      test_instructions=self.test_instructions,
      todo_progress=self._format_todos_for_analysis(state.claude_todos),
      recent_output=self._format_output_log_for_analysis(state.claude_log)
    )

    return self._call_llm("error_analysis", analysis_prompt)

  def _generate_feedback_guidance(self, state: WorkflowState) -> str:
    """Generate guidance for validation feedback using LLM"""
    example_output_section = f'Expected behavior: {self.example_output}' if self.example_output else ''

    # Get recent messages from Claude's log
    recent_messages = "\n".join(state.claude_log[-3:]) if state.claude_log else "No recent messages"

    analysis_prompt = prompts.feedback_guidance_template().format(
      problem_description=self.problem_description,
      example_output_section=example_output_section,
      validation_feedback=state.validation_feedback,
      test_instructions=self.test_instructions,
      recent_messages=recent_messages,
      todo_progress=self._format_todos_for_analysis(state.claude_todos)
    )

    return self._call_llm("feedback_analysis", analysis_prompt)

  def _format_todos_for_analysis(self, todos: list[dict]) -> str:
    """Format todo list for LLM analysis"""
    if not todos:
      return 'No todos available'

    formatted = []
    for todo in todos[-5:]:  # Last 5 todos
      status = todo.get('status', 'unknown')
      content = todo.get('content', 'Unknown task')
      formatted.append(f"- [{status.upper()}] {content}")

    return '\n'.join(formatted)

  def _format_output_log_for_analysis(self, output_log: list[str]) -> str:
    """Format output log for LLM analysis"""
    if not output_log:
      return 'No output available'

    # Get last few entries, excluding the initial prompt
    relevant_entries = [entry for entry in output_log[-3:] if not entry.startswith('PROMPT:')]
    if not relevant_entries:
      return 'No relevant output available'

    # Truncate long entries
    formatted = []
    for entry in relevant_entries:
      if len(entry) > 300:
        formatted.append(entry[:300] + '...')
      else:
        formatted.append(entry)

    return '\n'.join(formatted)

  def _call_llm(self, operation: str, prompt: str) -> str:
    """Wrapper for LLM calls for guidance analysis"""
    try:
      utils.print_with_timestamp(f"🤖 Calling LLM for {operation}...")
      messages = [HumanMessage(content=prompt)]
      response = self.llm.invoke(messages)
      content = response.content
      utils.print_success(f"LLM response received for {operation}")

      # Ensure we have string content
      if not isinstance(content, str):
        content = str(content)

      # Check if response is empty or error-like
      if not content.strip():
        return f"Empty response received for {operation}"

      return content
    except Exception as e:
      error_msg = f"Error in {operation}: {str(e)}"
      utils.print_error(error_msg)
      return error_msg

  def _run_tests(self, state: WorkflowState) -> WorkflowState:
    """Execute the tests and provide detailed feedback"""
    utils.print_with_timestamp("\n▶️  Running tests...")

    # Check if test file exists and has content
    if self.test_path is not None and not os.path.exists(self.test_path):
      state.test_results = f"Test file {self.test_path} does not exist"
      state.is_solved = False
      state.validation_feedback = f"Test file {self.test_path} was not created. Please create comprehensive tests for your solution."
      utils.print_error(f"Test file not found: {self.test_path}\n")
      return state

    # Check if solution file exists
    if self.solution_path is not None and not os.path.exists(self.solution_path):
      state.test_results = (f"Solution file {self.solution_path} "
                            'does not exist')
      state.is_solved = False
      state.validation_feedback = f"Solution file {self.solution_path} was not created. Please implement your solution first."
      utils.print_error(f"Solution file not found: {self.solution_path}\n")
      return state

    try:
      timeout = self.config.agent.test_timeout
      # First try to run a syntax check on both files
      for file_path, file_type in [(self.solution_path, 'solution'),
                                   (self.test_path, 'test')]:
        if file_path is None:
          continue
        try:
          with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
          error_msg = f"Syntax error in {file_type} file {file_path}: {e}"
          utils.print_error(error_msg)
          state.test_results = error_msg
          state.is_solved = False
          state.validation_feedback = f"Syntax error in {file_type} file: {e}. Please fix the syntax errors and ensure your code is valid Python."
          return state

      # Run the tests
      if self.test_path is None:
        state.test_results = "No test file specified"
        state.is_solved = False
        state.validation_feedback = "No test file specified for validation"
        return state

      result = subprocess.run(
        [sys.executable, '-m', 'pytest', self.test_path, '-v',
         '--tb=short', '--no-header'],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.getcwd()  # Ensure we're in the right directory
      )

      state.test_results = (f"Exit code: {result.returncode}\n"
                            f"STDOUT:\n{result.stdout}\n"
                            f"STDERR:\n{result.stderr}")

      if result.returncode == 0:
        state.is_solved = True
        utils.print_success(utils.green('All tests passed!'))
        print(f"Test output:\n{result.stdout}")
      else:
        state.is_solved = False
        # Generate detailed feedback instead of just error message
        state.validation_feedback = self._generate_test_failure_feedback(result, {'test_type': 'specific', 'test_files': [self.test_path]})
        utils.print_error(f"{utils.red('Tests failed')} (exit code: {result.returncode})")
        print(f"Test output:\n{result.stdout}")
        if result.stderr:
          print(f"Errors:\n{utils.red(result.stderr)}")
      print()

    except subprocess.TimeoutExpired:
      timeout = self.config.agent.test_timeout
      state.test_results = f"Tests timed out after {timeout} seconds"
      state.is_solved = False
      state.validation_feedback = f"Tests are taking too long to execute (>{timeout}s). Consider optimizing your implementation or breaking down complex tests."
      utils.print_with_timestamp(f"⏰ Tests timed out after {timeout} seconds\n")
    except FileNotFoundError:
      state.test_results = ("pytest not found. "
                            'Please install pytest: pip install pytest')
      state.is_solved = False
      state.validation_feedback = "Testing framework not available. Please ensure pytest is installed in your environment."
      utils.print_error("pytest not found. Install with: pip install pytest\n")
    except Exception as e:
      state.test_results = f"Error running tests: {str(e)}"
      state.is_solved = False
      state.validation_feedback = f"Error occurred while running tests: {str(e)}. Please check your test implementation."
      utils.print_error(f"Error running tests: {str(e)}\n")

    return state

  def finalize_solution(self, state: WorkflowState) -> WorkflowState:
    """Finalize the solution"""
    if state.is_solved:
      utils.print_with_timestamp(f"\n🎉 {utils.green(utils.bold('Solution completed successfully'))} after {state.current_iteration} iterations!")

      if self.integrate_into_codebase:
        utils.print_debug(utils.green('Solution integrated into existing codebase'))
        utils.print_debug(utils.green('Tests integrated into existing test structure'))
      else:
        utils.print_with_timestamp(f"💾 {utils.cyan('Code saved to:')} {self.solution_path}")
        utils.print_with_timestamp(f"💾 {utils.cyan('Tests saved to:')} {self.test_path}")

      if self.output_data is not None:
        utils.print_with_timestamp(f"📊 {utils.cyan('Output data:')} {type(self.output_data).__name__}")
        if hasattr(self.output_data, '__len__') and len(self.output_data) < 20:
          utils.print_with_timestamp(f"📊 {utils.cyan('Result:')} {self.output_data}")

      if not self.integrate_into_codebase:
        utils.print_with_timestamp(f"\n🚀 {utils.yellow('You can run the tests manually with:')} pytest {self.test_path}")
    else:
      # Check if this is a credit/quota error that caused early termination
      if state.should_terminate_early and state.error_message and "Credit/Quota Error" in state.error_message:
        utils.display_credit_quota_error(
          error_message=state.error_message,
          use_bedrock=self.config.claude_code.use_bedrock,
          current_iteration=state.current_iteration,
          claude_todos=state.claude_todos,
          claude_log=state.claude_log
        )
      else:
        utils.print_with_timestamp(f"\n❌ {utils.red('Maximum iterations')} ({self.config.agent.max_iterations}) {utils.red('reached without solving the problem.')}")
        utils.print_with_timestamp(f"📝 {utils.yellow('Last error:')} {utils.red(state.error_message)}")

      if not self.integrate_into_codebase:
        utils.print_with_timestamp("\n📁 Files generated (may contain partial solutions):")
        if self.solution_path and os.path.exists(self.solution_path):
          utils.print_with_timestamp(f"  - {self.solution_path}")
        if self.test_path and os.path.exists(self.test_path):
          utils.print_with_timestamp(f"  - {self.test_path}")

    # No cleanup needed - all data operations are in-memory only

    return state

  @abstractmethod
  def build_execution_graph(self):
    """
    Build the execution LangGraph workflow.

    Returns:
        Compiled LangGraph workflow for problem solving and iteration
    """
    raise NotImplementedError("Subclasses must implement the build_execution_graph method")

  def _get_process_start_message(self, problem_description: str) -> str:
    """
    Get the log message for the start of processing.

    Can be overridden by subclasses for custom messaging.

    Args:
        problem_description: The problem being solved

    Returns:
        Log message string
    """
    return f"🚀 Starting problem solving: {problem_description}"

  def process(
      self,
      problem_description: str, *,
      input_data: Any = None,
      output_data: Any = None,
      solution_path: str | None = None,
      test_path: str | None = None,
      enable_plan_mode: bool = False,
      **kwargs: Any,
    ) -> WorkflowState:
    """
    Main method to process a problem using two independent graphs.

    This method executes plan mode (if enabled) followed by execution mode.
    Plan mode generates an approved plan, which is then passed to execution mode.

    Args:
      problem_description: Description of the problem to solve
      input_data: Input data for Claude Code to work with (optional)
      output_data: Expected output for validation (optional)
      solution_path: Path to save solution file (optional, if None integrates into codebase)
      test_path: Path to save test file (optional, if None integrates into codebase)
      enable_plan_mode: Enable plan mode for this specific request

    Kwargs:
      development_guidelines: Custom development guidelines for Claude Code
      instruction_prompt: Custom instruction prompt for Claude Code
      plan_mode_instruction_prompt: Custom instruction prompt for Claude Code on plan mode
      test_instructions: Custom instructions for running tests

    Returns:
      WorkflowState with results after processing complete
    """
    # Set static configuration as instance attributes
    self.problem_description = problem_description
    self.input_data = input_data
    self.output_data = output_data
    self.solution_path = solution_path
    self.test_path = test_path
    self.integrate_into_codebase = solution_path is None and test_path is None

    # Prompt overrides
    self.development_guidelines = kwargs.get('development_guidelines') or prompts.development_guidelines()
    self.instruction_prompt = kwargs.get('instruction_prompt') or prompts.instruction_prompt()
    self.plan_mode_instruction_prompt = kwargs.get('plan_mode_instruction_prompt') or prompts.plan_mode_instruction_prompt()
    self.test_instructions = kwargs.get('test_instructions') or prompts.test_instructions(self.solution_path)

    # Create initial execution state
    execution_state = WorkflowState()

    utils.print_with_timestamp(self._get_process_start_message(problem_description))

    try:
      # Step 1: Execute plan mode if enabled
      if enable_plan_mode:
        utils.print_with_timestamp("📋 Starting plan mode...")

        plan_state = PlanState()
        plan_result = self.plan_graph.invoke(plan_state)
        if isinstance(plan_result, dict):
          final_plan_state = PlanState(**plan_result)
        else:
          final_plan_state = plan_result

        # Check for plan mode errors
        if final_plan_state.should_terminate_early or final_plan_state.error_message:
          execution_state.should_terminate_early = True
          execution_state.error_message = final_plan_state.error_message or "Plan mode terminated early"
          utils.print_with_timestamp("🚫 Plan mode terminated with error")
          return execution_state

        # Extract approved plan for execution
        if final_plan_state.plan_approved and final_plan_state.plan_content:
          execution_state.approved_plan = final_plan_state.plan_content
          utils.print_with_timestamp("✅ Plan mode completed successfully!")
          utils.print_with_timestamp(f"📋 Plan score: {final_plan_state.plan_review_score:.2f}/1.0")
        else:
          utils.print_with_timestamp("⚠️ Plan mode completed without approved plan")

      # Step 2: Execute main workflow
      utils.print_with_timestamp("🚀 Starting execution mode...")
      final_state = self.execution_graph.invoke(execution_state)

      # Ensure we return a WorkflowState object
      if isinstance(final_state, dict):
        final_state = WorkflowState(**final_state)

      return final_state

    except Exception as e:
      utils.print_error(f"Error during execution: {e}")
      execution_state.error_message = str(e)
      return execution_state


class FeedbackSupervisorAgent(BaseSupervisorAgent):
  """
  Intelligent supervisor with iterative feedback loops for automated problem-solving.

  This supervisor manages Claude Code sessions with comprehensive feedback mechanisms,
  analyzing failures and providing targeted guidance to improve solutions iteratively.
  Unlike single-shot execution, this supervisor continues refining the solution until
  it meets quality standards or reaches maximum iteration limits.

  The supervisor works by:
  1. Initiating Claude Code sessions with structured problem descriptions
  2. Monitoring Claude's real-time progress through SDK message streaming
  3. Tracking todo list updates to understand Claude's planning and execution
  4. Validating solutions by running automated tests with detailed feedback
  5. Providing LLM-powered guidance when Claude encounters issues
  6. Managing session continuity across multiple iterations with context management

  Key Features:
  - Iterative refinement with intelligent feedback loops
  - LLM-powered error analysis and guidance generation
  - Context length management with message reduction
  - Comprehensive validation with detailed feedback
  - Session resumption for continuity across iterations
  - Support for both file creation and codebase integration modes

  Example:
      >>> # Basic feedback supervisor
      >>> agent = FeedbackSupervisorAgent()
      >>> result = agent.process(
      >>>     'Create a function to calculate fibonacci numbers',
      >>>     solution_path='fib.py',
      >>>     test_path='test_fib.py'
      >>> )
      >>>
      >>> # With custom configuration
      >>> config = openai_config(model_name='gpt-4', max_iterations=10)
      >>> agent = FeedbackSupervisorAgent(config=config)
      >>> result = agent.process('Create a sorting algorithm')
      >>>
      >>> # Check results after feedback loops
      >>> if result.is_solved:
      >>>     print(f'Solved after {result.current_iteration} iterations')
      >>> else:
      >>>     print(f'Max iterations reached: {result.error_message}')

  Args:
    llm: Optional LangChain LLM model for guidance generation
    config: Optional SupervisorConfig instance for agent configuration
    append_system_prompt: Optional additional instructions for Claude Code
  """

  def _get_process_start_message(self, problem_description: str) -> str:
    """Get the log message for iterative problem solving start."""
    return f"🚀 Starting iterative problem solving: {problem_description}"

  def build_execution_graph(self):
    """Build the execution LangGraph workflow with feedback loops"""
    workflow = StateGraph(WorkflowState)

    # Core workflow nodes
    workflow.add_node(ExecutionModeNodes.INITIATE_CLAUDE, self.initiate_claude_code_session)
    workflow.add_node(ExecutionModeNodes.EXECUTE_CLAUDE, self.execute_claude_session)
    workflow.add_node(ExecutionModeNodes.REVIEW_SESSION, self.review_session)
    workflow.add_node(ExecutionModeNodes.TEST_AND_ANALYZE, self.test_and_analyze)
    workflow.add_node(ExecutionModeNodes.GENERATE_GUIDANCE, self.generate_guidance)
    workflow.add_node(ExecutionModeNodes.REDUCE_MESSAGE, self.reduce_message_and_retry)
    workflow.add_node(ExecutionModeNodes.FINALIZE, self.finalize_solution)

    # Execution workflow
    workflow.add_edge(START, ExecutionModeNodes.INITIATE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.INITIATE_CLAUDE, ExecutionModeNodes.EXECUTE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.EXECUTE_CLAUDE, ExecutionModeNodes.REVIEW_SESSION)
    workflow.add_conditional_edges(
      ExecutionModeNodes.REVIEW_SESSION,
      self.decide_next_action,
      {
        'validate': ExecutionModeNodes.TEST_AND_ANALYZE,
        'guide': ExecutionModeNodes.GENERATE_GUIDANCE,
        'reduce': ExecutionModeNodes.REDUCE_MESSAGE,
        'finish': ExecutionModeNodes.FINALIZE
      }
    )
    workflow.add_conditional_edges(
      ExecutionModeNodes.TEST_AND_ANALYZE,
      self.should_iterate,
      {
        'continue': ExecutionModeNodes.GENERATE_GUIDANCE,
        'finish': ExecutionModeNodes.FINALIZE,
      }
    )
    workflow.add_edge(ExecutionModeNodes.GENERATE_GUIDANCE, ExecutionModeNodes.EXECUTE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.REDUCE_MESSAGE, ExecutionModeNodes.EXECUTE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.FINALIZE, END)

    return workflow.compile()


class SingleShotSupervisorAgent(BaseSupervisorAgent):
  """
  Single-execution supervisor for Claude Code without iterative feedback loops.

  This supervisor executes Claude Code once and reports results without attempting
  to fix issues or provide guidance for iteration. It's designed for scenarios where
  you want a fast, single-attempt solution or where feedback loops are handled
  externally.

  The supervisor workflow is simplified to:
  1. Initiate Claude Code session with problem description
  2. Execute the session until completion
  3. Collect results and analyze outputs
  4. Validate solution by running tests
  5. Report final status (success or failure)

  Key Features:
  - Single execution without iteration
  - Fast and deterministic results
  - Full test validation and output data extraction
  - Same configuration and setup as feedback supervisor
  - Ideal for simple problems or external iteration control

  Example:
      >>> # Basic single-shot execution
      >>> agent = SingleShotSupervisorAgent()
      >>> result = agent.process(
      >>>     'Create a function to reverse a string',
      >>>     solution_path='reverse.py',
      >>>     test_path='test_reverse.py'
      >>> )
      >>>
      >>> # Check if solved in single attempt
      >>> if result.is_solved:
      >>>     print('Problem solved in single shot!')
      >>> else:
      >>>     print(f'Failed: {result.error_message}')
      >>>     print('Consider using FeedbackSupervisorAgent for iteration')

  Args:
    llm: Optional LangChain LLM model (not used in single-shot mode)
    config: Optional SupervisorConfig instance for agent configuration
    append_system_prompt: Optional additional instructions for Claude Code
  """

  def _get_process_start_message(self, problem_description: str) -> str:
    """Get the log message for single-shot problem solving start."""
    return f"🚀 Starting single-shot problem solving: {problem_description}"

  def test_solution(self, state: WorkflowState) -> WorkflowState:
    """Test the solution without generating feedback for iteration (single-shot only)"""
    utils.print_with_timestamp("\n🔍 Testing single-shot solution...")

    # Reset validation state
    state.is_solved = False

    if self.integrate_into_codebase:
      # In integration mode, we validate by running tests on the entire codebase
      utils.print_debug("Integration mode: testing codebase changes...")
      state = self._run_integration_tests(state)
    else:
      # Standard mode: check if both files exist first
      missing_files = []
      if self.solution_path is None:
        missing_files.append("Solution file not specified")
      elif not os.path.exists(self.solution_path):
        missing_files.append(f"Solution file {self.solution_path}")

      if self.test_path is None:
        missing_files.append("Test file not specified")
      elif not os.path.exists(self.test_path):
        missing_files.append(f"Test file {self.test_path}")

      if missing_files:
        utils.print_error(f"Missing files: {', '.join(missing_files)}")
        return state

      # Run the tests to validate
      state = self._run_tests(state)

    # For single-shot, we only care about pass/fail, no detailed feedback
    if state.is_solved:
      utils.print_success("✅ Single-shot solution validation successful")
      # Extract output data if needed (for data science workflows)
      if self.input_data is not None:
        state = self._extract_output_data(state)
    else:
      utils.print_error("❌ Single-shot solution validation failed")

    return state

  def build_execution_graph(self):
    """Build the simplified execution LangGraph workflow"""
    workflow = StateGraph(WorkflowState)

    workflow.add_node(ExecutionModeNodes.INITIATE_CLAUDE, self.initiate_claude_code_session)
    workflow.add_node(ExecutionModeNodes.EXECUTE_CLAUDE, self.execute_claude_session)
    workflow.add_node(ExecutionModeNodes.REVIEW_SESSION, self.review_session)
    workflow.add_node(ExecutionModeNodes.TEST_SOLUTION, self.test_solution)
    workflow.add_node(ExecutionModeNodes.FINALIZE, self.finalize_solution)

    # Linear workflow: initiate → execute → review → test → finalize
    workflow.add_edge(START, ExecutionModeNodes.INITIATE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.INITIATE_CLAUDE, ExecutionModeNodes.EXECUTE_CLAUDE)
    workflow.add_edge(ExecutionModeNodes.EXECUTE_CLAUDE, ExecutionModeNodes.REVIEW_SESSION)
    workflow.add_edge(ExecutionModeNodes.REVIEW_SESSION, ExecutionModeNodes.TEST_SOLUTION)
    workflow.add_edge(ExecutionModeNodes.TEST_SOLUTION, ExecutionModeNodes.FINALIZE)
    workflow.add_edge(ExecutionModeNodes.FINALIZE, END)

    return workflow.compile()

  def finalize_solution(self, state: WorkflowState) -> WorkflowState:
    """Finalize the single-shot solution without iteration checking"""
    if state.is_solved:
      utils.print_with_timestamp(f"\n🎉 {utils.green(utils.bold('Solution completed successfully'))} in single execution!")

      if self.integrate_into_codebase:
        utils.print_debug(utils.green('Solution integrated into existing codebase'))
        utils.print_debug(utils.green('Tests integrated into existing test structure'))
      else:
        utils.print_with_timestamp(f"💾 {utils.cyan('Code saved to:')} {self.solution_path}")
        utils.print_with_timestamp(f"💾 {utils.cyan('Tests saved to:')} {self.test_path}")

      if self.output_data is not None:
        utils.print_with_timestamp(f"📊 {utils.cyan('Output data:')} {type(self.output_data).__name__}")
        if hasattr(self.output_data, '__len__') and len(self.output_data) < 20:
          utils.print_with_timestamp(f"📊 {utils.cyan('Result:')} {self.output_data}")

      if not self.integrate_into_codebase:
        utils.print_with_timestamp(f"\n🚀 {utils.yellow('You can run the tests manually with:')} pytest {self.test_path}")
    else:
      # Single-shot failed - report the specific issues without mentioning iterations
      if state.should_terminate_early and state.error_message and "Credit/Quota Error" in state.error_message:
        utils.display_credit_quota_error(
          error_message=state.error_message,
          use_bedrock=self.config.claude_code.use_bedrock,
          current_iteration=state.current_iteration,
          claude_todos=state.claude_todos,
          claude_log=state.claude_log
        )
      elif state.error_message:
        utils.print_with_timestamp(f"\n❌ {utils.red('Single-shot execution encountered errors.')}")
        utils.print_with_timestamp(f"📝 {utils.yellow('Error:')} {utils.red(state.error_message)}")
      else:
        utils.print_with_timestamp(f"\n❌ {utils.red('Single-shot execution completed but solution validation failed.')}")
        utils.print_with_timestamp("📝 The solution may have issues that prevent tests from passing.")

      if not self.integrate_into_codebase:
        utils.print_with_timestamp("\n📁 Files generated (may contain partial solutions):")
        if self.solution_path is not None and os.path.exists(self.solution_path):
          utils.print_with_timestamp(f"  - {self.solution_path}")
        if self.test_path is not None and os.path.exists(self.test_path):
          utils.print_with_timestamp(f"  - {self.test_path}")

    return state
