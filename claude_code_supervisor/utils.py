"""
Utility functions for the Claude Code Supervisor.

Provides terminal coloring, formatting, and other helper functions
to enhance the supervisor's output and functionality.
"""

from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd

DataTypes = str | list | dict | tuple | np.ndarray | pd.DataFrame | pd.Series


class ToolsEnum(Enum):
  """
  Tools available for Claude Code
  Source: https://docs.anthropic.com/en/docs/claude-code/settings
  """
  AGENT = 'Agent'
  BASH = 'Bash'
  EDIT = 'Edit'
  GLOB = 'Glob'
  GREP = 'Grep'
  LS = 'LS'
  MULTIEDIT = 'MultiEdit'
  NOTEBOOKEDIT = 'NotebookEdit'
  NOTEBOOKREAD = 'NotebookRead'
  READ = 'Read'
  TODOREAD = 'TodoRead'
  TODOWRITE = 'TodoWrite'
  WEBFETCH = 'WebFetch'
  WEBSEARCH = 'WebSearch'
  WRITE = 'Write'

  @classmethod
  def all(cls) -> list[str]:
    """Get all tool names as a list"""
    return [tool.value for tool in cls]


def timestamp() -> str:
  """Get current timestamp for logging"""
  return datetime.now().strftime('%H:%M:%S')


def print_with_timestamp(message: str, color: str = None, line_break: bool = True) -> None:
  """
  Print message with timestamp and optional color.

  Args:
    message: The message to print
    color: Optional color ('blue', 'green', 'yellow', 'red', 'cyan', 'magenta', 'bold')
    line_break: Whether to add a line break after the message
  """
  timestamp_str = f"[{timestamp()}] "

  if color:
    color_func = globals().get(color)
    if color_func and callable(color_func):
      formatted_message = color_func(message)
    else:
      formatted_message = message
  else:
    formatted_message = message

  if line_break:
    print(timestamp_str + formatted_message)
  else:
    print(timestamp_str + formatted_message, end="")


def print_info(message: str, line_break: bool = True) -> None:
  """Print info message with timestamp and cyan color"""
  print_with_timestamp(f"ℹ️  {message}", "cyan", line_break)


def print_prompt(message: str, line_break: bool = True) -> None:
  """Print prompt message with timestamp and green color"""
  print_with_timestamp(f"💬 {message}", "green", line_break)


def print_success(message: str, line_break: bool = True) -> None:
  """Print success message with timestamp and green color"""
  print_with_timestamp(f"✅ {message}", "green", line_break)


def print_warning(message: str, line_break: bool = True) -> None:
  """Print warning message with timestamp and yellow color"""
  print_with_timestamp(f"⚠️  {message}", "yellow", line_break)


def print_error(message: str, line_break: bool = True) -> None:
  """Print error message with timestamp and red color"""
  print_with_timestamp(f"❌ {message}", "red", line_break)


def print_debug(message: str, line_break: bool = True) -> None:
  """Print debug message with timestamp and blue color"""
  print_with_timestamp(f"🔧 {message}", "blue", line_break)


def print_tool(message: str, line_break: bool = True) -> None:
  """Print tool message with timestamp and blue color"""
  print_with_timestamp(f"🔧 {message}", "blue", line_break)


def print_claude(message: str, line_break: bool = True) -> None:
  """Print Claude message with timestamp and blue color"""
  print_with_timestamp(f"💬 {blue('Claude:')} {blue(message)}", line_break=line_break)


def print_todo(message: str, line_break: bool = True) -> None:
  """Print todo message with timestamp and blue color"""
  print_with_timestamp(f"📋 {blue('Todo list updated:')} {blue(message)}", line_break=line_break)


def blue(text: str) -> str:
  """Add blue color to text for Claude Code outputs"""
  return f"\033[94m{text}\033[0m"


def green(text: str) -> str:
  """Add green color to text for success messages"""
  return f"\033[92m{text}\033[0m"


def yellow(text: str) -> str:
  """Add yellow color to text for warning messages"""
  return f"\033[93m{text}\033[0m"


def red(text: str) -> str:
  """Add red color to text for error messages"""
  return f"\033[91m{text}\033[0m"


def cyan(text: str) -> str:
  """Add cyan color to text for info messages"""
  return f"\033[96m{text}\033[0m"


def magenta(text: str) -> str:
  """Add magenta color to text for special messages"""
  return f"\033[95m{text}\033[0m"


def bold(text: str) -> str:
  """Make text bold"""
  return f"\033[1m{text}\033[0m"


def underline(text: str) -> str:
  """Underline text"""
  return f"\033[4m{text}\033[0m"


def reset() -> str:
  """Reset all formatting"""
  return "\033[0m"


def get_tool_info(tool_name: str, tool_input: dict) -> str:
  """Get detailed information about a tool being used"""
  try:
    if tool_name == 'Read':
      file_path = tool_input.get('file_path', 'unknown')
      return f"- {cyan('Reading:')} {file_path}"

    elif tool_name == 'Write':
      file_path = tool_input.get('file_path', 'unknown')
      content_length = len(tool_input.get('content', ''))
      return f"- {cyan('Writing:')} {file_path} ({content_length} chars)"

    elif tool_name == 'Edit':
      file_path = tool_input.get('file_path', 'unknown')
      old_string = tool_input.get('old_string', '')
      new_string = tool_input.get('new_string', '')
      return f"- {cyan('Editing:')} {file_path} ({len(old_string)}→{len(new_string)} chars)"

    elif tool_name == 'MultiEdit':
      file_path = tool_input.get('file_path', 'unknown')
      edits_count = len(tool_input.get('edits', []))
      return f"- {cyan('Multi-editing:')} {file_path} ({edits_count} edits)"

    elif tool_name == 'Bash':
      command = tool_input.get('command', 'unknown')
      command_preview = command[:50] + '...' if len(command) > 50 else command
      return f"- {cyan('Running:')} {command_preview}"

    elif tool_name == 'Glob':
      pattern = tool_input.get('pattern', 'unknown')
      path = tool_input.get('path', '.')
      return f"- {cyan('Finding files:')} {pattern} in {path}"

    elif tool_name == 'Grep':
      pattern = tool_input.get('pattern', 'unknown')
      include = tool_input.get('include', '')
      include_info = f" (*.{include})" if include else ""
      return f"- {cyan('Searching:')} '{pattern}'{include_info}"

    elif tool_name == 'LS':
      path = tool_input.get('path', '.')
      return f"- {cyan('Listing:')} {path}"

    elif tool_name == 'TodoWrite':
      todos = tool_input.get('todos', [])
      return f"- {cyan('Managing todos:')} {len(todos)} items"

    elif tool_name == 'TodoRead':
      return f"- {cyan('Reading todos')}"

    elif tool_name == 'Task':
      description = tool_input.get('description', 'unknown task')
      return f"- {cyan('Delegating:')} {description}"

    elif tool_name == 'mcp__ide__executeCode':
      code_preview = tool_input.get('code', '')[:30] + '...' if len(tool_input.get('code', '')) > 30 else tool_input.get('code', '')
      return f"- {cyan('Executing code:')} {code_preview}"

    elif tool_name == 'mcp__ide__getDiagnostics':
      uri = tool_input.get('uri', 'all files')
      return f"- {cyan('Getting diagnostics:')} {uri}"

    else:
      # Generic fallback for unknown tools
      return f"- {cyan('Parameters:')} {str(tool_input)[:100]}{'...' if len(str(tool_input)) > 100 else ''}"

  except Exception:
    # If anything goes wrong, just return empty string
    return ""


def display_credit_quota_error(error_message: str, use_bedrock: bool = False,
                               current_iteration: int = 0, claude_todos: list = None,
                               claude_log: list = None) -> None:
  """
  Display comprehensive error messaging for credit/quota issues.

  Args:
    error_message: The raw error message from the API
    use_bedrock: Whether using AWS Bedrock (affects provider-specific guidance)
    current_iteration: Number of completed iterations
    claude_todos: List of todos attempted
    claude_log: List of output log entries
  """
  if claude_todos is None:
    claude_todos = []
  if claude_log is None:
    claude_log = []

  print_with_timestamp(f"\n🚫 {red(bold('API Credit/Quota Error - Session Terminated'))}")
  print_with_timestamp("=" * 60)

  # Extract the actual error message
  error_details = error_message.replace("API Credit/Quota Error - Early Termination: ", "")
  print_with_timestamp(f"📝 {yellow('Error Details:')} {red(error_details)}")

  # Determine the likely provider
  provider = "Amazon Bedrock" if use_bedrock else "Claude API (Anthropic)"
  print_with_timestamp(f"🔧 {cyan('Provider:')} {provider}")

  # Provide comprehensive guidance based on error type
  error_lower = error_details.lower()

  # Credit/balance related errors
  credit_keywords = ['credit balance is too low', 'insufficient credits', 'balance insufficient',
                    'credits depleted', 'insufficient balance', 'credit limit']

  # Quota/rate limit related errors
  quota_keywords = ['quota exceeded', 'rate limit', 'api limit', 'usage limit', 'account limit',
                   'exceeded your quota', 'exceeded quota', 'rate limited', 'throttled',
                   'usage exceeded', 'monthly quota', 'daily quota', 'hourly quota',
                   'request limit', 'token limit exceeded']

  # Billing/payment related errors
  billing_keywords = ['billing', 'payment required', 'billing issue', 'payment issue',
                     'payment method', 'billing error', 'payment failed', 'upgrade your plan']

  # Account status related errors
  account_keywords = ['account suspended', 'account restricted', 'your account has been limited',
                     'account inactive', 'subscription expired']

  if any(keyword in error_lower for keyword in credit_keywords):
    print_with_timestamp(f"\n💰 {yellow('CREDIT BALANCE ISSUE:')}")
    print_with_timestamp("Your API account has insufficient credits to complete this request.")
    if use_bedrock:
      print_with_timestamp("🔗 Check your AWS billing: https://console.aws.amazon.com/billing/")
      print_with_timestamp("🔗 Bedrock pricing: https://aws.amazon.com/bedrock/pricing/")
    else:
      print_with_timestamp("🔗 Check your Anthropic credits: https://console.anthropic.com/settings/billing")
      print_with_timestamp("🔗 Add credits: https://console.anthropic.com/settings/billing")

  elif any(keyword in error_lower for keyword in quota_keywords):
    print_with_timestamp(f"\n⏰ {yellow('RATE LIMIT/QUOTA EXCEEDED:')}")
    print_with_timestamp("You have exceeded the API rate limits or quotas.")
    if use_bedrock:
      print_with_timestamp("🔗 Check service quotas: https://console.aws.amazon.com/servicequotas/")
      print_with_timestamp("📖 Bedrock quotas: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html")
    else:
      print_with_timestamp("🔗 Check rate limits: https://console.anthropic.com/settings/limits")
      print_with_timestamp("📖 API limits: https://docs.anthropic.com/en/api/rate-limits")

  elif any(keyword in error_lower for keyword in billing_keywords):
    print_with_timestamp(f"\n💳 {yellow('BILLING/PAYMENT ISSUE:')}")
    print_with_timestamp("There appears to be a billing or payment issue with your account.")
    if use_bedrock:
      print_with_timestamp("🔗 AWS billing console: https://console.aws.amazon.com/billing/")
      print_with_timestamp("📞 AWS support: https://support.aws.amazon.com/")
    else:
      print_with_timestamp("🔗 Anthropic billing: https://console.anthropic.com/settings/billing")
      print_with_timestamp("📧 Contact support: https://console.anthropic.com/support")

  elif any(keyword in error_lower for keyword in account_keywords):
    print_with_timestamp(f"\n🔒 {yellow('ACCOUNT STATUS ISSUE:')}")
    print_with_timestamp("Your account may be suspended, restricted, or inactive.")
    if use_bedrock:
      print_with_timestamp("🔗 AWS account status: https://console.aws.amazon.com/")
      print_with_timestamp("📞 AWS support: https://support.aws.amazon.com/")
    else:
      print_with_timestamp("🔗 Anthropic console: https://console.anthropic.com/")
      print_with_timestamp("📧 Contact support: https://console.anthropic.com/support")

  else:
    print_with_timestamp(f"\n❓ {yellow('UNKNOWN API ERROR:')}")
    print_with_timestamp("An API-related error occurred that prevented completion.")
    if use_bedrock:
      print_with_timestamp("🔗 AWS support: https://support.aws.amazon.com/")
      print_with_timestamp("📖 Bedrock docs: https://docs.aws.amazon.com/bedrock/")
    else:
      print_with_timestamp("📧 Anthropic support: https://console.anthropic.com/support")
      print_with_timestamp("📖 API docs: https://docs.anthropic.com/")

  # General recommendations
  print_with_timestamp(f"\n💡 {cyan('RECOMMENDATIONS:')}")
  print_with_timestamp("1. Check your account status and billing information")
  print_with_timestamp("2. Verify you have sufficient credits/quota for your usage")
  print_with_timestamp("3. If this is a temporary rate limit, wait a few minutes and try again")
  print_with_timestamp("4. Consider reducing the scope of your request if limits are an issue")
  print_with_timestamp("5. Contact support if the problem persists")

  # Show iteration progress
  print_with_timestamp(f"\n📊 {cyan('SESSION PROGRESS:')}")
  print_with_timestamp(f"- Completed iterations: {current_iteration}")
  print_with_timestamp(f"- Tasks attempted: {len(claude_todos)}")
  if claude_todos:
    completed_todos = [todo for todo in claude_todos if todo.get('status') == 'completed']
    print_with_timestamp(f"- Tasks completed: {len(completed_todos)}/{len(claude_todos)}")

  # Show any partial work
  if claude_log:
    print_with_timestamp("- Some work was completed before the error occurred")
    print_with_timestamp("- You may be able to resume or retry with a smaller scope")


def detect_errors_in_output(output_log: list) -> tuple[list, list, list]:
  """
  Detect error patterns in Claude's output log.

  Args:
    output_log: List of text responses from Claude

  Returns:
    tuple: (general_error_indicators, credit_quota_errors, context_length_errors)
      - general_error_indicators: List of responses with general errors
      - credit_quota_errors: List of responses with credit/quota errors
      - context_length_errors: List of responses with context length errors
  """
  error_indicators = []
  credit_quota_errors = []
  context_length_errors = []

  # Define error detection keywords - be more specific to avoid false positives
  general_error_keywords = [
    'error occurred', 'failed to', 'exception raised', 'traceback',
    'cannot', "can't", 'unable to', 'permission denied', 'file not found',
    'syntax error', 'import error', 'module not found', 'command not found'
  ]

  # Context length errors that should trigger message reduction
  context_length_keywords = [
    'input is too long', 'context length exceeded', 'maximum context length',
    'token limit exceeded', 'input too long for requested model',
    'context window exceeded', 'maximum sequence length'
  ]
  credit_quota_keywords = [
    'credit balance is too low', 'insufficient credits', 'quota exceeded',
    'rate limit', 'api limit', 'billing', 'payment required',
    'usage limit', 'account limit', 'balance insufficient', 'credits depleted',
    'exceeded your quota', 'exceeded quota', 'rate limited', 'throttled',
    'billing issue', 'payment issue', 'account suspended', 'account restricted',
    'insufficient balance', 'credit limit', 'usage exceeded', 'monthly quota',
    'daily quota', 'hourly quota', 'request limit', 'token limit exceeded',
    'your account has been limited', 'account inactive', 'subscription expired',
    'upgrade your plan', 'payment method', 'billing error', 'payment failed'
  ]

  # Check last 3 responses for error patterns
  for response in output_log[-3:]:
    response_lower = response.lower()

    # Check for credit/quota errors (these should terminate early)
    if any(keyword in response_lower for keyword in credit_quota_keywords):
      truncated_response = response[:200] + '...' if len(response) > 200 else response
      credit_quota_errors.append(truncated_response)

    # Check for context length errors (these should trigger message reduction)
    elif any(keyword in response_lower for keyword in context_length_keywords):
      truncated_response = response[:200] + '...' if len(response) > 200 else response
      context_length_errors.append(truncated_response)

    # Check for general errors
    elif any(keyword in response_lower for keyword in general_error_keywords):
      truncated_response = response[:200] + '...' if len(response) > 200 else response
      error_indicators.append(truncated_response)

  return error_indicators, credit_quota_errors, context_length_errors


def format_error_message(general_errors: list, credit_quota_errors: list, context_length_errors: list = None) -> tuple[str, bool, bool]:
  """
  Format error messages and determine if early termination or message reduction is needed.

  Args:
    general_errors: List of general error indicators
    credit_quota_errors: List of credit/quota error indicators
    context_length_errors: List of context length error indicators

  Returns:
    tuple: (error_message, should_terminate_early, should_reduce_message)
  """
  if context_length_errors is None:
    context_length_errors = []

  if credit_quota_errors:
    error_message = f"API Credit/Quota Error - Early Termination: {'; '.join(credit_quota_errors)}"
    should_terminate_early = True
    should_reduce_message = False
  elif context_length_errors:
    error_message = f"Context Length Error - Message Too Long: {'; '.join(context_length_errors)}"
    should_terminate_early = False
    should_reduce_message = True
  elif general_errors:
    error_message = f"Detected errors in output: {'; '.join(general_errors)}"
    should_terminate_early = False
    should_reduce_message = False
  else:
    error_message = ""
    should_terminate_early = False
    should_reduce_message = False

  return error_message, should_terminate_early, should_reduce_message


def reduce_message_length(message: str, reduction_factor: float = 0.7) -> str:
  """
  Reduce message length by removing non-essential parts while preserving core information.

  Args:
    message: The message to reduce
    reduction_factor: Target reduction factor (0.7 means reduce to 70% of original length)

  Returns:
    Reduced message that preserves essential information
  """
  if not message:
    return message

  lines = message.split('\n')
  target_length = int(len(message) * reduction_factor)

  # Priority order for content preservation:
  # 1. Problem description (usually at the beginning)
  # 2. Requirements/guidelines
  # 3. Input/output data (essential for data tasks)
  # 4. Examples (can be shortened)
  # 5. Verbose explanations (can be removed)

  # Find key sections
  problem_lines = []
  requirements_lines = []
  data_lines = []
  example_lines = []
  other_lines = []

  current_section = 'other'

  for line in lines:
    line_lower = line.lower()

    # Identify section types
    if any(keyword in line_lower for keyword in ['problem description', 'task:', 'create', 'implement', 'solve']):
      current_section = 'problem'
    elif any(keyword in line_lower for keyword in ['requirements', 'guidelines', 'development guidelines']):
      current_section = 'requirements'
    elif any(keyword in line_lower for keyword in ['input data', 'output data', 'expected output']):
      current_section = 'data'
    elif any(keyword in line_lower for keyword in ['example', 'demonstration', 'usage']):
      current_section = 'example'

    # Categorize lines
    if current_section == 'problem':
      problem_lines.append(line)
    elif current_section == 'requirements':
      requirements_lines.append(line)
    elif current_section == 'data':
      data_lines.append(line)
    elif current_section == 'example':
      example_lines.append(line)
    else:
      other_lines.append(line)

  # Build reduced message with priority
  reduced_lines = []
  current_length = 0

  # Always include problem description
  for line in problem_lines:
    if current_length + len(line) < target_length:
      reduced_lines.append(line)
      current_length += len(line)
    else:
      break

  # Include requirements (condensed)
  for line in requirements_lines:
    if current_length + len(line) < target_length:
      # Condense requirements by removing verbose explanations
      if len(line) > 100:
        condensed_line = line[:100] + '...'
      else:
        condensed_line = line
      reduced_lines.append(condensed_line)
      current_length += len(condensed_line)
    else:
      break

  # Include essential data info
  for line in data_lines:
    if current_length + len(line) < target_length:
      reduced_lines.append(line)
      current_length += len(line)
    else:
      break

  # Include shortened examples if space allows
  for line in example_lines:
    if current_length + len(line) < target_length:
      # Truncate long examples
      if len(line) > 150:
        shortened_line = line[:150] + '...'
      else:
        shortened_line = line
      reduced_lines.append(shortened_line)
      current_length += len(shortened_line)
    else:
      break

  # Include other essential lines if space allows
  for line in other_lines:
    if current_length + len(line) < target_length:
      if len(line) > 80:
        shortened_line = line[:80] + '...'
      else:
        shortened_line = line
      reduced_lines.append(shortened_line)
      current_length += len(shortened_line)
    else:
      break

  # Add a note about message reduction
  reduced_message = '\n'.join(reduced_lines)
  if len(reduced_message) < len(message):
    reduced_message += '\n\n[NOTE: This message was automatically reduced to fit context limits. Focus on the core requirements above.]'

  return reduced_message
