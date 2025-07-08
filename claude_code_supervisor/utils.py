"""
Utility functions for the Claude Code Supervisor.

Provides terminal coloring, formatting, and other helper functions
to enhance the supervisor's output and functionality.
"""

from datetime import datetime


def timestamp() -> str:
  """Get current timestamp for logging"""
  return datetime.now().strftime('%H:%M:%S')


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
                              claude_output_log: list = None) -> None:
  """
  Display comprehensive error messaging for credit/quota issues.
  
  Args:
    error_message: The raw error message from the API
    use_bedrock: Whether using AWS Bedrock (affects provider-specific guidance)
    current_iteration: Number of completed iterations
    claude_todos: List of todos attempted
    claude_output_log: List of output log entries
  """
  if claude_todos is None:
    claude_todos = []
  if claude_output_log is None:
    claude_output_log = []
    
  print(f"\n[{timestamp()}] 🚫 {red(bold('API Credit/Quota Error - Session Terminated'))}")
  print(f"[{timestamp()}] " + "=" * 60)
  
  # Extract the actual error message
  error_details = error_message.replace("API Credit/Quota Error - Early Termination: ", "")
  print(f"[{timestamp()}] 📝 {yellow('Error Details:')} {red(error_details)}")
  
  # Determine the likely provider
  provider = "Amazon Bedrock" if use_bedrock else "Claude API (Anthropic)"
  print(f"[{timestamp()}] 🔧 {cyan('Provider:')} {provider}")
  
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
    print(f"\n[{timestamp()}] 💰 {yellow('CREDIT BALANCE ISSUE:')}")
    print(f"[{timestamp()}] Your API account has insufficient credits to complete this request.")
    if use_bedrock:
      print(f"[{timestamp()}] 🔗 Check your AWS billing: https://console.aws.amazon.com/billing/")
      print(f"[{timestamp()}] 🔗 Bedrock pricing: https://aws.amazon.com/bedrock/pricing/")
    else:
      print(f"[{timestamp()}] 🔗 Check your Anthropic credits: https://console.anthropic.com/settings/billing")
      print(f"[{timestamp()}] 🔗 Add credits: https://console.anthropic.com/settings/billing")
  
  elif any(keyword in error_lower for keyword in quota_keywords):
    print(f"\n[{timestamp()}] ⏰ {yellow('RATE LIMIT/QUOTA EXCEEDED:')}")
    print(f"[{timestamp()}] You have exceeded the API rate limits or quotas.")
    if use_bedrock:
      print(f"[{timestamp()}] 🔗 Check service quotas: https://console.aws.amazon.com/servicequotas/")
      print(f"[{timestamp()}] 📖 Bedrock quotas: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html")
    else:
      print(f"[{timestamp()}] 🔗 Check rate limits: https://console.anthropic.com/settings/limits")
      print(f"[{timestamp()}] 📖 API limits: https://docs.anthropic.com/en/api/rate-limits")
  
  elif any(keyword in error_lower for keyword in billing_keywords):
    print(f"\n[{timestamp()}] 💳 {yellow('BILLING/PAYMENT ISSUE:')}")
    print(f"[{timestamp()}] There appears to be a billing or payment issue with your account.")
    if use_bedrock:
      print(f"[{timestamp()}] 🔗 AWS billing console: https://console.aws.amazon.com/billing/")
      print(f"[{timestamp()}] 📞 AWS support: https://support.aws.amazon.com/")
    else:
      print(f"[{timestamp()}] 🔗 Anthropic billing: https://console.anthropic.com/settings/billing")
      print(f"[{timestamp()}] 📧 Contact support: https://console.anthropic.com/support")
  
  elif any(keyword in error_lower for keyword in account_keywords):
    print(f"\n[{timestamp()}] 🔒 {yellow('ACCOUNT STATUS ISSUE:')}")
    print(f"[{timestamp()}] Your account may be suspended, restricted, or inactive.")
    if use_bedrock:
      print(f"[{timestamp()}] 🔗 AWS account status: https://console.aws.amazon.com/")
      print(f"[{timestamp()}] 📞 AWS support: https://support.aws.amazon.com/")
    else:
      print(f"[{timestamp()}] 🔗 Anthropic console: https://console.anthropic.com/")
      print(f"[{timestamp()}] 📧 Contact support: https://console.anthropic.com/support")
  
  else:
    print(f"\n[{timestamp()}] ❓ {yellow('UNKNOWN API ERROR:')}")
    print(f"[{timestamp()}] An API-related error occurred that prevented completion.")
    if use_bedrock:
      print(f"[{timestamp()}] 🔗 AWS support: https://support.aws.amazon.com/")
      print(f"[{timestamp()}] 📖 Bedrock docs: https://docs.aws.amazon.com/bedrock/")
    else:
      print(f"[{timestamp()}] 📧 Anthropic support: https://console.anthropic.com/support")
      print(f"[{timestamp()}] 📖 API docs: https://docs.anthropic.com/")
  
  # General recommendations
  print(f"\n[{timestamp()}] 💡 {cyan('RECOMMENDATIONS:')}")
  print(f"[{timestamp()}] 1. Check your account status and billing information")
  print(f"[{timestamp()}] 2. Verify you have sufficient credits/quota for your usage")
  print(f"[{timestamp()}] 3. If this is a temporary rate limit, wait a few minutes and try again")
  print(f"[{timestamp()}] 4. Consider reducing the scope of your request if limits are an issue")
  print(f"[{timestamp()}] 5. Contact support if the problem persists")
  
  # Show iteration progress
  print(f"\n[{timestamp()}] 📊 {cyan('SESSION PROGRESS:')}")
  print(f"[{timestamp()}] - Completed iterations: {current_iteration}")
  print(f"[{timestamp()}] - Tasks attempted: {len(claude_todos)}")
  if claude_todos:
    completed_todos = [todo for todo in claude_todos if todo.get('status') == 'completed']
    print(f"[{timestamp()}] - Tasks completed: {len(completed_todos)}/{len(claude_todos)}")
  
  # Show any partial work
  if claude_output_log:
    print(f"[{timestamp()}] - Some work was completed before the error occurred")
    print(f"[{timestamp()}] - You may be able to resume or retry with a smaller scope")


def detect_errors_in_output(output_log: list) -> tuple[list, list]:
  """
  Detect error patterns in Claude's output log.
  
  Args:
    output_log: List of text responses from Claude
    
  Returns:
    tuple: (general_error_indicators, credit_quota_errors)
      - general_error_indicators: List of responses with general errors
      - credit_quota_errors: List of responses with credit/quota errors
  """
  error_indicators = []
  credit_quota_errors = []
  
  # Define error detection keywords - be more specific to avoid false positives
  general_error_keywords = [
    'error occurred', 'failed to', 'exception raised', 'traceback',
    'cannot', "can't", 'unable to', 'permission denied', 'file not found',
    'syntax error', 'import error', 'module not found', 'command not found'
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
    
    # Check for general errors
    elif any(keyword in response_lower for keyword in general_error_keywords):
      truncated_response = response[:200] + '...' if len(response) > 200 else response
      error_indicators.append(truncated_response)
  
  return error_indicators, credit_quota_errors


def format_error_message(general_errors: list, credit_quota_errors: list) -> tuple[str, bool]:
  """
  Format error messages and determine if early termination is needed.
  
  Args:
    general_errors: List of general error indicators
    credit_quota_errors: List of credit/quota error indicators
    
  Returns:
    tuple: (error_message, should_terminate_early)
  """
  if credit_quota_errors:
    error_message = f"API Credit/Quota Error - Early Termination: {'; '.join(credit_quota_errors)}"
    should_terminate_early = True
  elif general_errors:
    error_message = f"Detected errors in output: {'; '.join(general_errors)}"
    should_terminate_early = False
  else:
    error_message = ""
    should_terminate_early = False
    
  return error_message, should_terminate_early