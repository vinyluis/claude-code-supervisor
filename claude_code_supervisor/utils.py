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