# Claude Development Guidelines

## General instructions

- Use 2-space indentation.
- Use single quotes for strings.
- Avoid magic numbers. Use constants, variables or parameters instead.
- Ensure that the code is well-documented, with docstrings one-liners for small functions, and more detailed docstrings for larger functions or classes.
- Use typehints whenever possible. Avoid using Any for types, preferring to omit the typing in these cases (e.g. use simply `dict` instead of `dict[str, Any]`). Use modern syntax (`list` instead of `List`). Prefer using `None` instead of `Optional` when possible.
- If you create new code, ensure that it is covered by tests. Follow the same testing conventions as the existing code.
- You have autonomy to change files, no need to ask for permission.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

# Claude Code SDK Usage Instructions

## Overview
The Claude Code SDK (`claude-code-sdk`) enables running Claude Code as a subprocess from Python scripts, allowing you to build AI-powered coding assistants and tools.

## Installation and Setup

### Prerequisites
- Python 3.8+ with `claude-code-sdk` installed: `pip install claude-code-sdk`
- Node.js (required for Claude Code CLI)
- Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

### Authentication
Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or use third-party providers:
```bash
export CLAUDE_CODE_USE_BEDROCK=1  # For Amazon Bedrock
export CLAUDE_CODE_USE_VERTEX=1   # For Google Vertex AI
```

## Basic Usage

### Simple Query
```python
from claude_code_sdk import query

async def main():
    async for message in query(prompt='Write a hello world function in Python'):
        print(message)
```

### With Options
```python
from claude_code_sdk import query, ClaudeCodeOptions

async def main():
    options = ClaudeCodeOptions(
        system_prompt='You are a Python expert',
        cwd='/path/to/project',
        max_turns=5,
        permission_mode='acceptEdits'
    )
    
    async for message in query(prompt='Refactor this code', options=options):
        print(message)
```

## Message Types

The SDK yields different message types:

### UserMessage
```python
from claude_code_sdk.types import UserMessage
# Contains: content (str)
```

### AssistantMessage
```python
from claude_code_sdk.types import AssistantMessage, TextBlock, ToolUseBlock
# Contains: content (list[ContentBlock])
# ContentBlock can be TextBlock, ToolUseBlock, or ToolResultBlock
```

### SystemMessage
```python
from claude_code_sdk.types import SystemMessage
# Contains: subtype (str), data (dict)
```

### ResultMessage
```python
from claude_code_sdk.types import ResultMessage
# Contains: session_id, num_turns, total_cost_usd, usage stats, etc.
```

## Advanced Configuration

### ClaudeCodeOptions Parameters
```python
ClaudeCodeOptions(
    # Tool control
    allowed_tools=['Read', 'Write', 'Bash'],           # Limit available tools
    disallowed_tools=['WebSearch'],                    # Block specific tools
    permission_mode='acceptEdits',                     # 'default', 'acceptEdits', 'bypassPermissions'
    
    # System prompts
    system_prompt='Custom system prompt',              # Replace default system prompt
    append_system_prompt='Additional instructions',    # Add to default system prompt
    
    # Conversation control
    max_turns=10,                                      # Limit conversation length
    continue_conversation=True,                        # Continue previous conversation
    resume='session-id-123',                          # Resume specific session
    
    # Model and performance
    model='claude-3-5-sonnet-20241022',              # Specify model
    max_thinking_tokens=8000,                         # Control thinking tokens
    
    # Working directory
    cwd='/path/to/project',                           # Set working directory
    
    # MCP (Model Context Protocol) integration
    mcp_servers={
        'filesystem': {
            'command': 'npx',
            'args': ['@modelcontextprotocol/server-filesystem', '/path/to/files']
        }
    },
    mcp_tools=['read_file', 'write_file']
)
```

## Message Processing Examples

### Extract Text Content
```python
from claude_code_sdk.types import AssistantMessage, TextBlock

async for message in query(prompt='Explain this code'):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f'Assistant: {block.text}')
```

### Monitor Tool Usage
```python
from claude_code_sdk.types import AssistantMessage, ToolUseBlock, ToolResultBlock

async for message in query(prompt='Fix the bug in main.py'):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                print(f'Tool used: {block.name} with input: {block.input}')
            elif isinstance(block, ToolResultBlock):
                print(f'Tool result: {block.content}')
```

### Track Session Information
```python
from claude_code_sdk.types import ResultMessage

async for message in query(prompt='Create a new feature'):
    if isinstance(message, ResultMessage):
        print(f'Session ID: {message.session_id}')
        print(f'Total turns: {message.num_turns}')
        print(f'Duration: {message.duration_ms}ms')
        print(f'Cost: ${message.total_cost_usd}')
```

## Error Handling

```python
from claude_code_sdk import query, ClaudeSDKError, CLINotFoundError, ProcessError

try:
    async for message in query(prompt='Help me code'):
        print(message)
except CLINotFoundError as e:
    print(f'Claude Code CLI not found: {e}')
except ProcessError as e:
    print(f'Process error: {e}')
except ClaudeSDKError as e:
    print(f'SDK error: {e}')
```

## Todo List Tracking

**Important**: The Claude Code SDK does not provide direct access to the agent's todo list. The todo list functionality is internal to the Claude Code CLI and is not exposed through the SDK's message types.

To track tasks programmatically, you would need to:
1. Parse `TextBlock` content from `AssistantMessage` for todo-related text
2. Monitor `ToolUseBlock` calls to the `TodoWrite` and `TodoRead` tools
3. Implement your own task tracking based on the conversation flow

Example of monitoring todo-related tool usage:
```python
from claude_code_sdk.types import AssistantMessage, ToolUseBlock

async for message in query(prompt='Plan and implement a new feature'):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                if block.name in ['TodoWrite', 'TodoRead']:
                    print(f'Todo tool used: {block.name}')
                    if block.name == 'TodoWrite':
                        todos = block.input.get('todos', [])
                        for todo in todos:
                            print(f'  - {todo["content"]} ({todo["status"]})')
```

## Complete Example

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage

async def main():
    options = ClaudeCodeOptions(
        system_prompt='You are a helpful coding assistant',
        cwd='/path/to/project',
        permission_mode='acceptEdits',
        max_turns=3
    )
    
    try:
        async for message in query(
            prompt='Review the code in main.py and suggest improvements',
            options=options
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f'Assistant: {block.text}')
                    elif isinstance(block, ToolUseBlock):
                        print(f'Using tool: {block.name}')
            elif isinstance(message, ResultMessage):
                print(f'Completed in {message.num_turns} turns')
                print(f'Session ID: {message.session_id}')
                
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())
```

## Best Practices

1. **Set appropriate `cwd`**: Always set the working directory to your project root
2. **Use `permission_mode='acceptEdits'`** for automated workflows to avoid interactive prompts
3. **Monitor `ResultMessage`** to track costs and performance
4. **Handle errors gracefully** with proper exception handling
5. **Limit `max_turns`** to prevent runaway conversations
6. **Use `allowed_tools`** to restrict functionality for security
7. **Process messages asynchronously** to handle real-time responses