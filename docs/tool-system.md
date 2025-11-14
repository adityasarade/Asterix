# Tool System

Asterix provides a powerful tool system that allows agents to interact with external functionality. Tools can be custom functions you define, or built-in memory management tools.

## Table of Contents

- [Built-in Memory Tools](#built-in-memory-tools)
- [Creating Custom Tools](#creating-custom-tools)
- [Parameter Validation](#parameter-validation)
- [Tool Categories](#tool-categories)
- [Retry Logic](#retry-logic)
- [Error Handling](#error-handling)
- [Auto-Documentation](#auto-documentation)
- [Tool Discovery](#tool-discovery)
- [Advanced Tool Development](#advanced-tool-development)

---

## Built-in Memory Tools

Agents have 5 built-in tools for managing their memory:

| Tool | Category | Description |
|------|----------|-------------|
| `core_memory_append` | memory | Add content to a memory block |
| `core_memory_replace` | memory | Replace content in a memory block |
| `archival_memory_insert` | memory | Store information in Qdrant for long-term retrieval |
| `archival_memory_search` | memory | Search archived memories semantically |
| `conversation_search` | memory | Search conversation history |

These tools are called automatically by the agent when needed.

---

## Creating Custom Tools

Register custom tools using the decorator pattern:

```python
from asterix import Agent

agent = Agent(...)

@agent.tool(
    name="execute_shell",
    description="Run a shell command and return output"
)
def execute_shell(command: str) -> str:
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

@agent.tool(name="search_web")
def search_web(query: str) -> str:
    # Your web search implementation
    return "Search results..."

# Agent can now use these tools
response = agent.chat("List all Python files in the current directory")
```

---

## Parameter Validation

Tools can define validation constraints for their parameters:

```python
from asterix.tools.base import Tool, ParameterConstraint

@agent.tool(
    name="create_user",
    description="Create a new user account",
    constraints={
        "username": ParameterConstraint(
            min_length=3,
            max_length=20,
            pattern=r'^[a-zA-Z0-9_]+$'
        ),
        "age": ParameterConstraint(
            min_value=13,
            max_value=120
        )
    }
)
def create_user(username: str, age: int) -> str:
    return f"Created user {username}, age {age}"
```

### Available Constraints

- **`min_length`** / **`max_length`** - String length validation
- **`min_value`** / **`max_value`** - Numeric range validation
- **`pattern`** - Regex pattern matching
- **`allowed_values`** - Whitelist of acceptable values

---

## Tool Categories

Organize tools by category for better discovery:

```python
from asterix.tools.base import ToolCategory

# Tools are automatically categorized
memory_tools = agent._tool_registry.get_by_category(ToolCategory.MEMORY)
file_tools = agent._tool_registry.get_by_category(ToolCategory.FILE_OPS)

# List all categories with counts
categories = agent._tool_registry.list_categories()
print(categories)  # {"memory": 5, "file_operations": 3, "custom": 2}
```

### Available Categories

- `ToolCategory.MEMORY` - Memory management tools
- `ToolCategory.FILE_OPS` - File operations
- `ToolCategory.WEB` - Web/API operations
- `ToolCategory.DATA` - Data processing
- `ToolCategory.CUSTOM` - User-defined tools

---

## Retry Logic

Enable automatic retries for transient failures:

```python
from asterix.tools.base import Tool

@agent.tool(
    name="fetch_data",
    description="Fetch data from API",
    retry_on_error=True,
    max_retries=3
)
def fetch_data(url: str) -> str:
    # Will retry up to 3 times with exponential backoff
    response = requests.get(url)
    return response.text
```

**Retry behavior:**
- Exponential backoff between retries
- Only retries on transient errors (network, timeouts)
- Preserves original error on final failure

---

## Error Handling

Rich error context and helpful suggestions:

```python
from asterix.tools.base import ToolNotFoundError, ToolExecutionError, ToolValidationError

try:
    # Typo in tool name
    agent._tool_registry.execute_tool("read_fle", filepath="test.txt")
except ToolNotFoundError as e:
    print(e)  # Suggests similar tool names

try:
    # Invalid parameter
    agent._tool_registry.execute_tool("create_user", username="ab", age=5)
except ToolValidationError as e:
    print(e)  # Shows validation constraints and provided value
```

### Validation Errors

Parameter validation errors include helpful context:

```
Tool 'create_user' parameter 'username' validation failed:
 username length must be >= 3, got 2
 Provided value: ab
```

### Error Context

All tool errors include rich metadata for debugging:

```python
try:
    result = tool.execute(invalid_param="value")
except Exception as e:
    # Error metadata includes:
    # - Tool name
    # - Exception type
    # - Parameters provided
    # - Retry attempts (if applicable)
    # - Full stack trace in logs
    pass
```

---

## Auto-Documentation

Generate documentation for your tools:

```python
# Single tool documentation
docs = agent._tool_registry.generate_tool_docs("read_file", format="markdown")
print(docs)

# Complete registry documentation
full_docs = agent._tool_registry.generate_registry_docs(
    format="markdown",
    group_by_category=True
)

# Save to file
with open("TOOL_REFERENCE.md", "w") as f:
    f.write(full_docs)

# Export as JSON catalog
catalog = agent._tool_registry.export_tool_catalog("json")

# Quick reference guide
quick_ref = agent._tool_registry.generate_quick_reference()
```

### Documentation Formats

- **Markdown** - Human-readable docs
- **JSON** - Machine-parseable catalog
- **YAML** - Configuration-style docs

---

## Tool Discovery

Find tools by various criteria:

```python
# Filter by category
memory_tools = agent._tool_registry.filter_tools(category=ToolCategory.MEMORY)

# Filter by name pattern
search_tools = agent._tool_registry.filter_tools(name_pattern="search")

# Filter by capabilities
validated_tools = agent._tool_registry.filter_tools(has_validation=True)
retry_tools = agent._tool_registry.filter_tools(has_retry=True)

# Get detailed tool info
info = agent._tool_registry.get_tool_info("core_memory_append")
print(f"Category: {info['category']}")
print(f"Constraints: {info['constraints']}")
print(f"Examples: {info['examples']}")
```

---

## Advanced Tool Development

Creating custom tools with full features:

```python
from asterix.tools.base import Tool, ToolCategory, ParameterConstraint

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Custom tool with validation",
            func=self.execute,
            category=ToolCategory.CUSTOM,
            constraints={
                "param": ParameterConstraint(min_length=5)
            },
            retry_on_error=True,
            max_retries=2
        )

    def execute(self, param: str) -> str:
        # Your tool logic here
        return f"Processed: {param}"

# Register with agent
agent.register_tool(MyCustomTool())
```

### Tool Configuration Options

When registering tools, you can configure:

```python
from asterix.tools.base import Tool, ToolCategory, ParameterConstraint

@agent.tool(
    name="advanced_tool",
    description="Tool with full configuration",
    category=ToolCategory.DATA,
    constraints={
        "query": ParameterConstraint(min_length=1, max_length=500)
    },
    examples=[
        "advanced_tool(query='search term')",
        "advanced_tool(query='another example')"
    ],
    retry_on_error=True,
    max_retries=3
)
def advanced_tool(query: str) -> str:
    return f"Processed: {query}"
```

### Tool System Features

- **Automatic Schema Generation** - Type hints → OpenAI function schemas
- **Parameter Validation** - Min/max values, lengths, patterns, allowed values
- **Category Organization** - Group tools by purpose (memory, file_ops, web, etc.)
- **Retry Logic** - Automatic retries with exponential backoff
- **Error Recovery** - Smart error messages with hints and suggestions
- **Auto-Documentation** - Generate markdown/JSON/YAML docs from metadata
- **Tool Discovery** - Filter and search tools by name, category, capabilities

---

## See Also

- [Memory System](memory-system.md) - How memory tools work
- [API Reference](api-reference.md) - Complete tool reference
- [Examples Guide](examples-guide.md) - Tool usage examples
