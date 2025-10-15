# Asterix

**Stateful AI agents with editable memory blocks and persistent storage.**

> **⚠️ EARLY DEVELOPMENT - NOT READY FOR PRODUCTION YET**
> 
> This library is under active development.

Asterix is a lightweight Python library for building AI agents that can remember, learn, and persist their state across sessions. No servers required - just `pip install` and start building.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- **🧠 Editable Memory Blocks** - Agents can read and write their own memory via built-in tools
- **💾 Persistent Storage** - State saves across sessions (JSON/SQLite backends)
- **🔍 Semantic Search** - Qdrant Cloud integration for long-term memory retrieval
- **🛠️ Enhanced Tool System** - Easy decorator pattern with parameter validation, retry logic, and categories
- **📚 Auto-Documentation** - Tools automatically generate markdown/JSON documentation
- **🚨 Smart Error Handling** - Helpful error messages with suggestions and context
- **🔄 Multi-Model Support** - Works with Groq, OpenAI, and extensible to others
- **📦 No Server Required** - Pure Python library, runs anywhere

---

## 🚀 Quick Start

### Installation

```bash
pip install asterix-agent
```

Or with UV (faster):
```bash
uv pip install asterix-agent
```

### Basic Usage

```python
from asterix import Agent, BlockConfig

# Create an agent with custom memory blocks
agent = Agent(
    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    },
    model="groq/llama-3.3-70b-versatile"
)

# Chat with your agent
response = agent.chat("Hello! Remember that I prefer Python over JavaScript.")
print(response)

# Agent automatically updates its memory
# Memory persists across conversations
```

### Add Custom Tools

```python
@agent.tool(name="read_file", description="Read a file from disk")
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

# Now your agent can read files
response = agent.chat("Read config.yaml and summarize the settings")
```

## 🛠️ Advanced Tool Features

### Parameter Validation

Tools can define validation constraints for their parameters:
```python
from asterix.tools import tool, ParameterConstraint

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

### Tool Categories

Organize tools by category for better discovery:
```python
from asterix.tools import ToolCategory

# Tools are automatically categorized
memory_tools = agent._tool_registry.get_by_category(ToolCategory.MEMORY)
file_tools = agent._tool_registry.get_by_category(ToolCategory.FILE_OPS)

# List all categories with counts
categories = agent._tool_registry.list_categories()
print(categories)  # {"memory": 5, "file_operations": 3, "custom": 2}
```

### Retry Logic

Enable automatic retries for transient failures:
```python
from asterix.tools import Tool

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

### Error Handling

Rich error context and helpful suggestions:
```python
# Automatic error suggestions
try:
    agent.get_tool("read_fil")  # Typo!
except ToolNotFoundError as e:
    print(e)  # "Tool 'read_fil' not found. Did you mean: read_file?"
```

### Auto-Documentation

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

### Tool Discovery

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

### Save & Load State

```python
# Save agent state
agent.save_state()

# Later session - load previous state
agent = Agent.load_state("agent_id")
agent.chat("What were we discussing?")  # Remembers everything!
```

## 🔍 Logging

Asterix uses Python's standard logging module. By default, logs are not displayed. To enable logging:

### Console Logging
```python
import logging

# Show all logs
logging.basicConfig(level=logging.INFO)

# Or just show errors
logging.basicConfig(level=logging.ERROR)
```

### File Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='asterix.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Fine-grained Control
```python
import logging

# Control specific modules
logging.getLogger('asterix.agent').setLevel(logging.DEBUG)
logging.getLogger('asterix.core').setLevel(logging.WARNING)
```

---

## 📚 Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Provider (at least one required)
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key

# Vector Storage (required)
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# Optional
ASTERIX_STATE_DIR=./agent_states
ASTERIX_LOG_LEVEL=INFO
```

### YAML Configuration (Optional)

```yaml
# agent_config.yaml
agent_id: "my_agent"
max_heartbeat_steps: 10

# LLM Configuration
llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"
  temperature: 0.1
  max_tokens: 1000

# Memory Blocks
blocks:
  task:
    size: 1500
    priority: 1
    description: "Current task and progress"
  
  notes:
    size: 1000
    priority: 2
    description: "Important notes and reminders"

# Storage
storage:
  qdrant_url: "${QDRANT_URL}"
  qdrant_api_key: "${QDRANT_API_KEY}"
  state_backend: "json"
  state_dir: "./agent_states"

# Memory Management
memory:
  eviction_strategy: "summarize_and_archive"
  context_window_threshold: 0.85

# Embeddings
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
```

Load from YAML:
```python
agent = Agent.from_yaml("agent_config.yaml")
```

### Tool Configuration

Configure tool behavior when registering:
```python
from asterix.tools import Tool, ToolCategory, ParameterConstraint

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

---

## 🧠 Memory System

### Built-in Memory Tools

Agents have 5 built-in tools for managing their memory:

1. **`core_memory_append`** - Add content to a memory block
2. **`core_memory_replace`** - Replace content in a memory block
3. **`archival_memory_insert`** - Store information in Qdrant for long-term retrieval
4. **`archival_memory_search`** - Search archived memories semantically
5. **`conversation_search`** - Search conversation history

These tools are called automatically by the agent when needed.

### Memory Blocks

Configure memory blocks with custom sizes and priorities:

```python
blocks = {
    "task": BlockConfig(
        size=2000,          # Max tokens before eviction
        priority=1,         # Lower = evicted first
        description="Current task context"
    ),
    "user_prefs": BlockConfig(
        size=500,
        priority=5,         # High priority = rarely evicted
        description="User preferences and settings"
    )
}
```

### Automatic Memory Management

When a block exceeds its token limit:
1. Content is summarized by LLM
2. Full content archived in Qdrant
3. Block replaced with summary
4. Original retrievable via semantic search

---

## 🛠️ Custom Tools

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

### Tool Development

Creating custom tools with full features:
```python
from asterix.tools import Tool, ToolCategory, ParameterConstraint

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

---

## 💾 State Persistence

### Save & Load

```python
# Save agent state to disk
agent.save_state()  # Saves to ./agent_states/{agent_id}.json

# Load from disk
agent = Agent.load_state("agent_id")

# Custom state directory
agent = Agent(..., state_dir="./my_agents")
agent.save_state()
```

### State Backends

```python
# JSON (default)
agent = Agent(..., state_backend="json")

# SQLite (better for many agents)
agent = Agent(..., state_backend="sqlite", state_db="agents.db")

# Custom backend
from asterix.storage import StateBackend

class RedisBackend(StateBackend):
    def save(self, agent_id: str, state: dict): ...
    def load(self, agent_id: str) -> dict: ...

agent = Agent(..., state_backend=RedisBackend())
```

## ⚠️ Error Handling

Asterix provides detailed error messages with context and suggestions:

### Tool Errors
```python
from asterix.tools import ToolNotFoundError, ToolExecutionError, ToolValidationError

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
```python
# Parameter validation errors include helpful context
# "Tool 'create_user' parameter 'username' validation failed: 
#  username length must be >= 3, got 2
#  Provided value: ab"
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

## 📖 Examples

For complete working examples, see the [`examples/`](examples/) directory:

- [`basic_agent.py`](examples/basic_agent.py) - Simple conversation agent
- [`custom_tools.py`](examples/custom_tools.py) - Tool registration with validation
- [`persistent_agent.py`](examples/persistent_agent.py) - State save/load demonstration
- [`tool_documentation.py`](examples/tool_documentation.py) - Auto-documentation generation
- [`cli_agent.py`](examples/cli_agent.py) - Full-featured CLI agent with file operations

### CLI Agent with File Operations

```python
from asterix import Agent, BlockConfig
import os

agent = Agent(
    blocks={
        "current_task": BlockConfig(size=2000, priority=1),
        "file_context": BlockConfig(size=3000, priority=2)
    },
    model="groq/llama-3.3-70b-versatile"
)

@agent.tool(name="list_files")
def list_files(directory: str = ".") -> str:
    files = os.listdir(directory)
    return "\n".join(files)

@agent.tool(name="read_file")
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

# Use the agent
agent.chat("List all Python files and review main.py for potential issues")
```

### Multi-Agent System

```python
# Orchestrator agent
main_agent = Agent(
    agent_id="orchestrator",
    blocks={"plan": BlockConfig(size=1500)},
    model="groq/llama-3.3-70b-versatile"
)

# Specialized agents
code_reviewer = Agent(
    agent_id="reviewer",
    blocks={"code": BlockConfig(size=3000)},
    model="groq/llama-3.3-70b-versatile"
)

# Coordination
task = "Review auth.py for security issues"
plan = main_agent.chat(f"Break down: {task}")
review = code_reviewer.chat(f"Execute: {plan}")
summary = main_agent.chat(f"Summarize: {review}")
```

---

## 🔧 Advanced Usage

### Direct Memory Access

```python
# Get all memory blocks
memory = agent.get_memory()
print(memory["task"])

# Update memory manually
agent.update_memory("task", "New content")

# Search archival memory
results = agent.search_archival("user preferences", k=5)
for result in results:
    print(f"Score: {result.score}, Text: {result.summary}")
```

### Heartbeat Control (Advanced)

```python
# Manual heartbeat loop control
controller = agent.create_heartbeat_controller()

for step in controller.run("Complex multi-step task"):
    if step.needs_tool_execution:
        # Custom tool execution logic
        results = my_custom_executor(step.tool_calls)
        controller.submit_tool_results(results)
    
    elif step.is_complete:
        response = step.get_response()
        break
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=asterix --cov-report=html

# Run specific test
pytest tests/test_agent.py::test_memory_tools
```

---

## 📊 Project Status

**Current Version:** 0.1.0 (Alpha)

**Roadmap:**
- [x] Core agent implementation
- [x] Memory tools system
- [x] State persistence
- [x] Qdrant integration
- [ ] Enhanced tool registration
- [ ] Performance optimizations
- [ ] Extended documentation
- [ ] Tutorial series

## 📚 Tool Reference

### Built-in Memory Tools

Asterix provides 5 built-in tools for memory management:

| Tool | Category | Description |
|------|----------|-------------|
| `core_memory_append` | memory | Add content to a memory block |
| `core_memory_replace` | memory | Replace content in a memory block |
| `archival_memory_insert` | memory | Store information in long-term memory (Qdrant) |
| `archival_memory_search` | memory | Search long-term memory |
| `conversation_search` | memory | Search conversation history |

### Tool System Features

- **Automatic Schema Generation** - Type hints → OpenAI function schemas
- **Parameter Validation** - Min/max values, lengths, patterns, allowed values
- **Category Organization** - Group tools by purpose (memory, file_ops, web, etc.)
- **Retry Logic** - Automatic retries with exponential backoff
- **Error Recovery** - Smart error messages with hints and suggestions
- **Auto-Documentation** - Generate markdown/JSON/YAML docs from metadata
- **Tool Discovery** - Filter and search tools by name, category, capabilities

### Generate Documentation
```bash
# In Python
from asterix import Agent

agent = Agent()

# Generate tool reference
docs = agent._tool_registry.generate_registry_docs(format="markdown")
with open("TOOL_REFERENCE.md", "w") as f:
    f.write(docs)

# Export tool catalog
catalog = agent._tool_registry.export_tool_catalog("json")
with open("tool_catalog.json", "w") as f:
    f.write(catalog)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Groq](https://groq.com/) and [OpenAI](https://openai.com/)
- Vector storage by [Qdrant](https://qdrant.tech/)
- Inspired by [Letta](https://www.letta.com/)

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/adityasarade/Asterix/issues)
- **Discussions:** [GitHub Discussions](https://github.com/adityasarade/Asterix/discussions)
- **Documentation:** [Full Docs](https://github.com/adityasarade/Asterix#readme)

---

**So that everyone can build better agents without worrying about memory (Let's hope OpenAI doesn't make this library meaningless)**