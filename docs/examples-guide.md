# Examples Guide

This guide walks through all the examples in the [`examples/`](../examples/) directory, explaining what each one demonstrates and how to use it.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Chat](#basic-chat)
- [Custom Tools](#custom-tools)
- [Persistent Agent (JSON)](#persistent-agent-json)
- [Persistent Agent (SQLite)](#persistent-agent-sqlite)
- [Tool Documentation](#tool-documentation)
- [CLI Agent](#cli-agent)
- [YAML Configuration](#yaml-configuration)
- [Advanced Examples](#advanced-examples)

---

## Getting Started

### Setup

```bash
# Clone the repository
git clone https://github.com/adityasarade/Asterix.git
cd Asterix

# Install in editable mode
pip install -e .

# Set up environment variables (create .env file)
# OPENAI_API_KEY=your-key
# QDRANT_URL=your-qdrant-url
# QDRANT_API_KEY=your-qdrant-key
```

### Running Examples

```bash
# Run any example
python examples/basic_chat.py
python examples/persistent_agent.py
python examples/custom_tools.py
```

---

## Basic Chat

**File:** [`examples/basic_chat.py`](../examples/basic_chat.py)

**Demonstrates:**
- Creating a basic agent
- Configuring memory blocks
- Simple conversation
- How agents use memory automatically

### Key Concepts

```python
from asterix import Agent, BlockConfig

# Create agent with custom memory blocks
agent = Agent(
    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    },
    model="openai/gpt-4o-mini"
)

# Chat with your agent
response = agent.chat("Hello! Remember that I prefer Python over JavaScript.")
print(response)

# Agent automatically updates its memory
# Memory persists within the session
```

**What happens:**
1. Agent receives your message
2. Automatically determines if memory should be updated
3. Uses `core_memory_append` to store "User prefers Python"
4. Responds with acknowledgment

**Run it:**
```bash
python examples/basic_chat.py
```

---

## Custom Tools

**File:** [`examples/custom_tools.py`](../examples/custom_tools.py)

**Demonstrates:**
- Registering custom tools
- Parameter validation
- Tool categories
- Retry logic
- Error handling

### Key Concepts

```python
from asterix import Agent
from asterix.tools.base import Tool, ParameterConstraint

agent = Agent(...)

# Simple tool
@agent.tool(name="read_file", description="Read a file from disk")
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

# Tool with validation
@agent.tool(
    name="create_user",
    description="Create a new user account",
    constraints={
        "username": ParameterConstraint(
            min_length=3,
            max_length=20,
            pattern=r'^[a-zA-Z0-9_]+$'
        ),
        "age": ParameterConstraint(min_value=13, max_value=120)
    }
)
def create_user(username: str, age: int) -> str:
    return f"Created user {username}, age {age}"

# Tool with retry logic
@agent.tool(
    name="fetch_data",
    description="Fetch data from API",
    retry_on_error=True,
    max_retries=3
)
def fetch_data(url: str) -> str:
    response = requests.get(url)
    return response.text

# Now your agent can use these tools
response = agent.chat("Create a user named 'john_doe' aged 25")
```

**Run it:**
```bash
python examples/custom_tools.py
```

---

## Persistent Agent (JSON)

**File:** [`examples/persistent_agent.py`](../examples/persistent_agent.py)

**Demonstrates:**
- Saving agent state to JSON
- Loading previous state
- Memory persistence across sessions
- State directory management

### Key Concepts

```python
from asterix import Agent, BlockConfig

# Session 1: Create and save
agent = Agent(
    agent_id="my_assistant",
    blocks={
        "user_prefs": BlockConfig(size=800, priority=5),
        "notes": BlockConfig(size=1200, priority=3)
    },
    model="openai/gpt-4o-mini"
)

agent.chat("Hi! I prefer Python over JavaScript.")
agent.save_state()  # Saves to ./agent_states/my_assistant.json

# Session 2: Load and continue
agent = Agent.load_state("my_assistant")
agent.chat("What language do I prefer?")  # Remembers!
```

**File structure:**
```
./agent_states/
└── my_assistant.json
```

**Run it:**
```bash
python examples/persistent_agent.py
```

---

## Persistent Agent (SQLite)

**File:** [`examples/persistent_agent_sqlite.py`](../examples/persistent_agent_sqlite.py)

**Demonstrates:**
- Using SQLite backend
- Managing multiple agents
- Querying agent metadata
- Production-ready persistence

### Key Concepts

```python
from asterix import Agent, BlockConfig, StorageConfig

# Create agent with SQLite backend
agent = Agent(
    agent_id="production_agent",
    blocks={"task": BlockConfig(size=2000, priority=1)},
    model="openai/gpt-4o-mini",
    storage=StorageConfig(
        state_backend="sqlite",
        state_db="./agent_states/agents.db"
    )
)

# Save to database
agent.save_state()

# Load from database
agent = Agent.load_state(
    "production_agent",
    state_backend="sqlite",
    state_db="./agent_states/agents.db"
)

# Query agent metadata
from asterix.storage import SQLiteStateBackend

backend = SQLiteStateBackend("./agent_states/agents.db")
agents = backend.list_agents()
info = backend.get_agent_info("production_agent")
```

**File structure:**
```
./agent_states/
└── agents.db  # Single database for all agents
```

**Run it:**
```bash
python examples/persistent_agent_sqlite.py
```

---

## Tool Documentation

**File:** [`examples/tool_documentation.py`](../examples/tool_documentation.py)

**Demonstrates:**
- Auto-generating tool documentation
- Exporting tool catalogs
- Creating quick references
- Different documentation formats

### Key Concepts

```python
from asterix import Agent

agent = Agent()

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

**Run it:**
```bash
python examples/tool_documentation.py
```

---

## CLI Agent

**File:** [`examples/cli_agent.py`](../examples/cli_agent.py)

**Demonstrates:**
- Full-featured CLI agent
- File operation tools
- Interactive conversation loop
- State persistence between runs

### Key Concepts

```python
from asterix import Agent, BlockConfig
import os

agent = Agent(
    blocks={
        "current_task": BlockConfig(size=2000, priority=1),
        "file_context": BlockConfig(size=3000, priority=2)
    },
    model="openai/gpt-4o-mini"
)

@agent.tool(name="list_files")
def list_files(directory: str = ".") -> str:
    files = os.listdir(directory)
    return "\n".join(files)

@agent.tool(name="read_file")
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = agent.chat(user_input)
    print(f"Agent: {response}")
```

**Run it:**
```bash
python examples/cli_agent.py
```

---

## YAML Configuration

**File:** [`examples/yaml_config.py`](../examples/yaml_config.py)

**Demonstrates:**
- Loading agent from YAML file
- Configuration management
- Environment variable substitution
- Complex agent setup

### Key Concepts

```python
from asterix import Agent

# Load agent from YAML
agent = Agent.from_yaml("agent_config.yaml")

# YAML file structure
"""
agent_id: "my_agent"
model: "openai/gpt-4o-mini"
blocks:
  task:
    size: 1500
    priority: 1
storage:
  qdrant_url: "${QDRANT_URL}"
  state_backend: "json"
"""
```

**See:** [example_agent_config.yaml](../example_agent_config.yaml) for full template.

**Run it:**
```bash
python examples/yaml_config.py
```

---

## Advanced Examples

### Multi-Agent System

```python
# Orchestrator agent
main_agent = Agent(
    agent_id="orchestrator",
    blocks={"plan": BlockConfig(size=1500)},
    model="openai/gpt-4o-mini"
)

# Specialized agents
code_reviewer = Agent(
    agent_id="reviewer",
    blocks={"code": BlockConfig(size=3000)},
    model="openai/gpt-4o-mini"
)

data_analyst = Agent(
    agent_id="analyst",
    blocks={"data": BlockConfig(size=2000)},
    model="openai/gpt-4o-mini"
)

# Coordination
task = "Review auth.py for security issues"
plan = main_agent.chat(f"Break down: {task}")
review = code_reviewer.chat(f"Execute: {plan}")
summary = main_agent.chat(f"Summarize: {review}")
```

### Custom Tool Class

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

### Direct Memory Access

```python
# Get all memory blocks
memory = agent.get_memory()
print(memory["task"])

# Update memory manually
agent.update_memory("task", "New content")

# Search archival memory
tool_result = agent._tool_registry.execute_tool(
    "archival_memory_search",
    query="user preferences",
    k=5
)
```

---

## Example Matrix

| Example | Memory | Tools | Persistence | Config | Complexity |
|---------|--------|-------|-------------|--------|------------|
| basic_chat.py | ✅ | ❌ | ❌ | Python | ⭐ |
| custom_tools.py | ✅ | ✅ | ❌ | Python | ⭐⭐ |
| persistent_agent.py | ✅ | ❌ | ✅ JSON | Python | ⭐⭐ |
| persistent_agent_sqlite.py | ✅ | ❌ | ✅ SQLite | Python | ⭐⭐ |
| tool_documentation.py | ✅ | ✅ | ❌ | Python | ⭐⭐ |
| cli_agent.py | ✅ | ✅ | ✅ | Python | ⭐⭐⭐ |
| yaml_config.py | ✅ | ❌ | ✅ | YAML | ⭐⭐⭐ |

---

## Next Steps

After running these examples:

1. **Customize memory blocks** - Adjust sizes and priorities for your use case
2. **Add custom tools** - Create tools specific to your application
3. **Configure persistence** - Choose JSON or SQLite based on your needs
4. **Explore advanced features** - Multi-agent systems, custom backends

---

## See Also

- [Tool System](tool-system.md) - Creating and using tools
- [Memory System](memory-system.md) - How memory works
- [Storage Backends](storage-backends.md) - Persistence options
- [Configuration](configuration.md) - All configuration options
