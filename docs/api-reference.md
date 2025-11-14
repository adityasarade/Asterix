# API Reference

Complete reference for Asterix classes, methods, and built-in tools.

## Table of Contents

- [Agent Class](#agent-class)
- [Configuration Classes](#configuration-classes)
- [Built-in Memory Tools](#built-in-memory-tools)
- [Tool Registry](#tool-registry)
- [Storage Backends](#storage-backends)
- [Exceptions](#exceptions)

---

## Agent Class

### `Agent`

Main agent class for creating stateful AI agents.

```python
from asterix import Agent, BlockConfig, StorageConfig, MemoryConfig

agent = Agent(
    agent_id="my_agent",
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    max_heartbeat_steps=10,
    blocks={"task": BlockConfig(size=1500, priority=1)},
    storage=StorageConfig(...),
    memory_config=MemoryConfig(...)
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | Generated | Unique identifier for the agent |
| `model` | `str` | Required | LLM model (e.g., "openai/gpt-4o-mini") |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int` | `1000` | Maximum tokens per completion |
| `max_heartbeat_steps` | `int` | `10` | Max tool call steps per turn |
| `blocks` | `dict[str, BlockConfig]` | `{}` | Memory block configuration |
| `storage` | `StorageConfig` | Default | Storage configuration |
| `memory_config` | `MemoryConfig` | Default | Memory management configuration |

#### Methods

##### `chat(message: str) -> str`

Send a message to the agent and get a response.

```python
response = agent.chat("Hello! Remember that I prefer Python.")
print(response)
```

**Parameters:**
- `message` (str): User message

**Returns:**
- `str`: Agent's response

---

##### `save_state(filepath: str = None) -> None`

Save agent state to persistent storage.

```python
# Save using configured backend
agent.save_state()

# Save to specific file (JSON only)
agent.save_state(filepath="./backups/agent.json")
```

**Parameters:**
- `filepath` (str, optional): Custom file path (JSON backend only)

---

##### `load_state(agent_id: str, **kwargs) -> Agent` (classmethod)

Load agent from persistent storage.

```python
# Load from default backend
agent = Agent.load_state("my_agent")

# Load from SQLite
agent = Agent.load_state(
    "my_agent",
    state_backend="sqlite",
    state_db="./agent_states/agents.db"
)
```

**Parameters:**
- `agent_id` (str): Agent identifier
- `**kwargs`: Backend-specific parameters

**Returns:**
- `Agent`: Loaded agent instance

---

##### `from_yaml(config_path: str) -> Agent` (classmethod)

Create agent from YAML configuration file.

```python
agent = Agent.from_yaml("agent_config.yaml")
```

**Parameters:**
- `config_path` (str): Path to YAML config file

**Returns:**
- `Agent`: Configured agent instance

---

##### `tool(name: str, description: str, **kwargs)` (decorator)

Register a custom tool with the agent.

```python
@agent.tool(name="read_file", description="Read a file")
def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `category` (ToolCategory, optional): Tool category
- `constraints` (dict, optional): Parameter constraints
- `examples` (list, optional): Usage examples
- `retry_on_error` (bool, optional): Enable retry logic
- `max_retries` (int, optional): Maximum retry attempts

---

##### `get_memory() -> dict[str, str]`

Get all memory block contents.

```python
memory = agent.get_memory()
print(memory["task"])
print(memory["notes"])
```

**Returns:**
- `dict[str, str]`: Dictionary of block names to contents

---

##### `update_memory(block: str, content: str) -> None`

Manually update a memory block.

```python
agent.update_memory("task", "New task content")
```

**Parameters:**
- `block` (str): Block name
- `content` (str): New content

---

## Configuration Classes

### `BlockConfig`

Configuration for a memory block.

```python
from asterix import BlockConfig

block = BlockConfig(
    size=1500,
    priority=2,
    description="Current task context",
    initial_value=""
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `int` | Required | Max tokens before eviction |
| `priority` | `int` | Required | Eviction priority (higher = kept longer) |
| `description` | `str` | `""` | Block description |
| `initial_value` | `str` | `""` | Initial content |

---

### `StorageConfig`

Configuration for storage backends.

```python
from asterix import StorageConfig

storage = StorageConfig(
    qdrant_url="https://cluster.cloud.qdrant.io:6333",
    qdrant_api_key="your-api-key",
    qdrant_collection_name="asterix_memory",
    vector_size=1536,
    state_backend="sqlite",
    state_dir="./agent_states",
    state_db="agents.db"
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qdrant_url` | `str` | From env | Qdrant Cloud URL |
| `qdrant_api_key` | `str` | From env | Qdrant API key |
| `qdrant_collection_name` | `str` | `"asterix_memory"` | Collection name |
| `vector_size` | `int` | `1536` | Embedding dimensions |
| `state_backend` | `str` | `"json"` | Backend type ("json" or "sqlite") |
| `state_dir` | `str` | `"./agent_states"` | State directory |
| `state_db` | `str` | `"agents.db"` | SQLite database filename |

---

### `MemoryConfig`

Configuration for memory management.

```python
from asterix import MemoryConfig

memory = MemoryConfig(
    eviction_strategy="summarize_and_archive",
    summary_token_limit=220,
    context_window_threshold=0.85,
    extraction_enabled=True,
    retrieval_k=6,
    score_threshold=0.7
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eviction_strategy` | `str` | `"summarize_and_archive"` | Eviction strategy |
| `summary_token_limit` | `int` | `220` | Target summary size |
| `context_window_threshold` | `float` | `0.85` | Extraction trigger |
| `extraction_enabled` | `bool` | `True` | Enable fact extraction |
| `retrieval_k` | `int` | `6` | Number of memories to retrieve |
| `score_threshold` | `float` | `0.7` | Minimum similarity score |

---

## Built-in Memory Tools

Asterix provides 5 built-in tools for memory management:

### 1. `core_memory_append`

Add content to a memory block.

**Parameters:**
- `block` (str): Block name (e.g., "task", "notes")
- `content` (str): Content to append

**Returns:**
- `str`: Confirmation message

**Example:**
```python
# Agent automatically calls this
# core_memory_append(block="task", content="User prefers Python")
```

---

### 2. `core_memory_replace`

Replace content in a memory block.

**Parameters:**
- `block` (str): Block name
- `old_content` (str): Content to replace
- `new_content` (str): New content

**Returns:**
- `str`: Confirmation message

**Example:**
```python
# Agent automatically calls this
# core_memory_replace(
#     block="task",
#     old_content="User prefers Python",
#     new_content="User prefers TypeScript"
# )
```

---

### 3. `archival_memory_insert`

Store information in Qdrant for long-term retrieval.

**Parameters:**
- `content` (str): Content to archive

**Returns:**
- `str`: Confirmation message with vector ID

**Example:**
```python
# Agent automatically calls this
# archival_memory_insert(content="Project X uses PostgreSQL with 10M records")
```

---

### 4. `archival_memory_search`

Search archived memories semantically.

**Parameters:**
- `query` (str): Search query
- `k` (int, optional): Number of results (default: 5)

**Returns:**
- `str`: Retrieved memories

**Example:**
```python
# Agent automatically calls this
# archival_memory_search(query="database details", k=5)
```

---

### 5. `conversation_search`

Search conversation history.

**Parameters:**
- `query` (str): Search query
- `k` (int, optional): Number of results (default: 3)

**Returns:**
- `str`: Relevant conversation turns

**Example:**
```python
# Agent automatically calls this
# conversation_search(query="API key location", k=3)
```

---

## Tool Registry

### `ToolRegistry`

Manages agent tools and provides discovery capabilities.

#### Methods

##### `execute_tool(name: str, **kwargs) -> str`

Execute a tool by name.

```python
result = agent._tool_registry.execute_tool(
    "archival_memory_search",
    query="user preferences",
    k=5
)
```

---

##### `get_by_category(category: ToolCategory) -> list[Tool]`

Get tools by category.

```python
from asterix.tools.base import ToolCategory

memory_tools = agent._tool_registry.get_by_category(ToolCategory.MEMORY)
```

---

##### `list_categories() -> dict[str, int]`

List all categories with tool counts.

```python
categories = agent._tool_registry.list_categories()
# {"memory": 5, "file_operations": 3, "custom": 2}
```

---

##### `generate_tool_docs(tool_name: str, format: str) -> str`

Generate documentation for a single tool.

```python
docs = agent._tool_registry.generate_tool_docs("read_file", format="markdown")
```

---

##### `generate_registry_docs(format: str, group_by_category: bool) -> str`

Generate documentation for all tools.

```python
full_docs = agent._tool_registry.generate_registry_docs(
    format="markdown",
    group_by_category=True
)
```

---

##### `export_tool_catalog(format: str) -> str`

Export tool catalog.

```python
catalog = agent._tool_registry.export_tool_catalog("json")
```

---

## Storage Backends

### `JSONStateBackend`

JSON file-based storage backend.

```python
from asterix.storage import JSONStateBackend

backend = JSONStateBackend(state_dir="./agent_states")
backend.save("agent_id", state_dict)
state = backend.load("agent_id")
```

---

### `SQLiteStateBackend`

SQLite database storage backend.

```python
from asterix.storage import SQLiteStateBackend

backend = SQLiteStateBackend("./agent_states/agents.db")
backend.save("agent_id", state_dict)
state = backend.load("agent_id")

# Query operations
agents = backend.list_agents()
info = backend.get_agent_info("agent_id")
all_info = backend.list_all_info(limit=10)
```

#### Additional Methods

##### `list_agents() -> list[str]`

List all agent IDs.

---

##### `get_agent_info(agent_id: str) -> dict`

Get agent metadata.

**Returns:**
```python
{
    'agent_id': 'agent1',
    'model': 'openai/gpt-4o-mini',
    'block_count': 3,
    'message_count': 42,
    'created_at': '2025-01-15T10:30:00',
    'last_updated': '2025-01-15T14:25:00'
}
```

---

##### `list_all_info(limit: int) -> list[dict]`

List all agents with metadata.

---

## Exceptions

### `ToolNotFoundError`

Raised when a tool is not found.

```python
from asterix.tools.base import ToolNotFoundError

try:
    agent._tool_registry.execute_tool("nonexistent_tool")
except ToolNotFoundError as e:
    print(e)  # Suggests similar tools
```

---

### `ToolExecutionError`

Raised when tool execution fails.

```python
from asterix.tools.base import ToolExecutionError

try:
    agent._tool_registry.execute_tool("read_file", filepath="missing.txt")
except ToolExecutionError as e:
    print(e)  # Includes error context
```

---

### `ToolValidationError`

Raised when parameter validation fails.

```python
from asterix.tools.base import ToolValidationError

try:
    agent._tool_registry.execute_tool("create_user", username="ab", age=5)
except ToolValidationError as e:
    print(e)  # Shows validation constraints
```

---

## Tool Categories

Available tool categories:

```python
from asterix.tools.base import ToolCategory

ToolCategory.MEMORY         # Memory management
ToolCategory.FILE_OPS       # File operations
ToolCategory.WEB            # Web/API operations
ToolCategory.DATA           # Data processing
ToolCategory.CUSTOM         # User-defined tools
```

---

## See Also

- [Tool System](tool-system.md) - Tool usage and development
- [Memory System](memory-system.md) - Memory management
- [Storage Backends](storage-backends.md) - Persistence
- [Configuration](configuration.md) - Configuration options
