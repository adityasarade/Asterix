# Configuration

Asterix agents can be configured using Python code, environment variables, or YAML files. Configuration options control the LLM, memory blocks, storage, embeddings, and more.

## Table of Contents

- [Configuration Methods](#configuration-methods)
- [Environment Variables](#environment-variables)
- [YAML Configuration](#yaml-configuration)
- [Python Configuration](#python-configuration)
- [LLM Configuration](#llm-configuration)
- [Memory Block Configuration](#memory-block-configuration)
- [Storage Configuration](#storage-configuration)
- [Embedding Configuration](#embedding-configuration)
- [Logging Configuration](#logging-configuration)

---

## Configuration Methods

Asterix supports three configuration methods with the following priority order:

**Priority:** Python overrides > Environment variables > YAML > Defaults

```python
# Method 1: Python (highest priority)
agent = Agent(model="gemini/gemini-2.5-flash", temperature=0.7)

# Method 2: Environment variables
# AGENT_MODEL=gemini/gemini-2.5-flash
# AGENT_TEMPERATURE=0.7

# Method 3: YAML file
agent = Agent.from_yaml("agent_config.yaml")
```

---

## Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Provider (at least one required)
GEMINI_API_KEY=your-gemini-api-key
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key

# Vector Storage (required for archival memory)
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=asterix_memory  # Optional

# Optional Settings
ASTERIX_STATE_DIR=./agent_states
ASTERIX_LOG_LEVEL=INFO
AGENT_MODEL=gemini/gemini-2.5-flash
AGENT_TEMPERATURE=0.1
```

### Available Environment Variables

#### LLM Settings
- `AGENT_MODEL` - Model to use (e.g., "gemini/gemini-2.5-flash")
- `AGENT_TEMPERATURE` - Sampling temperature (0.0-2.0)
- `AGENT_MAX_TOKENS` - Max tokens per completion
- `LLM_TIMEOUT` - Request timeout in seconds
- `GEMINI_API_KEY` - Google Gemini API key
- `GROQ_API_KEY` - Groq API key
- `OPENAI_API_KEY` - OpenAI API key

#### Storage Settings
- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant API key
- `QDRANT_COLLECTION_NAME` - Collection name (default: "asterix_memory")
- `ASTERIX_STATE_DIR` - Directory for agent state files
- `ASTERIX_STATE_BACKEND` - Backend type ("json" or "sqlite")
- `ASTERIX_STATE_DB` - SQLite database filename

#### Embedding Settings
- `EMBED_PROVIDER` - Embedding provider ("openai" or "sentence-transformers")
- `OPENAI_API_KEY` - Also used for embeddings

#### Agent Settings
- `AGENT_ID` - Agent identifier
- `AGENT_MAX_HEARTBEAT_STEPS` - Max tool call steps (default: 10)
- `ASTERIX_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

---

## YAML Configuration

For complex configurations, use a YAML file:

```yaml
# agent_config.yaml
agent_id: "my_agent"
max_heartbeat_steps: 10

# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1000
  timeout: 30

# Memory Blocks
blocks:
  persona:
    size: 1000
    priority: 10
    description: "Agent personality and behavior"
    initial_value: "I am a helpful AI assistant."

  user:
    size: 1000
    priority: 5
    description: "User information"

  task:
    size: 1500
    priority: 2
    description: "Current task and context"

  notes:
    size: 800
    priority: 3
    description: "Important notes"

# Storage
storage:
  qdrant_url: "${QDRANT_URL}"
  qdrant_api_key: "${QDRANT_API_KEY}"
  qdrant_collection_name: "asterix_memory"
  vector_size: 1536
  qdrant_timeout: 30
  auto_create_collection: true

  state_backend: "json"
  state_dir: "./agent_states"

# Memory Management
memory:
  eviction_strategy: "summarize_and_archive"
  summary_token_limit: 220
  context_window_threshold: 0.85
  extraction_enabled: true
  retrieval_k: 6
  score_threshold: 0.7

# Embeddings
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
  batch_size: 100
```

### Load from YAML

```python
from asterix import Agent

agent = Agent.from_yaml("agent_config.yaml")
```

### Environment Variable Substitution

YAML files support environment variable substitution using `${VAR_NAME}` syntax:

```yaml
storage:
  qdrant_url: "${QDRANT_URL}"
  qdrant_api_key: "${QDRANT_API_KEY}"
```

**See:** [example_agent_config.yaml](../example_agent_config.yaml) for a complete template with all options.

---

## Python Configuration

Configure agents directly in Python:

```python
from asterix import Agent, BlockConfig, StorageConfig, MemoryConfig

agent = Agent(
    agent_id="my_agent",
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000,

    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    },

    storage=StorageConfig(
        state_backend="sqlite",
        state_db="./agent_states/agents.db",
        qdrant_url="https://your-cluster.cloud.qdrant.io:6333",
        qdrant_api_key="your-api-key"
    ),

    memory_config=MemoryConfig(
        eviction_strategy="summarize_and_archive",
        context_window_threshold=0.85
    )
)
```

---

## LLM Configuration

### Supported Providers

Asterix supports multiple LLM providers:

- **Gemini** - Google's latest models with massive context windows (recommended)
- **Groq** - Fast inference with Llama, Qwen, Kimi K2
- **OpenAI** - GPT-4, GPT-4o, etc.

### Model Configuration

```python
# Gemini models (recommended - default provider)
agent = Agent(model="gemini/gemini-2.5-flash")  # Default, 1M context
agent = Agent(model="gemini/gemini-2.5-pro")    # Most capable
agent = Agent(model="gemini/gemini-3-pro")      # Latest, 1M context
agent = Agent(model="gemini/gemini-3-flash")    # Fast, 200K context

# Groq models (fast inference)
agent = Agent(model="groq/llama-3.3-70b-versatile")
agent = Agent(model="groq/qwen3-32b")           # 131K context
agent = Agent(model="groq/kimi-k2-instruct")    # 262K context
agent = Agent(model="groq/llama-3.1-8b-instant")

# OpenAI models
agent = Agent(model="openai/gpt-4o")
agent = Agent(model="openai/gpt-4o-mini")
agent = Agent(model="openai/gpt-4-turbo")
agent = Agent(model="openai/gpt-5-mini")
```

> **Note:** Mixtral models have been deprecated by Groq as of March 2025.

### Temperature and Tokens

```python
agent = Agent(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,     # 0.0 = deterministic, 2.0 = very creative
    max_tokens=1000,     # Maximum tokens per response
    timeout=30           # Request timeout in seconds
)
```

### YAML Format

```yaml
llm:
  provider: "gemini"
  model: "gemini-2.5-flash"
  temperature: 0.1
  max_tokens: 1000
  timeout: 30

# Legacy format also supported:
# model: "gemini/gemini-2.5-flash"
```

---

## Memory Block Configuration

Memory blocks are editable sections of agent memory. Configure size, priority, and initial content.

### Block Priority Guidelines

- **Priority 10** - Critical (persona, core identity) - never evicted
- **Priority 5-9** - Important (user preferences, key context) - rarely evicted
- **Priority 2-4** - Normal (task context, notes) - evicted when needed
- **Priority 1** - Low (temporary data) - evicted first

### Python Configuration

```python
from asterix import Agent, BlockConfig

agent = Agent(
    blocks={
        "persona": BlockConfig(
            size=1000,
            priority=10,
            description="Agent personality",
            initial_value="I am a helpful assistant."
        ),
        "user": BlockConfig(
            size=1000,
            priority=5,
            description="User information"
        ),
        "task": BlockConfig(
            size=1500,
            priority=2,
            description="Current task"
        )
    }
)
```

### YAML Configuration

```yaml
blocks:
  persona:
    size: 1000
    priority: 10
    description: "Agent personality and behavior"
    initial_value: "I am a helpful AI assistant."

  user:
    size: 1000
    priority: 5
    description: "Information about the user"

  task:
    size: 1500
    priority: 2
    description: "Current task and context"
```

---

## Storage Configuration

Configure Qdrant for archival memory and state persistence backends.

### Python Configuration

```python
from asterix import Agent, StorageConfig

agent = Agent(
    storage=StorageConfig(
        # Qdrant settings
        qdrant_url="https://your-cluster.cloud.qdrant.io:6333",
        qdrant_api_key="your-api-key",
        qdrant_collection_name="asterix_memory",
        vector_size=1536,
        auto_create_collection=True,

        # State persistence
        state_backend="sqlite",  # or "json"
        state_dir="./agent_states",
        state_db="agents.db"
    )
)
```

### YAML Configuration

```yaml
storage:
  # Qdrant Cloud
  qdrant_url: "${QDRANT_URL}"
  qdrant_api_key: "${QDRANT_API_KEY}"
  qdrant_collection_name: "asterix_memory"
  vector_size: 1536
  qdrant_timeout: 30
  auto_create_collection: true

  # State persistence
  state_backend: "json"
  state_dir: "./agent_states"
  state_db: "agents.db"
```

### State Backend Options

- **`json`** - Simple file-based storage, one file per agent
- **`sqlite`** - Database storage, better for multiple agents

**See:** [Storage Backends](storage-backends.md) for detailed comparison.

---

## Embedding Configuration

Configure embedding models for semantic search in Qdrant.

### Python Configuration

```python
from asterix import Agent, EmbeddingConfig

agent = Agent(
    embedding_config=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536
    )
)
```

### YAML Configuration

```yaml
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
  batch_size: 100
```

### Embedding Providers

**OpenAI** (requires API key):
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
```

**Sentence Transformers** (local, no API key):
```yaml
embedding:
  provider: "sentence-transformers"
  model: "all-MiniLM-L6-v2"
  dimensions: 384
```

**Important:** `embedding.dimensions` must match `storage.vector_size`.

---

## Logging Configuration

Asterix uses Python's standard logging module.

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

### Environment Variable

Set log level via environment:

```bash
ASTERIX_LOG_LEVEL=DEBUG
```

---

## Memory Management Configuration

Configure how agent memory is managed and archived.

### Python Configuration

```python
from asterix import Agent, MemoryConfig

agent = Agent(
    memory_config=MemoryConfig(
        eviction_strategy="summarize_and_archive",
        summary_token_limit=220,
        context_window_threshold=0.85,
        extraction_enabled=True,
        retrieval_k=6,
        score_threshold=0.7
    )
)
```

### YAML Configuration

```yaml
memory:
  eviction_strategy: "summarize_and_archive"
  summary_token_limit: 220
  context_window_threshold: 0.85
  extraction_enabled: true
  retrieval_k: 6
  score_threshold: 0.7
```

### Configuration Options

- **`eviction_strategy`** - How to handle full memory blocks
  - `"summarize_and_archive"` - Summarize and store in Qdrant (default)
  - `"truncate"` - Simply truncate (not recommended)

- **`summary_token_limit`** - Target size for summarized blocks (default: 220)

- **`context_window_threshold`** - Trigger extraction at % full (default: 0.85)

- **`extraction_enabled`** - Enable automatic fact extraction (default: true)

- **`retrieval_k`** - Number of memories to retrieve from Qdrant (default: 6)

- **`score_threshold`** - Minimum similarity score for retrieval (default: 0.7)

---

## Complete Configuration Example

**Python:**
```python
from asterix import Agent, BlockConfig, StorageConfig, MemoryConfig

agent = Agent(
    agent_id="production_agent",
    model="gemini/gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1000,
    max_heartbeat_steps=10,
    system_prompt="You are a production assistant. Be precise and concise.",

    blocks={
        "persona": BlockConfig(size=1000, priority=10),
        "user": BlockConfig(size=1000, priority=5),
        "task": BlockConfig(size=1500, priority=2),
        "notes": BlockConfig(size=800, priority=3)
    },

    storage=StorageConfig(
        qdrant_url="https://cluster.cloud.qdrant.io:6333",
        qdrant_api_key="your-api-key",
        state_backend="sqlite",
        state_db="./agent_states/agents.db"
    ),

    memory_config=MemoryConfig(
        eviction_strategy="summarize_and_archive",
        context_window_threshold=0.85
    ),

    on_before_tool_call=lambda name, args: True,  # Approve all tool calls
    on_after_tool_call=lambda name, args, result: print(f"Tool {name} completed"),
    on_step=lambda step_num, info: print(f"Step {step_num}: {info}")
)
```

**YAML:** See [example_agent_config.yaml](../example_agent_config.yaml)

---

## See Also

- [Storage Backends](storage-backends.md) - Persistence options
- [Memory System](memory-system.md) - How memory works
- [Examples Guide](examples-guide.md) - Configuration examples
