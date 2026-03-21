# Storage Backends

Asterix supports persistent agent state across sessions using multiple storage backends. This allows your agents to remember conversations, memory blocks, and context even after restarting your application.

## Table of Contents

- [Overview](#overview)
- [JSON Backend (Default)](#json-backend-default)
- [SQLite Backend](#sqlite-backend)
- [Choosing a Backend](#choosing-a-backend)
- [Custom Backend](#custom-backend)
- [Save and Load Operations](#save-and-load-operations)

---

## Overview

Asterix persists the following agent state:

- **Memory blocks** - All configured memory blocks and their contents
- **Conversation history** - Complete message history
- **Agent configuration** - Model, settings, and metadata
- **Tool registry** - Custom tools and their configurations

Two built-in backends are available:
- **JSON** - Simple, human-readable files (default)
- **SQLite** - Database storage for multiple agents

---

## JSON Backend (Default)

Perfect for single agents, prototyping, and human-readable storage.

### Basic Usage

```python
from asterix import Agent, BlockConfig

# Create agent (JSON backend is default)
agent = Agent(
    agent_id="my_assistant",
    blocks={
        "user_prefs": BlockConfig(size=800, priority=5),
        "notes": BlockConfig(size=1200, priority=3)
    },
    model="gemini/gemini-2.5-flash"
)

# Chat with agent
agent.chat("Hi! I prefer Python over JavaScript.")

# Save state to ./agent_states/my_assistant.json
agent.save_state()

# Later session - load previous state
agent = Agent.load_state("my_assistant")
agent.chat("What language do I prefer?")  # Remembers everything!
```

### Custom Directory

```python
from asterix import Agent, StorageConfig

agent = Agent(
    agent_id="my_assistant",
    storage=StorageConfig(
        state_backend="json",
        state_dir="./custom_states"
    )
)

# Saves to: ./custom_states/my_assistant.json
agent.save_state()
```

### JSON File Structure

```json
{
  "agent_id": "my_assistant",
  "model": "gemini/gemini-2.5-flash",
  "blocks": {
    "task": {
      "content": "Current task content...",
      "size": 1500,
      "priority": 1
    }
  },
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ],
  "metadata": {
    "created_at": "2025-01-15T10:30:00",
    "last_updated": "2025-01-15T14:25:00"
  }
}
```

### Advantages

- ✅ Human-readable
- ✅ Easy to inspect and debug
- ✅ Simple file-based storage
- ✅ Git-friendly (can version control)
- ✅ No database setup required

### Limitations

- ❌ No atomic updates
- ❌ No querying capabilities
- ❌ One file per agent (can clutter directories)
- ❌ Not ideal for 10+ agents

---

## SQLite Backend

Better for production, multiple agents, and querying capabilities.

### Basic Usage

```python
from asterix import Agent, BlockConfig, StorageConfig

# Create agent with SQLite backend
agent = Agent(
    agent_id="production_agent",
    blocks={"task": BlockConfig(size=2000, priority=1)},
    model="gemini/gemini-2.5-flash",
    storage=StorageConfig(
        state_backend="sqlite",
        state_db="./agent_states/agents.db"
    )
)

# Save to SQLite database
agent.save_state()

# Load from SQLite
agent = Agent.load_state(
    "production_agent",
    state_backend="sqlite",
    state_db="./agent_states/agents.db"
)
```

### Advanced Features

Query agent metadata without loading full state:

```python
from asterix.storage import SQLiteStateBackend

backend = SQLiteStateBackend("./agent_states/agents.db")

# List all agents
agents = backend.list_agents()
print(agents)  # ['agent1', 'agent2', 'agent3']

# Get agent metadata
info = backend.get_agent_info("agent1")
print(info)

# Output:
# {
#   'agent_id': 'agent1',
#   'model': 'openai/gpt-4o-mini',
#   'block_count': 3,
#   'message_count': 42,
#   'created_at': '2025-01-15T10:30:00',
#   'last_updated': '2025-01-15T14:25:00'
# }

# List all agents with metadata
all_agents = backend.list_all_info(limit=10)
for agent in all_agents:
    print(f"{agent['agent_id']}: {agent['message_count']} messages")
```

### Managing Multiple Agents

```python
from asterix import Agent, StorageConfig

# Shared database for all agents
db_path = "./agent_states/agents.db"

# Create multiple agents
agent1 = Agent(
    agent_id="customer_support",
    storage=StorageConfig(state_backend="sqlite", state_db=db_path)
)

agent2 = Agent(
    agent_id="code_reviewer",
    storage=StorageConfig(state_backend="sqlite", state_db=db_path)
)

agent3 = Agent(
    agent_id="data_analyst",
    storage=StorageConfig(state_backend="sqlite", state_db=db_path)
)

# All agents share one database file
agent1.save_state()
agent2.save_state()
agent3.save_state()
```

### Database Schema

SQLite backend creates the following tables:

```sql
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);

CREATE INDEX idx_last_updated ON agents(last_updated);
```

### Advantages

- ✅ Atomic updates (transaction-safe)
- ✅ Query capabilities
- ✅ Single database file for all agents
- ✅ Better for 10+ agents
- ✅ Metadata queries without full load
- ✅ Production-ready

### Limitations

- ❌ Not human-readable
- ❌ Requires SQLite (usually available)
- ❌ Slightly more complex setup

---

## Choosing a Backend

| Feature | JSON | SQLite |
|---------|------|--------|
| **Best for** | 1-10 agents, prototyping | 10+ agents, production |
| **Performance** | Fast for single agent | Fast for many agents |
| **Querying** | ❌ | ✅ |
| **Human-readable** | ✅ | ❌ |
| **Atomic updates** | ❌ | ✅ |
| **File structure** | One file per agent | Single database file |
| **Setup complexity** | None | Minimal |
| **Git-friendly** | ✅ | ❌ |

### Decision Guide

**Choose JSON if:**
- You have 1-10 agents
- You're prototyping
- You want to inspect state manually
- You want version control on agent state

**Choose SQLite if:**
- You have 10+ agents
- You're in production
- You need to query agent metadata
- You want atomic updates
- You need multi-agent management

---

## Custom Backend

Implement your own storage backend by following this interface:

```python
class CustomBackend:
    """Custom storage backend interface"""

    def save(self, agent_id: str, state_dict: dict) -> None:
        """Save agent state"""
        # Your implementation here
        pass

    def load(self, agent_id: str) -> dict:
        """Load agent state"""
        # Your implementation here
        pass

    def exists(self, agent_id: str) -> bool:
        """Check if agent exists"""
        # Your implementation here
        pass

    def delete(self, agent_id: str) -> None:
        """Delete agent state"""
        # Your implementation here
        pass

# Use custom backend
agent = Agent(
    agent_id="my_agent",
    storage=StorageConfig(state_backend=CustomBackend())
)
```

### Redis Backend Example

```python
import redis
import json

class RedisBackend:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)

    def save(self, agent_id: str, state_dict: dict) -> None:
        key = f"agent:{agent_id}"
        self.redis.set(key, json.dumps(state_dict))

    def load(self, agent_id: str) -> dict:
        key = f"agent:{agent_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def exists(self, agent_id: str) -> bool:
        key = f"agent:{agent_id}"
        return self.redis.exists(key)

    def delete(self, agent_id: str) -> None:
        key = f"agent:{agent_id}"
        self.redis.delete(key)

# Use Redis backend
agent = Agent(
    agent_id="redis_agent",
    storage=StorageConfig(state_backend=RedisBackend())
)
```

### PostgreSQL Backend Example

```python
import psycopg2
import json

class PostgreSQLBackend:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    state_json JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            self.conn.commit()

    def save(self, agent_id: str, state_dict: dict) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agents (agent_id, state_json)
                VALUES (%s, %s)
                ON CONFLICT (agent_id)
                DO UPDATE SET state_json = %s, updated_at = NOW()
            """, (agent_id, json.dumps(state_dict), json.dumps(state_dict)))
            self.conn.commit()

    def load(self, agent_id: str) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT state_json FROM agents WHERE agent_id = %s", (agent_id,))
            result = cur.fetchone()
            return result[0] if result else None

    def exists(self, agent_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM agents WHERE agent_id = %s", (agent_id,))
            return cur.fetchone() is not None

    def delete(self, agent_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM agents WHERE agent_id = %s", (agent_id,))
            self.conn.commit()
```

---

## Save and Load Operations

### Saving State

```python
# Automatic save (recommended)
agent.save_state()  # Uses configured backend

# With explicit path (JSON only)
agent.save_state(filepath="./backups/agent_backup.json")
```

### Loading State

```python
# Load from default backend
agent = Agent.load_state("agent_id")

# Load from specific backend
agent = Agent.load_state(
    "agent_id",
    state_backend="sqlite",
    state_db="./agent_states/agents.db"
)

# Load from JSON file
agent = Agent.load_state("agent_id", state_backend="json", state_dir="./custom_states")
```

### Checking if State Exists

```python
from asterix import Agent

# Check if agent state exists
if Agent.state_exists("my_agent", state_backend="json"):
    agent = Agent.load_state("my_agent")
else:
    agent = Agent(agent_id="my_agent")
```

### Deleting State

```python
# JSON backend - delete file
import os
os.remove("./agent_states/my_agent.json")

# SQLite backend - delete from database
from asterix.storage import SQLiteStateBackend
backend = SQLiteStateBackend("./agent_states/agents.db")
backend.delete("my_agent")
```

---

## Environment Configuration

Configure storage via environment variables:

```bash
# .env file
ASTERIX_STATE_DIR=./agent_states
ASTERIX_STATE_BACKEND=sqlite  # or "json"
ASTERIX_STATE_DB=./agent_states/agents.db  # for SQLite
```

```python
from asterix import Agent
import os

# Uses environment variables
agent = Agent(agent_id="my_agent")
agent.save_state()  # Uses ASTERIX_STATE_DIR and ASTERIX_STATE_BACKEND
```

---

## See Also

- [Configuration](configuration.md) - Environment and YAML configuration
- [Memory System](memory-system.md) - What gets persisted
- [Examples Guide](examples-guide.md) - Complete examples with persistence
