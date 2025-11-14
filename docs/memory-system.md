# Memory System

Asterix agents have a sophisticated memory system that allows them to remember, learn, and persist their knowledge across sessions. The memory system consists of memory blocks, archival storage, and conversation history.

## Table of Contents

- [Overview](#overview)
- [Memory Blocks](#memory-blocks)
- [Built-in Memory Tools](#built-in-memory-tools)
- [Automatic Memory Management](#automatic-memory-management)
- [Direct Memory Access](#direct-memory-access)
- [Archival Memory](#archival-memory)
- [Conversation Search](#conversation-search)

---

## Overview

The Asterix memory system has three layers:

1. **Memory Blocks** - Short-term, editable memory that agents can read and write
2. **Archival Memory** - Long-term storage in Qdrant with semantic search
3. **Conversation History** - Complete record of all interactions

```
┌─────────────────┐
│  Memory Blocks  │  ← Short-term, editable (task, notes, etc.)
└─────────────────┘
        ↓ (when full)
┌─────────────────┐
│ Archival Memory │  ← Long-term, searchable (Qdrant)
└─────────────────┘
        +
┌─────────────────┐
│ Conversation    │  ← Full interaction history
│    History      │
└─────────────────┘
```

---

## Memory Blocks

Memory blocks are editable sections of the agent's short-term memory. Each block has:

- **Name** - Identifier (e.g., "task", "notes", "user_prefs")
- **Size** - Maximum tokens before eviction
- **Priority** - Lower priority blocks are evicted first
- **Description** - Purpose of the block

### Configuring Memory Blocks

```python
from asterix import Agent, BlockConfig

agent = Agent(
    blocks={
        "task": BlockConfig(
            size=2000,          # Max tokens before eviction
            priority=1,         # Lower = evicted first
            description="Current task context"
        ),
        "user_prefs": BlockConfig(
            size=500,
            priority=5,         # High priority = rarely evicted
            description="User preferences and settings"
        ),
        "notes": BlockConfig(
            size=1000,
            priority=2,
            description="Important notes and reminders"
        )
    },
    model="openai/gpt-4o-mini"
)
```

### Memory Block Best Practices

- **Task blocks** - Use for current work context (priority: 1-2)
- **User preferences** - Keep user settings (priority: 4-5)
- **Notes** - Temporary information (priority: 2-3)
- **Context** - Relevant background info (priority: 2-3)

---

## Built-in Memory Tools

Agents have 5 built-in tools for managing their memory:

### 1. `core_memory_append`

Add content to a memory block.

```python
# Agent automatically calls this when needed
agent.chat("Remember that I prefer Python over JavaScript")

# The agent uses: core_memory_append(block="user_prefs", content="User prefers Python over JavaScript")
```

### 2. `core_memory_replace`

Replace content in a memory block.

```python
# Agent automatically calls this when needed
agent.chat("Actually, I prefer TypeScript now")

# The agent uses: core_memory_replace(
#     block="user_prefs",
#     old_content="User prefers Python",
#     new_content="User prefers TypeScript"
# )
```

### 3. `archival_memory_insert`

Store information in Qdrant for long-term retrieval.

```python
# Agent stores important information for later
# archival_memory_insert(content="Project X uses PostgreSQL database with 10M records")
```

### 4. `archival_memory_search`

Search archived memories semantically.

```python
# Agent searches when it needs to recall something
# archival_memory_search(query="database details", k=5)
```

### 5. `conversation_search`

Search conversation history.

```python
# Agent searches past conversations
# conversation_search(query="API key", k=3)
```

**Note:** These tools are called automatically by the agent. You don't need to invoke them manually.

---

## Automatic Memory Management

When a memory block exceeds its token limit, Asterix automatically:

1. **Summarizes** the content using the LLM
2. **Archives** the full content in Qdrant
3. **Replaces** the block with the summary
4. **Makes it searchable** via `archival_memory_search`

### Eviction Process

```python
# Block "task" has 2000 token limit
# Current content: 1950 tokens
# Agent tries to append 200 tokens → Exceeds limit!

# Asterix automatically:
# 1. Summarizes the 1950 token content → 500 tokens
# 2. Archives original 1950 tokens in Qdrant
# 3. Replaces block with 500 token summary
# 4. Appends new 200 tokens
# Result: Block now has 700 tokens
```

### Eviction Strategies

You can configure how memory is managed:

```python
from asterix import Agent, MemoryConfig

agent = Agent(
    ...,
    memory_config=MemoryConfig(
        eviction_strategy="summarize_and_archive",  # Default
        context_window_threshold=0.85  # Trigger at 85% full
    )
)
```

**Available strategies:**
- `summarize_and_archive` - Summarize and store in Qdrant (default)
- `archive_only` - Store in Qdrant without summarizing
- `discard` - Remove oldest content (not recommended)

---

## Direct Memory Access

You can manually access and update memory blocks:

### Get Memory Contents

```python
# Get all memory blocks
memory = agent.get_memory()
print(memory["task"])
print(memory["notes"])

# Access specific block
task_content = memory.get("task", "")
```

### Update Memory Manually

```python
# Update a memory block directly
agent.update_memory("task", "New task: Build user authentication system")

# Clear a memory block
agent.update_memory("notes", "")
```

### Search Archival Memory

```python
# Manually search archived memories
tool_result = agent._tool_registry.execute_tool(
    "archival_memory_search",
    query="user preferences",
    k=5
)
print(tool_result)
```

---

## Archival Memory

Archival memory uses Qdrant for long-term, semantic storage.

### Configuration

Set up Qdrant in your `.env`:

```bash
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
```

### How It Works

1. **Storage** - Content is embedded using OpenAI embeddings
2. **Indexing** - Vectors stored in Qdrant collection
3. **Retrieval** - Semantic search finds relevant memories
4. **Context** - Retrieved memories are added to agent context

### Semantic Search

Unlike keyword search, semantic search understands meaning:

```python
# Stored: "User prefers Python for data analysis"
# Search: "programming language preference"
# ✅ Finds the memory (semantically similar)

# Search: "favorite food"
# ❌ Doesn't find it (not semantically related)
```

---

## Conversation Search

Search through the agent's conversation history:

```python
# Agent automatically searches when needed
# conversation_search(query="API key location", k=3)

# Returns the 3 most relevant conversation turns
```

### Use Cases

- Recall previous instructions
- Find mentioned file paths
- Remember user preferences from earlier in conversation
- Locate specific information discussed before

---

## Memory System Architecture

```
┌──────────────────────────────────────┐
│           Agent Context              │
├──────────────────────────────────────┤
│  Memory Blocks (Short-term)          │
│  ┌────────┐ ┌────────┐ ┌──────────┐ │
│  │  task  │ │ notes  │ │user_prefs││ │
│  └────────┘ └────────┘ └──────────┘ │
│                 ↓                    │
│  When block is full: summarize       │
│                 ↓                    │
│  ┌────────────────────────────────┐ │
│  │   Archival Memory (Qdrant)     │ │
│  │   - Semantic search            │ │
│  │   - Long-term storage          │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │   Conversation History         │ │
│  │   - Full interaction record    │ │
│  │   - Keyword search             │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

## Best Practices

### Block Sizing

- **Small blocks (500-1000 tokens)** - Frequently changing info
- **Medium blocks (1000-2000 tokens)** - Current work context
- **Large blocks (2000-3000 tokens)** - Comprehensive background

### Priority Settings

- **Priority 5** - Critical info (user preferences, requirements)
- **Priority 3-4** - Important context
- **Priority 1-2** - Temporary working memory

### When to Use Each Layer

| Memory Type | Use For | Lifespan |
|-------------|---------|----------|
| **Blocks** | Current task, active context | Until evicted |
| **Archival** | Long-term facts, important info | Persistent |
| **Conversation** | Recent interactions | Session |

---

## See Also

- [Tool System](tool-system.md) - Memory tools documentation
- [Storage Backends](storage-backends.md) - Persisting memory across sessions
- [Configuration](configuration.md) - Memory configuration options
