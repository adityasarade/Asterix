# Asterix!

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
- **🛠️ Tool System** - Easy decorator pattern for custom capabilities
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

### Save & Load State

```python
# Save agent state
agent.save_state()

# Later session - load previous state
agent = Agent.load_state("agent_id")
agent.chat("What were we discussing?")  # Remembers everything!
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
agent:
  model: "groq/llama-3.3-70b-versatile"
  temperature: 0.1
  max_tokens: 1000
  max_heartbeat_steps: 10

blocks:
  task:
    size: 1500
    priority: 1
    description: "Current task and progress"
  
  notes:
    size: 1000
    priority: 2
    description: "Important notes and reminders"

storage:
  qdrant_url: "${QDRANT_URL}"
  qdrant_api_key: "${QDRANT_API_KEY}"
  state_backend: "json"
  state_dir: "./agent_states"
```

Load from YAML:
```python
agent = Agent.from_yaml("agent_config.yaml")
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

---

## 📖 Examples

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