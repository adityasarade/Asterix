# Asterix

**Stateful AI agents with editable memory blocks and persistent storage.**

> **Note:** Asterix is in Beta (v0.1.4). Core features are stable and production-ready.
> Enhanced features and optimizations are in active development.

Asterix is a lightweight Python library for building AI agents that can remember, learn, and persist their state across sessions. No servers required - just `pip install` and start building.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Editable Memory Blocks** - Agents can read and write their own memory via built-in tools
- **Persistent Storage** - State saves across sessions (JSON/SQLite backends)
- **Semantic Search** - Qdrant Cloud integration for long-term memory retrieval
- **Enhanced Tool System** - Easy decorator pattern with parameter validation, retry logic, and categories
- **Auto-Documentation** - Tools automatically generate markdown/JSON documentation
- **Smart Error Handling** - Helpful error messages with suggestions and context
- **Multi-Model Support** - Works with Groq, OpenAI, and extensible to others
- **No Server Required** - Pure Python library, runs anywhere

---

## Quick Start

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
    model="openai/gpt-4o-mini"
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

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### Core Concepts
- **[Memory System](docs/memory-system.md)** - How agent memory works (blocks, archival, conversation search)
- **[Tool System](docs/tool-system.md)** - Creating and using tools (validation, categories, retry logic)
- **[Storage Backends](docs/storage-backends.md)** - State persistence (JSON, SQLite, custom backends)
- **[Configuration](docs/configuration.md)** - Environment variables, YAML, and Python configuration

### Guides
- **[Examples Guide](docs/examples-guide.md)** - Walkthrough of all examples with explanations
- **[API Reference](docs/api-reference.md)** - Complete API documentation

### Quick Links
- [Environment Setup](docs/configuration.md#environment-variables)
- [YAML Configuration Template](example_agent_config.yaml)
- [Built-in Memory Tools](docs/memory-system.md#built-in-memory-tools)
- [Custom Tool Development](docs/tool-system.md#advanced-tool-development)
- [JSON vs SQLite Backends](docs/storage-backends.md#choosing-a-backend)

---

## 📦 Examples

Complete working examples in [`examples/`](examples/):

```bash
# Clone and setup
git clone https://github.com/adityasarade/Asterix.git
cd Asterix && pip install -e .

# Run examples
python examples/basic_chat.py              # Simple conversation
python examples/custom_tools.py            # Tool registration
python examples/persistent_agent.py        # JSON persistence
python examples/persistent_agent_sqlite.py # SQLite persistence
python examples/cli_agent.py               # Full-featured CLI
```

See the **[Examples Guide](docs/examples-guide.md)** for detailed walkthroughs.

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest --cov=asterix --cov-report=html
```

---

## Project Status

**Current Version:** 0.1.4 (Beta)

**Roadmap:**
- [x] Core agent implementation
- [x] Memory tools system
- [x] State persistence (JSON & SQLite)
- [x] Qdrant integration
- [x] Enhanced tool system with validation
- [x] Auto-documentation
- [ ] Performance optimizations
- [ ] Advanced monitoring
- [ ] Streaming responses
- [ ] Multi-agent collaboration
- [ ] Additional backends (Redis, PostgreSQL)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [Groq](https://groq.com/) and [OpenAI](https://openai.com/)
- Vector storage by [Qdrant](https://qdrant.tech/)
- Inspired by [Letta](https://www.letta.com/)

---

## Support

- **Documentation:** [Full documentation](docs/) with guides and API reference
- **Issues:** [GitHub Issues](https://github.com/adityasarade/Asterix/issues) - Report bugs or request features
- **Examples:** [Examples directory](examples/) - Working code examples
- **Changelog:** [CHANGELOG.md](CHANGELOG.md) - Version history and updates

---

**So that everyone can build better agents without worrying about memory (Let's hope OpenAI doesn't make this library meaningless)**
