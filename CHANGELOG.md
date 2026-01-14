# Changelog

All notable changes to the Asterix Agent Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-01-14

### Added
- **Gemini Provider Support** - Added Google Gemini as a new LLM provider with full tool calling support
- **Dynamic Model Configuration** - Models specified in config are now actually used instead of hardcoded defaults
- **New Models** - Added support for Qwen3-32B, Kimi K2-Instruct, Gemini 2.5/3.0 Flash/Pro
- **Gemini Health Check** - Added health monitoring for Gemini API

### Changed
- **Default Model** - Changed from `openai/gpt-5-mini` to `gemini/gemini-2.5-flash` (1M context window)
- **Updated Context Limits** - Added accurate context limits for all supported models

### Deprecated
- Mixtral models are no longer available on Groq (deprecated March 2025)

## [0.1.4] - 2025-11-14

- Fixed persistent agent example filename bug
- Added SQLite backend example
- Improved state persistence documentation

## [0.1.3] - 2025-11-13

### Fixed
- Added missing asterix/storage/__init__.py
- Fixed import: from asterix.storage import SQLiteStateBackend

## [0.1.2] - 2025-11-04

### 📝 Documentation Update

**Changes:**
- Updated README.md to reflect Beta status
- Removed "not ready for production" warning
- Package is stable and ready for community use

---

## [0.1.1] - 2025-11-04

### 📦 Maintenance Release

**Changes:**
- Updated package metadata for stable release
- Added MANIFEST.in for proper file distribution
- Enhanced PyPI classifiers (Development Status: Beta)
- Added comprehensive publishing documentation
- Improved package structure documentation

**Technical:**
- All files properly included in distribution
- Examples and tests included in package
- Documentation improvements

---

## [0.1.0] - 2025-11-04

### 🎉 Initial Stable Release

The first stable release of Asterix - a lightweight Python library for building stateful AI agents with editable memory blocks and persistent storage.

### ✨ Features

#### Core Agent System
- **Agent Class** - Main entry point for creating and managing AI agents
- **Memory Block System** - Configurable memory blocks with size limits and priorities
- **Heartbeat Loop** - Automatic multi-turn conversation handling with tool execution
- **Context Window Monitoring** - Tracks token usage and prevents context overflow
- **Multi-Model Support** - Works with Groq and OpenAI LLM providers

#### Memory Management
- **Block Eviction** - Automatic summarization when blocks exceed size limits
- **Context Extraction** - LLM-driven fact extraction from conversations
- **Archival Storage** - Long-term memory storage in Qdrant Cloud
- **Agent Isolation** - Separate memory spaces per agent with ID filtering
- **Smart Summarization** - Reduces block size (e.g., 2000→220 tokens) while preserving key information

#### Memory Tools (5 Built-in Tools)
- `core_memory_append` - Add content to memory blocks
- `core_memory_replace` - Replace content in memory blocks
- `archival_memory_insert` - Store information in long-term memory (Qdrant)
- `archival_memory_search` - Retrieve relevant memories from Qdrant
- `conversation_search` - Search through conversation history

#### Tool System
- **Decorator Pattern** - Easy tool registration with `@agent.tool()`
- **Automatic Schema Generation** - Type hints automatically converted to OpenAI function schemas
- **Parameter Validation** - Constraints for min/max values, lengths, patterns, and allowed values
- **Tool Categories** - Organize tools by purpose (memory, file_ops, web, custom, etc.)
- **Retry Logic** - Automatic retries with exponential backoff for transient failures
- **Error Recovery** - Smart error messages with suggestions and context
- **Auto-Documentation** - Generate markdown/JSON/YAML docs from tool metadata
- **Tool Discovery** - Filter and search tools by name, category, or capabilities

#### State Persistence
- **JSON Backend** - Simple file-based state storage (default)
- **SQLite Backend** - Database storage for multi-agent applications
- **Full State Preservation** - Saves conversation history, memory blocks, and configuration
- **Save/Load API** - `agent.save_state()` and `Agent.load_state(agent_id)`
- **Custom State Directory** - Configurable storage location

#### Configuration
- **YAML Configuration** - Load agent settings from YAML files
- **Environment Variables** - Support for `.env` files
- **Python Configuration** - Direct configuration via `Agent()` constructor
- **Flexible Defaults** - Sensible defaults that work out of the box

#### Integrations
- **Qdrant Cloud** - Vector database for semantic memory search
- **OpenAI API** - LLM and embeddings support
- **Groq API** - Fast inference with Llama models
- **Sentence Transformers** - Local embedding fallback option

### 📚 Documentation

- **Comprehensive README** - Installation, quick start, and feature overview
- **6 Example Scripts** - Demonstrating all major features:
  - `basic_chat.py` - Simple conversation agent
  - `custom_tools.py` - Tool registration with validation
  - `persistent_agent.py` - State save/load demonstration
  - `yaml_config.py` - YAML configuration loading
  - `cli_agent.py` - Full-featured CLI agent with file operations
  - `tool_documentation.py` - Auto-documentation generation
- **API Documentation** - Inline docstrings for all public methods
- **Type Hints** - Full type coverage for IDE support

### 🧪 Testing

- **Diagnostic Tests** - Core functionality verification
- **Memory Management Tests** - Block eviction and context extraction
- **Tool System Tests** - Parameter validation and retry logic
- **State Persistence Tests** - JSON and SQLite backend verification
- **Integration Tests** - End-to-end conversation flows

### 🔧 Technical Details

- **Python 3.10+** - Modern Python with type hints
- **No Server Required** - Pure Python library, no external servers
- **Minimal Dependencies** - Only essential packages required
- **Production Ready** - Error handling, logging, and state management
- **Framework Agnostic** - Works with any LLM provider

### 📦 Installation

```bash
pip install asterix-agent
```

### 🔗 Links

- **PyPI**: https://pypi.org/project/asterix-agent/
- **GitHub**: https://github.com/adityasarade/Asterix
- **Issues**: https://github.com/adityasarade/Asterix/issues
- **Documentation**: https://github.com/adityasarade/Asterix#readme

---

## [0.1.0a1] - 2025-10-15

### 🚧 Alpha Release

Initial alpha release for testing and feedback.

#### Added
- Core agent implementation with memory blocks
- Basic memory tools (append, replace, insert, search)
- Simple state persistence (JSON only)
- Groq and OpenAI integration
- Basic tool registration system

#### Known Limitations
- Limited documentation
- No parameter validation
- Basic error handling
- No retry logic

---

## Planned for Future Releases

### [0.2.0] - Planned
- **Enhanced Tool System** - Tool versioning and deprecation support
- **Advanced Memory Strategies** - Configurable eviction and archival strategies
- **Performance Optimizations** - Caching and batch processing
- **Extended Documentation** - Tutorial series and advanced usage guides
- **Monitoring & Observability** - Built-in metrics and tracing

### [0.3.0] - Planned
- **Multi-Agent Coordination** - Agent-to-agent communication
- **Streaming Responses** - Real-time response streaming
- **Plugin System** - Third-party tool packages
- **Web UI** - Optional web interface for agent management

---

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/adityasarade/Asterix) for guidelines.

## License

This project is licensed under the MIT License.