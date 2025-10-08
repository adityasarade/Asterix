"""
Asterix - Stateful AI agents with editable memory blocks

A lightweight Python library for building AI agents that can remember,
learn, and persist their state across sessions.
"""

__version__ = "0.1.0"
__author__ = "Aditya Sarade"

from .agent import Agent, MemoryBlock
from .core.config import (
    AgentConfig,
    BlockConfig,
    MemoryConfig,
    StorageConfig,
    LLMConfig,
    EmbeddingConfig,
    ConfigurationManager,
    get_config_manager,
    create_default_blocks
)

__all__ = [
    # Main classes
    "Agent",
    "MemoryBlock",
    
    # Configuration
    "AgentConfig",
    "BlockConfig",
    "MemoryConfig",
    "StorageConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "ConfigurationManager",
    "get_config_manager",
    "create_default_blocks",
]