"""
Asterix Configuration System

Provides configuration classes and management for the Asterix library.
Supports both YAML configuration files and direct Python configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Core Configuration Dataclasses
# ============================================================================

@dataclass
class BlockConfig:
    """
    Configuration for a memory block.
    
    Memory blocks are editable storage areas that the agent can read and write.
    Each block has a token limit and priority for eviction.
    
    Args:
        size: Maximum tokens before eviction is triggered
        priority: Eviction priority (lower = evicted first, higher = kept longer)
        description: Human-readable description of the block's purpose
        initial_value: Initial content for the block (optional)
    
    Example:
        >>> code_block = BlockConfig(
        ...     size=2000,
        ...     priority=1,
        ...     description="Code being reviewed or edited"
        ... )
    """
    size: int
    priority: int = 1
    description: str = ""
    initial_value: str = ""
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.size <= 0:
            raise ValueError("Block size must be positive")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")


@dataclass
class MemoryConfig:
    """
    Configuration for memory management behavior.
    
    Controls how the agent manages its memory blocks, including eviction,
    summarization, and archival strategies.
    
    Args:
        eviction_strategy: Strategy for handling full blocks ("summarize_and_archive", "truncate")
        summary_token_limit: Maximum tokens for block summaries
        context_window_threshold: Trigger memory extraction at this % of context window
        extraction_enabled: Whether to automatically extract memories
        retrieval_k: Default number of memories to retrieve from archival
        score_threshold: Minimum similarity score for archival retrieval
    """
    eviction_strategy: str = "summarize_and_archive"
    summary_token_limit: int = 220
    context_window_threshold: float = 0.85
    extraction_enabled: bool = True
    retrieval_k: int = 6
    score_threshold: float = 0.7
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.eviction_strategy not in ["summarize_and_archive", "truncate"]:
            raise ValueError(f"Invalid eviction strategy: {self.eviction_strategy}")
        if not 0.0 < self.context_window_threshold <= 1.0:
            raise ValueError("Context window threshold must be between 0 and 1")
        if self.summary_token_limit <= 0:
            raise ValueError("Summary token limit must be positive")


@dataclass
class StorageConfig:
    """
    Configuration for storage backends (Qdrant and state persistence).
    
    Args:
        qdrant_url: Qdrant Cloud URL
        qdrant_api_key: Qdrant API key
        qdrant_collection_name: Collection name for this agent's memories
        vector_size: Embedding dimension (1536 for OpenAI, 384 for sentence-transformers)
        qdrant_timeout: Timeout for Qdrant operations (seconds)
        auto_create_collection: Whether to auto-create collection if missing
        
        state_backend: State persistence backend ("json", "sqlite", or custom)
        state_dir: Directory for state files (for json backend)
        state_db: Database path (for sqlite backend)
    """
    # Qdrant configuration
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "asterix_memory"
    vector_size: int = 1536
    qdrant_timeout: int = 30
    auto_create_collection: bool = True
    
    # State persistence configuration
    state_backend: str = "json"
    state_dir: str = "./agent_states"
    state_db: str = "agents.db"
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.state_backend not in ["json", "sqlite"]:
            # Allow custom backends without validation
            if not hasattr(self.state_backend, 'save'):
                logger.warning(f"Custom state backend should implement 'save' and 'load' methods")


@dataclass
class LLMConfig:
    """
    Configuration for LLM provider.
    
    Args:
        provider: LLM provider name ("groq", "openai")
        model: Model identifier
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens for completion
        timeout: Request timeout (seconds)
        api_key: API key for the provider (optional, can use env var)
    """
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.provider not in ["groq", "openai"]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding provider.
    
    Args:
        provider: Embedding provider ("openai", "sentence_transformers")
        model: Model identifier
        dimensions: Embedding dimensions
        batch_size: Batch size for processing
        api_key: API key (optional, can use env var)
    """
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.provider not in ["openai", "sentence_transformers"]:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be positive")


@dataclass
class AgentConfig:
    """
    Main configuration for an Asterix.
    
    This is the primary configuration class that bundles all settings for an agent.
    
    Args:
        agent_id: Unique identifier for this agent
        blocks: Dictionary mapping block names to BlockConfig objects
        
        model: LLM model string (format: "provider/model-name")
        temperature: LLM temperature
        max_tokens: Maximum tokens for LLM responses
        max_heartbeat_steps: Maximum tool execution loop iterations
        
        llm: Full LLM configuration (optional, overrides model/temperature/max_tokens)
        embedding: Embedding configuration (optional, uses defaults if not provided)
        memory: Memory management configuration (optional, uses defaults)
        storage: Storage configuration (optional, must provide Qdrant details)
        
    Example:
        >>> config = AgentConfig(
        ...     agent_id="my_agent",
        ...     blocks={
        ...         "task": BlockConfig(size=1500, priority=1),
        ...         "notes": BlockConfig(size=1000, priority=2)
        ...     },
        ...     model="groq/llama-3.3-70b-versatile",
        ...     storage=StorageConfig(
        ...         qdrant_url="https://...",
        ...         qdrant_api_key="..."
        ...     )
        ... )
    """
    # Agent identity
    agent_id: str
    blocks: Dict[str, BlockConfig] = field(default_factory=dict)
    
    # LLM settings (simple)
    model: str = "groq/llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1000
    max_heartbeat_steps: int = 10
    
    # Full configurations (optional, for advanced use)
    llm: Optional[LLMConfig] = None
    embedding: Optional[EmbeddingConfig] = None
    memory: Optional[MemoryConfig] = None
    storage: Optional[StorageConfig] = None
    
    def __post_init__(self):
        """Set up derived configurations."""
        # Parse model string to create LLMConfig if not provided
        if self.llm is None:
            provider, model_name = self._parse_model_string(self.model)
            self.llm = LLMConfig(
                provider=provider,
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        # Set defaults for other configs if not provided
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        
        if self.memory is None:
            self.memory = MemoryConfig()
        
        if self.storage is None:
            self.storage = StorageConfig()
        
        # Validate agent_id
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")
    
    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """
        Parse model string in format 'provider/model-name'.
        
        Args:
            model: Model string (e.g., "groq/llama-3.3-70b-versatile")
            
        Returns:
            Tuple of (provider, model_name)
        """
        if "/" in model:
            provider, model_name = model.split("/", 1)
            return provider.strip(), model_name.strip()
        else:
            # Assume groq if no provider specified
            return "groq", model.strip()


# ============================================================================
# Environment Variable Loading
# ============================================================================

def load_environment():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.debug(".env file not found, using system environment variables")


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional type conversion.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not found")
    
    # Type conversion for boolean strings
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
    
    return value


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_blocks() -> Dict[str, BlockConfig]:
    """
    Create a set of default memory blocks for general use.
    
    Returns:
        Dictionary of default blocks
    """
    return {
        "persona": BlockConfig(
            size=1000,
            priority=10,  # High priority - rarely evicted
            description="Agent's personality and behavior guidelines",
            initial_value="I am a helpful AI assistant with persistent memory."
        ),
        "user": BlockConfig(
            size=1000,
            priority=5,  # Medium-high priority
            description="Information about the user",
            initial_value="User information will be stored here."
        ),
        "task": BlockConfig(
            size=1500,
            priority=2,  # Lower priority - can be evicted
            description="Current task and context",
            initial_value=""
        )
    }


# Load environment on module import
load_environment()