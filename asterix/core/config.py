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
import yaml

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
    state_db: str = "./agent_states/agents.db"
    
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
        ...     model="openai/gpt-5-mini",
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
    model: str = "openai/gpt-5-mini"
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
            model: Model string (e.g., "openai/gpt-5-mini")
            
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

# ============================================================================
# YAML Configuration Manager
# ============================================================================

class ConfigurationManager:
    """
    Manages configuration loading from YAML files and environment variables.
    
    Priority order:
    1. Python overrides (highest)
    2. Environment variables
    3. YAML configuration files
    4. Default values (lowest)
    
    Example:
        >>> manager = ConfigurationManager("config")
        >>> config = manager.load_agent_config("my_agent.yaml")
        >>> # Or with overrides:
        >>> config = manager.load_agent_config(
        ...     "my_agent.yaml",
        ...     model="openai/gpt-4",
        ...     temperature=0.2
        ... )
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing YAML config files (default: "./config")
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Ensure environment is loaded
        load_environment()
    
    def load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file (e.g., "agent_config.yaml")
            
        Returns:
            Dictionary with configuration data
        """
        # Check cache first
        if filename in self._config_cache:
            return self._config_cache[filename]
        
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Replace environment variable references (${VAR_NAME})
            config_data = self._resolve_env_vars(config_data)
            
            # Cache the config
            self._config_cache[filename] = config_data
            
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}")
            return {}
    
    def _resolve_env_vars(self, config: Any) -> Any:
        """
        Recursively resolve environment variable references in config.
        
        Replaces ${VAR_NAME} with the value of the environment variable.
        
        Args:
            config: Configuration value (dict, list, string, etc.)
            
        Returns:
            Configuration with resolved environment variables
        """
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} patterns
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.getenv(var_name, config)  # Return original if not found
            return config
        else:
            return config
    
    def load_agent_config(self, 
                         filename: Optional[str] = None,
                         **overrides) -> AgentConfig:
        """
        Load agent configuration from YAML file with optional overrides.
        
        Args:
            filename: YAML file to load (optional, can create from overrides only)
            **overrides: Override specific configuration values
            
        Returns:
            AgentConfig object
            
        Example:
            >>> # Load from YAML
            >>> config = manager.load_agent_config("my_agent.yaml")
            
            >>> # Load with overrides
            >>> config = manager.load_agent_config(
            ...     "my_agent.yaml",
            ...     model="openai/gpt-4",
            ...     temperature=0.2
            ... )
            
            >>> # Create from scratch with overrides only
            >>> config = manager.load_agent_config(
            ...     agent_id="new_agent",
            ...     model="openai/gpt-5-mini",
            ...     blocks={"task": BlockConfig(size=1500)}
            ... )
        """
        # Load YAML config if provided
        if filename:
            yaml_config = self.load_yaml_file(filename)
        else:
            yaml_config = {}
        
        # Build configuration with priority: overrides > env vars > yaml > defaults
        
        # 1. Load agent_id and max_heartbeat_steps (top-level settings)
        agent_id = yaml_config.get("agent_id", "agent")
        max_heartbeat_steps = yaml_config.get("max_heartbeat_steps", 10)
        
        # Apply environment variable overrides
        agent_id = get_env("AGENT_ID", agent_id)
        max_heartbeat_steps = int(get_env("AGENT_MAX_HEARTBEAT_STEPS", max_heartbeat_steps))
        
        # Apply Python overrides
        agent_id = overrides.pop("agent_id", agent_id)
        max_heartbeat_steps = overrides.pop("max_heartbeat_steps", max_heartbeat_steps)
        
        # 2. Load LLM configuration (using new method)
        model, temperature, max_tokens, timeout = self._load_llm_config(yaml_config, overrides)
        
        # 3. Load all other configuration sections
        blocks = self._load_blocks_config(yaml_config, overrides)
        storage = self._load_storage_config(yaml_config, overrides)
        memory = self._load_memory_config(yaml_config, overrides)
        embedding = self._load_embedding_config(yaml_config, overrides)
        
        # Create AgentConfig
        return AgentConfig(
            agent_id=agent_id,
            blocks=blocks,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_heartbeat_steps=max_heartbeat_steps,
            storage=storage,
            memory=memory,
            embedding=embedding
        )
    
    def _load_blocks_config(self, 
                           yaml_config: Dict[str, Any],
                           overrides: Dict[str, Any]) -> Dict[str, BlockConfig]:
        """Load blocks configuration from YAML and overrides."""
        # Check for override first
        if "blocks" in overrides:
            return overrides.pop("blocks")
        
        # Load from YAML
        blocks_config = yaml_config.get("blocks", {})
        
        if not blocks_config:
            # Return default blocks if none specified
            return create_default_blocks()
        
        # Convert YAML blocks to BlockConfig objects
        blocks = {}
        for block_name, block_data in blocks_config.items():
            blocks[block_name] = BlockConfig(
                size=block_data.get("size", 1500),
                priority=block_data.get("priority", 1),
                description=block_data.get("description", ""),
                initial_value=block_data.get("initial_value", "")
            )
        
        return blocks
    
    def _load_llm_config(self,
                        yaml_config: Dict[str, Any],
                        overrides: Dict[str, Any]) -> tuple[str, float, int, int]:
        """
        Load LLM configuration from YAML and overrides.
        
        Supports two formats:
        1. New format (preferred):
           llm:
             provider: "openai"
             model: "gpt-5-mini"
             temperature: 0.1
             max_tokens: 1000
             timeout: 30
        
        2. Legacy format (backward compatible):
           model: "openai/gpt-5-mini"
           temperature: 0.1
           max_tokens: 1000
        
        Args:
            yaml_config: YAML configuration dictionary
            overrides: Python override dictionary
            
        Returns:
            Tuple of (model, temperature, max_tokens, timeout)
        """
        # Check for new LLM section format
        llm_config = yaml_config.get("llm", {})
        
        if llm_config:
            # New format: llm section exists
            provider = llm_config.get("provider", "openai")
            model_name = llm_config.get("model", "gpt-5-mini")
            
            # Combine provider/model if provider specified
            if provider and model_name:
                model = f"{provider}/{model_name}"
            else:
                model = model_name
            
            temperature = float(llm_config.get("temperature", 0.1))
            max_tokens = int(llm_config.get("max_tokens", 1000))
            timeout = int(llm_config.get("timeout", 30))
        else:
            # Legacy format: settings at top level
            model = yaml_config.get("model", "openai/gpt-5-mini")
            temperature = float(yaml_config.get("temperature", 0.1))
            max_tokens = int(yaml_config.get("max_tokens", 1000))
            timeout = 30  # Default for legacy format
        
        # Apply environment variable overrides
        model = get_env("AGENT_MODEL", model)
        temperature = float(get_env("AGENT_TEMPERATURE", temperature))
        max_tokens = int(get_env("AGENT_MAX_TOKENS", max_tokens))
        timeout = int(get_env("LLM_TIMEOUT", timeout))
        
        # Apply Python overrides
        model = overrides.pop("model", model)
        temperature = overrides.pop("temperature", temperature)
        max_tokens = overrides.pop("max_tokens", max_tokens)
        timeout = overrides.pop("timeout", timeout) if "timeout" in overrides else timeout
        
        return model, temperature, max_tokens, timeout
    
    def _load_storage_config(self,
                            yaml_config: Dict[str, Any],
                            overrides: Dict[str, Any]) -> StorageConfig:
        """Load storage configuration from YAML and overrides."""
        # Check for override first
        if "storage" in overrides:
            return overrides.pop("storage")
        
        # Load from YAML
        storage_config = yaml_config.get("storage", {})
        
        # Build StorageConfig with env var fallbacks
        return StorageConfig(
            qdrant_url=storage_config.get("qdrant_url") or get_env("QDRANT_URL", ""),
            qdrant_api_key=storage_config.get("qdrant_api_key") or get_env("QDRANT_API_KEY", ""),
            qdrant_collection_name=storage_config.get("qdrant_collection_name") or 
                                   get_env("QDRANT_COLLECTION_NAME", "asterix_memory"),
            vector_size=int(storage_config.get("vector_size", 1536)),
            qdrant_timeout=int(storage_config.get("qdrant_timeout", 30)),
            auto_create_collection=storage_config.get("auto_create_collection", True),
            state_backend=storage_config.get("state_backend", "json"),
            state_dir=storage_config.get("state_dir", "./agent_states"),
            state_db=storage_config.get("state_db", "agents.db")
        )
    
    def _load_memory_config(self,
                           yaml_config: Dict[str, Any],
                           overrides: Dict[str, Any]) -> MemoryConfig:
        """Load memory configuration from YAML and overrides."""
        # Check for override first
        if "memory" in overrides:
            return overrides.pop("memory")
        
        # Load from YAML
        memory_config = yaml_config.get("memory", {})
        
        return MemoryConfig(
            eviction_strategy=memory_config.get("eviction_strategy", "summarize_and_archive"),
            summary_token_limit=int(memory_config.get("summary_token_limit", 220)),
            context_window_threshold=float(memory_config.get("context_window_threshold", 0.85)),
            extraction_enabled=memory_config.get("extraction_enabled", True),
            retrieval_k=int(memory_config.get("retrieval_k", 6)),
            score_threshold=float(memory_config.get("score_threshold", 0.7))
        )
    
    def _load_embedding_config(self,
                              yaml_config: Dict[str, Any],
                              overrides: Dict[str, Any]) -> EmbeddingConfig:
        """Load embedding configuration from YAML and overrides."""
        # Check for override first
        if "embedding" in overrides:
            return overrides.pop("embedding")
        
        # Load from YAML
        embedding_config = yaml_config.get("embedding", {})
        
        return EmbeddingConfig(
            provider=embedding_config.get("provider") or get_env("EMBED_PROVIDER", "openai"),
            model=embedding_config.get("model", "text-embedding-3-small"),
            dimensions=int(embedding_config.get("dimensions", 1536)),
            batch_size=int(embedding_config.get("batch_size", 100)),
            api_key=embedding_config.get("api_key") or get_env("OPENAI_API_KEY")
        )
    
    def save_agent_config(self, config: AgentConfig, filename: str):
        """
        Save agent configuration to YAML file.
        
        Args:
            config: AgentConfig to save
            filename: Output filename
        """
        output_path = self.config_dir / filename
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = {
            "agent_id": config.agent_id,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_heartbeat_steps": config.max_heartbeat_steps,
            "blocks": {
                name: {
                    "size": block.size,
                    "priority": block.priority,
                    "description": block.description,
                    "initial_value": block.initial_value
                }
                for name, block in config.blocks.items()
            },
            "storage": {
                "qdrant_url": "${QDRANT_URL}",  # Use env var reference
                "qdrant_api_key": "${QDRANT_API_KEY}",
                "qdrant_collection_name": config.storage.qdrant_collection_name,
                "vector_size": config.storage.vector_size,
                "state_backend": config.storage.state_backend,
                "state_dir": config.storage.state_dir
            },
            "memory": {
                "eviction_strategy": config.memory.eviction_strategy,
                "summary_token_limit": config.memory.summary_token_limit,
                "context_window_threshold": config.memory.context_window_threshold,
                "retrieval_k": config.memory.retrieval_k,
                "score_threshold": config.memory.score_threshold
            },
            "embedding": {
                "provider": config.embedding.provider,
                "model": config.embedding.model,
                "dimensions": config.embedding.dimensions,
                "batch_size": config.embedding.batch_size
            }
        }
        
        # Write to YAML file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_path}: {e}")
            raise


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """
    Get or create the global configuration manager instance.
    
    Args:
        config_dir: Directory containing config files (optional)
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    
    return _config_manager