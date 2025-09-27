"""
MemGPT Configuration Loader

Loads and manages configuration from YAML files and environment variables.
Provides a unified interface for accessing all configuration settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """Service health status information"""
    name: str
    available: bool
    error: Optional[str] = None
    last_check: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    api_key: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration"""
    provider: str
    model: str
    dimensions: int = 1536
    batch_size: int = 100
    api_key: Optional[str] = None


@dataclass
class QdrantConfig:
    """Qdrant configuration"""
    url: str
    api_key: str
    collection_name: str = "memgpt_archive"
    vector_size: int = 1536
    timeout: int = 30
    auto_create: bool = True


@dataclass
class MemoryConfig:
    """Memory management configuration"""
    core_block_token_limit: int = 1500
    summary_token_limit: int = 220
    retrieval_k: int = 6
    eviction_strategy: str = "summarize_and_archive"
    token_counter: str = "tiktoken"


@dataclass
class ControllerConfig:
    """Controller service configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    timeout: int = 60
    max_concurrent_requests: int = 10
    heartbeat_max_steps: int = 6


class ConfigurationManager:
    """
    Manages configuration loading from YAML files and environment variables.
    
    Priority order:
    1. Environment variables (highest)
    2. YAML configuration files
    3. Default values (lowest)
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing YAML config files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._config_cache: Dict[str, Any] = {}
        self._load_environment()
        self._load_yaml_configs()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.warning(".env file not found, using system environment variables")
    
    def _load_yaml_configs(self):
        """Load YAML configuration files"""
        config_files = [
            "agent_config.yaml",
            "service_config.yaml", 
            "memory_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        self._config_cache[config_file] = config_data
                        logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logger.error(f"Failed to load {config_path}: {e}")
                    self._config_cache[config_file] = {}
            else:
                logger.warning(f"Configuration file not found: {config_path}")
                self._config_cache[config_file] = {}
    
    def get_env(self, key: str, default: Any = None, required: bool = False) -> Any:
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
    
    def get_yaml_config(self, file: str, path: str, default: Any = None) -> Any:
        """
        Get configuration value from YAML file using dot notation.
        
        Args:
            file: YAML filename (e.g., "agent_config.yaml")
            path: Dot-separated path (e.g., "agent.llm.groq.model")
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        config = self._config_cache.get(file, {})
        
        # Navigate through nested dictionaries
        current = config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_llm_config(self, provider: Optional[str] = None) -> LLMConfig:
        """
        Get LLM configuration for specified provider.
        
        Args:
            provider: LLM provider name (groq, openai) or None for primary
            
        Returns:
            LLM configuration object
        """
        # Determine provider
        if not provider:
            provider = self.get_env("LLM_PROVIDER") or \
                      self.get_yaml_config("agent_config.yaml", "agent.llm.primary_provider", "groq")
        
        # Get provider-specific config
        if provider == "groq":
            model = self.get_env("GROQ_MODEL") or \
                   self.get_yaml_config("agent_config.yaml", "agent.llm.groq.model", "llama-3.3-70b-versatile")
            
            return LLMConfig(
                provider=provider,
                model=model,
                temperature=float(self.get_env("GROQ_TEMPERATURE") or 
                                self.get_yaml_config("agent_config.yaml", "agent.llm.groq.temperature", 0.1)),
                max_tokens=int(self.get_env("GROQ_MAX_TOKENS") or 
                             self.get_yaml_config("agent_config.yaml", "agent.llm.groq.max_tokens", 1000)),
                timeout=int(self.get_env("GROQ_TIMEOUT") or 
                          self.get_yaml_config("agent_config.yaml", "agent.llm.groq.timeout", 30)),
                api_key=self.get_env("GROQ_API_KEY", required=True)
            )
        
        elif provider == "openai":
            model = self.get_env("OPENAI_MODEL") or \
                   self.get_yaml_config("agent_config.yaml", "agent.llm.openai.model", "gpt-4-turbo-preview")
            
            return LLMConfig(
                provider=provider,
                model=model,
                temperature=float(self.get_env("OPENAI_TEMPERATURE") or 
                                self.get_yaml_config("agent_config.yaml", "agent.llm.openai.temperature", 0.1)),
                max_tokens=int(self.get_env("OPENAI_MAX_TOKENS") or 
                             self.get_yaml_config("agent_config.yaml", "agent.llm.openai.max_tokens", 1000)),
                timeout=int(self.get_env("OPENAI_TIMEOUT") or 
                          self.get_yaml_config("agent_config.yaml", "agent.llm.openai.timeout", 30)),
                api_key=self.get_env("OPENAI_API_KEY", required=True)
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """
        Get embedding configuration.
        
        Returns:
            Embedding configuration object
        """
        provider = self.get_env("EMBED_PROVIDER") or \
                  self.get_yaml_config("agent_config.yaml", "agent.embeddings.provider", "openai")
        
        if provider == "openai":
            return EmbeddingConfig(
                provider=provider,
                model=self.get_env("OPENAI_EMBEDDING_MODEL") or 
                     self.get_yaml_config("agent_config.yaml", "agent.embeddings.openai.model", "text-embedding-3-small"),
                dimensions=int(self.get_env("OPENAI_EMBEDDING_DIMENSIONS") or 
                             self.get_yaml_config("agent_config.yaml", "agent.embeddings.openai.dimensions", 1536)),
                batch_size=int(self.get_env("OPENAI_EMBEDDING_BATCH_SIZE") or 
                             self.get_yaml_config("agent_config.yaml", "agent.embeddings.openai.batch_size", 100)),
                api_key=self.get_env("OPENAI_API_KEY", required=True)
            )
        
        elif provider == "sentence_transformers":
            return EmbeddingConfig(
                provider=provider,
                model=self.get_env("SBERT_MODEL") or 
                     self.get_yaml_config("agent_config.yaml", "agent.embeddings.sentence_transformers.model", "all-MiniLM-L6-v2"),
                batch_size=int(self.get_env("SBERT_BATCH_SIZE") or 
                             self.get_yaml_config("agent_config.yaml", "agent.embeddings.sentence_transformers.batch_size", 32))
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def get_qdrant_config(self) -> QdrantConfig:
        """
        Get Qdrant configuration.
        
        Returns:
            Qdrant configuration object
        """
        return QdrantConfig(
            url=self.get_env("QDRANT_URL", required=True),
            api_key=self.get_env("QDRANT_API_KEY", required=True),
            collection_name=self.get_env("QDRANT_COLLECTION_NAME") or 
                          self.get_yaml_config("service_config.yaml", "qdrant.collection_name", "memgpt_archive"),
            vector_size=int(self.get_env("QDRANT_VECTOR_SIZE") or 
                          self.get_yaml_config("service_config.yaml", "qdrant.collection.vector_size", 1536)),
            timeout=int(self.get_env("QDRANT_TIMEOUT") or 
                      self.get_yaml_config("service_config.yaml", "qdrant.timeout", 30)),
            auto_create=self.get_env("QDRANT_AUTO_CREATE", True)
        )
    
    def get_memory_config(self) -> MemoryConfig:
        """
        Get memory management configuration.
        
        Returns:
            Memory configuration object
        """
        return MemoryConfig(
            core_block_token_limit=int(self.get_env("CORE_BLOCK_TOKEN_LIMIT") or 
                                     self.get_yaml_config("memory_config.yaml", "tokens.core_memory.block_limit", 1500)),
            summary_token_limit=int(self.get_env("SUMMARY_TOKEN_LIMIT") or 
                                  self.get_yaml_config("memory_config.yaml", "tokens.core_memory.summary_limit", 220)),
            retrieval_k=int(self.get_env("RETRIEVAL_K") or 
                          self.get_yaml_config("memory_config.yaml", "retrieval.semantic.default_k", 6)),
            eviction_strategy=self.get_env("EVICTION_STRATEGY") or 
                            self.get_yaml_config("memory_config.yaml", "eviction.strategy", "summarize_and_archive"),
            token_counter=self.get_env("TOKEN_COUNTER") or 
                        self.get_yaml_config("memory_config.yaml", "tokens.counter", "tiktoken")
        )
    
    def get_controller_config(self) -> ControllerConfig:
        """
        Get controller configuration.
        
        Returns:
            Controller configuration object
        """
        return ControllerConfig(
            host=self.get_env("CONTROLLER_HOST") or 
                self.get_yaml_config("service_config.yaml", "controller.host", "127.0.0.1"),
            port=int(self.get_env("CONTROLLER_PORT") or 
                   self.get_yaml_config("service_config.yaml", "controller.port", 8000)),
            timeout=int(self.get_env("CONTROLLER_TIMEOUT") or 
                      self.get_yaml_config("service_config.yaml", "controller.timeout", 60)),
            max_concurrent_requests=int(self.get_env("MAX_CONCURRENT_REQUESTS") or 
                                      self.get_yaml_config("service_config.yaml", "controller.max_concurrent_requests", 10)),
            heartbeat_max_steps=int(self.get_env("HEARTBEAT_MAX_STEPS") or 
                                  self.get_yaml_config("service_config.yaml", "heartbeat.max_steps", 6))
        )
    
    def get_letta_config(self) -> Dict[str, Any]:
        """
        Get Letta service configuration.
        
        Returns:
            Dictionary with Letta configuration
        """
        return {
            "url": self.get_env("LETTA_URL") or 
                  self.get_yaml_config("service_config.yaml", "letta.url", "http://127.0.0.1:8283"),
            "api_key": self.get_env("LETTA_API_KEY"),
            "timeout": int(self.get_env("LETTA_TIMEOUT") or 
                         self.get_yaml_config("service_config.yaml", "letta.timeout", 30)),
            "max_retries": int(self.get_env("LETTA_MAX_RETRIES") or 
                             self.get_yaml_config("service_config.yaml", "letta.max_retries", 3)),
            "retry_delay": float(self.get_env("LETTA_RETRY_DELAY") or 
                               self.get_yaml_config("service_config.yaml", "letta.retry_delay", 1.0))
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dictionary with logging configuration
        """
        return {
            "level": self.get_env("LOG_LEVEL") or 
                    self.get_yaml_config("service_config.yaml", "logging.level", "INFO"),
            "directory": self.get_env("LOG_DIR") or 
                        self.get_yaml_config("service_config.yaml", "logging.directory", "logs"),
            "files": self.get_yaml_config("service_config.yaml", "logging.files", {
                "controller": "memgpt_controller.log",
                "memory_ops": "memory_operations.log", 
                "service_health": "service_health.log",
                "errors": "errors.log"
            }),
            "format": self.get_yaml_config("service_config.yaml", "logging.format", {}),
            "rotation": self.get_yaml_config("service_config.yaml", "logging.rotation", {}),
            "console": self.get_yaml_config("service_config.yaml", "logging.console", {
                "enabled": True,
                "level": "INFO",
                "colored": True
            })
        }
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate all configuration settings.
        
        Returns:
            Dictionary mapping config sections to validation status
        """
        validation_results = {}
        
        try:
            # Validate LLM configuration
            llm_config = self.get_llm_config()
            validation_results["llm"] = bool(llm_config.api_key)
        except Exception as e:
            logger.error(f"LLM configuration validation failed: {e}")
            validation_results["llm"] = False
        
        try:
            # Validate embedding configuration
            embed_config = self.get_embedding_config()
            validation_results["embeddings"] = bool(embed_config.provider)
        except Exception as e:
            logger.error(f"Embedding configuration validation failed: {e}")
            validation_results["embeddings"] = False
        
        try:
            # Validate Qdrant configuration
            qdrant_config = self.get_qdrant_config()
            validation_results["qdrant"] = bool(qdrant_config.url and qdrant_config.api_key)
        except Exception as e:
            logger.error(f"Qdrant configuration validation failed: {e}")
            validation_results["qdrant"] = False
        
        try:
            # Validate Letta configuration
            letta_config = self.get_letta_config()
            validation_results["letta"] = bool(letta_config["url"])
        except Exception as e:
            logger.error(f"Letta configuration validation failed: {e}")
            validation_results["letta"] = False
        
        return validation_results
    
    def get_core_memory_blocks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get core memory blocks configuration.
        
        Returns:
            Dictionary mapping block names to their configuration
        """
        return self.get_yaml_config("agent_config.yaml", "core_memory.blocks", {
            "human": {
                "label": "Human",
                "description": "Information about the user",
                "initial_value": "The user's information will be stored here.",
                "max_tokens": 1500,
                "priority": 1
            },
            "persona": {
                "label": "Persona", 
                "description": "Agent's personality and behavior",
                "initial_value": "I am a helpful AI assistant with persistent memory.",
                "max_tokens": 1500,
                "priority": 1
            }
        })
    
    def reload_configuration(self):
        """Reload all configuration from files and environment"""
        self._config_cache.clear()
        self._load_environment()
        self._load_yaml_configs()
        logger.info("Configuration reloaded")


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    return config_manager