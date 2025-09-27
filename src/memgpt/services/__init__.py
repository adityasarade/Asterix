"""
MemGPT Services Package

Provides service wrappers for external integrations:
- Letta client wrapper for agent operations
- Qdrant Cloud wrapper for vector storage and search
- Embedding service wrapper with provider switching
- LLM provider manager for intelligent routing
"""

from .letta_client import (
    LettaClientWrapper,
    LettaAgentConfig,
    LettaConnectionError,
    LettaAgentError,
    letta_client
)

from .qdrant_client import (
    QdrantCloudWrapper,
    VectorSearchResult,
    ArchivalRecord,
    QdrantConnectionError,
    QdrantOperationError,
    qdrant_client
)

from .embedding_service import (
    EmbeddingServiceWrapper,
    EmbeddingResult,
    EmbeddingError,
    embedding_service
)

from .llm_manager import (
    LLMProviderManager,
    LLMResponse,
    LLMMessage,
    LLMError,
    llm_manager
)

__all__ = [
    # Letta Client
    "LettaClientWrapper",
    "LettaAgentConfig",
    "LettaConnectionError",
    "LettaAgentError",
    "letta_client",
    
    # Qdrant Client
    "QdrantCloudWrapper",
    "VectorSearchResult",
    "ArchivalRecord",
    "QdrantConnectionError",
    "QdrantOperationError",
    "qdrant_client",
    
    # Embedding Service
    "EmbeddingServiceWrapper",
    "EmbeddingResult",
    "EmbeddingError",
    "embedding_service",
    
    # LLM Manager
    "LLMProviderManager",
    "LLMResponse",
    "LLMMessage",
    "LLMError",
    "llm_manager"
]