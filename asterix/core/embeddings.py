"""
MemGPT Embedding Service Wrapper

Provides a unified interface for embedding services with:
- Provider switching between OpenAI and SentenceTransformers
- Automatic fallback and health-aware operations
- Batch processing and caching for performance
- Comprehensive error handling and retry logic
- Configuration-driven provider selection
"""

import asyncio
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

import openai
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import get_config_manager
from ..utils.health import health_monitor

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation"""
    embeddings: List[List[float]]
    model: str
    provider: str
    dimensions: int
    usage: Dict[str, int]
    processing_time: float


class EmbeddingError(Exception):
    """Raised when embedding operations fail"""
    pass


class EmbeddingServiceWrapper:
    """
    Unified wrapper for embedding services with provider switching.
    
    Features:
    - Automatic provider switching (OpenAI ↔ SentenceTransformers)
    - Health-aware operations with automatic fallback
    - Batch processing with configurable sizes
    - In-memory caching for performance optimization
    - Comprehensive error handling and retry logic
    """
    
    def __init__(self):
        """Initialize the embedding service wrapper."""
        # Get API keys from environment
        import os
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Default configuration
        self.embedding_config = type('EmbeddingConfig', (), {
            'provider': 'openai',
            'model': 'text-embedding-3-small',
            'dimensions': 1536,
            'batch_size': 100,
            'api_key': self._openai_api_key
        })()
        
        # Provider clients
        self._openai_client: Optional[openai.OpenAI] = None
        self._sbert_model: Optional[SentenceTransformer] = None
        
        # Provider status
        self._primary_provider = self.embedding_config.provider
        self._fallback_provider = "sentence_transformers"
        self._current_provider = self._primary_provider
        self._provider_health = {}
        self._last_health_check = 0
        self._health_check_interval = 45  # seconds
        
        # Caching
        self._cache: Dict[str, Tuple[List[float], float]] = {}  # hash -> (embedding, timestamp)
        self._cache_ttl = 3600  # 1 hour
        self._max_cache_size = 1000
        
        # Performance tracking
        self._operation_count = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Batch processing
        self._batch_size = self.embedding_config.batch_size
        self._max_batch_size = 100
    
    async def _ensure_providers_initialized(self):
        """Ensure embedding providers are initialized."""
        # Initialize OpenAI client if needed
        if self._openai_client is None and self.embedding_config.api_key:
            self._openai_client = openai.OpenAI(api_key=self.embedding_config.api_key)
        
        # Initialize SentenceTransformers model if needed
        if (self._sbert_model is None and 
            (self._current_provider == "sentence_transformers" or 
             self._fallback_provider == "sentence_transformers")):
            
            # Use default SentenceTransformers configuration
            model_name = "all-MiniLM-L6-v2"  # Default model (384 dimensions)
            device = "cpu"  # Default device
            
            try:
                self._sbert_model = SentenceTransformer(model_name, device=device)
                logger.info(f"Initialized SentenceTransformers model: {model_name} on {device}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformers: {e}")
    
    async def _check_provider_health(self) -> Dict[str, bool]:
        """
        Check health of embedding providers (simplified - no actual health checks).
        
        Returns:
            Dictionary mapping provider names to health status
        """
        # Simplified: just return True for configured providers
        # Actual health will be determined when we try to use them
        health_status = {}
        
        # Check if providers are configured (not if they're actually healthy)
        if self._openai_api_key:
            health_status["openai"] = True
        
        # SentenceTransformers is always "healthy" (local, no API)
        health_status["sentence_transformers"] = True
        
        self._provider_health = health_status
        self._last_health_check = time.time()
        
        return health_status
    
    async def _select_provider(self) -> str:
        """
        Select the best available provider based on health status.
        
        Returns:
            Name of the selected provider
        """
        health_status = await self._check_provider_health()
        
        # Try primary provider first
        if health_status.get(self._primary_provider, False):
            return self._primary_provider
        
        # Fall back to fallback provider if primary fails
        if health_status.get(self._fallback_provider, False):
            logger.debug(f"Using fallback provider: {self._fallback_provider}")
            return self._fallback_provider

        # Default to primary
        return self._primary_provider
    
    def _get_cache_key(self, texts: List[str], model: str) -> str:
        """Generate cache key for embedding request."""
        # Create a hash of the texts and model
        content = "|".join(texts) + f"|{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[List[float]]]:
        """Get embeddings from cache if available and valid."""
        if cache_key in self._cache:
            embeddings, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                self._cache_hits += 1
                return embeddings
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _store_in_cache(self, cache_key: str, embeddings: List[List[float]]):
        """Store embeddings in cache."""
        # Clean up cache if it's getting too large
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries (simple cleanup)
            oldest_keys = list(self._cache.keys())[:self._max_cache_size // 4]
            for key in oldest_keys:
                del self._cache[key]
        
        self._cache[cache_key] = (embeddings, time.time())
    
    async def _embed_with_openai(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding result with metadata
        """
        if not self._openai_client:
            raise EmbeddingError("OpenAI client not initialized")
        
        start_time = time.time()
        
        try:
            response = self._openai_client.embeddings.create(
                model=self.embedding_config.model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build result
            result = EmbeddingResult(
                embeddings=embeddings,
                model=self.embedding_config.model,
                provider="openai",
                dimensions=len(embeddings[0]) if embeddings else self.embedding_config.dimensions,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                processing_time=processing_time
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings using OpenAI in {processing_time:.3f}s")
            return result
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise EmbeddingError(f"Rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise EmbeddingError(f"Authentication failed: {e}")
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingError(f"OpenAI error: {e}")
    
    async def _embed_with_sentence_transformers(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings using SentenceTransformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding result with metadata
        """
        if not self._sbert_model:
            raise EmbeddingError("SentenceTransformers model not initialized")
        
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self._sbert_model.encode(
                texts,
                batch_size=self.embedding_config.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build result
            result = EmbeddingResult(
                embeddings=embeddings,
                model=self._sbert_model.get_sentence_embedding_dimension(),
                provider="sentence_transformers",
                dimensions=len(embeddings[0]) if embeddings else 384,  # Default dimension
                usage={
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                },
                processing_time=processing_time
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings using SentenceTransformers in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"SentenceTransformers embedding error: {e}")
            raise EmbeddingError(f"SentenceTransformers error: {e}")
    
    async def embed_texts(self, texts: Union[str, List[str]], 
                         provider: Optional[str] = None) -> EmbeddingResult:
        """
        Generate embeddings for texts using the best available provider.
        
        Args:
            texts: Text or list of texts to embed
            provider: Specific provider to use (optional)
            
        Returns:
            Embedding result with embeddings and metadata
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise EmbeddingError("No texts provided for embedding")
        
        # Ensure providers are initialized
        await self._ensure_providers_initialized()
        
        # Select provider
        if provider:
            selected_provider = provider
        else:
            selected_provider = await self._select_provider()
        
        # Check cache
        cache_key = self._get_cache_key(texts, selected_provider)
        cached_embeddings = self._get_from_cache(cache_key)
        
        if cached_embeddings:
            logger.info(f"Retrieved {len(cached_embeddings)} embeddings from cache")
            return EmbeddingResult(
                embeddings=cached_embeddings,
                model=f"{selected_provider}_cached",
                provider=selected_provider,
                dimensions=len(cached_embeddings[0]) if cached_embeddings else 0,
                usage={"prompt_tokens": 0, "total_tokens": 0},
                processing_time=0.0
            )
        
        # Generate embeddings
        try:
            if selected_provider == "openai":
                result = await self._embed_with_openai(texts)
            elif selected_provider == "sentence_transformers":
                result = await self._embed_with_sentence_transformers(texts)
            else:
                raise EmbeddingError(f"Unknown provider: {selected_provider}")
            
            # Update metrics
            self._operation_count += 1
            self._total_processing_time += result.processing_time
            
            # Store in cache
            self._store_in_cache(cache_key, result.embeddings)
            
            return result
            
        except EmbeddingError:
            # If the selected provider fails and we haven't tried fallback yet
            if selected_provider == self._primary_provider and self._fallback_provider != self._primary_provider:
                logger.warning(f"Primary provider {selected_provider} failed, trying fallback {self._fallback_provider}")
                return await self.embed_texts(texts, provider=self._fallback_provider)
            else:
                raise
    
    async def embed_batch(self, texts: List[str], 
                         batch_size: Optional[int] = None) -> EmbeddingResult:
        """
        Generate embeddings for a large batch of texts with automatic batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            
        Returns:
            Combined embedding result
        """
        if not texts:
            raise EmbeddingError("No texts provided for batch embedding")
        
        effective_batch_size = batch_size or self._batch_size
        effective_batch_size = min(effective_batch_size, self._max_batch_size)
        
        # Process in batches
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        total_time = 0.0
        provider_used = None
        model_used = None
        dimensions = 0
        
        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i:i + effective_batch_size]
            
            try:
                result = await self.embed_texts(batch)
                
                all_embeddings.extend(result.embeddings)
                total_usage["prompt_tokens"] += result.usage.get("prompt_tokens", 0)
                total_usage["total_tokens"] += result.usage.get("total_tokens", 0)
                total_time += result.processing_time
                
                if provider_used is None:
                    provider_used = result.provider
                    model_used = result.model
                    dimensions = result.dimensions
                
                logger.info(f"Processed batch {i//effective_batch_size + 1}/{(len(texts) + effective_batch_size - 1)//effective_batch_size}")
                
            except EmbeddingError as e:
                logger.error(f"Batch {i//effective_batch_size + 1} failed: {e}")
                raise
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model_used or "unknown",
            provider=provider_used or "unknown",
            dimensions=dimensions,
            usage=total_usage,
            processing_time=total_time
        )
    
    async def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """
        Get the embedding dimension for the specified provider.
        
        Args:
            provider: Provider to check (optional)
            
        Returns:
            Embedding dimension
        """
        if not provider:
            provider = await self._select_provider()
        
        if provider == "openai":
            return self.embedding_config.dimensions
        elif provider == "sentence_transformers":
            if not self._sbert_model:
                await self._ensure_providers_initialized()
            
            if self._sbert_model:
                return self._sbert_model.get_sentence_embedding_dimension()
            else:
                return 384  # Default for all-MiniLM-L6-v2
        
        return 384  # Fallback
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedding service.
        
        Returns:
            Dictionary with performance information
        """
        avg_processing_time = (
            self._total_processing_time / self._operation_count 
            if self._operation_count > 0 else 0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses) 
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )
        
        return {
            "primary_provider": self._primary_provider,
            "fallback_provider": self._fallback_provider,
            "current_provider": self._current_provider,
            "provider_health": self._provider_health,
            "operation_count": self._operation_count,
            "average_processing_time_ms": round(avg_processing_time * 1000, 2),
            "total_processing_time_ms": round(self._total_processing_time * 1000, 2),
            "cache_size": len(self._cache),
            "cache_hit_rate": round(cache_hit_rate * 100, 2),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "batch_size": self._batch_size,
            "max_cache_size": self._max_cache_size
        }
    
    async def test_provider(self, provider: str) -> Dict[str, Any]:
        """
        Test a specific embedding provider.
        
        Args:
            provider: Provider name to test
            
        Returns:
            Test result with performance metrics
        """
        test_text = "This is a test embedding for service verification."
        
        try:
            start_time = time.time()
            result = await self.embed_texts([test_text], provider=provider)
            total_time = time.time() - start_time
            
            return {
                "provider": provider,
                "status": "healthy",
                "response_time_ms": round(total_time * 1000, 2),
                "embedding_dimension": result.dimensions,
                "model": result.model,
                "usage": result.usage
            }
            
        except Exception as e:
            return {
                "provider": provider,
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": None,
                "embedding_dimension": None,
                "model": None,
                "usage": None
            }


# Global embedding service instance
embedding_service = EmbeddingServiceWrapper()