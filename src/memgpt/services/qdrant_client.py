"""
MemGPT Qdrant Cloud Wrapper

Provides a robust wrapper around Qdrant Cloud operations with:
- Auto-collection creation with proper schema management
- Vector operations with batch processing and optimization
- Comprehensive error handling and retry logic
- Health-aware operations with service status integration
- Metadata management and search optimization
"""

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ..utils.config import get_config
from ..utils.health import health_monitor
from ..utils.tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search operation"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class ArchivalRecord:
    """Record for archival memory storage"""
    text: str
    summary: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    id: Optional[str] = None


class QdrantConnectionError(Exception):
    """Raised when Qdrant connection fails"""
    pass


class QdrantOperationError(Exception):
    """Raised when Qdrant operations fail"""
    pass


class QdrantCloudWrapper:
    """
    Robust wrapper for Qdrant Cloud operations.
    
    Features:
    - Auto-collection creation with optimized schema
    - Vector operations with batch processing
    - Health-aware operations with service status checking
    - Comprehensive error handling and retry logic
    - Metadata management and search optimization
    """
    
    def __init__(self):
        """Initialize the Qdrant Cloud wrapper."""
        self.config = get_config()
        self.qdrant_config = self.config.get_qdrant_config()
        self._client: Optional[QdrantClient] = None
        self._connected = False
        self._collection_exists = False
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds
        self._retry_attempts = 0
        self._max_retries = 3
        self._retry_delay = 1.0
        
        # Performance tracking
        self._operation_count = 0
        self._total_latency = 0.0
        self._last_operation_time = 0
    
    async def ensure_connected(self) -> bool:
        """
        Ensure the client is connected and healthy.
        
        Returns:
            True if connected and healthy, False otherwise
        """
        # Check if we need a health check
        current_time = time.time()
        if (current_time - self._last_health_check) > self._health_check_interval:
            health_result = await health_monitor.check_qdrant_health()
            self._last_health_check = current_time
            
            if health_result.status != "healthy":
                logger.error(f"Qdrant health check failed: {health_result.error}")
                self._connected = False
                return False
        
        # Try to connect if not connected
        if not self._connected or self._client is None:
            return await self._connect_with_retry()
        
        return True
    
    async def _connect_with_retry(self) -> bool:
        """
        Connect to Qdrant with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(self._max_retries):
            try:
                self._client = QdrantClient(
                    url=self.qdrant_config.url,
                    api_key=self.qdrant_config.api_key,
                    timeout=self.qdrant_config.timeout
                )
                
                # Test the connection
                collections = self._client.get_collections()
                logger.info(f"Connected to Qdrant successfully, {len(collections.collections)} collections found")
                
                self._connected = True
                self._retry_attempts = 0
                
                # Check/create collection
                await self._ensure_collection_exists()
                return True
                
            except Exception as e:
                self._retry_attempts = attempt + 1
                wait_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                
                logger.warning(
                    f"Qdrant connection attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )
                
                if attempt < self._max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All Qdrant connection attempts failed")
                    self._connected = False
                    
        return False
    
    async def _ensure_collection_exists(self) -> bool:
        """
        Ensure the target collection exists with proper configuration.
        
        Returns:
            True if collection exists or was created successfully
        """
        if not self._client:
            return False
        
        try:
            # Check if collection exists
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.qdrant_config.collection_name in collection_names:
                logger.info(f"Collection '{self.qdrant_config.collection_name}' already exists")
                self._collection_exists = True
                return True
            
            # Create collection if auto-create is enabled
            if self.qdrant_config.auto_create:
                return await self._create_collection()
            else:
                logger.error(f"Collection '{self.qdrant_config.collection_name}' does not exist and auto-create is disabled")
                return False
                
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    async def _create_collection(self) -> bool:
        """
        Create the collection with optimized configuration.
        
        Returns:
            True if creation successful
        """
        try:
            # Get service config for advanced settings
            service_config = self.config.get_yaml_config(
                "service_config.yaml", 
                "qdrant.collection", 
                {}
            )
            
            # Configure vector parameters
            vectors_config = models.VectorParams(
                size=self.qdrant_config.vector_size,
                distance=models.Distance.COSINE  # Use cosine distance for semantic similarity
            )
            
            # Configure HNSW for performance
            hnsw_config = models.HnswConfigDiff(
                m=service_config.get("hnsw_config", {}).get("m", 16),
                ef_construct=service_config.get("hnsw_config", {}).get("ef_construct", 100),
                full_scan_threshold=service_config.get("hnsw_config", {}).get("full_scan_threshold", 10000)
            )
            
            # Configure optimizers
            optimizer_config = models.OptimizersConfigDiff(
                default_segment_number=2,
                max_segment_size=None,
                memmap_threshold=None,
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=1
            )
            
            # Create collection
            result = self._client.create_collection(
                collection_name=self.qdrant_config.collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config,
                timeout=60
            )
            
            if result:
                logger.info(f"Created collection '{self.qdrant_config.collection_name}' successfully")
                self._collection_exists = True
                
                # Create indexes for common search fields
                await self._create_payload_indexes()
                return True
            else:
                logger.error("Collection creation returned False")
                return False
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    async def _create_payload_indexes(self):
        """Create indexes on payload fields for faster searching."""
        try:
            # Index for filtering by source
            self._client.create_payload_index(
                collection_name=self.qdrant_config.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for filtering by type
            self._client.create_payload_index(
                collection_name=self.qdrant_config.collection_name,
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for filtering by user_id
            self._client.create_payload_index(
                collection_name=self.qdrant_config.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for filtering by agent_id
            self._client.create_payload_index(
                collection_name=self.qdrant_config.collection_name,
                field_name="agent_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for date range filtering
            self._client.create_payload_index(
                collection_name=self.qdrant_config.collection_name,
                field_name="created_at",
                field_schema=models.PayloadSchemaType.DATETIME
            )
            
            logger.info("Created payload indexes for optimized searching")
            
        except Exception as e:
            logger.warning(f"Error creating payload indexes: {e}")
    
    async def insert_vectors(self, records: List[ArchivalRecord]) -> List[str]:
        """
        Insert multiple vectors into the collection.
        
        Args:
            records: List of archival records to insert
            
        Returns:
            List of inserted point IDs
            
        Raises:
            QdrantConnectionError: If unable to connect to Qdrant
            QdrantOperationError: If insertion fails
        """
        if not await self.ensure_connected():
            raise QdrantConnectionError("Unable to connect to Qdrant service")
        
        if not self._collection_exists:
            raise QdrantOperationError("Collection does not exist")
        
        start_time = time.time()
        
        try:
            points = []
            point_ids = []
            
            for record in records:
                # Generate ID if not provided
                point_id = record.id or str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare payload
                payload = {
                    "text": record.text,
                    "summary": record.summary,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **record.metadata
                }
                
                # Create point
                point = models.PointStruct(
                    id=point_id,
                    vector=record.embedding,
                    payload=payload
                )
                points.append(point)
            
            # Batch insert
            operation_result = self._client.upsert(
                collection_name=self.qdrant_config.collection_name,
                points=points,
                wait=True
            )
            
            # Update performance metrics
            self._update_metrics(time.time() - start_time)
            
            logger.info(f"Inserted {len(points)} vectors successfully")
            return point_ids
            
        except ResponseHandlingException as e:
            logger.error(f"Qdrant API error inserting vectors: {e}")
            raise QdrantOperationError(f"Failed to insert vectors: {e}")
        except Exception as e:
            logger.error(f"Unexpected error inserting vectors: {e}")
            raise QdrantOperationError(f"Unexpected error: {e}")
    
    async def search_vectors(self, query_vector: List[float], 
                           limit: int = 10,
                           score_threshold: Optional[float] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filters to apply
            
        Returns:
            List of search results
            
        Raises:
            QdrantConnectionError: If unable to connect to Qdrant
            QdrantOperationError: If search fails
        """
        if not await self.ensure_connected():
            raise QdrantConnectionError("Unable to connect to Qdrant service")
        
        if not self._collection_exists:
            raise QdrantOperationError("Collection does not exist")
        
        start_time = time.time()
        
        try:
            # Prepare search filters
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        # Multiple values - use "any" condition
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        # Single value - use exact match
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    search_filter = models.Filter(must=conditions)
            
            # Perform search
            search_result = self._client.search(
                collection_name=self.qdrant_config.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors unless needed
            )
            
            # Format results
            results = []
            for point in search_result:
                result = VectorSearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                    vector=point.vector if hasattr(point, 'vector') else None
                )
                results.append(result)
            
            # Update performance metrics
            self._update_metrics(time.time() - start_time)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except ResponseHandlingException as e:
            logger.error(f"Qdrant API error searching vectors: {e}")
            raise QdrantOperationError(f"Failed to search vectors: {e}")
        except Exception as e:
            logger.error(f"Unexpected error searching vectors: {e}")
            raise QdrantOperationError(f"Unexpected error: {e}")
    
    async def get_vector(self, point_id: str, with_vector: bool = False) -> Optional[VectorSearchResult]:
        """
        Get a specific vector by ID.
        
        Args:
            point_id: ID of the point to retrieve
            with_vector: Whether to include the vector data
            
        Returns:
            Vector search result or None if not found
        """
        if not await self.ensure_connected():
            return None
        
        try:
            points = self._client.retrieve(
                collection_name=self.qdrant_config.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=with_vector
            )
            
            if points:
                point = points[0]
                return VectorSearchResult(
                    id=str(point.id),
                    score=1.0,  # Not applicable for direct retrieval
                    payload=point.payload or {},
                    vector=point.vector if hasattr(point, 'vector') else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving vector {point_id}: {e}")
            return None
    
    async def delete_vectors(self, point_ids: List[str]) -> bool:
        """
        Delete vectors by their IDs.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            True if deletion successful
        """
        if not await self.ensure_connected():
            return False
        
        try:
            operation_result = self._client.delete(
                collection_name=self.qdrant_config.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                ),
                wait=True
            )
            
            logger.info(f"Deleted {len(point_ids)} vectors successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        if not await self.ensure_connected():
            return {}
        
        try:
            collection_info = self._client.get_collection(self.qdrant_config.collection_name)
            
            return {
                "name": self.qdrant_config.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def count_vectors(self, filter_conditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Count vectors in the collection with optional filtering.
        
        Args:
            filter_conditions: Optional filters to apply
            
        Returns:
            Number of vectors matching the conditions
        """
        if not await self.ensure_connected():
            return 0
        
        try:
            # Prepare search filters
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if conditions:
                    search_filter = models.Filter(must=conditions)
            
            # Count points
            count_result = self._client.count(
                collection_name=self.qdrant_config.collection_name,
                count_filter=search_filter,
                exact=False  # Use approximate counting for performance
            )
            
            return count_result.count
            
        except Exception as e:
            logger.error(f"Error counting vectors: {e}")
            return 0
    
    async def scroll_vectors(self, 
                           limit: int = 100,
                           offset: Optional[str] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None,
                           with_vectors: bool = False) -> Tuple[List[VectorSearchResult], Optional[str]]:
        """
        Scroll through vectors in the collection.
        
        Args:
            limit: Maximum number of results per page
            offset: Offset ID for pagination
            filter_conditions: Optional filters to apply
            with_vectors: Whether to include vector data
            
        Returns:
            Tuple of (results, next_offset)
        """
        if not await self.ensure_connected():
            return [], None
        
        try:
            # Prepare search filters
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if conditions:
                    search_filter = models.Filter(must=conditions)
            
            # Scroll points
            scroll_result = self._client.scroll(
                collection_name=self.qdrant_config.collection_name,
                limit=limit,
                offset=offset,
                scroll_filter=search_filter,
                with_payload=True,
                with_vectors=with_vectors
            )
            
            # Format results
            results = []
            for point in scroll_result[0]:  # Points are in first element
                result = VectorSearchResult(
                    id=str(point.id),
                    score=1.0,  # Not applicable for scrolling
                    payload=point.payload or {},
                    vector=point.vector if hasattr(point, 'vector') else None
                )
                results.append(result)
            
            next_offset = scroll_result[1]  # Next offset is in second element
            
            logger.info(f"Scrolled {len(results)} vectors")
            return results, next_offset
            
        except Exception as e:
            logger.error(f"Error scrolling vectors: {e}")
            return [], None
    
    def _update_metrics(self, operation_time: float):
        """Update performance metrics."""
        self._operation_count += 1
        self._total_latency += operation_time
        self._last_operation_time = operation_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the client.
        
        Returns:
            Dictionary with performance information
        """
        avg_latency = (self._total_latency / self._operation_count) if self._operation_count > 0 else 0
        
        return {
            "connected": self._connected,
            "collection_exists": self._collection_exists,
            "operation_count": self._operation_count,
            "average_latency_ms": round(avg_latency * 1000, 2),
            "last_operation_latency_ms": round(self._last_operation_time * 1000, 2),
            "retry_attempts": self._retry_attempts,
            "max_retries": self._max_retries,
            "collection_name": self.qdrant_config.collection_name,
            "server_url": self.qdrant_config.url
        }
    
    async def cleanup_old_vectors(self, 
                                days_old: int = 30,
                                dry_run: bool = True) -> int:
        """
        Clean up old vectors based on creation date.
        
        Args:
            days_old: Delete vectors older than this many days
            dry_run: If True, only count vectors that would be deleted
            
        Returns:
            Number of vectors cleaned up (or would be cleaned up if dry_run)
        """
        if not await self.ensure_connected():
            return 0
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            cutoff_date = cutoff_date.replace(
                day=cutoff_date.day - days_old
            ).isoformat()
            
            # Create filter for old vectors
            old_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.DatetimeRange(
                            lt=cutoff_date
                        )
                    )
                ]
            )
            
            if dry_run:
                # Just count the vectors
                count_result = self._client.count(
                    collection_name=self.qdrant_config.collection_name,
                    count_filter=old_filter,
                    exact=True
                )
                
                logger.info(f"Found {count_result.count} vectors older than {days_old} days")
                return count_result.count
            else:
                # Delete old vectors
                delete_result = self._client.delete(
                    collection_name=self.qdrant_config.collection_name,
                    points_selector=models.FilterSelector(filter=old_filter),
                    wait=True
                )
                
                logger.info(f"Cleaned up old vectors older than {days_old} days")
                return delete_result.operation_id  # This might not be the count
                
        except Exception as e:
            logger.error(f"Error cleaning up old vectors: {e}")
            return 0


# Global Qdrant client instance
qdrant_client = QdrantCloudWrapper()