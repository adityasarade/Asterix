"""
Asterix Archival Memory Tools

Tools for long-term memory storage and retrieval using Qdrant:
- archival_memory_insert: Store information in vector database
- archival_memory_search: Retrieve relevant memories semantically

These tools enable agents to remember information beyond their context window.
"""

import logging
import asyncio
from typing import TYPE_CHECKING, Dict, Any, List
from datetime import datetime, timezone

from .base import Tool, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# Archival Memory Insert Tool
# ============================================================================

class ArchivalMemoryInsertTool(Tool):
    """
    Tool for inserting memories into long-term storage (Qdrant).
    
    When memory blocks get full or the agent wants to remember something
    for later retrieval, this tool stores it in Qdrant with semantic embeddings.
    
    The agent can later search these memories using archival_memory_search.
    
    Schema:
        {
            "name": "archival_memory_insert",
            "description": "Store information in long-term memory",
            "parameters": {
                "content": "Information to store",
                "summary": "Optional short summary (auto-generated if omitted)",
                "importance": "Importance score 0.0-1.0 (optional)"
            }
        }
    
    Example usage by agent:
        >>> # Agent wants to remember user preferences long-term
        >>> archival_memory_insert(
        ...     content="User prefers dark mode, uses VS Code, likes Python",
        ...     summary="User preferences",
        ...     importance=0.8
        ... )
        >>> # Returns: "Successfully stored memory in archival storage"
    """
    
    def __init__(self, agent: 'Agent'):
        """
        Initialize the archival_memory_insert tool.
        
        Args:
            agent: Agent instance that owns this tool
        """
        self.agent = agent
        
        # Define tool schema for LLM
        schema = {
            "name": "archival_memory_insert",
            "description": "Store information in long-term archival memory for later retrieval. Use this for facts, preferences, or context you want to remember permanently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store in archival memory. Be specific and detailed."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional short summary of the content (will be auto-generated if omitted)"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Optional importance score between 0.0 (low) and 1.0 (high). Defaults to 0.5."
                    }
                },
                "required": ["content"]
            }
        }
        
        super().__init__(
            name="archival_memory_insert",
            description="Store information in long-term archival memory",
            schema=schema
        )
    
    def execute(self, content: str, summary: str = None, importance: float = 0.5) -> ToolResult:
        """
        Execute the archival_memory_insert operation.
        
        Args:
            content: Information to store
            summary: Optional short summary (auto-generated if None)
            importance: Importance score 0.0-1.0
            
        Returns:
            ToolResult with operation status
        """
        try:
            # Validate content
            if not content or not content.strip():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Content cannot be empty"
                )
            
            # Validate importance score
            if not 0.0 <= importance <= 1.0:
                logger.warning(f"Importance {importance} out of range, clamping to [0.0, 1.0]")
                importance = max(0.0, min(1.0, importance))
            
            # Import services (lazy import to avoid circular dependencies)
            from ..storage.qdrant_client import qdrant_client, ArchivalRecord
            from ..core.embeddings import embedding_service
            from ..core.llm_manager import llm_manager
            
            # Generate summary if not provided
            if not summary or not summary.strip():
                logger.debug("No summary provided, generating automatically")
                try:
                    # Use LLM to generate a concise summary
                    summary_result = asyncio.run(
                        llm_manager.summarize_text(content, max_tokens=50)
                    )
                    summary = summary_result.content.strip()
                    logger.debug(f"Generated summary: {summary[:100]}")
                except Exception as e:
                    logger.warning(f"Failed to generate summary: {e}, using truncated content")
                    # Fallback: use first 100 chars
                    summary = content[:100] + "..." if len(content) > 100 else content
            
            # Generate embedding for the content
            logger.debug("Generating embedding for archival content")
            embedding_result = asyncio.run(
                embedding_service.embed_texts([content])
            )
            
            if not embedding_result.embeddings:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Failed to generate embedding for content"
                )
            
            embedding = embedding_result.embeddings[0]
            
            # Prepare metadata
            metadata = {
                "agent_id": self.agent.id,
                "source": "agent_insertion",
                "type": "manual_memory",
                "importance": importance,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create archival record
            record = ArchivalRecord(
                text=content.strip(),
                summary=summary.strip(),
                metadata=metadata,
                embedding=embedding
            )
            
            # Insert into Qdrant
            logger.debug("Inserting memory into Qdrant")
            point_ids = asyncio.run(
                qdrant_client.insert_vectors([record])
            )
            
            if not point_ids:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Failed to insert memory into archival storage"
                )
            
            # Success response
            result_message = f"Successfully stored memory in archival storage with ID: {point_ids[0]}"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=result_message,
                metadata={
                    "operation": "archival_insert",
                    "memory_id": point_ids[0],
                    "summary": summary.strip(),
                    "importance": importance,
                    "embedding_provider": embedding_result.provider,
                    "embedding_dimensions": embedding_result.dimensions,
                    "content_length": len(content)
                }
            )
            
        except Exception as e:
            logger.error(f"archival_memory_insert failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Failed to store memory in archival storage: {str(e)}"
            )


# ============================================================================
# Archival Memory Search Tool
# ============================================================================

class ArchivalMemorySearchTool(Tool):
    """
    Tool for searching long-term memory storage (Qdrant).
    
    Performs semantic search over all memories stored in Qdrant.
    Returns the most relevant memories based on the query.
    
    Schema:
        {
            "name": "archival_memory_search",
            "description": "Search long-term memory",
            "parameters": {
                "query": "Search query",
                "k": "Number of results (default 5)",
                "score_threshold": "Minimum similarity score (optional)"
            }
        }
    
    Example usage by agent:
        >>> # Agent wants to recall user preferences
        >>> archival_memory_search(
        ...     query="user editor preferences",
        ...     k=3
        ... )
        >>> # Returns: List of relevant memories with scores
    """
    
    def __init__(self, agent: 'Agent'):
        """
        Initialize the archival_memory_search tool.
        
        Args:
            agent: Agent instance that owns this tool
        """
        self.agent = agent
        
        # Get default search parameters from config
        default_k = agent.config.memory.retrieval_k
        default_threshold = agent.config.memory.score_threshold
        
        # Define tool schema for LLM
        schema = {
            "name": "archival_memory_search",
            "description": "Search long-term archival memory to retrieve relevant information. Use this to recall facts, preferences, or context from previous conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what you want to remember. Be specific."
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {default_k})"
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": f"Minimum similarity score 0.0-1.0 (default: {default_threshold}). Higher = stricter matches."
                    }
                },
                "required": ["query"]
            }
        }
        
        super().__init__(
            name="archival_memory_search",
            description="Search long-term archival memory",
            schema=schema
        )
    
    def execute(self, query: str, k: int = None, score_threshold: float = None) -> ToolResult:
        """
        Execute the archival_memory_search operation.
        
        Args:
            query: Search query
            k: Number of results to return (uses config default if None)
            score_threshold: Minimum similarity score (uses config default if None)
            
        Returns:
            ToolResult with search results
        """
        try:
            # Validate query
            if not query or not query.strip():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Search query cannot be empty"
                )
            
            # Use config defaults if not provided
            if k is None:
                k = self.agent.config.memory.retrieval_k
            if score_threshold is None:
                score_threshold = self.agent.config.memory.score_threshold
            
            # Validate parameters
            if k <= 0:
                k = 5
                logger.warning("Invalid k value, using default: 5")
            
            if not 0.0 <= score_threshold <= 1.0:
                score_threshold = 0.7
                logger.warning("Invalid score_threshold, using default: 0.7")
            
            # Import services
            from ..storage.qdrant_client import qdrant_client
            from ..core.embeddings import embedding_service
            
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:100]}")
            embedding_result = asyncio.run(
                embedding_service.embed_texts([query.strip()])
            )
            
            if not embedding_result.embeddings:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Failed to generate query embedding"
                )
            
            query_embedding = embedding_result.embeddings[0]
            
            # Search Qdrant with agent_id filter
            logger.debug(f"Searching Qdrant with k={k}, threshold={score_threshold}")
            filter_conditions = {"agent_id": self.agent.id}
            
            search_results = asyncio.run(
                qdrant_client.search_vectors(
                    query_vector=query_embedding,
                    limit=k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions
                )
            )
            
            # Format results for agent
            if not search_results:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content="No relevant memories found in archival storage.",
                    metadata={
                        "operation": "archival_search",
                        "query": query.strip(),
                        "results_count": 0,
                        "k": k,
                        "score_threshold": score_threshold
                    }
                )
            
            # Build formatted response
            result_lines = [f"Found {len(search_results)} relevant memories:\n"]
            
            for i, result in enumerate(search_results, 1):
                summary = result.payload.get("summary", "No summary")
                importance = result.payload.get("importance", 0.5)
                created_at = result.payload.get("created_at", "Unknown")
                
                result_lines.append(
                    f"{i}. [{result.score:.3f}] {summary} "
                    f"(importance: {importance:.2f}, stored: {created_at[:10]})"
                )
            
            formatted_results = "\n".join(result_lines)
            
            # Prepare detailed metadata
            results_metadata = []
            for result in search_results:
                results_metadata.append({
                    "id": result.id,
                    "score": result.score,
                    "summary": result.payload.get("summary"),
                    "text": result.payload.get("text"),
                    "importance": result.payload.get("importance"),
                    "created_at": result.payload.get("created_at")
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=formatted_results,
                metadata={
                    "operation": "archival_search",
                    "query": query.strip(),
                    "results_count": len(search_results),
                    "k": k,
                    "score_threshold": score_threshold,
                    "results": results_metadata,
                    "embedding_provider": embedding_result.provider
                }
            )
            
        except Exception as e:
            logger.error(f"archival_memory_search failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Failed to search archival memory: {str(e)}"
            )


# ============================================================================
# Factory Function
# ============================================================================

def create_archival_memory_tools(agent: 'Agent') -> Dict[str, Tool]:
    """
    Create both archival memory tools for an agent.
    
    Args:
        agent: Agent instance
        
    Returns:
        Dictionary mapping tool names to Tool instances
    
    Example:
        >>> tools = create_archival_memory_tools(agent)
        >>> agent.register_tool(tools["archival_memory_insert"])
        >>> agent.register_tool(tools["archival_memory_search"])
    """
    return {
        "archival_memory_insert": ArchivalMemoryInsertTool(agent),
        "archival_memory_search": ArchivalMemorySearchTool(agent)
    }