"""
Asterix Conversation Search Tool

Tool for searching through conversation history:
- conversation_search: Find relevant past messages semantically

Allows agents to reference what was discussed earlier in the conversation.
"""

import logging
import asyncio
from typing import TYPE_CHECKING, Dict, Any, List
from datetime import datetime

from .base import Tool, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# Conversation Search Tool
# ============================================================================

class ConversationSearchTool(Tool):
    """
    Tool for searching through conversation history.
    
    Performs semantic search over the agent's conversation_history to find
    relevant past messages. Useful when the agent needs to recall what was
    discussed earlier in the conversation.
    
    Unlike archival_memory_search (which searches long-term Qdrant storage),
    this searches the current session's conversation history.
    
    Schema:
        {
            "name": "conversation_search",
            "description": "Search conversation history",
            "parameters": {
                "query": "What to search for in the conversation",
                "k": "Number of results to return (default 3)"
            }
        }
    
    Example usage by agent:
        >>> # User references something mentioned earlier
        >>> # Agent wants to recall the context
        >>> conversation_search(
        ...     query="coffee preferences",
        ...     k=3
        ... )
        >>> # Returns: Relevant past messages from this conversation
    """
    
    def __init__(self, agent: 'Agent'):
        """
        Initialize the conversation_search tool.
        
        Args:
            agent: Agent instance that owns this tool
        """
        self.agent = agent
        
        # Define tool schema for LLM
        schema = {
            "name": "conversation_search",
            "description": "Search through this conversation's history to find relevant past messages. Use this to recall what was discussed earlier in the current session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what you want to find in the conversation history"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)"
                    }
                },
                "required": ["query"]
            }
        }
        
        super().__init__(
            name="conversation_search",
            description="Search conversation history",
            schema=schema
        )
    
    def execute(self, query: str, k: int = 3) -> ToolResult:
        """
        Execute the conversation_search operation.
        
        Args:
            query: Search query
            k: Number of results to return
            
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
            
            # Validate k
            if k <= 0:
                k = 3
                logger.warning("Invalid k value, using default: 3")
            
            # Get conversation history
            conversation_history = self.agent.conversation_history
            
            if not conversation_history:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content="No conversation history available yet.",
                    metadata={
                        "operation": "conversation_search",
                        "query": query.strip(),
                        "results_count": 0
                    }
                )
            
            # Import embedding service for semantic search
            from ..core.embeddings import embedding_service
            
            # Two search strategies:
            # 1. Semantic search (better quality, requires embeddings)
            # 2. Simple text search (fallback)
            
            # Try semantic search first
            try:
                results = self._semantic_search(query.strip(), k, conversation_history)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, falling back to text search")
                results = self._text_search(query.strip(), k, conversation_history)
            
            if not results:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content=f"No relevant messages found for query: '{query}'",
                    metadata={
                        "operation": "conversation_search",
                        "query": query.strip(),
                        "results_count": 0,
                        "total_messages": len(conversation_history)
                    }
                )
            
            # Format results for agent
            result_lines = [f"Found {len(results)} relevant messages:\n"]
            
            for i, (message, score) in enumerate(results, 1):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                # Truncate long messages
                display_content = content[:150] + "..." if len(content) > 150 else content
                
                # Format timestamp if available
                time_str = ""
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = f" at {dt.strftime('%H:%M')}"
                    except:
                        pass
                
                result_lines.append(
                    f"{i}. [{score:.3f}] {role.upper()}{time_str}: {display_content}"
                )
            
            formatted_results = "\n".join(result_lines)
            
            # Prepare metadata
            results_metadata = []
            for message, score in results:
                results_metadata.append({
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp"),
                    "score": score
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=formatted_results,
                metadata={
                    "operation": "conversation_search",
                    "query": query.strip(),
                    "results_count": len(results),
                    "total_messages": len(conversation_history),
                    "results": results_metadata
                }
            )
            
        except Exception as e:
            logger.error(f"conversation_search failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Failed to search conversation history: {str(e)}"
            )
    
    def _semantic_search(self, query: str, k: int, conversation_history: List[Dict[str, Any]]) -> List[tuple]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            k: Number of results
            conversation_history: List of conversation messages
            
        Returns:
            List of (message, score) tuples
        """
        from ..core.embeddings import embedding_service
        import numpy as np
        
        # Generate query embedding
        query_result = asyncio.run(
            embedding_service.embed_texts([query])
        )
        
        if not query_result.embeddings:
            raise ValueError("Failed to generate query embedding")
        
        query_embedding = np.array(query_result.embeddings[0])
        
        # Generate embeddings for all messages
        message_texts = []
        message_indices = []
        
        for i, message in enumerate(conversation_history):
            content = message.get("content", "")
            if content.strip():  # Only process non-empty messages
                message_texts.append(content)
                message_indices.append(i)
        
        if not message_texts:
            return []
        
        # Batch embed all messages
        messages_result = asyncio.run(
            embedding_service.embed_texts(message_texts)
        )
        
        if not messages_result.embeddings:
            raise ValueError("Failed to generate message embeddings")
        
        message_embeddings = np.array(messages_result.embeddings)
        
        # Compute cosine similarities
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        message_norms = message_embeddings / np.linalg.norm(message_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(message_norms, query_norm)
        
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # Descending order
        
        results = []
        for idx in top_k_indices:
            message_idx = message_indices[idx]
            message = conversation_history[message_idx]
            score = float(similarities[idx])
            results.append((message, score))
        
        return results
    
    def _text_search(self, query: str, k: int, conversation_history: List[Dict[str, Any]]) -> List[tuple]:
        """
        Perform simple text-based search (fallback).
        
        Args:
            query: Search query
            k: Number of results
            conversation_history: List of conversation messages
            
        Returns:
            List of (message, score) tuples
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_messages = []
        
        for message in conversation_history:
            content = message.get("content", "")
            if not content.strip():
                continue
            
            content_lower = content.lower()
            
            # Simple scoring:
            # - Exact phrase match: high score
            # - Word overlap: medium score
            # - No match: zero score
            
            if query_lower in content_lower:
                score = 1.0  # Exact phrase match
            else:
                content_words = set(content_lower.split())
                overlap = query_words & content_words
                if overlap:
                    score = len(overlap) / len(query_words)  # Proportion of query words found
                else:
                    score = 0.0
            
            if score > 0:
                scored_messages.append((message, score))
        
        # Sort by score and return top k
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return scored_messages[:k]


# ============================================================================
# Factory Function
# ============================================================================

def create_conversation_search_tool(agent: 'Agent') -> Tool:
    """
    Create the conversation search tool for an agent.
    
    Args:
        agent: Agent instance
        
    Returns:
        ConversationSearchTool instance
    
    Example:
        >>> tool = create_conversation_search_tool(agent)
        >>> agent.register_tool(tool)
    """
    return ConversationSearchTool(agent)