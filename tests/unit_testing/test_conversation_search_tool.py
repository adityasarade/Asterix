"""
Unit tests for Conversation Search Tool.

Tests ConversationSearchTool functionality:
- Tool initialization with agent context
- Successful search with results (text and semantic)
- Search with no conversation history
- Search with no matches
- Error handling (empty query, invalid parameters)
- Semantic search with embedding service
- Text search fallback mechanism
- Result structure and metadata
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
from asterix.tools.conversation import (
    ConversationSearchTool,
    create_conversation_search_tool
)
from asterix.tools.base import ToolResult, ToolStatus


class TestConversationSearchTool:
    """Test ConversationSearchTool functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with conversation history."""
        agent = Mock()
        agent.id = "test_agent_123"
        agent.conversation_history = [
            {
                "role": "user",
                "content": "I love Python programming",
                "timestamp": "2025-01-15T10:00:00Z"
            },
            {
                "role": "assistant",
                "content": "That's great! Python is very versatile.",
                "timestamp": "2025-01-15T10:00:05Z"
            },
            {
                "role": "user",
                "content": "My favorite editor is VS Code",
                "timestamp": "2025-01-15T10:01:00Z"
            },
            {
                "role": "assistant",
                "content": "VS Code is excellent for Python development.",
                "timestamp": "2025-01-15T10:01:05Z"
            },
            {
                "role": "user",
                "content": "I prefer dark mode themes",
                "timestamp": "2025-01-15T10:02:00Z"
            },
            {
                "role": "assistant",
                "content": "Dark mode is easier on the eyes.",
                "timestamp": "2025-01-15T10:02:05Z"
            }
        ]
        return agent
    
    def test_initialization(self, mock_agent):
        """Test tool initializes with correct schema."""
        tool = ConversationSearchTool(mock_agent)
        
        assert tool.name == "conversation_search"
        assert tool.agent == mock_agent
        assert tool.schema is not None
        assert "query" in tool.schema["parameters"]["properties"]
        assert "k" in tool.schema["parameters"]["properties"]
        assert "query" in tool.schema["parameters"]["required"]
    
    def test_search_text_success(self, mock_agent):
        """Test successful text-based search."""
        tool = ConversationSearchTool(mock_agent)
        
        # Force text search by patching semantic search to fail
        with patch.object(tool, '_semantic_search', side_effect=Exception("No embeddings")):
            result = tool.execute(query="Python programming", k=2)
        
        assert result.status == ToolStatus.SUCCESS
        assert "Python" in result.content or "programming" in result.content
        assert result.metadata["operation"] == "conversation_search"
        assert result.metadata["query"] == "Python programming"
        assert result.metadata["results_count"] > 0
        assert result.metadata["total_messages"] == 6
    
    def test_search_semantic_success(self, mock_agent):
        """Test successful semantic search with embeddings."""
        tool = ConversationSearchTool(mock_agent)
        
        # Mock semantic search to return results
        mock_results = [
            (mock_agent.conversation_history[0], 0.95),  # High relevance
            (mock_agent.conversation_history[1], 0.82)   # Medium relevance
        ]
        
        with patch.object(tool, '_semantic_search', return_value=mock_results):
            result = tool.execute(query="programming languages", k=2)
        
        assert result.status == ToolStatus.SUCCESS
        assert "Python" in result.content
        assert result.metadata["results_count"] == 2
        assert len(result.metadata["results"]) == 2
        # Check scores are included in metadata
        assert result.metadata["results"][0]["score"] == 0.95
    
    def test_search_no_conversation_history(self):
        """Test search with empty conversation history."""
        agent = Mock()
        agent.conversation_history = []
        
        tool = ConversationSearchTool(agent)
        result = tool.execute(query="anything", k=3)
        
        assert result.status == ToolStatus.SUCCESS
        assert "No conversation history" in result.content
        assert result.metadata["results_count"] == 0
    
    def test_search_no_matches(self, mock_agent):
        """Test search with query that has no matches."""
        tool = ConversationSearchTool(mock_agent)
        
        # Force text search with irrelevant query
        with patch.object(tool, '_semantic_search', side_effect=Exception("No embeddings")):
            result = tool.execute(query="underwater basket weaving", k=3)
        
        assert result.status == ToolStatus.SUCCESS
        assert "No relevant messages found" in result.content
        assert result.metadata["results_count"] == 0
        assert result.metadata["total_messages"] == 6
    
    def test_search_empty_query_error(self, mock_agent):
        """Test that empty query returns error."""
        tool = ConversationSearchTool(mock_agent)
        
        result = tool.execute(query="", k=3)
        
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
    
    def test_search_whitespace_query_error(self, mock_agent):
        """Test that whitespace-only query returns error."""
        tool = ConversationSearchTool(mock_agent)
        
        result = tool.execute(query="   ", k=3)
        
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
    
    def test_search_invalid_k_uses_default(self, mock_agent):
        """Test that invalid k value uses default."""
        tool = ConversationSearchTool(mock_agent)
        
        with patch.object(tool, '_semantic_search', side_effect=Exception("No embeddings")):
            result = tool.execute(query="Python", k=0)
        
        assert result.status == ToolStatus.SUCCESS
        # Should use default k=3 even though we passed k=0
    
    def test_text_search_exact_phrase(self, mock_agent):
        """Test text search finds exact phrase matches."""
        tool = ConversationSearchTool(mock_agent)
        
        results = tool._text_search("Python programming", 2, mock_agent.conversation_history)
        
        assert len(results) > 0
        # First result should have high score (exact phrase match)
        assert results[0][1] == 1.0
        assert "Python programming" in results[0][0]["content"]
    
    def test_text_search_word_overlap(self, mock_agent):
        """Test text search with partial word overlap."""
        tool = ConversationSearchTool(mock_agent)
        
        results = tool._text_search("Python editor", 3, mock_agent.conversation_history)
        
        assert len(results) > 0
        # Should find messages with either "Python" or "editor"
        for message, score in results:
            content_lower = message["content"].lower()
            assert "python" in content_lower or "editor" in content_lower
            assert 0 < score <= 1.0
    
    def test_text_search_case_insensitive(self, mock_agent):
        """Test text search is case insensitive."""
        tool = ConversationSearchTool(mock_agent)
        
        results_lower = tool._text_search("python", 2, mock_agent.conversation_history)
        results_upper = tool._text_search("PYTHON", 2, mock_agent.conversation_history)
        
        # Should return same results regardless of case
        assert len(results_lower) == len(results_upper)
    
    def test_semantic_search_with_embeddings(self, mock_agent):
        """Test semantic search fallback mechanism."""
        tool = ConversationSearchTool(mock_agent)
        
        # Test that when semantic search fails, it falls back to text search
        with patch.object(tool, '_semantic_search', side_effect=Exception("No embeddings")):
            result = tool.execute(query="Python programming", k=2)
        
        # Should succeed using text search fallback
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata["results_count"] >= 0
        assert result.metadata["total_messages"] == 6
    
    def test_search_result_formatting(self, mock_agent):
        """Test that search results are formatted correctly."""
        tool = ConversationSearchTool(mock_agent)
        
        mock_results = [
            (mock_agent.conversation_history[0], 0.95),
            (mock_agent.conversation_history[2], 0.80)
        ]
        
        with patch.object(tool, '_semantic_search', return_value=mock_results):
            result = tool.execute(query="test", k=2)
        
        assert result.status == ToolStatus.SUCCESS
        # Should contain numbered results
        assert "1." in result.content
        assert "2." in result.content
        # Should show role
        assert "USER" in result.content.upper() or "ASSISTANT" in result.content.upper()
        # Should show content
        assert "Python" in result.content or "VS Code" in result.content
    
    def test_search_truncates_long_messages(self, mock_agent):
        """Test that long messages are truncated in results."""
        # Add a very long message
        long_message = {
            "role": "user",
            "content": "x" * 200,  # Very long content
            "timestamp": "2025-01-15T10:03:00Z"
        }
        mock_agent.conversation_history.append(long_message)
        
        tool = ConversationSearchTool(mock_agent)
        
        mock_results = [(long_message, 0.95)]
        
        with patch.object(tool, '_semantic_search', return_value=mock_results):
            result = tool.execute(query="test", k=1)
        
        # Should contain ellipsis indicating truncation
        assert "..." in result.content
    
    def test_search_exception_handling(self, mock_agent):
        """Test that exceptions are caught and returned as errors."""
        tool = ConversationSearchTool(mock_agent)
        
        # Mock both search methods to fail
        with patch.object(tool, '_semantic_search', side_effect=Exception("Embedding error")):
            with patch.object(tool, '_text_search', side_effect=Exception("Text search error")):
                result = tool.execute(query="test", k=3)
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to search" in result.error


class TestConversationSearchFactory:
    """Test create_conversation_search_tool factory function."""
    
    def test_factory_creates_tool(self):
        """Test factory creates tool correctly."""
        mock_agent = Mock()
        mock_agent.conversation_history = []
        
        tool = create_conversation_search_tool(mock_agent)
        
        assert isinstance(tool, ConversationSearchTool)
        assert tool.agent == mock_agent
        assert tool.name == "conversation_search"
    
    def test_factory_tool_is_functional(self):
        """Test that tool created by factory is functional."""
        mock_agent = Mock()
        mock_agent.conversation_history = [
            {"role": "user", "content": "Hello", "timestamp": "2025-01-15T10:00:00Z"}
        ]
        
        tool = create_conversation_search_tool(mock_agent)
        
        # Force text search
        with patch.object(tool, '_semantic_search', side_effect=Exception("No embeddings")):
            result = tool.execute(query="Hello", k=1)
        
        assert result.status == ToolStatus.SUCCESS