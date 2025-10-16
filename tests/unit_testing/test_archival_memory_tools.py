"""
Unit tests for Archival Memory Tools.

Tests ArchivalMemoryInsertTool and ArchivalMemorySearchTool functionality:
- Tool initialization with agent context
- Successful insert operations with Qdrant
- Successful search operations
- Error handling (empty content, invalid parameters)
- Edge cases (importance clamping, summary generation)
- Result structure and metadata
- Integration with embedding service and Qdrant
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from collections import namedtuple
from asterix.tools.archival import (
    ArchivalMemoryInsertTool,
    ArchivalMemorySearchTool,
    create_archival_memory_tools
)
from asterix.tools.base import ToolResult, ToolStatus


class TestArchivalMemoryInsertTool:
    """Test ArchivalMemoryInsertTool functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.id = "test_agent_123"
        agent.config = Mock()
        agent.config.memory = Mock()
        return agent
    
    def test_initialization(self, mock_agent):
        """Test tool initializes with correct schema."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        assert tool.name == "archival_memory_insert"
        assert tool.agent == mock_agent
        assert tool.schema is not None
        assert "content" in tool.schema["parameters"]["properties"]
        assert "summary" in tool.schema["parameters"]["properties"]
        assert "importance" in tool.schema["parameters"]["properties"]
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_insert_success(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test successful memory insertion."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Mock embedding generation
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2, 0.3]]
        mock_embed_result.provider = "openai"
        mock_embed_result.dimensions = 1536
        mock_asyncio.side_effect = [
            mock_embed_result,  # First call: embedding
            ["memory_id_123"]   # Second call: insert
        ]
        
        result = tool.execute(
            content="Python is great for AI",
            summary="Programming preference",
            importance=0.8
        )
        
        assert result.status == ToolStatus.SUCCESS
        assert "Successfully stored" in result.content
        assert "memory_id_123" in result.content
        assert result.metadata["memory_id"] == "memory_id_123"
        assert result.metadata["importance"] == 0.8
        assert result.metadata["summary"] == "Programming preference"
        assert result.metadata["embedding_provider"] == "openai"
        assert result.metadata["embedding_dimensions"] == 1536
    
    def test_insert_empty_content(self, mock_agent):
        """Test insert with empty content returns error."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Empty string
        result = tool.execute(content="", summary="test")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
        
        # Whitespace only
        result = tool.execute(content="   \n\t  ", summary="test")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
    
    def test_insert_importance_clamping(self, mock_agent):
        """Test importance score is clamped to [0.0, 1.0]."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        with patch('asterix.tools.archival.asyncio.run') as mock_asyncio, \
             patch('asterix.storage.qdrant_client.qdrant_client'), \
             patch('asterix.core.embeddings.embedding_service'):
            
            # Mock successful responses
            mock_embed_result = Mock()
            mock_embed_result.embeddings = [[0.1, 0.2]]
            mock_embed_result.provider = "openai"
            mock_embed_result.dimensions = 1536
            mock_asyncio.side_effect = [mock_embed_result, ["id_1"]]
            
            # Test value > 1.0
            result = tool.execute(content="test", importance=1.5)
            assert result.status == ToolStatus.SUCCESS
            assert result.metadata["importance"] == 1.0  # Clamped
            
            # Reset mock
            mock_asyncio.side_effect = [mock_embed_result, ["id_2"]]
            
            # Test value < 0.0
            result = tool.execute(content="test", importance=-0.5)
            assert result.status == ToolStatus.SUCCESS
            assert result.metadata["importance"] == 0.0  # Clamped
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_insert_auto_generates_summary(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test summary is auto-generated when not provided."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Mock successful responses
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1]]
        mock_embed_result.provider = "openai"
        mock_embed_result.dimensions = 1536
        mock_asyncio.side_effect = [mock_embed_result, ["id_1"]]
        
        # Short content - should use full content
        result = tool.execute(content="Short text", summary=None)
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata["summary"] == "Short text"
        
        # Reset mock
        mock_asyncio.side_effect = [mock_embed_result, ["id_2"]]
        
        # Long content - should truncate
        long_content = "x" * 150
        result = tool.execute(content=long_content, summary=None)
        assert result.status == ToolStatus.SUCCESS
        assert len(result.metadata["summary"]) < len(long_content)
        assert result.metadata["summary"].endswith("...")
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.core.embeddings.embedding_service')
    def test_insert_embedding_failure(self, mock_embed_service, mock_asyncio, mock_agent):
        """Test handling when embedding generation fails."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Mock embedding failure
        mock_embed_result = Mock()
        mock_embed_result.embeddings = []  # Empty embeddings
        mock_asyncio.return_value = mock_embed_result
        
        result = tool.execute(content="test content")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to generate embedding" in result.error
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_insert_qdrant_failure(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test handling when Qdrant insertion fails."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Mock embedding success but Qdrant failure
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1]]
        mock_embed_result.provider = "openai"
        mock_embed_result.dimensions = 1536
        mock_asyncio.side_effect = [
            mock_embed_result,  # Embedding succeeds
            []                  # Qdrant returns empty list (failure)
        ]
        
        result = tool.execute(content="test content")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to insert memory" in result.error
    
    @patch('asterix.tools.archival.asyncio.run')
    def test_insert_exception_handling(self, mock_asyncio, mock_agent):
        """Test that exceptions are caught and returned as errors."""
        tool = ArchivalMemoryInsertTool(mock_agent)
        
        # Mock exception during embedding
        mock_asyncio.side_effect = RuntimeError("Connection timeout")
        
        result = tool.execute(content="test content")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to store memory" in result.error
        assert "Connection timeout" in result.error


class TestArchivalMemorySearchTool:
    """Test ArchivalMemorySearchTool functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.id = "test_agent_456"
        agent.config = Mock()
        agent.config.memory = Mock()
        agent.config.memory.retrieval_k = 5
        agent.config.memory.score_threshold = 0.7
        return agent
    
    def test_initialization(self, mock_agent):
        """Test tool initializes with correct schema."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        assert tool.name == "archival_memory_search"
        assert tool.agent == mock_agent
        assert tool.schema is not None
        assert "query" in tool.schema["parameters"]["properties"]
        assert "k" in tool.schema["parameters"]["properties"]
        assert "score_threshold" in tool.schema["parameters"]["properties"]
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_search_success(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test successful memory search."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Mock embedding generation
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2]]
        
        # Mock search results using namedtuple for proper attribute access
        SearchResult = namedtuple('SearchResult', ['text', 'summary', 'score'])
        mock_search_results = [
            SearchResult(text="Python is great", summary="Programming", score=0.95),
            SearchResult(text="AI development", summary="AI", score=0.85)
        ]
        
        mock_asyncio.side_effect = [
            mock_embed_result,      # Embedding generation
            mock_search_results     # Search results
        ]
        
        result = tool.execute(query="Python programming", k=5)
        
        assert result.status == ToolStatus.SUCCESS
        assert "Found 2 memories" in result.content
        assert "Python is great" in result.content
        assert result.metadata["results_count"] == 2
        assert len(result.metadata["results"]) == 2
        assert result.metadata["results"][0]["score"] == 0.95
    
    def test_search_empty_query(self, mock_agent):
        """Test search with empty query returns error."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Empty string
        result = tool.execute(query="")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
        
        # Whitespace only
        result = tool.execute(query="   \n\t  ")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
    
    def test_search_invalid_k(self, mock_agent):
        """Test search with invalid k parameter logs warning but continues."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        with patch('asterix.tools.archival.asyncio.run') as mock_asyncio, \
             patch('asterix.storage.qdrant_client.qdrant_client'), \
             patch('asterix.core.embeddings.embedding_service'):
            
            # Mock responses
            mock_embed_result = Mock()
            mock_embed_result.embeddings = [[0.1]]
            mock_asyncio.side_effect = [mock_embed_result, []]
            
            # k <= 0 logs warning but uses default (doesn't error)
            result = tool.execute(query="test", k=0)
            assert result.status == ToolStatus.SUCCESS  # Uses default k, doesn't error
            
            # Reset for second test
            mock_asyncio.side_effect = [mock_embed_result, []]
            result = tool.execute(query="test", k=-5)
            assert result.status == ToolStatus.SUCCESS  # Uses default k, doesn't error
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_search_no_results(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test search when no results found."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Mock embedding generation
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1, 0.2]]
        
        # Mock empty search results
        mock_asyncio.side_effect = [
            mock_embed_result,  # Embedding generation
            []                  # No search results
        ]
        
        result = tool.execute(query="nonexistent topic")
        
        assert result.status == ToolStatus.SUCCESS
        assert "No relevant memories found" in result.content  # Actual message
        assert result.metadata["results_count"] == 0
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_search_with_score_threshold(self, mock_embed_service, mock_qdrant, mock_asyncio, mock_agent):
        """Test search filters results by score threshold."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Mock embedding generation
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1]]
        
        # Mock search results with varying scores using namedtuple
        SearchResult = namedtuple('SearchResult', ['text', 'summary', 'score'])
        mock_search_results = [
            SearchResult(text="High score", summary="High", score=0.95),
            SearchResult(text="Medium score", summary="Medium", score=0.75),
            SearchResult(text="Low score", summary="Low", score=0.5)
        ]
        
        mock_asyncio.side_effect = [
            mock_embed_result,
            mock_search_results
        ]
        
        # Search with threshold 0.7
        result = tool.execute(query="test", k=10, score_threshold=0.7)
        
        assert result.status == ToolStatus.SUCCESS
        # Should only include results >= 0.7
        assert result.metadata["results_count"] == 2
        assert "High score" in result.content
        assert "Medium score" in result.content
        assert "Low score" not in result.content
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.core.embeddings.embedding_service')
    def test_search_embedding_failure(self, mock_embed_service, mock_asyncio, mock_agent):
        """Test handling when embedding generation fails."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Mock embedding failure
        mock_embed_result = Mock()
        mock_embed_result.embeddings = []  # Empty embeddings
        mock_asyncio.return_value = mock_embed_result
        
        result = tool.execute(query="test query")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to generate query embedding" in result.error  # Actual message
    
    @patch('asterix.tools.archival.asyncio.run')
    def test_search_exception_handling(self, mock_asyncio, mock_agent):
        """Test that exceptions are caught and returned as errors."""
        tool = ArchivalMemorySearchTool(mock_agent)
        
        # Mock exception during search
        mock_asyncio.side_effect = RuntimeError("Database connection failed")
        
        result = tool.execute(query="test query")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to search" in result.error
        assert "Database connection failed" in result.error


class TestArchivalMemoryToolsFactory:
    """Test create_archival_memory_tools factory function."""
    
    def test_create_archival_memory_tools(self):
        """Test factory creates both tools correctly."""
        mock_agent = Mock()
        mock_agent.id = "factory_test"
        mock_agent.config = Mock()
        mock_agent.config.memory = Mock()
        mock_agent.config.memory.retrieval_k = 5
        mock_agent.config.memory.score_threshold = 0.7
        
        tools = create_archival_memory_tools(mock_agent)
        
        assert isinstance(tools, dict)
        assert "archival_memory_insert" in tools
        assert "archival_memory_search" in tools
        
        # Verify types
        assert isinstance(tools["archival_memory_insert"], ArchivalMemoryInsertTool)
        assert isinstance(tools["archival_memory_search"], ArchivalMemorySearchTool)
        
        # Verify they reference the same agent
        assert tools["archival_memory_insert"].agent == mock_agent
        assert tools["archival_memory_search"].agent == mock_agent
    
    @patch('asterix.tools.archival.asyncio.run')
    @patch('asterix.storage.qdrant_client.qdrant_client')
    @patch('asterix.core.embeddings.embedding_service')
    def test_factory_tools_are_functional(self, mock_embed_service, mock_qdrant, mock_asyncio):
        """Test that tools created by factory are functional."""
        mock_agent = Mock()
        mock_agent.id = "functional_test"
        mock_agent.config = Mock()
        mock_agent.config.memory = Mock()
        mock_agent.config.memory.retrieval_k = 5
        mock_agent.config.memory.score_threshold = 0.7
        
        tools = create_archival_memory_tools(mock_agent)
        
        # Mock responses for insert
        mock_embed_result = Mock()
        mock_embed_result.embeddings = [[0.1]]
        mock_embed_result.provider = "openai"
        mock_embed_result.dimensions = 1536
        mock_asyncio.side_effect = [mock_embed_result, ["id_1"]]
        
        # Test insert tool works
        insert_tool = tools["archival_memory_insert"]
        result = insert_tool.execute(content="test memory")
        assert result.status == ToolStatus.SUCCESS
        
        # Mock responses for search
        mock_asyncio.side_effect = [mock_embed_result, []]
        
        # Test search tool works
        search_tool = tools["archival_memory_search"]
        result = search_tool.execute(query="test query")
        assert result.status == ToolStatus.SUCCESS