"""
Unit tests for Core Memory Tools.

Tests CoreMemoryAppendTool and CoreMemoryReplaceTool functionality:
- Tool initialization with agent context
- Successful append operations
- Successful replace operations
- Error handling (invalid blocks, empty content, content not found)
- Edge cases (multiple occurrences, eviction detection)
- Result structure and metadata
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from asterix.tools.core_memory import (
    CoreMemoryAppendTool,
    CoreMemoryReplaceTool,
    create_core_memory_tools
)
from asterix.tools.base import ToolResult, ToolStatus
from asterix.agent import BlockConfig


class MockMemoryBlock:
    """Mock memory block for testing."""
    
    def __init__(self, content: str = "", tokens: int = 0, config: BlockConfig = None):
        self.content = content
        self.tokens = tokens
        self.config = config or BlockConfig(size=1000, priority=1)


class TestCoreMemoryAppendTool:
    """Test CoreMemoryAppendTool functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.blocks = {
            "task": MockMemoryBlock(content="Current task: Code review", tokens=50),
            "user": MockMemoryBlock(content="User: Alice", tokens=20)
        }
        agent.append_to_memory = Mock()
        return agent
    
    def test_initialization(self, mock_agent):
        """Test tool initializes with correct schema."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        assert tool.name == "core_memory_append"
        assert tool.agent == mock_agent
        assert tool.schema is not None
        assert "block" in tool.schema["parameters"]["properties"]
        assert "content" in tool.schema["parameters"]["properties"]
        # Schema should include available blocks as enum
        assert list(mock_agent.blocks.keys()) == tool.schema["parameters"]["properties"]["block"]["enum"]
    
    def test_append_success(self, mock_agent):
        """Test successful append operation."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        # Mock the append behavior
        def mock_append(block, content):
            mock_agent.blocks[block].content += "\n" + content
            mock_agent.blocks[block].tokens += len(content) // 4
        
        mock_agent.append_to_memory.side_effect = mock_append
        
        result = tool.execute(block="task", content="Review auth module")
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert "Successfully appended" in result.content
        assert "task" in result.content
        assert result.metadata["block"] == "task"
        assert result.metadata["operation"] == "append"
        assert result.metadata["content_added"] == "Review auth module"
        
        # Verify agent method was called
        mock_agent.append_to_memory.assert_called_once_with("task", "Review auth module")
    
    def test_append_invalid_block(self, mock_agent):
        """Test append to non-existent block returns error."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        result = tool.execute(block="nonexistent", content="Some content")
        
        assert result.status == ToolStatus.ERROR
        assert "does not exist" in result.error
        assert "nonexistent" in result.error
        assert "Available blocks" in result.error
    
    def test_append_empty_content(self, mock_agent):
        """Test append with empty content returns error."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        # Test with empty string
        result = tool.execute(block="task", content="")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
        
        # Test with whitespace only
        result = tool.execute(block="task", content="   \n\t  ")
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
    
    def test_append_trims_whitespace(self, mock_agent):
        """Test that append trims whitespace from content."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        def mock_append(block, content):
            mock_agent.blocks[block].content += "\n" + content
        
        mock_agent.append_to_memory.side_effect = mock_append
        
        result = tool.execute(block="task", content="  Content with spaces  ")
        
        assert result.status == ToolStatus.SUCCESS
        # Should be called with trimmed content
        mock_agent.append_to_memory.assert_called_once_with("task", "Content with spaces")
    
    def test_append_detects_eviction(self, mock_agent):
        """Test that append detects when eviction was triggered."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        original_tokens = 500
        content_to_add = "New content" * 50  # Would add ~125 tokens
        
        mock_agent.blocks["task"].tokens = original_tokens
        
        # Mock append that triggers eviction (block gets summarized)
        def mock_append_with_eviction(block, content):
            # Simulate eviction: tokens reduced significantly
            mock_agent.blocks[block].content = "Summarized content"
            mock_agent.blocks[block].tokens = 200  # Much less than expected
        
        mock_agent.append_to_memory.side_effect = mock_append_with_eviction
        
        result = tool.execute(block="task", content=content_to_add)
        
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata["eviction_triggered"] is True
        assert result.metadata["old_tokens"] == original_tokens
        assert result.metadata["new_tokens"] == 200
        assert "summarized" in result.content.lower()
    
    def test_append_exception_handling(self, mock_agent):
        """Test that exceptions are caught and returned as errors."""
        tool = CoreMemoryAppendTool(mock_agent)
        
        # Mock append to raise exception
        mock_agent.append_to_memory.side_effect = RuntimeError("Database error")
        
        result = tool.execute(block="task", content="Some content")
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to append" in result.error
        assert "Database error" in result.error


class TestCoreMemoryReplaceTool:
    """Test CoreMemoryReplaceTool functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.blocks = {
            "task": MockMemoryBlock(
                content="Status: In Progress\nTask: Review code",
                tokens=60
            ),
            "notes": MockMemoryBlock(
                content="Important: Follow style guide",
                tokens=40
            )
        }
        agent.update_memory = Mock()
        return agent
    
    def test_initialization(self, mock_agent):
        """Test tool initializes with correct schema."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        assert tool.name == "core_memory_replace"
        assert tool.agent == mock_agent
        assert tool.schema is not None
        assert "block" in tool.schema["parameters"]["properties"]
        assert "old_content" in tool.schema["parameters"]["properties"]
        assert "new_content" in tool.schema["parameters"]["properties"]
    
    def test_replace_success(self, mock_agent):
        """Test successful replace operation."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        # Mock the update behavior
        def mock_update(block, new_full_content):
            mock_agent.blocks[block].content = new_full_content
            mock_agent.blocks[block].tokens = len(new_full_content) // 4
        
        mock_agent.update_memory.side_effect = mock_update
        
        result = tool.execute(
            block="task",
            old_content="Status: In Progress",
            new_content="Status: Complete"
        )
        
        assert result.status == ToolStatus.SUCCESS
        assert "Successfully replaced" in result.content
        assert result.metadata["block"] == "task"
        assert result.metadata["operation"] == "replace"
        assert result.metadata["old_content"] == "Status: In Progress"
        assert result.metadata["new_content"] == "Status: Complete"
        assert result.metadata["occurrences_replaced"] == 1
        
        # Verify update was called with correct content
        expected_content = "Status: Complete\nTask: Review code"
        mock_agent.update_memory.assert_called_once_with("task", expected_content)
    
    def test_replace_invalid_block(self, mock_agent):
        """Test replace on non-existent block returns error."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        result = tool.execute(
            block="nonexistent",
            old_content="something",
            new_content="something else"
        )
        
        assert result.status == ToolStatus.ERROR
        assert "does not exist" in result.error
        assert "nonexistent" in result.error
    
    def test_replace_empty_old_content(self, mock_agent):
        """Test replace with empty old_content returns error."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        result = tool.execute(
            block="task",
            old_content="",
            new_content="new content"
        )
        
        assert result.status == ToolStatus.ERROR
        assert "cannot be empty" in result.error
        assert "must specify what to replace" in result.error
    
    def test_replace_content_not_found(self, mock_agent):
        """Test replace when old_content doesn't exist in block."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        result = tool.execute(
            block="task",
            old_content="This text does not exist",
            new_content="new content"
        )
        
        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error
        assert "task" in result.error
        assert "matches exactly" in result.error
    
    def test_replace_multiple_occurrences(self, mock_agent):
        """Test replace handles multiple occurrences of old_content."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        # Set up block with duplicate content
        mock_agent.blocks["notes"].content = "TODO: Fix bug\nTODO: Write tests\nTODO: Deploy"
        
        def mock_update(block, new_full_content):
            mock_agent.blocks[block].content = new_full_content
            mock_agent.blocks[block].tokens = len(new_full_content) // 4
        
        mock_agent.update_memory.side_effect = mock_update
        
        result = tool.execute(
            block="notes",
            old_content="TODO:",
            new_content="DONE:"
        )
        
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata["occurrences_replaced"] == 3
        assert "3 occurrences" in result.content
        
        # Verify all occurrences were replaced
        expected_content = "DONE: Fix bug\nDONE: Write tests\nDONE: Deploy"
        mock_agent.update_memory.assert_called_once_with("notes", expected_content)
    
    def test_replace_trims_whitespace(self, mock_agent):
        """Test that replace trims whitespace from old and new content."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        def mock_update(block, new_full_content):
            mock_agent.blocks[block].content = new_full_content
        
        mock_agent.update_memory.side_effect = mock_update
        
        result = tool.execute(
            block="task",
            old_content="  Status: In Progress  ",
            new_content="  Status: Complete  "
        )
        
        assert result.status == ToolStatus.SUCCESS
        # Verify trimmed versions were used
        assert result.metadata["old_content"] == "Status: In Progress"
        assert result.metadata["new_content"] == "Status: Complete"
    
    def test_replace_detects_eviction(self, mock_agent):
        """Test that replace detects when eviction was triggered."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        # Mock count_tokens to return predictable values
        with patch('asterix.utils.tokens.count_tokens') as mock_count:
            mock_count.return_value = Mock(tokens=100)
            
            # Mock update that triggers eviction
            def mock_update_with_eviction(block, content):
                # Simulate eviction: final tokens less than expected
                mock_agent.blocks[block].content = "Summarized"
                mock_agent.blocks[block].tokens = 50  # Less than expected 100
            
            mock_agent.update_memory.side_effect = mock_update_with_eviction
            
            result = tool.execute(
                block="task",
                old_content="Status: In Progress",
                new_content="Status: Complete"
            )
            
            assert result.status == ToolStatus.SUCCESS
            assert result.metadata["eviction_triggered"] is True
            assert "summarized" in result.content.lower()
    
    def test_replace_exception_handling(self, mock_agent):
        """Test that exceptions are caught and returned as errors."""
        tool = CoreMemoryReplaceTool(mock_agent)
        
        # Mock update to raise exception
        mock_agent.update_memory.side_effect = RuntimeError("Memory error")
        
        result = tool.execute(
            block="task",
            old_content="Status: In Progress",
            new_content="Status: Complete"
        )
        
        assert result.status == ToolStatus.ERROR
        assert "Failed to replace" in result.error
        assert "Memory error" in result.error


class TestCoreMemoryToolsFactory:
    """Test create_core_memory_tools factory function."""
    
    def test_create_core_memory_tools(self):
        """Test factory creates both tools correctly."""
        mock_agent = Mock()
        mock_agent.blocks = {"task": MockMemoryBlock()}
        
        tools = create_core_memory_tools(mock_agent)
        
        assert isinstance(tools, dict)
        assert "core_memory_append" in tools
        assert "core_memory_replace" in tools
        
        # Verify types
        assert isinstance(tools["core_memory_append"], CoreMemoryAppendTool)
        assert isinstance(tools["core_memory_replace"], CoreMemoryReplaceTool)
        
        # Verify they reference the same agent
        assert tools["core_memory_append"].agent == mock_agent
        assert tools["core_memory_replace"].agent == mock_agent
    
    def test_factory_tools_are_functional(self):
        """Test that tools created by factory are functional."""
        mock_agent = Mock()
        mock_agent.blocks = {
            "test": MockMemoryBlock(content="initial content", tokens=30)
        }
        mock_agent.append_to_memory = Mock()
        mock_agent.update_memory = Mock()
        
        tools = create_core_memory_tools(mock_agent)
        
        # Test append tool works
        append_tool = tools["core_memory_append"]
        result = append_tool.execute(block="test", content="new content")
        assert result.status == ToolStatus.SUCCESS
        
        # Test replace tool works
        replace_tool = tools["core_memory_replace"]
        result = replace_tool.execute(
            block="test",
            old_content="initial",
            new_content="updated"
        )
        assert result.status == ToolStatus.SUCCESS


class TestCoreMemoryToolsIntegration:
    """Test core memory tools with more realistic agent integration."""
    
    def test_append_and_replace_sequence(self):
        """Test a realistic sequence of append and replace operations."""
        mock_agent = Mock()
        mock_agent.blocks = {
            "task": MockMemoryBlock(content="Task list:", tokens=20)
        }
        
        # Mock methods to actually update content
        def mock_append(block, content):
            mock_agent.blocks[block].content += "\n" + content
            mock_agent.blocks[block].tokens += len(content) // 4
        
        def mock_update(block, content):
            mock_agent.blocks[block].content = content
            mock_agent.blocks[block].tokens = len(content) // 4
        
        mock_agent.append_to_memory = Mock(side_effect=mock_append)
        mock_agent.update_memory = Mock(side_effect=mock_update)
        
        # Create tools
        append_tool = CoreMemoryAppendTool(mock_agent)
        replace_tool = CoreMemoryReplaceTool(mock_agent)
        
        # 1. Append a task
        result1 = append_tool.execute(block="task", content="1. Review PR")
        assert result1.status == ToolStatus.SUCCESS
        assert "Review PR" in mock_agent.blocks["task"].content
        
        # 2. Append another task
        result2 = append_tool.execute(block="task", content="2. Write tests")
        assert result2.status == ToolStatus.SUCCESS
        assert "Write tests" in mock_agent.blocks["task"].content
        
        # 3. Mark first task as complete using replace
        result3 = replace_tool.execute(
            block="task",
            old_content="1. Review PR",
            new_content="1. Review PR ✓"
        )
        assert result3.status == ToolStatus.SUCCESS
        assert "Review PR ✓" in mock_agent.blocks["task"].content
        assert "1. Review PR\n" not in mock_agent.blocks["task"].content  # Old version gone