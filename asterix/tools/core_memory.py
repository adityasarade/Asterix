"""
Asterix Core Memory Tools

Tools for agents to edit their own memory blocks:
- core_memory_append: Add content to a block
- core_memory_replace: Replace content in a block

These tools allow agents to manage their short-term memory during conversations.
"""

import logging
from typing import TYPE_CHECKING, Dict, Any

from .base import Tool, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# Core Memory Append Tool
# ============================================================================

class CoreMemoryAppendTool(Tool):
    """
    Tool for appending content to a memory block.
    
    This is a non-destructive operation - content is added to the end
    of the block. If the block exceeds its token limit after appending,
    eviction will be triggered (handled by the agent in Step 4).
    
    Schema:
        {
            "name": "core_memory_append",
            "description": "Append content to a memory block",
            "parameters": {
                "block": "Name of the memory block",
                "content": "Content to append"
            }
        }
    
    Example usage by agent:
        >>> # Agent wants to remember something new
        >>> core_memory_append(block="user", content="User prefers Python over JavaScript")
        >>> # Returns: "Successfully appended to block 'user'"
    """
    
    def __init__(self, agent: 'Agent'):
        """
        Initialize the core_memory_append tool.
        
        Args:
            agent: Agent instance that owns this tool
        """
        self.agent = agent
        
        # Define tool schema for LLM
        schema = {
            "name": "core_memory_append",
            "description": "Append content to a memory block. Use this to add new information without removing existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block": {
                        "type": "string",
                        "description": f"Name of the memory block to append to. Available blocks: {list(agent.blocks.keys())}",
                        "enum": list(agent.blocks.keys())
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append to the block"
                    }
                },
                "required": ["block", "content"]
            }
        }
        
        super().__init__(
            name="core_memory_append",
            description="Append content to a memory block",
            schema=schema
        )
    
    def execute(self, block: str, content: str) -> ToolResult:
        """
        Execute the core_memory_append operation.
        
        Args:
            block: Name of the memory block
            content: Content to append
            
        Returns:
            ToolResult with operation status
        """
        try:
            # Validate block exists
            if block not in self.agent.blocks:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error=f"Memory block '{block}' does not exist. Available blocks: {list(self.agent.blocks.keys())}"
                )
            
            # Validate content is not empty
            if not content or not content.strip():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="Content cannot be empty"
                )
            
            # Get current block state
            memory_block = self.agent.blocks[block]
            old_content = memory_block.content
            old_tokens = memory_block.tokens
            
            # Append content using the agent's method
            # (agent.append_to_memory handles eviction automatically)
            self.agent.append_to_memory(block, content.strip())
            
            # Get updated state after potential eviction
            new_content = memory_block.content
            new_tokens = memory_block.tokens
            
            # Detect if eviction was triggered
            eviction_triggered = new_tokens < (old_tokens + len(content.strip()) // 4)
            
            # Success response
            result_message = f"Successfully appended content to memory block '{block}'"
            if eviction_triggered:
                result_message += f" (block was summarized: {old_tokens}→{new_tokens} tokens)"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=result_message,
                metadata={
                    "block": block,
                    "operation": "append",
                    "content_added": content.strip(),
                    "old_tokens": old_tokens,
                    "new_tokens": new_tokens,
                    "eviction_triggered": eviction_triggered,
                    "block_priority": memory_block.config.priority
                }
            )
            
        except Exception as e:
            logger.error(f"core_memory_append failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Failed to append to memory block: {str(e)}"
            )


# ============================================================================
# Core Memory Replace Tool
# ============================================================================

class CoreMemoryReplaceTool(Tool):
    """
    Tool for replacing specific content in a memory block.
    
    This is a destructive operation - finds and replaces text.
    Useful for updating existing information (e.g., marking tasks complete,
    updating status, correcting information).
    
    Schema:
        {
            "name": "core_memory_replace",
            "description": "Replace content in a memory block",
            "parameters": {
                "block": "Name of the memory block",
                "old_content": "Content to find and replace",
                "new_content": "Content to replace with"
            }
        }
    
    Example usage by agent:
        >>> # Agent wants to mark a task as complete
        >>> core_memory_replace(
        ...     block="task",
        ...     old_content="Status: In Progress",
        ...     new_content="Status: Complete"
        ... )
        >>> # Returns: "Successfully replaced content in block 'task'"
    """
    
    def __init__(self, agent: 'Agent'):
        """
        Initialize the core_memory_replace tool.
        
        Args:
            agent: Agent instance that owns this tool
        """
        self.agent = agent
        
        # Define tool schema for LLM
        schema = {
            "name": "core_memory_replace",
            "description": "Replace specific content in a memory block. Use this to update or correct existing information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "block": {
                        "type": "string",
                        "description": f"Name of the memory block to modify. Available blocks: {list(agent.blocks.keys())}",
                        "enum": list(agent.blocks.keys())
                    },
                    "old_content": {
                        "type": "string",
                        "description": "Exact text to find and replace. Must match existing content exactly."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New text to replace the old content with"
                    }
                },
                "required": ["block", "old_content", "new_content"]
            }
        }
        
        super().__init__(
            name="core_memory_replace",
            description="Replace content in a memory block",
            schema=schema
        )
    
    def execute(self, block: str, old_content: str, new_content: str) -> ToolResult:
        """
        Execute the core_memory_replace operation.
        
        Args:
            block: Name of the memory block
            old_content: Content to find and replace
            new_content: Content to replace with
            
        Returns:
            ToolResult with operation status
        """
        try:
            # Validate block exists
            if block not in self.agent.blocks:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error=f"Memory block '{block}' does not exist. Available blocks: {list(self.agent.blocks.keys())}"
                )
            
            # Validate old_content is not empty
            if not old_content or not old_content.strip():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error="old_content cannot be empty. You must specify what to replace."
                )
            
            # Get current block state
            memory_block = self.agent.blocks[block]
            current_content = memory_block.content
            
            # Check if old_content exists in the block
            if old_content.strip() not in current_content:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error=f"Content to replace not found in block '{block}'. Make sure the text matches exactly."
                )
            
            # Count occurrences
            occurrence_count = current_content.count(old_content.strip())
            
            if occurrence_count > 1:
                # Multiple occurrences - warn but proceed with replace all
                logger.warning(f"Found {occurrence_count} occurrences of old_content in block '{block}', replacing all")
            
            # Get token count before replacement
            old_tokens = memory_block.tokens
            
            # Perform replacement
            updated_content = current_content.replace(old_content.strip(), new_content.strip())
            
            # Update the block using agent's method
            # (agent.update_memory handles eviction automatically)
            self.agent.update_memory(block, updated_content)
            
            # Get updated state after potential eviction
            new_tokens = memory_block.tokens
            final_content = memory_block.content
            
            # Detect if eviction was triggered
            from ..utils.tokens import count_tokens
            expected_tokens = count_tokens(updated_content).tokens
            eviction_triggered = new_tokens < expected_tokens
            
            # Success response
            result_message = f"Successfully replaced content in memory block '{block}'"
            if occurrence_count > 1:
                result_message += f" ({occurrence_count} occurrences)"
            if eviction_triggered:
                result_message += f" (block was summarized: {expected_tokens}→{new_tokens} tokens)"
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=result_message,
                metadata={
                    "block": block,
                    "operation": "replace",
                    "old_content": old_content.strip(),
                    "new_content": new_content.strip(),
                    "occurrences_replaced": occurrence_count,
                    "old_tokens": old_tokens,
                    "new_tokens": new_tokens,
                    "eviction_triggered": eviction_triggered,
                    "block_priority": memory_block.config.priority
                }
            )
            
        except Exception as e:
            logger.error(f"core_memory_replace failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Failed to replace memory block content: {str(e)}"
            )


# ============================================================================
# Factory Function
# ============================================================================

def create_core_memory_tools(agent: 'Agent') -> Dict[str, Tool]:
    """
    Create both core memory tools for an agent.
    
    Args:
        agent: Agent instance
        
    Returns:
        Dictionary mapping tool names to Tool instances
    
    Example:
        >>> tools = create_core_memory_tools(agent)
        >>> agent.register_tool(tools["core_memory_append"])
        >>> agent.register_tool(tools["core_memory_replace"])
    """
    return {
        "core_memory_append": CoreMemoryAppendTool(agent),
        "core_memory_replace": CoreMemoryReplaceTool(agent)
    }