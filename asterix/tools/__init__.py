"""
Asterix Tool System

Provides the tool system for agent capabilities:
- Base classes for tools
- Memory management tools (core and archival)
- Conversation search
- Tool registry and execution
"""

from .base import (
    Tool,
    ToolResult,
    ToolStatus,
    ToolRegistry,
    tool,
    generate_tool_schema
)

from .core_memory import (
    CoreMemoryAppendTool,
    CoreMemoryReplaceTool,
    create_core_memory_tools
)

from .archival import (
    ArchivalMemoryInsertTool,
    ArchivalMemorySearchTool,
    create_archival_memory_tools
)

from .conversation import (
    ConversationSearchTool,
    create_conversation_search_tool
)

__all__ = [
    # Base classes
    "Tool",
    "ToolResult",
    "ToolStatus",
    "ToolRegistry",
    
    # Utilities
    "tool",
    "generate_tool_schema",
    
    # Core memory tools
    "CoreMemoryAppendTool",
    "CoreMemoryReplaceTool",
    "create_core_memory_tools",
    
    # Archival memory tools
    "ArchivalMemoryInsertTool",
    "ArchivalMemorySearchTool",
    "create_archival_memory_tools",
    
    # Conversation search
    "ConversationSearchTool",
    "create_conversation_search_tool",
]