"""
Asterix Tool System - Base Classes

Provides the foundation for the tool system with:
- Base Tool class for all agent tools
- ToolResult for structured returns
- Automatic schema generation from function signatures
- Tool registration and validation
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Result Structures
# ============================================================================

class ToolStatus(Enum):
    """Status codes for tool execution results."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"  # Some operations succeeded, some failed


@dataclass
class ToolResult:
    """
    Structured result from tool execution.
    
    All tools should return this structure for consistent handling
    by the agent and heartbeat loop.
    
    Args:
        status: Execution status (success/error/partial)
        content: Main content/result from the tool
        error: Error message if status is ERROR
        metadata: Additional metadata about the execution
        timestamp: When the tool was executed
    
    Example:
        >>> result = ToolResult(
        ...     status=ToolStatus.SUCCESS,
        ...     content="Updated memory block 'task' with new content",
        ...     metadata={"block_name": "task", "tokens_used": 150}
        ... )
    """
    status: ToolStatus
    content: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation for LLM consumption."""
        if self.status == ToolStatus.SUCCESS:
            return self.content
        elif self.status == ToolStatus.ERROR:
            return f"Error: {self.error}"
        else:
            return f"{self.content}\nWarning: {self.error}"


# ============================================================================
# Tool Schema Generation
# ============================================================================

def python_type_to_json_schema(python_type: type) -> Dict[str, Any]:
    """
    Convert Python type to JSON Schema type for OpenAI function calling.
    
    Args:
        python_type: Python type annotation
        
    Returns:
        JSON Schema type definition
    """
    # Handle Optional types
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Optional[X] is Union[X, None]
        if type(None) in args:
            # Get the non-None type
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return python_type_to_json_schema(non_none_types[0])
    
    # Handle List types
    if origin is list or python_type is list:
        args = get_args(python_type)
        if args:
            return {
                "type": "array",
                "items": python_type_to_json_schema(args[0])
            }
        return {"type": "array"}
    
    # Handle Dict types
    if origin is dict or python_type is dict:
        return {"type": "object"}
    
    # Basic type mapping
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    json_type = type_map.get(python_type, "string")
    return {"type": json_type}


def generate_tool_schema(func: Callable, 
                         name: Optional[str] = None,
                         description: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate OpenAI function calling schema from a Python function.
    
    Automatically extracts:
    - Function name
    - Description from docstring
    - Parameters with types from type hints
    - Required parameters
    
    Args:
        func: Function to generate schema for
        name: Override function name (optional)
        description: Override description (optional)
        
    Returns:
        OpenAI function schema
    
    Example:
        >>> def read_file(filepath: str, encoding: str = "utf-8") -> str:
        ...     '''Read a file from disk'''
        ...     pass
        >>> schema = generate_tool_schema(read_file)
        >>> # Returns:
        >>> # {
        >>> #     "name": "read_file",
        >>> #     "description": "Read a file from disk",
        >>> #     "parameters": {
        >>> #         "type": "object",
        >>> #         "properties": {
        >>> #             "filepath": {"type": "string"},
        >>> #             "encoding": {"type": "string"}
        >>> #         },
        >>> #         "required": ["filepath"]
        >>> #     }
        >>> # }
    """
    # Get function name
    tool_name = name or func.__name__
    
    # Get description from docstring or override
    if description:
        tool_description = description
    else:
        tool_description = inspect.getdoc(func) or f"Execute {tool_name}"
        # Take only the first line of docstring
        tool_description = tool_description.split('\n')[0].strip()
    
    # Get function signature
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Build parameters schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip self/cls parameters
        if param_name in ('self', 'cls'):
            continue
        
        # Get type hint
        param_type = type_hints.get(param_name, str)
        
        # Convert to JSON schema type
        param_schema = python_type_to_json_schema(param_type)
        
        # Add description from docstring if available
        # (We'll keep it simple for now - can enhance later)
        param_schema["description"] = f"The {param_name} parameter"
        
        properties[param_name] = param_schema
        
        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    # Build complete schema in OpenAI function calling format
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    return schema


# ============================================================================
# Base Tool Class
# ============================================================================

class Tool:
    """
    Base class for all agent tools.
    
    Tools are callable objects that the agent can use during conversations.
    Each tool has:
    - A name (unique identifier)
    - A description (for LLM understanding)
    - A schema (OpenAI function calling format)
    - An execute method (the actual implementation)
    
    Subclass this to create new tools, or use the @tool decorator on functions.
    
    Example:
        >>> class ReadFileTool(Tool):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="read_file",
        ...             description="Read contents of a file from disk"
        ...         )
        ...     
        ...     def execute(self, filepath: str) -> ToolResult:
        ...         try:
        ...             with open(filepath, 'r') as f:
        ...                 content = f.read()
        ...             return ToolResult(
        ...                 status=ToolStatus.SUCCESS,
        ...                 content=content,
        ...                 metadata={"filepath": filepath}
        ...             )
        ...         except Exception as e:
        ...             return ToolResult(
        ...                 status=ToolStatus.ERROR,
        ...                 content="",
        ...                 error=str(e)
        ...             )
    """
    
    def __init__(self, 
                name: str,
                description: str,
                func: Optional[Callable] = None,
                schema: Optional[Dict[str, Any]] = None):
        """
        Initialize a tool.
        
        Args:
            name: Tool name (unique identifier)
            description: Tool description for LLM
            func: Optional function implementation
            schema: Optional pre-computed schema (auto-generated if not provided)
        """
        self.name = name
        self.description = description
        self._func = func
        
        # Generate or use provided schema
        if schema:
            # Schema already provided - ensure it has the correct format
            if "type" not in schema:
                # Wrap it in OpenAI format
                self.schema = {
                    "type": "function",
                    "function": schema
                }
            else:
                self.schema = schema
        elif func:
            self.schema = generate_tool_schema(func, name, description)
        else:
            # Default schema - will be overridden by subclasses
            self.schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        
        logger.debug(f"Initialized tool: {name}")
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with provided arguments.
        
        This method should be overridden by subclasses, or will use
        the provided function if initialized with one.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            ToolResult with execution status and output
        """
        if self._func:
            try:
                result = self._func(**kwargs)
                
                # If function returns ToolResult, use it directly
                if isinstance(result, ToolResult):
                    return result
                
                # Otherwise wrap in ToolResult
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content=str(result),
                    metadata={"tool": self.name}
                )
                
            except Exception as e:
                logger.error(f"Tool {self.name} execution failed: {e}")
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error=f"Tool execution failed: {str(e)}",
                    metadata={"tool": self.name}
                )
        else:
            raise NotImplementedError(f"Tool {self.name} does not implement execute()")
    
    def __call__(self, **kwargs) -> ToolResult:
        """Make tool callable."""
        return self.execute(**kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tool(name='{self.name}', description='{self.description}')"


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """
    Registry for managing available tools.
    
    Keeps track of all registered tools and provides methods for:
    - Adding/removing tools
    - Getting tool schemas for LLM
    - Executing tools by name
    - Validating tool calls
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, Tool] = {}
        logger.debug("Initialized tool registry")
    
    def register(self, tool: Tool):
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool to register
            
        Raises:
            ValueError: If tool name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str):
        """
        Remove a tool from the registry.
        
        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Tool name
            
        Returns:
            True if tool exists
        """
        return tool_name in self._tools
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all tools
        """
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function schemas for all registered tools.
        
        Returns:
            List of tool schemas for LLM function calling
        """
        return [tool.schema for tool in self._tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments
            
        Returns:
            ToolResult from execution
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get(tool_name)
        
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Tool execution failed: {str(e)}"
            )
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# ============================================================================
# Tool Decorator (for easy registration)
# ============================================================================

def tool(name: Optional[str] = None, 
         description: Optional[str] = None):
    """
    Decorator to convert a function into a Tool.
    
    This is the easiest way to create custom tools. Just decorate
    a function and it will automatically:
    - Generate the tool schema from type hints
    - Wrap it in a Tool object
    - Handle ToolResult conversion
    
    Args:
        name: Tool name (uses function name if not provided)
        description: Tool description (uses docstring if not provided)
        
    Returns:
        Decorator function
    
    Example:
        >>> @tool(name="read_file", description="Read a file from disk")
        >>> def read_file(filepath: str) -> str:
        ...     with open(filepath, 'r') as f:
        ...         return f.read()
        >>> 
        >>> # Now read_file is a Tool that can be registered with an agent
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or (inspect.getdoc(func) or f"Execute {tool_name}").split('\n')[0].strip()
        
        return Tool(
            name=tool_name,
            description=tool_description,
            func=func
        )
    
    return decorator