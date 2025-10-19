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
# Custom Exceptions
# ============================================================================

class ToolError(Exception):
    """Base exception for all tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when attempting to use a tool that doesn't exist."""
    
    def __init__(self, tool_name: str, available_tools: Optional[List[str]] = None):
        self.tool_name = tool_name
        self.available_tools = available_tools
        
        message = f"Tool '{tool_name}' not found in registry"
        
        if available_tools:
            # Suggest similar tool names (simple similarity check)
            suggestions = [t for t in available_tools if tool_name.lower() in t.lower()]
            if suggestions:
                message += f". Did you mean: {', '.join(suggestions)}?"
            elif len(available_tools) <= 10:
                message += f". Available tools: {', '.join(available_tools)}"
        
        super().__init__(message)


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(self, tool_name: str, original_error: Exception, context: Optional[Dict[str, Any]] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        self.context = context or {}
        
        message = f"Tool '{tool_name}' execution failed: {str(original_error)}"
        
        if context:
            message += f"\nContext: {context}"
        
        super().__init__(message)


class ToolValidationError(ToolError):
    """Raised when tool parameters fail validation."""
    
    def __init__(self, tool_name: str, param_name: str, validation_message: str, value: Any = None):
        self.tool_name = tool_name
        self.param_name = param_name
        self.validation_message = validation_message
        self.value = value
        
        message = f"Tool '{tool_name}' parameter '{param_name}' validation failed: {validation_message}"
        
        if value is not None:
            message += f"\nProvided value: {value}"
        
        super().__init__(message)


class ToolRegistrationError(ToolError):
    """Raised when tool registration fails."""
    
    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        
        message = f"Failed to register tool '{tool_name}': {reason}"
        super().__init__(message)

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
# Parameter Validation
# ============================================================================

@dataclass
class ParameterConstraint:
    """
    Constraints for tool parameters.
    
    Allows defining validation rules for parameters beyond just type hints.
    
    Args:
        min_value: Minimum value for numbers
        max_value: Maximum value for numbers
        min_length: Minimum length for strings/lists
        max_length: Maximum length for strings/lists
        pattern: Regex pattern for string validation
        allowed_values: Specific allowed values (enum-like)
        custom_validator: Custom validation function
    """
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    
    def validate(self, value: Any, param_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate a value against constraints.
        
        Args:
            value: Value to validate
            param_name: Parameter name (for error messages)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Min/max value validation
        if self.min_value is not None and value < self.min_value:
            return False, f"{param_name} must be >= {self.min_value}, got {value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"{param_name} must be <= {self.max_value}, got {value}"
        
        # Length validation
        if hasattr(value, '__len__'):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False, f"{param_name} length must be >= {self.min_length}, got {length}"
            if self.max_length is not None and length > self.max_length:
                return False, f"{param_name} length must be <= {self.max_length}, got {length}"
        
        # Pattern validation for strings
        if self.pattern and isinstance(value, str):
            import re
            if not re.match(self.pattern, value):
                return False, f"{param_name} must match pattern '{self.pattern}'"
        
        # Allowed values (enum-like)
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"{param_name} must be one of {self.allowed_values}, got {value}"
        
        # Custom validator
        if self.custom_validator:
            try:
                if not self.custom_validator(value):
                    return False, f"{param_name} failed custom validation"
            except Exception as e:
                return False, f"{param_name} validation error: {str(e)}"
        
        return True, None


class ToolCategory(Enum):
    """
    Categories for organizing tools.
    
    Helps users find relevant tools and allows filtering by category.
    """
    MEMORY = "memory"              # Memory management tools
    FILE_OPS = "file_operations"   # File reading/writing
    WEB = "web"                    # Web scraping, API calls
    SHELL = "shell"                # Shell commands
    DATA = "data_processing"       # Data analysis, transformation
    COMMUNICATION = "communication" # Email, messaging
    CUSTOM = "custom"              # User-defined tools
    OTHER = "other"                # Uncategorized

# ============================================================================
# Base Tool Class
# ============================================================================

class Tool:
    """
    Base class for all agent tools.
    
    Tools are callable functions that agents can invoke during conversations.
    They have a name, description, and automatically generated schema for LLM function calling.
    
    Args:
        name: Unique tool name
        description: What the tool does (shown to LLM)
        func: The actual function to execute
        schema: OpenAI function schema (auto-generated if not provided)
        category: Tool category for organization
        constraints: Parameter validation constraints
        examples: Usage examples for documentation
        retry_on_error: Whether to retry on transient failures
        max_retries: Maximum retry attempts
    """
    
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 schema: Optional[Dict[str, Any]] = None,
                 category: ToolCategory = ToolCategory.CUSTOM,
                 constraints: Optional[Dict[str, ParameterConstraint]] = None,
                 examples: Optional[List[str]] = None,
                 retry_on_error: bool = False,
                 max_retries: int = 3):
        """Initialize tool with enhanced features."""
        self.name = name
        self.description = description
        self.func = func
        self.category = category
        self.constraints = constraints or {}
        self.examples = examples or []
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries
        self.schema = schema or generate_tool_schema(func, name, description)
        
        # Add category to schema metadata
        if 'metadata' not in self.schema:
            self.schema['metadata'] = {}
        self.schema['metadata']['category'] = category.value
        self.schema['metadata']['examples'] = self.examples
        
        logger.debug(f"Initialized tool '{name}' in category '{category.value}'")
    
    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against constraints.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param_name, constraint in self.constraints.items():
            if param_name in kwargs:
                is_valid, error_msg = constraint.validate(kwargs[param_name], param_name)
                if not is_valid:
                    return False, error_msg
        
        return True, None
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with enhanced error handling.
        
        Includes:
        - Parameter validation with detailed error messages
        - Retry logic for transient failures (if enabled)
        - Rich error context for debugging
        - Graceful error recovery
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            ToolResult from execution
        """
        # Step 1: Validate parameters first
        try:
            is_valid, error_msg = self.validate_parameters(**kwargs)
            if not is_valid:
                # Return detailed validation error
                logger.error(f"Validation failed for tool '{self.name}': {error_msg}")
                return ToolResult(
                    status=ToolStatus.ERROR,
                    content="",
                    error=f"Parameter validation failed: {error_msg}",
                    metadata={
                        "tool": self.name,
                        "error_type": "validation",
                        "validation_error": True,
                        "parameters": kwargs
                    }
                )
        except Exception as e:
            # Validation itself failed (shouldn't happen, but be safe)
            logger.error(f"Validation error for tool '{self.name}': {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                content="",
                error=f"Parameter validation error: {str(e)}",
                metadata={"tool": self.name, "error_type": "validation_exception"}
            )
        
        # Step 2: Execute with retry logic if enabled
        attempts = 0
        last_error = None
        
        while attempts <= (self.max_retries if self.retry_on_error else 0):
            try:
                # Execute the actual function
                logger.debug(f"Executing tool '{self.name}' (attempt {attempts + 1})")
                result = self.func(**kwargs)
                
                # Step 3: Wrap result in ToolResult if needed
                if isinstance(result, ToolResult):
                    # Already a ToolResult, add attempt metadata
                    if attempts > 0:
                        result.metadata["attempts"] = attempts + 1
                        result.metadata["retried"] = True
                    return result
                else:
                    # Wrap simple return value in ToolResult
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        content=str(result),
                        metadata={
                            "tool": self.name,
                            "attempts": attempts + 1,
                            "retried": attempts > 0
                        }
                    )
            
            except Exception as e:
                last_error = e
                attempts += 1
                
                # Log the error with context
                error_context = {
                    "tool": self.name,
                    "attempt": attempts,
                    "max_retries": self.max_retries,
                    "parameters": kwargs,
                    "exception_type": type(e).__name__
                }
                
                # Check if we should retry
                if self.retry_on_error and attempts <= self.max_retries:
                    logger.warning(
                        f"Tool '{self.name}' failed (attempt {attempts}/{self.max_retries}): {e}. "
                        f"Retrying in {0.5 * attempts}s...",
                        extra=error_context
                    )
                    import time
                    time.sleep(0.5 * attempts)  # Exponential backoff
                    continue
                else:
                    # No more retries - return detailed error
                    logger.error(
                        f"Tool '{self.name}' execution failed after {attempts} attempt(s): {e}",
                        extra=error_context,
                        exc_info=True  # Include stack trace in logs
                    )
                    
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        content="",
                        error=self._format_error_message(e, attempts),
                        metadata=error_context
                    )
        
        # This should never be reached, but just in case
        return ToolResult(
            status=ToolStatus.ERROR,
            content="",
            error=f"Tool execution failed unexpectedly: {last_error}",
            metadata={
                "tool": self.name,
                "error_type": "unexpected_failure",
                "exception": str(last_error)
            }
        )

    def _format_error_message(self, exception: Exception, attempts: int) -> str:
        """
        Format a user-friendly error message from an exception.
        
        Args:
            exception: The exception that occurred
            attempts: Number of attempts made
            
        Returns:
            Formatted error message
        """
        error_msg = f"Tool execution failed: {str(exception)}"
        
        if attempts > 1:
            error_msg += f" (after {attempts} attempts)"
        
        # Add helpful hints based on exception type
        if isinstance(exception, FileNotFoundError):
            error_msg += "\nHint: Check that the file path is correct and the file exists."
        elif isinstance(exception, PermissionError):
            error_msg += "\nHint: Check file permissions or try running with appropriate access."
        elif isinstance(exception, ValueError):
            error_msg += "\nHint: Check that parameter values are in the correct format."
        elif isinstance(exception, TypeError):
            error_msg += "\nHint: Check that parameter types match the expected types."
        
        return error_msg
    
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
    Registry for managing available tools with category support.
    
    Keeps track of all registered tools and provides methods for:
    - Adding/removing tools
    - Getting tool schemas for LLM
    - Executing tools by name
    - Filtering by category
    - Tool discovery and organization
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, Tool] = {}
        logger.debug("Initialized tool registry")
    
    def register(self, tool: Tool):
        """
        Register a tool with validation and error handling.
        
        Args:
            tool: Tool to register
            
        Raises:
            ToolRegistrationError: If registration fails
        """
        # Validate tool before registration
        if not tool.name:
            raise ToolRegistrationError("", "Tool name cannot be empty")
        
        if not tool.description:
            logger.warning(f"Tool '{tool.name}' registered without description")
        
        if tool.name in self._tools:
            raise ToolRegistrationError(
                tool.name,
                "Tool with this name already exists"
            )
        
        # Register the tool
        try:
            self._tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name} (category: {tool.category.value})")
        except Exception as e:
            raise ToolRegistrationError(
                tool.name,
                f"Registration failed: {str(e)}"
            )
    
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
        
        Handles both wrapped and unwrapped schema formats for compatibility.
        
        Returns:
            List of tool schemas for LLM function calling
        """
        schemas = []
        
        for tool in self._tools.values():
            # Check if schema is already in OpenAI format with wrapper
            if isinstance(tool.schema, dict) and 'type' in tool.schema and tool.schema.get('type') == 'function':
                # Already wrapped - use as-is
                schemas.append(tool.schema)
            else:
                # Unwrapped - wrap it
                schemas.append({
                    "type": "function",
                    "function": tool.schema
                })
        
        return schemas
    
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

    # ========================================================================
    # Category-Based Methods
    # ========================================================================
    
    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """
        Get all tools in a specific category.
        
        Useful for showing users only tools relevant to their current task
        (e.g., only show file operation tools when working with files).
        
        Args:
            category: ToolCategory enum value to filter by
            
        Returns:
            List of tools in the specified category
            
        Example:
            >>> memory_tools = registry.get_by_category(ToolCategory.MEMORY)
            >>> for tool in memory_tools:
            ...     print(f"- {tool.name}: {tool.description}")
        """
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]
    
    def list_tools(self, include_schemas: bool = False) -> List[Dict[str, Any]]:
        """
        List all tools with their metadata.
        
        Returns a structured list of all tools with key information
        for display or documentation purposes.
        
        Args:
            include_schemas: If True, include full OpenAI function schemas
            
        Returns:
            List of dictionaries with tool information
            
        Example:
            >>> tools = registry.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool['name']} ({tool['category']})")
            ...     print(f"  {tool['description']}")
        """
        tools_list = []
        
        for tool in self._tools.values():
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "has_validation": len(tool.constraints) > 0,
                "retry_enabled": tool.retry_on_error,
                "max_retries": tool.max_retries if tool.retry_on_error else 0,
                "examples": tool.examples
            }
            
            if include_schemas:
                tool_info["schema"] = tool.schema
            
            tools_list.append(tool_info)
        
        return tools_list
    
    def list_categories(self) -> Dict[str, int]:
        """
        List all tool categories with counts.
        
        Shows how many tools are registered in each category,
        useful for understanding the distribution of capabilities.
        
        Returns:
            Dictionary mapping category names to tool counts
            
        Example:
            >>> categories = registry.list_categories()
            >>> for category, count in categories.items():
            ...     print(f"{category}: {count} tools")
            # memory: 5 tools
            # file_operations: 3 tools
            # custom: 2 tools
        """
        category_counts: Dict[str, int] = {}
        
        for tool in self._tools.values():
            category_name = tool.category.value
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        return category_counts
    
    def count_by_category(self, category: ToolCategory) -> int:
        """
        Count tools in a specific category.
        
        Args:
            category: ToolCategory to count
            
        Returns:
            Number of tools in that category
            
        Example:
            >>> memory_count = registry.count_by_category(ToolCategory.MEMORY)
            >>> print(f"Memory tools available: {memory_count}")
        """
        return len(self.get_by_category(category))
    
    def filter_tools(self,
                    category: Optional[ToolCategory] = None,
                    name_pattern: Optional[str] = None,
                    has_retry: Optional[bool] = None,
                    has_validation: Optional[bool] = None) -> List[Tool]:
        """
        Advanced filtering of tools by multiple criteria.
        
        Allows combining multiple filters for precise tool discovery.
        
        Args:
            category: Filter by tool category
            name_pattern: Filter by name (case-insensitive substring match)
            has_retry: Filter by retry capability (True/False/None for any)
            has_validation: Filter by validation constraints (True/False/None for any)
            
        Returns:
            List of tools matching all specified criteria
            
        Example:
            >>> # Find all memory tools with validation
            >>> tools = registry.filter_tools(
            ...     category=ToolCategory.MEMORY,
            ...     has_validation=True
            ... )
            >>> 
            >>> # Find all tools with 'search' in the name
            >>> search_tools = registry.filter_tools(name_pattern="search")
        """
        filtered = list(self._tools.values())
        
        # Filter by category
        if category is not None:
            filtered = [t for t in filtered if t.category == category]
        
        # Filter by name pattern (case-insensitive)
        if name_pattern is not None:
            pattern_lower = name_pattern.lower()
            filtered = [t for t in filtered if pattern_lower in t.name.lower()]
        
        # Filter by retry capability
        if has_retry is not None:
            filtered = [t for t in filtered if t.retry_on_error == has_retry]
        
        # Filter by validation constraints
        if has_validation is not None:
            if has_validation:
                filtered = [t for t in filtered if len(t.constraints) > 0]
            else:
                filtered = [t for t in filtered if len(t.constraints) == 0]
        
        return filtered
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool.
        
        Returns comprehensive metadata about a tool including its
        category, validation rules, examples, and schema.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool information or None if not found
            
        Example:
            >>> info = registry.get_tool_info("core_memory_append")
            >>> print(f"Category: {info['category']}")
            >>> print(f"Examples: {info['examples']}")
        """
        tool = self.get(tool_name)
        
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "constraints": {
                param: {
                    "min_value": c.min_value,
                    "max_value": c.max_value,
                    "min_length": c.min_length,
                    "max_length": c.max_length,
                    "pattern": c.pattern,
                    "allowed_values": c.allowed_values
                }
                for param, c in tool.constraints.items()
            },
            "examples": tool.examples,
            "retry_enabled": tool.retry_on_error,
            "max_retries": tool.max_retries,
            "schema": tool.schema
        }

    # ========================================================================
    # Documentation Generation
    # ========================================================================

    def generate_tool_docs(self, tool_name: str, format: str = "markdown") -> Optional[str]:
        """
        Generate documentation for a specific tool.
        
        Creates human-readable documentation including description,
        parameters, constraints, examples, and usage information.
        
        Args:
            tool_name: Name of the tool to document
            format: Output format ('markdown' or 'text')
            
        Returns:
            Formatted documentation string or None if tool not found
            
        Example:
            >>> docs = registry.generate_tool_docs("core_memory_append")
            >>> print(docs)
            # core_memory_append
            
            **Category:** memory
            
            **Description:** Add content to a memory block...
            ...
        """
        tool = self.get(tool_name)
        if not tool:
            logger.warning(f"Cannot generate docs: tool '{tool_name}' not found")
            return None
        
        if format == "markdown":
            return self._generate_markdown_docs(tool)
        else:
            return self._generate_text_docs(tool)

    def _generate_markdown_docs(self, tool: Tool) -> str:
        """Generate markdown documentation for a tool."""
        lines = []
        
        # Header
        lines.append(f"# {tool.name}\n")
        
        # Category
        lines.append(f"**Category:** `{tool.category.value}`\n")
        
        # Description
        lines.append(f"**Description:** {tool.description}\n")
        
        # Parameters
        if tool.schema and 'function' in tool.schema:
            params = tool.schema['function'].get('parameters', {})
            properties = params.get('properties', {})
            required = params.get('required', [])
            
            if properties:
                lines.append("## Parameters\n")
                
                for param_name, param_info in properties.items():
                    # Parameter name with required indicator
                    req_marker = " *(required)*" if param_name in required else " *(optional)*"
                    lines.append(f"### `{param_name}`{req_marker}\n")
                    
                    # Type
                    param_type = param_info.get('type', 'any')
                    lines.append(f"- **Type:** `{param_type}`")
                    
                    # Description
                    if 'description' in param_info:
                        lines.append(f"- **Description:** {param_info['description']}")
                    
                    # Constraints
                    if param_name in tool.constraints:
                        constraint = tool.constraints[param_name]
                        lines.append("- **Constraints:**")
                        
                        if constraint.min_value is not None:
                            lines.append(f"  - Minimum value: `{constraint.min_value}`")
                        if constraint.max_value is not None:
                            lines.append(f"  - Maximum value: `{constraint.max_value}`")
                        if constraint.min_length is not None:
                            lines.append(f"  - Minimum length: `{constraint.min_length}`")
                        if constraint.max_length is not None:
                            lines.append(f"  - Maximum length: `{constraint.max_length}`")
                        if constraint.pattern is not None:
                            lines.append(f"  - Pattern: `{constraint.pattern}`")
                        if constraint.allowed_values is not None:
                            lines.append(f"  - Allowed values: `{constraint.allowed_values}`")
                    
                    lines.append("")  # Blank line
        
        # Examples
        if tool.examples:
            lines.append("## Examples\n")
            for idx, example in enumerate(tool.examples, 1):
                lines.append(f"**Example {idx}:**")
                lines.append(f"```python\n{example}\n```\n")
        
        # Retry configuration
        if tool.retry_on_error:
            lines.append("## Retry Configuration\n")
            lines.append(f"- **Retry on error:** Yes")
            lines.append(f"- **Max retries:** {tool.max_retries}\n")
        
        return "\n".join(lines)

    def _generate_text_docs(self, tool: Tool) -> str:
        """Generate plain text documentation for a tool."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append(f"TOOL: {tool.name}")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Category: {tool.category.value}")
        lines.append(f"Description: {tool.description}")
        lines.append("")
        
        # Parameters
        if tool.schema and 'function' in tool.schema:
            params = tool.schema['function'].get('parameters', {})
            properties = params.get('properties', {})
            required = params.get('required', [])
            
            if properties:
                lines.append("PARAMETERS:")
                lines.append("-" * 60)
                
                for param_name, param_info in properties.items():
                    req = "[REQUIRED]" if param_name in required else "[OPTIONAL]"
                    param_type = param_info.get('type', 'any')
                    lines.append(f"  {param_name} ({param_type}) {req}")
                    
                    if 'description' in param_info:
                        lines.append(f"    {param_info['description']}")
                    
                    if param_name in tool.constraints:
                        constraint = tool.constraints[param_name]
                        constraints_list = []
                        
                        if constraint.min_value is not None:
                            constraints_list.append(f"min: {constraint.min_value}")
                        if constraint.max_value is not None:
                            constraints_list.append(f"max: {constraint.max_value}")
                        if constraint.min_length is not None:
                            constraints_list.append(f"min_length: {constraint.min_length}")
                        if constraint.max_length is not None:
                            constraints_list.append(f"max_length: {constraint.max_length}")
                        
                        if constraints_list:
                            lines.append(f"    Constraints: {', '.join(constraints_list)}")
                    
                    lines.append("")
        
        # Examples
        if tool.examples:
            lines.append("EXAMPLES:")
            lines.append("-" * 60)
            for idx, example in enumerate(tool.examples, 1):
                lines.append(f"Example {idx}:")
                lines.append(f"  {example}")
                lines.append("")
        
        return "\n".join(lines)

    def generate_registry_docs(self, 
                            format: str = "markdown",
                            group_by_category: bool = True,
                            include_examples: bool = True) -> str:
        """
        Generate complete documentation for all tools in the registry.
        
        Creates a comprehensive documentation file covering all registered
        tools, organized by category.
        
        Args:
            format: Output format ('markdown' or 'text')
            group_by_category: Whether to group tools by category
            include_examples: Whether to include usage examples
            
        Returns:
            Complete documentation string
            
        Example:
            >>> docs = registry.generate_registry_docs()
            >>> with open("TOOL_REFERENCE.md", "w") as f:
            ...     f.write(docs)
        """
        if format == "markdown":
            return self._generate_markdown_registry_docs(group_by_category, include_examples)
        else:
            return self._generate_text_registry_docs(group_by_category, include_examples)

    def _generate_markdown_registry_docs(self, 
                                        group_by_category: bool,
                                        include_examples: bool) -> str:
        """Generate markdown documentation for entire registry."""
        lines = []
        
        # Header
        lines.append("# Tool Registry Documentation\n")
        lines.append(f"**Total Tools:** {len(self._tools)}\n")
        
        # Category summary
        lines.append("## Categories\n")
        categories = self.list_categories()
        for category, count in sorted(categories.items()):
            lines.append(f"- **{category}:** {count} tool{'s' if count != 1 else ''}")
        lines.append("\n---\n")
        
        # Tools documentation
        if group_by_category:
            # Group by category
            for category in ToolCategory:
                tools_in_category = self.get_by_category(category)
                
                if tools_in_category:
                    lines.append(f"## {category.value.replace('_', ' ').title()} Tools\n")
                    
                    for tool in sorted(tools_in_category, key=lambda t: t.name):
                        # Add tool documentation
                        tool_docs = self._generate_markdown_docs(tool)
                        
                        # Remove examples if not requested
                        if not include_examples and "## Examples" in tool_docs:
                            tool_docs = tool_docs.split("## Examples")[0]
                        
                        lines.append(tool_docs)
                        lines.append("\n---\n")
        else:
            # Alphabetical order
            lines.append("## All Tools\n")
            for tool_name in sorted(self._tools.keys()):
                tool = self._tools[tool_name]
                tool_docs = self._generate_markdown_docs(tool)
                
                if not include_examples and "## Examples" in tool_docs:
                    tool_docs = tool_docs.split("## Examples")[0]
                
                lines.append(tool_docs)
                lines.append("\n---\n")
        
        return "\n".join(lines)

    def _generate_text_registry_docs(self, 
                                    group_by_category: bool,
                                    include_examples: bool) -> str:
        """Generate text documentation for entire registry."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("TOOL REGISTRY DOCUMENTATION")
        lines.append("=" * 60)
        lines.append(f"Total Tools: {len(self._tools)}")
        lines.append("")
        
        # Category summary
        lines.append("CATEGORIES:")
        categories = self.list_categories()
        for category, count in sorted(categories.items()):
            lines.append(f"  - {category}: {count} tool(s)")
        lines.append("")
        lines.append("=" * 60)
        lines.append("")
        
        # Tools documentation
        if group_by_category:
            for category in ToolCategory:
                tools_in_category = self.get_by_category(category)
                
                if tools_in_category:
                    lines.append(f"{category.value.upper()} TOOLS")
                    lines.append("-" * 60)
                    lines.append("")
                    
                    for tool in sorted(tools_in_category, key=lambda t: t.name):
                        tool_docs = self._generate_text_docs(tool)
                        if not include_examples and "EXAMPLES:" in tool_docs:
                            tool_docs = tool_docs.split("EXAMPLES:")[0]
                        lines.append(tool_docs)
                        lines.append("")
        else:
            for tool_name in sorted(self._tools.keys()):
                tool = self._tools[tool_name]
                tool_docs = self._generate_text_docs(tool)
                if not include_examples and "EXAMPLES:" in tool_docs:
                    tool_docs = tool_docs.split("EXAMPLES:")[0]
                lines.append(tool_docs)
                lines.append("")
        
        return "\n".join(lines)

    def export_tool_catalog(self, format: str = "json") -> str:
        """
        Export tool catalog in structured format (JSON or YAML).
        
        Useful for tool discovery, IDE integration, or external documentation
        generators.
        
        Args:
            format: Export format ('json' or 'yaml')
            
        Returns:
            Serialized tool catalog
            
        Example:
            >>> catalog = registry.export_tool_catalog("json")
            >>> with open("tool_catalog.json", "w") as f:
            ...     f.write(catalog)
        """
        catalog = {
            "tool_count": len(self._tools),
            "categories": self.list_categories(),
            "tools": self.list_tools(include_schemas=True)
        }
        
        if format == "json":
            import json
            return json.dumps(catalog, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump(catalog, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")

    def generate_quick_reference(self) -> str:
        """
        Generate a quick reference guide for all tools.
        
        Creates a compact one-page reference showing tool names,
        categories, and brief descriptions.
        
        Returns:
            Quick reference markdown string
            
        Example:
            >>> ref = registry.generate_quick_reference()
            >>> print(ref)
        """
        lines = []
        
        lines.append("# Tool Quick Reference\n")
        lines.append(f"**Total Tools:** {len(self._tools)}\n")
        
        # Group by category
        for category in ToolCategory:
            tools_in_category = self.get_by_category(category)
            
            if tools_in_category:
                lines.append(f"## {category.value.replace('_', ' ').title()}\n")
                
                for tool in sorted(tools_in_category, key=lambda t: t.name):
                    # Get parameter count
                    param_count = 0
                    if tool.schema and 'function' in tool.schema:
                        params = tool.schema['function'].get('parameters', {})
                        param_count = len(params.get('properties', {}))
                    
                    # Brief one-liner
                    lines.append(f"- **`{tool.name}`** ({param_count} params) - {tool.description[:80]}{'...' if len(tool.description) > 80 else ''}")
                
                lines.append("")
        
        return "\n".join(lines)
    
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