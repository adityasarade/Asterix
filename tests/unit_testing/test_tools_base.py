"""
Unit tests for Tool class.

Tests Tool class functionality including:
- Tool initialization and configuration
- Schema generation from function signatures
- Tool execution (success and error cases)
- Parameter validation integration
- Retry logic with exponential backoff
- Error handling and formatting
- ToolResult wrapping
- Tool as callable
"""

import pytest
import time
from unittest.mock import Mock, patch
from asterix.tools.base import (
    Tool,
    ToolResult,
    ToolStatus,
    ToolCategory,
    ParameterConstraint,
    generate_tool_schema
)


class TestToolInitialization:
    """Test Tool initialization with various configurations."""
    
    def test_basic_initialization(self):
        """Test basic tool creation with minimal parameters."""
        def sample_func(x: int) -> str:
            return f"Result: {x}"
        
        tool = Tool(
            name="sample_tool",
            description="A sample tool",
            func=sample_func
        )
        
        assert tool.name == "sample_tool"
        assert tool.description == "A sample tool"
        assert tool.func == sample_func
        assert tool.category == ToolCategory.CUSTOM  # Default
        assert tool.constraints == {}
        assert tool.examples == []
        assert tool.retry_on_error is False
        assert tool.max_retries == 3
        assert tool.schema is not None
    
    def test_initialization_with_category(self):
        """Test tool creation with specific category."""
        def file_reader(path: str) -> str:
            return ""
        
        tool = Tool(
            name="read_file",
            description="Read a file",
            func=file_reader,
            category=ToolCategory.FILE_OPS
        )
        
        assert tool.category == ToolCategory.FILE_OPS
        assert tool.schema['metadata']['category'] == ToolCategory.FILE_OPS.value
    
    def test_initialization_with_constraints(self):
        """Test tool creation with parameter constraints."""
        def create_user(username: str, age: int) -> str:
            return f"User {username}"
        
        constraints = {
            "username": ParameterConstraint(min_length=3, max_length=20),
            "age": ParameterConstraint(min_value=0, max_value=120)
        }
        
        tool = Tool(
            name="create_user",
            description="Create a user",
            func=create_user,
            constraints=constraints
        )
        
        assert len(tool.constraints) == 2
        assert "username" in tool.constraints
        assert "age" in tool.constraints
    
    def test_initialization_with_examples(self):
        """Test tool creation with usage examples."""
        def calculator(x: int, y: int) -> int:
            return x + y
        
        examples = [
            "calculator(x=5, y=3)",
            "calculator(x=10, y=20)"
        ]
        
        tool = Tool(
            name="calculator",
            description="Add two numbers",
            func=calculator,
            examples=examples
        )
        
        assert tool.examples == examples
        assert tool.schema['metadata']['examples'] == examples
    
    def test_initialization_with_retry(self):
        """Test tool creation with retry configuration."""
        def flaky_api() -> str:
            return "data"
        
        tool = Tool(
            name="flaky_api",
            description="Flaky API call",
            func=flaky_api,
            retry_on_error=True,
            max_retries=5
        )
        
        assert tool.retry_on_error is True
        assert tool.max_retries == 5
    
    def test_schema_auto_generation(self):
        """Test that schema is auto-generated from function signature."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        tool = Tool(
            name="add",
            description="Add numbers",
            func=add
        )
        
        assert tool.schema is not None
        assert tool.schema['type'] == 'function'
        assert 'function' in tool.schema
        assert tool.schema['function']['name'] == 'add'
        assert 'parameters' in tool.schema['function']


class TestToolExecution:
    """Test Tool execution for success cases."""
    
    def test_execute_simple_function(self):
        """Test executing a simple function that returns a string."""
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        tool = Tool(name="greet", description="Greet someone", func=greet)
        result = tool.execute(name="Alice")
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.content == "Hello, Alice!"
        assert result.error is None
        assert result.metadata['tool'] == 'greet'
    
    def test_execute_function_with_multiple_params(self):
        """Test executing function with multiple parameters."""
        def calculate(x: int, y: int, operation: str) -> int:
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            return 0
        
        tool = Tool(name="calc", description="Calculate", func=calculate)
        result = tool.execute(x=5, y=3, operation="add")
        
        assert result.status == ToolStatus.SUCCESS
        assert result.content == "8"
    
    def test_execute_function_returning_toolresult(self):
        """Test function that already returns ToolResult."""
        def advanced_tool() -> ToolResult:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                content="Custom result",
                metadata={"custom_key": "custom_value"}
            )
        
        tool = Tool(name="advanced", description="Advanced tool", func=advanced_tool)
        result = tool.execute()
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.content == "Custom result"
        assert result.metadata['custom_key'] == "custom_value"
    
    def test_execute_function_with_side_effects(self):
        """Test function that has side effects but returns None."""
        executed = []
        
        def side_effect_tool(value: str):
            executed.append(value)
            return f"Added {value}"
        
        tool = Tool(name="side_effect", description="Has side effects", func=side_effect_tool)
        result = tool.execute(value="test")
        
        assert "test" in executed
        assert result.status == ToolStatus.SUCCESS
        assert "Added test" in result.content


class TestToolParameterValidation:
    """Test parameter validation within Tool.execute()."""
    
    def test_execute_with_valid_parameters(self):
        """Test execution passes when parameters are valid."""
        def create_user(username: str, age: int) -> str:
            return f"Created user {username}"
        
        tool = Tool(
            name="create_user",
            description="Create user",
            func=create_user,
            constraints={
                "username": ParameterConstraint(min_length=3),
                "age": ParameterConstraint(min_value=18)
            }
        )
        
        result = tool.execute(username="alice", age=25)
        assert result.status == ToolStatus.SUCCESS
    
    def test_execute_with_invalid_min_length(self):
        """Test execution fails when string too short."""
        def create_user(username: str) -> str:
            return f"Created {username}"
        
        tool = Tool(
            name="create_user",
            description="Create user",
            func=create_user,
            constraints={
                "username": ParameterConstraint(min_length=5)
            }
        )
        
        result = tool.execute(username="ab")
        
        assert result.status == ToolStatus.ERROR
        assert "Parameter validation failed" in result.error
        assert "length must be >= 5" in result.error
        assert result.metadata['validation_error'] is True
    
    def test_execute_with_invalid_max_value(self):
        """Test execution fails when value exceeds maximum."""
        def set_volume(level: int) -> str:
            return f"Volume: {level}"
        
        tool = Tool(
            name="set_volume",
            description="Set volume",
            func=set_volume,
            constraints={
                "level": ParameterConstraint(max_value=100)
            }
        )
        
        result = tool.execute(level=150)
        
        assert result.status == ToolStatus.ERROR
        assert "level must be <= 100" in result.error
    
    def test_execute_with_invalid_pattern(self):
        """Test execution fails when pattern doesn't match."""
        def validate_code(code: str) -> str:
            return f"Code: {code}"
        
        tool = Tool(
            name="validate_code",
            description="Validate code",
            func=validate_code,
            constraints={
                "code": ParameterConstraint(pattern=r'^[A-Z]{3}\d{3}$')
            }
        )
        
        result = tool.execute(code="invalid")
        
        assert result.status == ToolStatus.ERROR
        assert "must match pattern" in result.error


class TestToolRetryLogic:
    """Test retry logic with exponential backoff."""
    
    def test_no_retry_on_success(self):
        """Test no retries when function succeeds first time."""
        call_count = []
        
        def succeeds_immediately() -> str:
            call_count.append(1)
            return "success"
        
        tool = Tool(
            name="succeeds",
            description="Succeeds",
            func=succeeds_immediately,
            retry_on_error=True,
            max_retries=3
        )
        
        result = tool.execute()
        
        assert len(call_count) == 1
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata.get('retried', False) is False
    
    def test_retry_on_failure_then_success(self):
        """Test retries until success within max attempts."""
        attempts = []
        
        def fails_twice() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")
            return "success after retries"
        
        tool = Tool(
            name="flaky",
            description="Flaky tool",
            func=fails_twice,
            retry_on_error=True,
            max_retries=3
        )
        
        # Mock sleep to avoid actual delays in tests
        with patch('time.sleep'):
            result = tool.execute()
        
        assert len(attempts) == 3
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata['retried'] is True
        assert result.metadata['attempts'] == 3
    
    def test_retry_exhausted_returns_error(self):
        """Test returns error when all retries exhausted."""
        def always_fails() -> str:
            raise RuntimeError("Persistent failure")
        
        tool = Tool(
            name="broken",
            description="Broken tool",
            func=always_fails,
            retry_on_error=True,
            max_retries=2
        )
        
        with patch('time.sleep'):
            result = tool.execute()
        
        assert result.status == ToolStatus.ERROR
        assert "Persistent failure" in result.error
        assert result.metadata['attempt'] == 3  # Initial + 2 retries
    
    def test_no_retry_when_disabled(self):
        """Test no retries when retry_on_error is False."""
        attempts = []
        
        def fails_always() -> str:
            attempts.append(1)
            raise ValueError("Error")
        
        tool = Tool(
            name="no_retry",
            description="No retry",
            func=fails_always,
            retry_on_error=False
        )
        
        result = tool.execute()
        
        assert len(attempts) == 1  # Only one attempt
        assert result.status == ToolStatus.ERROR


class TestToolErrorHandling:
    """Test error handling and error message formatting."""
    
    def test_error_message_includes_exception(self):
        """Test error message includes original exception."""
        def raises_error() -> str:
            raise ValueError("Something went wrong")
        
        tool = Tool(name="error_tool", description="Errors", func=raises_error)
        result = tool.execute()
        
        assert result.status == ToolStatus.ERROR
        assert "Something went wrong" in result.error
        assert result.metadata['exception_type'] == "ValueError"
    
    def test_error_hint_for_file_not_found(self):
        """Test helpful hint added for FileNotFoundError."""
        def read_missing_file() -> str:
            raise FileNotFoundError("File not found")
        
        tool = Tool(name="read_file", description="Read file", func=read_missing_file)
        result = tool.execute()
        
        assert result.status == ToolStatus.ERROR
        assert "file path is correct" in result.error.lower()
    
    def test_error_hint_for_permission_error(self):
        """Test helpful hint added for PermissionError."""
        def restricted_operation() -> str:
            raise PermissionError("Access denied")
        
        tool = Tool(name="restricted", description="Restricted", func=restricted_operation)
        result = tool.execute()
        
        assert result.status == ToolStatus.ERROR
        assert "permission" in result.error.lower()
    
    def test_error_metadata_includes_context(self):
        """Test error result includes execution context."""
        def fails_with_params(x: int, y: str) -> str:
            raise RuntimeError("Failed")
        
        tool = Tool(name="context_tool", description="Context", func=fails_with_params)
        result = tool.execute(x=10, y="test")
        
        assert result.status == ToolStatus.ERROR
        assert result.metadata['tool'] == 'context_tool'
        assert result.metadata['parameters'] == {'x': 10, 'y': 'test'}


class TestToolCallable:
    """Test Tool as a callable object."""
    
    def test_tool_is_callable(self):
        """Test tool can be called directly using () syntax."""
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(name="add", description="Add", func=add)
        result = tool(a=5, b=3)
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.content == "8"
    
    def test_callable_with_validation(self):
        """Test callable syntax works with validation."""
        def restricted(value: int) -> str:
            return f"Value: {value}"
        
        tool = Tool(
            name="restricted",
            description="Restricted",
            func=restricted,
            constraints={"value": ParameterConstraint(max_value=10)}
        )
        
        result = tool(value=5)
        assert result.status == ToolStatus.SUCCESS
        
        result_invalid = tool(value=20)
        assert result_invalid.status == ToolStatus.ERROR


class TestToolSchemaGeneration:
    """Test automatic schema generation from function signatures."""
    
    def test_schema_generation_basic(self):
        """Test schema generated for basic function."""
        def simple_func(x: int, y: str) -> str:
            return f"{y}: {x}"
        
        schema = generate_tool_schema(simple_func, "simple", "A simple function")
        
        assert schema['type'] == 'function'
        assert schema['function']['name'] == 'simple'
        assert schema['function']['description'] == 'A simple function'
        assert 'parameters' in schema['function']
        
        params = schema['function']['parameters']
        assert 'x' in params['properties']
        assert 'y' in params['properties']
        assert params['properties']['x']['type'] == 'integer'
        assert params['properties']['y']['type'] == 'string'
    
    def test_schema_required_parameters(self):
        """Test required vs optional parameters in schema."""
        def func_with_defaults(required: str, optional: int = 5) -> str:
            return f"{required}-{optional}"
        
        schema = generate_tool_schema(func_with_defaults, "func", "Function")
        
        required_params = schema['function']['parameters']['required']
        assert 'required' in required_params
        assert 'optional' not in required_params


class TestToolRepr:
    """Test string representation of tools."""
    
    def test_repr_format(self):
        """Test __repr__ returns useful information."""
        def sample() -> str:
            return "test"
        
        tool = Tool(name="sample_tool", description="A sample tool", func=sample)
        repr_str = repr(tool)
        
        assert "Tool" in repr_str
        assert "sample_tool" in repr_str
        assert "A sample tool" in repr_str