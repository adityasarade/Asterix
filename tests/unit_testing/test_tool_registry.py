"""
Unit tests for ToolRegistry class.

Tests ToolRegistry functionality including:
- Tool registration and unregistration
- Tool retrieval (get, get_all, get_schemas)
- Tool execution by name
- Category-based operations
- Tool filtering and discovery
- Error handling with helpful suggestions
- Documentation generation (markdown, text, JSON)
- Tool catalog export
"""

import pytest
import json
from asterix.tools.base import (
    Tool,
    ToolResult,
    ToolStatus,
    ToolRegistry,
    ToolCategory,
    ParameterConstraint,
    ToolNotFoundError,
    ToolRegistrationError
)


class TestToolRegistryBasics:
    """Test basic ToolRegistry operations."""
    
    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = ToolRegistry()
        
        assert len(registry) == 0
        assert registry.get_all_tools() == []
    
    def test_register_single_tool(self):
        """Test registering a single tool."""
        registry = ToolRegistry()
        
        def sample_func(x: int) -> str:
            return f"Result: {x}"
        
        tool = Tool(name="sample", description="Sample tool", func=sample_func)
        registry.register(tool)
        
        assert len(registry) == 1
        assert "sample" in registry
        # Test retrieval works
        assert registry.get("sample") is not None
    
    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        
        def func1() -> str:
            return "1"
        
        def func2() -> str:
            return "2"
        
        tool1 = Tool(name="tool1", description="First", func=func1)
        tool2 = Tool(name="tool2", description="Second", func=func2)
        
        registry.register(tool1)
        registry.register(tool2)
        
        assert len(registry) == 2
        assert "tool1" in registry
        assert "tool2" in registry
    
    def test_register_duplicate_raises_error(self):
        """Test registering duplicate tool name raises error."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool1 = Tool(name="duplicate", description="First", func=func)
        tool2 = Tool(name="duplicate", description="Second", func=func)
        
        registry.register(tool1)
        
        with pytest.raises(ToolRegistrationError) as exc_info:
            registry.register(tool2)
        
        assert "duplicate" in str(exc_info.value).lower()
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(name="removable", description="Remove me", func=func)
        registry.register(tool)
        
        assert "removable" in registry
        
        registry.unregister("removable")
        
        assert "removable" not in registry
        assert len(registry) == 0
    
    def test_unregister_nonexistent_tool(self):
        """Test unregistering non-existent tool handles gracefully."""
        registry = ToolRegistry()
        
        # Should not raise error, just log warning
        # This is the actual behavior - unregister silently handles missing tools
        registry.unregister("nonexistent")
        
        # Registry should still be empty
        assert len(registry) == 0
    
    def test_contains_operator(self):
        """Test 'in' operator for checking tool existence."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(name="exists", description="Exists", func=func)
        registry.register(tool)
        
        assert "exists" in registry
        assert "nonexistent" not in registry


class TestToolRegistryRetrieval:
    """Test retrieving tools from registry."""
    
    def test_get_existing_tool(self):
        """Test getting an existing tool."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        original_tool = Tool(name="getter", description="Get me", func=func)
        registry.register(original_tool)
        
        retrieved_tool = registry.get("getter")
        
        assert retrieved_tool is not None
        assert retrieved_tool.name == "getter"
        assert retrieved_tool.description == "Get me"
    
    def test_get_nonexistent_tool_returns_none(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()
        
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_get_all_tools(self):
        """Test getting all registered tools."""
        registry = ToolRegistry()
        
        def func1() -> str:
            return "1"
        
        def func2() -> str:
            return "2"
        
        tool1 = Tool(name="tool1", description="First", func=func1)
        tool2 = Tool(name="tool2", description="Second", func=func2)
        
        registry.register(tool1)
        registry.register(tool2)
        
        all_tools = registry.get_all_tools()
        
        assert len(all_tools) == 2
        assert any(t.name == "tool1" for t in all_tools)
        assert any(t.name == "tool2" for t in all_tools)
    
    def test_get_tool_schemas(self):
        """Test getting OpenAI function schemas for all tools."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(name="add", description="Add numbers", func=add)
        registry.register(tool)
        
        schemas = registry.get_tool_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]['type'] == 'function'
        assert schemas[0]['function']['name'] == 'add'
    
    def test_execute_tool_by_name(self):
        """Test executing a tool by name."""
        registry = ToolRegistry()
        
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        tool = Tool(name="greet", description="Greet", func=greet)
        registry.register(tool)
        
        result = registry.execute_tool("greet", name="Alice")
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert "Hello, Alice!" in result.content
    
    def test_execute_nonexistent_tool_returns_error(self):
        """Test executing non-existent tool returns error result."""
        registry = ToolRegistry()
        
        result = registry.execute_tool("nonexistent", arg="value")
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.ERROR
        assert "not found" in result.error.lower()


class TestToolRegistryCategories:
    """Test category-based operations."""
    
    def test_get_by_category_empty(self):
        """Test getting tools from empty category."""
        registry = ToolRegistry()
        
        memory_tools = registry.get_by_category(ToolCategory.MEMORY)
        
        assert memory_tools == []
    
    def test_get_by_category_with_tools(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        
        def mem_func() -> str:
            return "memory"
        
        def file_func() -> str:
            return "file"
        
        mem_tool = Tool(
            name="memory_tool",
            description="Memory",
            func=mem_func,
            category=ToolCategory.MEMORY
        )
        
        file_tool = Tool(
            name="file_tool",
            description="File",
            func=file_func,
            category=ToolCategory.FILE_OPS
        )
        
        registry.register(mem_tool)
        registry.register(file_tool)
        
        memory_tools = registry.get_by_category(ToolCategory.MEMORY)
        file_tools = registry.get_by_category(ToolCategory.FILE_OPS)
        
        assert len(memory_tools) == 1
        assert memory_tools[0].name == "memory_tool"
        
        assert len(file_tools) == 1
        assert file_tools[0].name == "file_tool"
    
    def test_list_categories(self):
        """Test listing categories with counts."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        # Register multiple tools in different categories
        for i in range(3):
            tool = Tool(
                name=f"mem_{i}",
                description="Memory",
                func=func,
                category=ToolCategory.MEMORY
            )
            registry.register(tool)
        
        for i in range(2):
            tool = Tool(
                name=f"file_{i}",
                description="File",
                func=func,
                category=ToolCategory.FILE_OPS
            )
            registry.register(tool)
        
        categories = registry.list_categories()
        
        assert categories["memory"] == 3
        assert categories["file_operations"] == 2
    
    def test_count_by_category(self):
        """Test counting tools in a specific category."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        for i in range(5):
            tool = Tool(
                name=f"web_{i}",
                description="Web",
                func=func,
                category=ToolCategory.WEB
            )
            registry.register(tool)
        
        count = registry.count_by_category(ToolCategory.WEB)
        
        assert count == 5


class TestToolRegistryFiltering:
    """Test advanced filtering operations."""
    
    def test_filter_by_category(self):
        """Test filtering tools by category."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        mem_tool = Tool(name="mem", description="Memory", func=func, category=ToolCategory.MEMORY)
        file_tool = Tool(name="file", description="File", func=func, category=ToolCategory.FILE_OPS)
        
        registry.register(mem_tool)
        registry.register(file_tool)
        
        filtered = registry.filter_tools(category=ToolCategory.MEMORY)
        
        assert len(filtered) == 1
        assert filtered[0].name == "mem"
    
    def test_filter_by_name_pattern(self):
        """Test filtering by name pattern."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tools = [
            Tool(name="search_files", description="Search", func=func),
            Tool(name="search_memory", description="Search", func=func),
            Tool(name="read_file", description="Read", func=func)
        ]
        
        for tool in tools:
            registry.register(tool)
        
        filtered = registry.filter_tools(name_pattern="search")
        
        assert len(filtered) == 2
        assert all("search" in t.name for t in filtered)
    
    def test_filter_by_retry_capability(self):
        """Test filtering by retry capability."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        retry_tool = Tool(
            name="retry_tool",
            description="Retry",
            func=func,
            retry_on_error=True
        )
        
        no_retry_tool = Tool(
            name="no_retry_tool",
            description="No retry",
            func=func,
            retry_on_error=False
        )
        
        registry.register(retry_tool)
        registry.register(no_retry_tool)
        
        filtered = registry.filter_tools(has_retry=True)
        
        assert len(filtered) == 1
        assert filtered[0].name == "retry_tool"
    
    def test_filter_by_validation(self):
        """Test filtering by validation constraints."""
        registry = ToolRegistry()
        
        def func(x: str) -> str:
            return x
        
        validated_tool = Tool(
            name="validated",
            description="Validated",
            func=func,
            constraints={"x": ParameterConstraint(min_length=3)}
        )
        
        unvalidated_tool = Tool(
            name="unvalidated",
            description="Unvalidated",
            func=func
        )
        
        registry.register(validated_tool)
        registry.register(unvalidated_tool)
        
        filtered = registry.filter_tools(has_validation=True)
        
        assert len(filtered) == 1
        assert filtered[0].name == "validated"
    
    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria."""
        registry = ToolRegistry()
        
        def func(x: str) -> str:
            return x
        
        # Tool matching all criteria
        matching_tool = Tool(
            name="memory_search_validated",
            description="Search",
            func=func,
            category=ToolCategory.MEMORY,
            constraints={"x": ParameterConstraint(min_length=1)},
            retry_on_error=True
        )
        
        # Tool not matching
        non_matching_tool = Tool(
            name="file_read",
            description="Read",
            func=func,
            category=ToolCategory.FILE_OPS
        )
        
        registry.register(matching_tool)
        registry.register(non_matching_tool)
        
        filtered = registry.filter_tools(
            category=ToolCategory.MEMORY,
            name_pattern="search",
            has_validation=True,
            has_retry=True
        )
        
        assert len(filtered) == 1
        assert filtered[0].name == "memory_search_validated"
    
    def test_list_tools(self):
        """Test listing all tools with metadata."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(
            name="list_test",
            description="Test",
            func=func,
            category=ToolCategory.DATA,
            retry_on_error=True,
            max_retries=5
        )
        
        registry.register(tool)
        
        tools_list = registry.list_tools()
        
        assert len(tools_list) == 1
        tool_info = tools_list[0]
        
        assert tool_info['name'] == "list_test"
        assert tool_info['category'] == "data_processing"
        assert tool_info['retry_enabled'] is True
        assert tool_info['max_retries'] == 5


class TestToolRegistryErrors:
    """Test error handling and helpful suggestions."""
    
    def test_tool_not_found_error_with_suggestions(self):
        """Test helpful error messages when tool not found during execution."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(name="read_file", description="Read", func=func)
        registry.register(tool)
        
        # Execute non-existent tool returns error result with context
        result = registry.execute_tool("read_fil")  # Typo
        
        assert result.status == ToolStatus.ERROR
        assert "read_fil" in result.error
        assert "not found" in result.error.lower()
    
    def test_execute_tool_error_includes_context(self):
        """Test execution errors include helpful context."""
        registry = ToolRegistry()
        
        def failing_func(x: int) -> str:
            raise ValueError("Invalid value")
        
        tool = Tool(name="failer", description="Fails", func=failing_func)
        registry.register(tool)
        
        result = registry.execute_tool("failer", x=5)
        
        assert result.status == ToolStatus.ERROR
        assert "Invalid value" in result.error
    
    def test_get_tool_info_for_nonexistent(self):
        """Test getting info for non-existent tool returns None."""
        registry = ToolRegistry()
        
        info = registry.get_tool_info("nonexistent")
        
        assert info is None
    
    def test_get_tool_info_complete(self):
        """Test get_tool_info returns complete metadata."""
        registry = ToolRegistry()
        
        def func(x: str) -> str:
            return x
        
        tool = Tool(
            name="info_test",
            description="Test info",
            func=func,
            category=ToolCategory.CUSTOM,
            constraints={"x": ParameterConstraint(min_length=3, max_length=10)},
            examples=["info_test(x='hello')"],
            retry_on_error=True,
            max_retries=2
        )
        
        registry.register(tool)
        
        info = registry.get_tool_info("info_test")
        
        assert info is not None
        assert info['name'] == "info_test"
        assert info['category'] == "custom"
        assert 'x' in info['constraints']
        assert info['constraints']['x']['min_length'] == 3
        assert info['constraints']['x']['max_length'] == 10
        assert len(info['examples']) == 1
        assert info['retry_enabled'] is True
        assert info['max_retries'] == 2


class TestToolRegistryDocumentation:
    """Test documentation generation features."""
    
    def test_generate_tool_docs_markdown(self):
        """Test generating markdown documentation for a tool."""
        registry = ToolRegistry()
        
        def func(x: str, y: int) -> str:
            return f"{x}-{y}"
        
        tool = Tool(
            name="doc_test",
            description="Test documentation",
            func=func,
            category=ToolCategory.DATA,
            constraints={
                "x": ParameterConstraint(min_length=3),
                "y": ParameterConstraint(min_value=0, max_value=100)
            },
            examples=["doc_test(x='test', y=50)"]
        )
        
        registry.register(tool)
        
        docs = registry.generate_tool_docs("doc_test", format="markdown")
        
        assert docs is not None
        assert "# doc_test" in docs
        assert "Test documentation" in docs
        assert "data_processing" in docs
        assert "Parameters" in docs
        assert "Examples" in docs
    
    def test_generate_tool_docs_text(self):
        """Test generating text documentation for a tool."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(name="text_doc", description="Text docs", func=func)
        registry.register(tool)
        
        docs = registry.generate_tool_docs("text_doc", format="text")
        
        assert docs is not None
        assert "TOOL: text_doc" in docs
        assert "Text docs" in docs
    
    def test_generate_tool_docs_nonexistent(self):
        """Test generating docs for non-existent tool returns None."""
        registry = ToolRegistry()
        
        docs = registry.generate_tool_docs("nonexistent")
        
        assert docs is None
    
    def test_generate_registry_docs_markdown(self):
        """Test generating complete registry documentation."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool1 = Tool(name="tool1", description="First", func=func, category=ToolCategory.MEMORY)
        tool2 = Tool(name="tool2", description="Second", func=func, category=ToolCategory.FILE_OPS)
        
        registry.register(tool1)
        registry.register(tool2)
        
        docs = registry.generate_registry_docs(format="markdown", group_by_category=True)
        
        assert "# Tool Registry Documentation" in docs
        assert "**Total Tools:** 2" in docs  # Markdown bold format
        assert "memory" in docs.lower()
        assert "file_operations" in docs.lower()
    
    def test_export_tool_catalog_json(self):
        """Test exporting tool catalog as JSON."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool = Tool(name="catalog_test", description="Catalog", func=func)
        registry.register(tool)
        
        catalog_json = registry.export_tool_catalog(format="json")
        catalog = json.loads(catalog_json)
        
        assert catalog['tool_count'] == 1
        assert 'categories' in catalog
        assert 'tools' in catalog
        assert len(catalog['tools']) == 1
        assert catalog['tools'][0]['name'] == "catalog_test"
    
    def test_export_tool_catalog_invalid_format(self):
        """Test exporting with invalid format raises error."""
        registry = ToolRegistry()
        
        with pytest.raises(ValueError) as exc_info:
            registry.export_tool_catalog(format="invalid")
        
        assert "Unsupported format" in str(exc_info.value)


class TestToolRegistryRepr:
    """Test string representation of registry."""
    
    def test_repr_empty_registry(self):
        """Test repr of empty registry."""
        registry = ToolRegistry()
        
        repr_str = repr(registry)
        
        assert "ToolRegistry" in repr_str
        assert "[]" in repr_str
    
    def test_repr_with_tools(self):
        """Test repr includes tool names."""
        registry = ToolRegistry()
        
        def func() -> str:
            return "test"
        
        tool1 = Tool(name="tool_a", description="A", func=func)
        tool2 = Tool(name="tool_b", description="B", func=func)
        
        registry.register(tool1)
        registry.register(tool2)
        
        repr_str = repr(registry)
        
        assert "ToolRegistry" in repr_str
        assert "tool_a" in repr_str
        assert "tool_b" in repr_str