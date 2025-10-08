"""
Asterix - Main Agent Class

The core Agent class that provides stateful AI agents with editable memory blocks
and persistent storage.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from pathlib import Path

from .core.config import (
    AgentConfig,
    BlockConfig,
    MemoryConfig,
    StorageConfig,
    ConfigurationManager,
    get_config_manager,
    create_default_blocks
)

from .tools import (
    ToolRegistry,
    Tool,
    ToolResult,
    create_core_memory_tools,
    create_archival_memory_tools,
    create_conversation_search_tool
)

logger = logging.getLogger(__name__)


class MemoryBlock:
    """
    Represents a single memory block with content and metadata.
    
    Memory blocks are the core storage units that agents can read and write.
    Each block has a size limit and priority for eviction management.
    """
    
    def __init__(self, name: str, config: BlockConfig):
        """
        Initialize a memory block.
        
        Args:
            name: Block identifier
            config: Block configuration
        """
        self.name = name
        self.config = config
        self.content = config.initial_value
        self.tokens = 0  # Will be calculated on first update
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at
    
    def update_content(self, new_content: str):
        """
        Update the block's content.
        
        Args:
            new_content: New content for the block
        """
        self.content = new_content
        self.last_updated = datetime.now(timezone.utc)
        # Token count will be calculated by agent
    
    def append_content(self, additional_content: str):
        """
        Append content to the block.
        
        Args:
            additional_content: Content to append
        """
        if self.content:
            self.content += "\n" + additional_content
        else:
            self.content = additional_content
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for serialization."""
        return {
            "name": self.name,
            "content": self.content,
            "tokens": self.tokens,
            "size_limit": self.config.size,
            "priority": self.config.priority,
            "description": self.config.description,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Create block from dictionary."""
        config = BlockConfig(
            size=data["size_limit"],
            priority=data["priority"],
            description=data.get("description", ""),
            initial_value=""
        )
        
        block = cls(data["name"], config)
        block.content = data["content"]
        block.tokens = data.get("tokens", 0)
        block.created_at = datetime.fromisoformat(data["created_at"])
        block.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return block


class Agent:
    """
    Main Agent class.
    
    An agent is a stateful AI entity with:
    - Editable memory blocks (persona, task, user info, etc.)
    - Persistent storage via Qdrant Cloud
    - Tool execution capabilities
    - State persistence across sessions
    
    Example:
        >>> agent = Agent(
        ...     agent_id="my_agent",
        ...     blocks={
        ...         "task": BlockConfig(size=1500, priority=1),
        ...         "notes": BlockConfig(size=1000, priority=2)
        ...     },
        ...     model="groq/llama-3.3-70b-versatile"
        ... )
        >>> response = agent.chat("Hello! Remember that I like Python.")
        >>> agent.save_state()
    """
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 blocks: Optional[Dict[str, BlockConfig]] = None,
                 model: str = "groq/llama-3.3-70b-versatile",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 max_heartbeat_steps: int = 10,
                 config: Optional[AgentConfig] = None,
                 **kwargs):
        """
        Initialize an Agent.
        
        Args:
            agent_id: Unique identifier (auto-generated if not provided)
            blocks: Dictionary of memory blocks (uses defaults if not provided)
            model: LLM model string (format: "provider/model-name")
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens for LLM responses
            max_heartbeat_steps: Maximum tool execution loop iterations
            config: Full AgentConfig object (overrides other args if provided)
            **kwargs: Additional config options (storage, memory, embedding)
        
        Example:
            >>> # Simple initialization
            >>> agent = Agent(model="groq/llama-3.3-70b-versatile")
            
            >>> # With custom blocks
            >>> agent = Agent(
            ...     agent_id="coder",
            ...     blocks={
            ...         "code": BlockConfig(size=2000, priority=1),
            ...         "plan": BlockConfig(size=1000, priority=2)
            ...     }
            ... )
            
            >>> # With full config
            >>> config = AgentConfig(...)
            >>> agent = Agent(config=config)
        """
        # Use provided config or build from arguments
        if config is not None:
            self.config = config
        else:
            # Generate agent_id if not provided
            if agent_id is None:
                agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            
            # Use provided blocks or create defaults
            if blocks is None:
                blocks = create_default_blocks()
            
            # Build config from arguments
            self.config = AgentConfig(
                agent_id=agent_id,
                blocks=blocks,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_heartbeat_steps=max_heartbeat_steps,
                **kwargs
            )
        
        # Set agent identity
        self.id = self.config.agent_id
        
        # Initialize memory blocks
        self.blocks: Dict[str, MemoryBlock] = {}
        self._initialize_blocks()
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize tool registry
        self._tool_registry = ToolRegistry()
        
        # Auto-register built-in memory tools
        self._register_memory_tools()
        
        # Legacy tool tracking (will be replaced by registry in later steps)
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Service connections (will be initialized on first use)
        self._llm_manager = None
        self._embedding_service = None
        self._qdrant_client = None
        
        # State tracking
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at
        
        logger.info(f"Initialized agent '{self.id}' with {len(self.blocks)} memory blocks")
    
    def _initialize_blocks(self):
        """Initialize memory blocks from configuration."""
        for block_name, block_config in self.config.blocks.items():
            self.blocks[block_name] = MemoryBlock(block_name, block_config)
        
        logger.debug(f"Initialized blocks: {list(self.blocks.keys())}")
    
    # ========================================================================
    # Memory Block Management
    # ========================================================================
    
    def get_memory(self, block_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get memory block content.
        
        Args:
            block_name: Specific block to retrieve (None = all blocks)
            
        Returns:
            Dictionary mapping block names to their content
            
        Example:
            >>> # Get all blocks
            >>> memory = agent.get_memory()
            >>> print(memory["task"])
            
            >>> # Get specific block
            >>> task_content = agent.get_memory("task")
        """
        if block_name:
            if block_name not in self.blocks:
                raise ValueError(f"Block '{block_name}' does not exist")
            return {block_name: self.blocks[block_name].content}
        
        # Return all blocks
        return {name: block.content for name, block in self.blocks.items()}
    
    def update_memory(self, block_name: str, content: str):
        """
        Update a memory block's content.
        
        Args:
            block_name: Name of the block to update
            content: New content for the block
            
        Raises:
            ValueError: If block doesn't exist
            
        Example:
            >>> agent.update_memory("task", "Review authentication code")
        """
        if block_name not in self.blocks:
            raise ValueError(f"Block '{block_name}' does not exist")
        
        self.blocks[block_name].update_content(content)
        self.last_updated = datetime.now(timezone.utc)
        
        # TODO: Check token limits and trigger eviction if needed (Step 4)
        
        logger.debug(f"Updated block '{block_name}'")
    
    def append_to_memory(self, block_name: str, content: str):
        """
        Append content to a memory block.
        
        Args:
            block_name: Name of the block to append to
            content: Content to append
            
        Raises:
            ValueError: If block doesn't exist
            
        Example:
            >>> agent.append_to_memory("notes", "User prefers dark mode")
        """
        if block_name not in self.blocks:
            raise ValueError(f"Block '{block_name}' does not exist")
        
        self.blocks[block_name].append_content(content)
        self.last_updated = datetime.now(timezone.utc)
        
        # TODO: Check token limits and trigger eviction if needed (Step 4)
        
        logger.debug(f"Appended to block '{block_name}'")
    
    def get_block_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all memory blocks.
        
        Returns:
            Dictionary with block metadata (tokens, limits, priorities)
            
        Example:
            >>> info = agent.get_block_info()
            >>> for block_name, data in info.items():
            ...     print(f"{block_name}: {data['tokens']}/{data['size_limit']} tokens")
        """
        return {
            name: {
                "tokens": block.tokens,
                "size_limit": block.config.size,
                "priority": block.config.priority,
                "description": block.config.description,
                "last_updated": block.last_updated.isoformat()
            }
            for name, block in self.blocks.items()
        }
    
    # ========================================================================
    # Tool Management
    # ========================================================================
    
    def _register_memory_tools(self):
        """
        Register the 5 built-in memory tools.
        
        Called automatically during Agent initialization. These tools allow
        the agent to:
        - Edit its own memory blocks (core_memory_append, core_memory_replace)
        - Store/retrieve long-term memories (archival_memory_insert, archival_memory_search)
        - Search conversation history (conversation_search)
        
        Internal method - users don't need to call this.
        """
        try:
            # Register core memory tools (append, replace)
            core_tools = create_core_memory_tools(self)
            for tool_name, tool in core_tools.items():
                self._tool_registry.register(tool)
                logger.debug(f"Registered core memory tool: {tool_name}")
            
            # Register archival memory tools (insert, search)
            archival_tools = create_archival_memory_tools(self)
            for tool_name, tool in archival_tools.items():
                self._tool_registry.register(tool)
                logger.debug(f"Registered archival memory tool: {tool_name}")
            
            # Register conversation search tool
            conversation_tool = create_conversation_search_tool(self)
            self._tool_registry.register(conversation_tool)
            logger.debug(f"Registered conversation search tool: {conversation_tool.name}")
            
            logger.info(f"Registered {len(self._tool_registry)} built-in memory tools")
            
        except Exception as e:
            logger.error(f"Failed to register memory tools: {e}")
            raise RuntimeError(f"Memory tool registration failed: {e}")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function schemas for all registered tools.
        
        This is used by the heartbeat loop to tell the LLM what tools
        are available for function calling.
        
        Returns:
            List of tool schemas in OpenAI function calling format
            
        Example:
            >>> schemas = agent.get_tool_schemas()
            >>> # Returns schemas for all registered tools
            >>> # Used in Step 2 for LLM completion calls
        """
        return self._tool_registry.get_tool_schemas()
    
    # ========================================================================
    # Chat Interface (Stub - will implement in Step 2)
    # ========================================================================
    
    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        This is the main interface for interacting with the agent.
        The agent will process the message, potentially call tools to update
        its memory or retrieve information, and return a response.
        
        Args:
            message: User message
            
        Returns:
            Agent's response
            
        Example:
            >>> response = agent.chat("What's the current task?")
            >>> print(response)
        
        Note:
            Full implementation coming in Step 2 (Heartbeat Loop)
        """
        # TODO: Implement full heartbeat loop in Step 2
        logger.warning("chat() not fully implemented yet - coming in Step 2")
        
        # For now, just store message and return placeholder
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return "Agent chat functionality will be implemented in Step 2 (Heartbeat Loop)"
    
    # ========================================================================
    # Tool Registration
    # ========================================================================
    
    def tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Decorator for registering custom tools.
        
        Allows users to easily add custom capabilities to the agent by
        decorating functions. The function signature is automatically
        converted to an OpenAI tool schema.
        
        Args:
            name: Tool name (uses function name if not provided)
            description: Tool description for LLM (uses docstring if not provided)
            
        Returns:
            Decorator function
            
        Example:
            >>> @agent.tool(name="read_file", description="Read a file from disk")
            >>> def read_file(filepath: str) -> str:
            ...     '''Read contents of a file'''
            ...     with open(filepath, 'r') as f:
            ...         return f.read()
            >>> 
            >>> # Now agent can call read_file() during conversations
            >>> response = agent.chat("Read config.yaml and summarize it")
            >>> # Agent will automatically call the read_file tool
            
        Note:
            The function can return either a string/value (wrapped in ToolResult)
            or a ToolResult directly for more control over status and metadata.
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__
            
            # Create Tool object from function
            tool_obj = Tool(
                name=tool_name,
                description=tool_description,
                func=func
            )
            
            # Register with tool registry
            try:
                self._tool_registry.register(tool_obj)
                logger.info(f"Registered custom tool: {tool_name}")
            except ValueError as e:
                # Tool name already exists
                logger.warning(f"Tool '{tool_name}' already registered, skipping")
            
            # Also store in legacy _tools dict for backward compatibility
            self._tools[tool_name] = func
            
            # Return original function so it can still be called normally
            return func
        
        return decorator
    
    def register_tool(self, tool: Tool):
        """
        Register a Tool object directly with the agent.
        
        Use this when you have a Tool object (not just a function).
        Most users will prefer the @agent.tool() decorator.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool name already exists
            
        Example:
            >>> from asterix.tools import Tool, ToolResult, ToolStatus
            >>> 
            >>> class CustomTool(Tool):
            ...     def execute(self, arg: str) -> ToolResult:
            ...         return ToolResult(
            ...             status=ToolStatus.SUCCESS,
            ...             content=f"Processed: {arg}"
            ...         )
            >>> 
            >>> my_tool = CustomTool(name="custom", description="Custom tool")
            >>> agent.register_tool(my_tool)
        """
        try:
            self._tool_registry.register(tool)
            logger.info(f"Registered tool: {tool.name}")
        except ValueError as e:
            logger.error(f"Failed to register tool '{tool.name}': {e}")
            raise
    
    def unregister_tool(self, tool_name: str):
        """
        Remove a tool from the agent.
        
        Note: Cannot unregister built-in memory tools (core_memory_*, 
        archival_memory_*, conversation_search) as they are essential
        for agent functionality.
        
        Args:
            tool_name: Name of tool to remove
            
        Example:
            >>> agent.unregister_tool("my_custom_tool")
        """
        # Protect built-in memory tools
        builtin_tools = {
            "core_memory_append",
            "core_memory_replace", 
            "archival_memory_insert",
            "archival_memory_search",
            "conversation_search"
        }
        
        if tool_name in builtin_tools:
            logger.warning(f"Cannot unregister built-in memory tool: {tool_name}")
            raise ValueError(f"Cannot unregister built-in memory tool: {tool_name}")
        
        self._tool_registry.unregister(tool_name)
        
        # Also remove from legacy dict
        if tool_name in self._tools:
            del self._tools[tool_name]
        
        logger.info(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
            
        Example:
            >>> tool = agent.get_tool("read_file")
            >>> if tool:
            ...     print(f"Found: {tool.description}")
        """
        return self._tool_registry.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool exists, False otherwise
            
        Example:
            >>> if agent.has_tool("read_file"):
            ...     response = agent.chat("Read config.yaml")
        """
        return self._tool_registry.has_tool(tool_name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all Tool instances
            
        Example:
            >>> tools = agent.get_all_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        return self._tool_registry.get_all_tools()
    
    # ========================================================================
    # State Persistence (Stubs - will implement in Step 3)
    # ========================================================================
    
    def save_state(self, filepath: Optional[str] = None):
        """
        Save agent state to disk.
        
        Args:
            filepath: Custom filepath (uses default from config if not provided)
            
        Example:
            >>> agent.save_state()  # Saves to ./agent_states/{agent_id}.json
            >>> agent.save_state("my_agent.json")  # Custom path
        
        Note:
            Full implementation coming in Step 3 (State Persistence)
        """
        # TODO: Implement in Step 3
        logger.warning("save_state() not fully implemented yet - coming in Step 3")
        
        if filepath:
            logger.info(f"Would save to: {filepath}")
        else:
            default_path = Path(self.config.storage.state_dir) / f"{self.id}.json"
            logger.info(f"Would save to: {default_path}")
    
    @classmethod
    def load_state(cls, agent_id: str, state_dir: Optional[str] = None) -> 'Agent':
        """
        Load agent state from disk.
        
        Args:
            agent_id: Agent identifier
            state_dir: Directory containing state files (uses default if not provided)
            
        Returns:
            Loaded Agent instance
            
        Example:
            >>> agent = Agent.load_state("my_agent")
            >>> agent.chat("What were we discussing?")
        
        Note:
            Full implementation coming in Step 3 (State Persistence)
        """
        # TODO: Implement in Step 3
        logger.warning("load_state() not fully implemented yet - coming in Step 3")
        
        # For now, just create a new agent with that ID
        return cls(agent_id=agent_id)
    
    @classmethod
    def from_yaml(cls, filename: str, config_dir: Optional[str] = None, **overrides) -> 'Agent':
        """
        Create agent from YAML configuration file.
        
        Args:
            filename: YAML config filename
            config_dir: Directory containing config files
            **overrides: Override specific config values
            
        Returns:
            New Agent instance
            
        Example:
            >>> agent = Agent.from_yaml("my_agent.yaml")
            >>> # With overrides:
            >>> agent = Agent.from_yaml("my_agent.yaml", model="openai/gpt-4")
        """
        manager = get_config_manager(config_dir)
        config = manager.load_agent_config(filename, **overrides)
        return cls(config=config)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"Agent(id='{self.id}', "
                f"model='{self.config.model}', "
                f"blocks={list(self.blocks.keys())})")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Dictionary with agent metadata and statistics
        """
        return {
            "agent_id": self.id,
            "model": self.config.model,
            "blocks": list(self.blocks.keys()),
            "conversation_turns": len(self.conversation_history),
            "registered_tools": [tool.name for tool in self._tool_registry.get_all_tools()],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "max_heartbeat_steps": self.config.max_heartbeat_steps
            }
        }