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
from .utils.tokens import count_tokens

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
        Update the block's content and recalculate token count.
        
        Args:
            new_content: New content for the block
        """
        self.content = new_content
        self.last_updated = datetime.now(timezone.utc)
        
        # Calculate tokens for the new content
        try:
            token_result = count_tokens(new_content)
            self.tokens = token_result.tokens
            logger.debug(f"Block '{self.name}' updated: {self.tokens} tokens")
        except Exception as e:
            # Fallback to rough estimate if token counting fails
            self.tokens = len(new_content) // 4
            logger.warning(f"Token counting failed for block '{self.name}', using estimate: {e}")
    
    def append_content(self, additional_content: str):
        """
        Append content to the block and recalculate token count.
        
        Args:
            additional_content: Content to append
        """
        if self.content:
            self.content += "\n" + additional_content
        else:
            self.content = additional_content
        self.last_updated = datetime.now(timezone.utc)
        
        # Recalculate tokens for updated content
        try:
            token_result = count_tokens(self.content)
            self.tokens = token_result.tokens
            logger.debug(f"Block '{self.name}' appended: {self.tokens} tokens")
        except Exception as e:
            # Fallback to rough estimate if token counting fails
            self.tokens = len(self.content) // 4
            logger.warning(f"Token counting failed for block '{self.name}', using estimate: {e}")
    
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
    
    def _ensure_llm_manager(self):
        """
        Initialize LLM manager if not already initialized (lazy loading).
        
        This is called on first use to avoid initializing services until needed.
        """
        if self._llm_manager is None:
            from .core.llm_manager import llm_manager
            self._llm_manager = llm_manager
            logger.debug("Initialized LLM manager")
    
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
        
        # Check if block exceeds token limit and evict if needed (Step 4.1)
        if self._check_block_eviction(block_name):
            self._evict_memory_block(block_name)
        
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
        
        # Check if block exceeds token limit (Step 4.1)
        if self._check_block_eviction(block_name):
            self._evict_memory_block(block_name)
        
        logger.debug(f"Appended to block '{block_name}'")
    
    def _check_block_eviction(self, block_name: str) -> bool:
        """
        Check if a memory block exceeds its token limit and needs eviction.
        
        Args:
            block_name: Name of the block to check
            
        Returns:
            True if block needs eviction, False otherwise
        """
        block = self.blocks[block_name]
        exceeds_limit = block.tokens > block.config.size
        
        if exceeds_limit:
            logger.warning(
                f"Block '{block_name}' exceeds limit: "
                f"{block.tokens}/{block.config.size} tokens "
                f"(+{block.tokens - block.config.size} over)"
            )
        
        return exceeds_limit
    
    def _evict_memory_block(self, block_name: str):
        """
        Evict (summarize and replace) a memory block that exceeds its token limit.
        
        This is Step 4.1's local summarization - it does NOT store in Qdrant.
        The block content is replaced in-place with a concise summary.
        
        Args:
            block_name: Name of the block to evict
        """
        block = self.blocks[block_name]
        original_content = block.content
        original_tokens = block.tokens
        
        logger.info(
            f"Starting eviction for block '{block_name}': "
            f"{original_tokens} tokens → target {self.config.memory.summary_token_limit} tokens"
        )
        
        try:
            # Ensure LLM manager is initialized
            self._ensure_llm_manager()
            
            # Build summarization prompt
            summary_prompt = f"""Summarize the following content concisely in approximately {self.config.memory.summary_token_limit} tokens or less.
    Focus on the most important information. Be clear and direct.

    Content to summarize:
    {original_content}

    Concise summary:"""
            
            # Import LLMMessage for formatting
            from .core.llm_manager import LLMMessage
            import asyncio
            
            # Call LLM for summarization
            logger.debug(f"Requesting summary from LLM for block '{block_name}'")
            response = asyncio.run(
                self._llm_manager.complete(
                    messages=[LLMMessage(role="user", content=summary_prompt)],
                    temperature=0.1,  # Low temperature for consistent summaries
                    max_tokens=self.config.memory.summary_token_limit + 50  # Small buffer
                )
            )
            
            summary = response.content.strip()
            
            # Validate summary is actually shorter
            summary_token_count = count_tokens(summary).tokens
            
            if summary_token_count >= original_tokens:
                logger.warning(
                    f"Summary ({summary_token_count} tokens) not shorter than original ({original_tokens} tokens). "
                    f"Using truncated version instead."
                )
                # Fallback: just truncate to fit
                from .utils.tokens import truncate_to_tokens
                summary = truncate_to_tokens(
                    original_content, 
                    self.config.memory.summary_token_limit,
                    preserve_end=False
                )
                summary_token_count = count_tokens(summary).tokens
            
            # Replace block content with summary
            block.update_content(summary)
            
            logger.info(
                f"Eviction complete for block '{block_name}': "
                f"{original_tokens} → {summary_token_count} tokens "
                f"({summary_token_count / original_tokens * 100:.1f}% of original)"
            )
            
            logger.debug(f"Summary preview: {summary[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to evict block '{block_name}': {e}")
            # Fallback: truncate without summarization
            logger.warning(f"Falling back to truncation for block '{block_name}'")
            
            from .utils.tokens import truncate_to_tokens
            truncated = truncate_to_tokens(
                original_content,
                self.config.memory.summary_token_limit,
                preserve_end=False
            )
            block.update_content(truncated)
            
            logger.info(f"Truncated block '{block_name}' to {block.tokens} tokens")
    
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
        """
        try:
            # Ensure LLM manager is initialized
            self._ensure_llm_manager()
            
            # Add user message to conversation history
            self._add_to_conversation_history("user", message)
            
            logger.info(f"User message: {message[:100]}")
            
            # Build messages for LLM (system prompt + memory + conversation)
            llm_messages = self._build_llm_messages()
            
            import asyncio
            import json
            from .core.llm_manager import LLMMessage
            
            # Get tool schemas for LLM
            tool_schemas = self.get_tool_schemas()
            
            # Heartbeat loop - allow multiple tool calls
            max_steps = self.config.max_heartbeat_steps
            assistant_response = None
            
            for step in range(max_steps):
                logger.debug(f"Heartbeat step {step + 1}/{max_steps}")
                
                # Convert to LLMMessage format
                formatted_messages = [
                    LLMMessage(role=msg["role"], content=msg["content"])
                    for msg in llm_messages
                ]
                
                # Get LLM response (with tools)
                response = asyncio.run(
                    self._llm_manager.complete(
                        messages=formatted_messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=tool_schemas if tool_schemas else None
                    )
                )
                
                # Check if response has tool calls
                if self._has_tool_calls(response):
                    # Extract tool calls
                    tool_calls = self._extract_tool_calls(response)
                    logger.info(f"LLM requested {len(tool_calls)} tool calls")
                    
                    # Add assistant message with tool calls to history
                    # (LLM needs to see its own tool call request)
                    llm_messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_calls  # Store for conversation context
                    })
                    
                    # Execute tools
                    tool_results = self._execute_tool_calls(tool_calls)
                    
                    # Add tool results to messages
                    for tool_result in tool_results:
                        llm_messages.append(tool_result)
                    
                    # Continue loop - LLM will see tool results and respond
                    continue
                
                else:
                    # LLM provided final text response
                    assistant_response = response.content
                    logger.info(f"LLM provided final response after {step + 1} steps")
                    break
            
            # Check if we exhausted max steps without final response
            if assistant_response is None:
                logger.warning(f"Reached max heartbeat steps ({max_steps}) without final response")
                assistant_response = "I need more time to process this request. Please try asking again or breaking it into smaller parts."
            
            # Add final response to conversation history
            self._add_to_conversation_history("assistant", assistant_response)
            
            # Update last_updated timestamp
            self.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"Assistant response: {assistant_response[:100]}")
            
            # Check context window after conversation turn
            context_status = self._check_context_window()

            if context_status["action_needed"]:
                logger.warning(
                    f"Context window at {context_status['percentage']:.1f}% - "
                    f"triggering context extraction"
                )
                # Step 4.2 - Extract and archive context
                self._extract_and_archive_context()
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _build_llm_messages(self) -> List[Dict[str, str]]:
        """
        Build the message list for LLM completion.
        
        Structure:
        1. System prompt (agent persona + memory blocks)
        2. Conversation history
        
        Returns:
            List of message dictionaries for LLM
        """
        messages = []
        
        # 1. System prompt with memory blocks
        system_content = self._build_system_prompt()
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # 2. Add conversation history
        messages.extend(self.conversation_history)
        
        return messages
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt including memory blocks.
        
        Format:
        - Agent instructions
        - Memory blocks (formatted)
        - Tool usage guidelines
        
        Returns:
            System prompt string
        """
        lines = [
            "You are a helpful AI assistant with persistent memory.",
            "",
            "# Memory Blocks",
            "You have access to the following memory blocks that you can read and modify:",
            ""
        ]
        
        # Add each memory block
        for block_name, block in self.blocks.items():
            lines.append(f"## {block_name}")
            if block.config.description:
                lines.append(f"*{block.config.description}*")
            lines.append(f"```")
            lines.append(block.content if block.content else "(empty)")
            lines.append(f"```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _add_to_conversation_history(self, role: str, content: str, **metadata):
        """
        Add a message to conversation history with timestamp and metadata.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional metadata (tool_calls, tool_call_id, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add any additional metadata
        message.update(metadata)
        
        self.conversation_history.append(message)
        
        logger.debug(f"Added {role} message to history (length: {len(self.conversation_history)})")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dictionary with conversation metrics
            
        Example:
            >>> stats = agent.get_conversation_stats()
            >>> print(f"Total turns: {stats['total_messages']}")
        """
        stats = {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg.get("role") == "user"),
            "assistant_messages": sum(1 for msg in self.conversation_history if msg.get("role") == "assistant"),
            "tool_messages": sum(1 for msg in self.conversation_history if msg.get("role") == "tool"),
        }
        
        # Calculate estimated token count (rough estimate: 4 chars per token)
        total_chars = sum(len(msg.get("content", "")) for msg in self.conversation_history)
        stats["estimated_tokens"] = total_chars // 4
        
        return stats
    
    def get_context_status(self) -> Dict[str, Any]:
        """
        Get current context window status.
        
        Useful for monitoring and debugging.
        
        Returns:
            Dictionary with context window information
            
        Example:
            >>> status = agent.get_context_status()
            >>> if status['exceeds_threshold']:
            ...     print(f"Warning: Context at {status['percentage']}%")
        """
        return self._check_context_window()
    
    def _trim_conversation_history(self, keep_recent: int = 10):
        """
        Trim conversation history, keeping only recent messages.
        
        Used when context window gets too large (will be called in Step 2.6).
        
        Args:
            keep_recent: Number of recent conversation turns to keep
            
        Note:
            This removes old messages from active context, but they should
            be preserved in state persistence (Step 3) and extracted to
            Qdrant before trimming (Step 4).
        """
        if len(self.conversation_history) <= keep_recent:
            logger.debug(f"Conversation history size ({len(self.conversation_history)}) within limit")
            return
        
        old_count = len(self.conversation_history)
        
        # Keep only recent messages
        self.conversation_history = self.conversation_history[-keep_recent:]
        
        logger.info(f"Trimmed conversation history from {old_count} to {len(self.conversation_history)} messages")
    
    def _calculate_context_tokens(self) -> int:
        """
        Calculate total tokens in current context.
        
        Context includes:
        - System prompt (with memory blocks)
        - Conversation history
        - Tool schemas
        
        Returns:
            Estimated total tokens
        """
        from .utils.tokens import count_tokens
        
        total_tokens = 0
        
        # 1. System prompt tokens
        system_prompt = self._build_system_prompt()
        total_tokens += count_tokens(system_prompt).tokens
        
        # 2. Conversation history tokens
        for message in self.conversation_history:
            content = message.get("content", "")
            total_tokens += count_tokens(content).tokens
            
            # Add small overhead for message structure
            total_tokens += 4
        
        # 3. Tool schemas tokens (rough estimate)
        tool_schemas = self.get_tool_schemas()
        if tool_schemas:
            import json
            schemas_text = json.dumps(tool_schemas)
            total_tokens += count_tokens(schemas_text).tokens
        
        logger.debug(f"Total context tokens: {total_tokens}")
        
        return total_tokens
    
    def _check_context_window(self) -> Dict[str, Any]:
        """
        Check if context window is approaching limit.
        
        Model context limits:
        - llama-3.3-70b: 8192 tokens
        - gpt-4-turbo: 128000 tokens
        
        Threshold: 85% of context limit
        
        Returns:
            Dictionary with context status:
            - current_tokens: Current context size
            - max_tokens: Model's context limit
            - percentage: Current usage percentage
            - exceeds_threshold: Whether 85% threshold is exceeded
            - action_needed: Whether context extraction is needed
        """
        # Model context limits (tokens)
        MODEL_CONTEXT_LIMITS = {
            "llama-3.3-70b-versatile": 8192,
            "llama-3.1-70b-versatile": 131072,
            "llama-3.1-8b-instant": 131072,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        
        # Get model context limit
        model_name = self.config.llm.model
        max_tokens = MODEL_CONTEXT_LIMITS.get(model_name, 131072)  # Default to 131072
        
        # Calculate current usage
        current_tokens = self._calculate_context_tokens()
        
        # Calculate percentage
        percentage = (current_tokens / max_tokens) * 100
        
        # Check if exceeds threshold
        threshold_percentage = self.config.memory.context_window_threshold * 100
        exceeds_threshold = percentage >= threshold_percentage
        
        status = {
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "percentage": round(percentage, 2),
            "threshold_percentage": threshold_percentage,
            "exceeds_threshold": exceeds_threshold,
            "action_needed": exceeds_threshold
        }
        
        if exceeds_threshold:
            logger.warning(
                f"Context window at {percentage:.1f}% "
                f"({current_tokens}/{max_tokens} tokens) - "
                f"exceeds {threshold_percentage}% threshold"
            )
        else:
            logger.debug(
                f"Context window at {percentage:.1f}% "
                f"({current_tokens}/{max_tokens} tokens)"
            )
        
        return status
    
    def _extract_and_archive_context(self):
        """
        Extract important facts from conversation that's about to be trimmed.
        
        Called when context window reaches 85% threshold.
        
        Process:
        1. Identify messages that will be trimmed (all except recent 10)
        2. Extract facts from those OLD messages (they're about to disappear)
        3. Store extracted facts in Qdrant via archival_memory_insert
        4. Trim conversation to keep only recent 10 turns
        
        This preserves information before it's deleted from active context.
        Key insight: Extract from what's LEAVING context, not what's STAYING.
        """
        logger.info("Starting context extraction and archival process")
        
        try:
            # Ensure LLM manager is initialized before using it
            self._ensure_llm_manager()
            # 1. Identify which messages will be trimmed
            keep_recent = 10
            total_messages = len(self.conversation_history)
            
            if total_messages <= keep_recent:
                logger.info(f"Only {total_messages} messages, no extraction needed")
                return
            
            # Messages that will be DELETED (extract facts from these before they're gone)
            messages_to_archive = self.conversation_history[:-keep_recent]
            
            logger.info(
                f"Extracting facts from {len(messages_to_archive)} old messages "
                f"(keeping recent {keep_recent} in active context)"
            )
            
            # 2. Also include memory blocks in extraction
            memory_blocks_content = {
                name: block.content 
                for name, block in self.blocks.items() 
                if block.content.strip()  # Only non-empty blocks
            }
            
            # 3. Build extraction prompt from OLD messages (about to be deleted)
            extraction_prompt = self._build_extraction_prompt(
                messages_to_archive,  # Extract from OLD messages
                memory_blocks_content
            )
            
            # 4. Call LLM to extract facts
            logger.debug("Requesting fact extraction from LLM")
            
            from .core.llm_manager import LLMMessage
            import asyncio
            import json
            
            response = asyncio.run(
                self._llm_manager.complete(
                    messages=[LLMMessage(role="user", content=extraction_prompt)],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=1500  # Increased for more facts
                )
            )
            
            # 5. Parse extracted facts
            try:
                # LLM should return JSON array of facts
                facts_text = response.content.strip()
                
                # Handle markdown code blocks if present
                if facts_text.startswith("```json"):
                    facts_text = facts_text.split("```json")[1].split("```")[0].strip()
                elif facts_text.startswith("```"):
                    facts_text = facts_text.split("```")[1].split("```")[0].strip()
                
                facts = json.loads(facts_text)
                
                if not isinstance(facts, list):
                    logger.error(f"Expected list of facts, got: {type(facts)}")
                    facts = []  # Continue with empty list rather than failing
                
                logger.info(f"LLM extracted {len(facts)} facts from old context")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted facts as JSON: {e}")
                logger.debug(f"LLM response preview: {response.content[:200]}")
                facts = []  # Continue with empty list
            
            # 6. Store each fact in Qdrant via archival_memory_insert
            stored_count = 0
            failed_count = 0
            
            for i, fact in enumerate(facts, 1):
                if not isinstance(fact, dict):
                    logger.warning(f"Skipping invalid fact #{i}: {fact}")
                    failed_count += 1
                    continue
                
                content = fact.get("content", "")
                fact_type = fact.get("type", "extracted_fact")
                importance = fact.get("importance", 0.5)
                
                if not content or not content.strip():
                    logger.warning(f"Skipping empty fact #{i}")
                    failed_count += 1
                    continue
                
                # Validate importance is in range
                if not isinstance(importance, (int, float)):
                    importance = 0.5
                importance = max(0.0, min(1.0, float(importance)))
                
                try:
                    # Use the archival_memory_insert tool
                    insert_tool = self.get_tool("archival_memory_insert")
                    
                    if not insert_tool:
                        logger.error("archival_memory_insert tool not available")
                        break
                    
                    # Generate a concise summary (first 100 chars)
                    summary = content[:100] + "..." if len(content) > 100 else content
                    
                    result = insert_tool.execute(
                        content=content,
                        summary=summary,
                        importance=importance
                    )
                    
                    if result.status.value == "success":
                        stored_count += 1
                        logger.debug(f"Stored fact {i}/{len(facts)}: {content[:60]}...")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to store fact #{i}: {result.error}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error storing fact #{i} in archival: {e}")
            
            if stored_count > 0:
                logger.info(
                    f"Successfully stored {stored_count}/{len(facts)} facts in Qdrant "
                    f"({failed_count} failed)"
                )
            else:
                logger.warning(f"No facts were stored (extracted {len(facts)}, all failed)")
            
            # 7. Trim conversation history (keep recent 10 turns)
            old_count = len(self.conversation_history)
            self._trim_conversation_history(keep_recent=keep_recent)
            new_count = len(self.conversation_history)
            
            logger.info(
                f"Context extraction complete: "
                f"{stored_count} facts archived to Qdrant, "
                f"conversation trimmed from {old_count} to {new_count} messages"
            )
            
            # Update last_updated timestamp
            self.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - extraction failure shouldn't crash the agent
            # Just log the error and continue

    def _build_extraction_prompt(self, 
                                conversation: List[Dict[str, Any]], 
                                memory_blocks: Dict[str, str]) -> str:
        """
        Build prompt for LLM to extract important facts.
        
        Args:
            conversation: Conversation turns to extract from (typically old messages)
            memory_blocks: Current memory blocks content
            
        Returns:
            Extraction prompt string
        """
        prompt_lines = [
            "You are extracting important facts from conversation history that is about to be archived.",
            "",
            "Extract ONLY facts that should be remembered long-term:",
            "✓ User preferences, habits, and settings",
            "✓ Decisions and commitments made",
            "✓ Important requirements and constraints",
            "✓ Goals, intentions, and plans",
            "✓ Key context needed for future conversations",
            "",
            "DO NOT extract:",
            "✗ Casual greetings or pleasantries",
            "✗ Temporary/one-time information",
            "✗ General knowledge or common facts",
            "✗ Redundant information already captured",
            "",
            "Return ONLY a valid JSON array (no other text):",
            "[",
            '  {"content": "User prefers Python over JavaScript", "type": "preference", "importance": 0.8},',
            '  {"content": "Working on authentication module review", "type": "goal", "importance": 0.7},',
            '  ...',
            "]",
            "",
            "Types: preference, fact, goal, decision",
            "Importance: 0.0 (low) to 1.0 (high)",
            "",
            "---",
            "",
            "# MEMORY BLOCKS TO CONSIDER",
            ""
        ]
        
        # Add memory blocks (these provide context but don't need extraction)
        if memory_blocks:
            for block_name, content in memory_blocks.items():
                prompt_lines.append(f"## {block_name}")
                # Truncate long blocks for prompt efficiency
                truncated = content[:400] + "..." if len(content) > 400 else content
                prompt_lines.append(truncated)
                prompt_lines.append("")
        else:
            prompt_lines.append("(No memory blocks)")
            prompt_lines.append("")
        
        prompt_lines.append("# CONVERSATION HISTORY TO EXTRACT FROM")
        prompt_lines.append("")
        
        # Add conversation turns (these are what we're extracting from)
        if conversation:
            for msg in conversation:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                
                # Truncate very long messages
                if len(content) > 300:
                    content = content[:297] + "..."
                
                prompt_lines.append(f"{role}: {content}")
        else:
            prompt_lines.append("(No conversation)")
        
        prompt_lines.extend([
            "",
            "---",
            "",
            "Extract important facts as JSON array:"
        ])
        
        return "\n".join(prompt_lines)
    
    def _has_tool_calls(self, llm_response) -> bool:
        """
        Check if LLM response contains tool calls.
        
        Args:
            llm_response: Response from LLM manager
            
        Returns:
            True if response has tool calls, False otherwise
        """
        # Check if response has raw_response with tool_calls
        if hasattr(llm_response, 'raw_response') and llm_response.raw_response:
            raw = llm_response.raw_response
            
            # OpenAI/Groq format: choices[0].message.tool_calls
            if isinstance(raw, dict):
                choices = raw.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    tool_calls = message.get('tool_calls')
                    if tool_calls:
                        return True
        
        return False
    
    def _extract_tool_calls(self, llm_response) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response.
        
        Args:
            llm_response: Response from LLM manager
            
        Returns:
            List of tool call dictionaries with 'id', 'name', and 'arguments'
        """
        tool_calls = []
        
        if hasattr(llm_response, 'raw_response') and llm_response.raw_response:
            raw = llm_response.raw_response
            
            # OpenAI/Groq format: choices[0].message.tool_calls
            if isinstance(raw, dict):
                choices = raw.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    raw_tool_calls = message.get('tool_calls', [])
                    
                    for tool_call in raw_tool_calls:
                        tool_calls.append({
                            'id': tool_call.get('id', ''),
                            'name': tool_call.get('function', {}).get('name', ''),
                            'arguments': tool_call.get('function', {}).get('arguments', '{}')
                        })
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls and return results.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of tool result dictionaries for LLM consumption
        """
        import json
        
        results = []
        
        for tool_call in tool_calls:
            tool_id = tool_call['id']
            tool_name = tool_call['name']
            
            try:
                # Parse arguments
                arguments = json.loads(tool_call['arguments'])
                
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                
                # Execute tool via registry
                tool_result = self._tool_registry.execute_tool(tool_name, **arguments)
                
                # Format result for LLM
                results.append({
                    'tool_call_id': tool_id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': str(tool_result)  # ToolResult.__str__ returns formatted content
                })
                
                logger.info(f"Tool {tool_name} result: {str(tool_result)[:100]}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {tool_name}: {e}")
                results.append({
                    'tool_call_id': tool_id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': f"Error: Invalid tool arguments - {str(e)}"
                })
            
            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")
                results.append({
                    'tool_call_id': tool_id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': f"Error: {str(e)}"
                })
        
        return results
    
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
    # State Persistence - JSON Backend
    # ========================================================================

    def save_state(self, filepath: Optional[str] = None):
        """
        Save agent state to disk.
        
        Uses the configured state backend (JSON or SQLite).
        
        Args:
            filepath: Custom filepath (only for JSON backend)
            
        Example:
            >>> # JSON backend (default)
            >>> agent.save_state()  # Saves to ./agent_states/{agent_id}.json
            
            >>> # SQLite backend
            >>> agent = Agent(..., storage=StorageConfig(state_backend="sqlite"))
            >>> agent.save_state()  # Saves to agents.db
        """
        import json
        
        # Check backend type
        backend = self.config.storage.state_backend
        
        if backend == "sqlite":
            # Use SQLite backend
            from .storage.agent_state import SQLiteStateBackend
            
            db_path = self.config.storage.state_db
            sqlite_backend = SQLiteStateBackend(db_path)
            
            state_dict = self._to_state_dict()
            sqlite_backend.save(self.id, state_dict)
            
            logger.info(f"Saved agent state to SQLite database: {db_path}")
            
        else:
            # Use JSON backend (default)
            try:
                # Determine filepath
                if filepath:
                    save_path = Path(filepath)
                else:
                    state_dir = Path(self.config.storage.state_dir)
                    save_path = state_dir / f"{self.id}.json"
                
                # Create directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Serialize state
                state_dict = self._to_state_dict()
                
                # Write to file
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved agent state to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save agent state: {e}")
                raise IOError(f"Failed to save agent state: {e}")

    @classmethod
    def load_state(cls, agent_id: str, 
                state_dir: Optional[str] = None,
                state_backend: str = "json",
                state_db: str = "./agent_states/agents.db") -> 'Agent':
        """
        Load agent state from storage.
        
        Args:
            agent_id: Agent identifier
            state_dir: Directory for JSON files (JSON backend only)
            state_backend: "json" or "sqlite"
            state_db: Database path (SQLite backend only)
            
        Returns:
            Loaded Agent instance
            
        Example:
            >>> # JSON backend
            >>> agent = Agent.load_state("my_agent")
            
            >>> # SQLite backend
            >>> agent = Agent.load_state("my_agent", state_backend="sqlite")
        """
        import json
        
        if state_backend == "sqlite":
            # Use SQLite backend
            from .storage.agent_state import SQLiteStateBackend
            
            sqlite_backend = SQLiteStateBackend(state_db)
            state_dict = sqlite_backend.load(agent_id)
            
            agent = cls._from_state_dict(state_dict)
            logger.info(f"Loaded agent from SQLite database: {state_db}")
            
            return agent
            
        else:
            # Use JSON backend (existing code)
            try:
                if state_dir:
                    load_path = Path(state_dir) / f"{agent_id}.json"
                else:
                    load_path = Path("./agent_states") / f"{agent_id}.json"
                
                if not load_path.exists():
                    raise FileNotFoundError(
                        f"State file not found: {load_path}\n"
                        f"Make sure the agent was saved previously with save_state()"
                    )
                
                with open(load_path, 'r', encoding='utf-8') as f:
                    state_dict = json.load(f)
                
                required_keys = ["agent_id", "config", "blocks", "conversation_history"]
                missing_keys = [key for key in required_keys if key not in state_dict]
                
                if missing_keys:
                    raise ValueError(f"Invalid state file: missing keys {missing_keys}")
                
                agent = cls._from_state_dict(state_dict)
                logger.info(f"Loaded agent state from {load_path}")
                
                return agent
                
            except FileNotFoundError:
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in state file: {e}")
                raise ValueError(f"State file contains invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to load agent state: {e}")
                raise

    @classmethod
    def load_state_from_file(cls, filepath: str) -> 'Agent':
        """
        Load agent state from a specific file path.
        
        Args:
            filepath: Full path to state file
            
        Returns:
            Loaded Agent instance
            
        Example:
            >>> agent = Agent.load_state_from_file("./backups/my_agent_2025.json")
        """
        import json
        
        try:
            load_path = Path(filepath)
            
            if not load_path.exists():
                raise FileNotFoundError(f"State file not found: {load_path}")
            
            with open(load_path, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            
            agent = cls._from_state_dict(state_dict)
            
            logger.info(f"Loaded agent state from {load_path}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to load agent state from {filepath}: {e}")
            raise
    
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
        conversation_stats = self.get_conversation_stats()
        
        context_status = self._check_context_window()
        
        return {
            "agent_id": self.id,
            "model": self.config.model,
            "blocks": list(self.blocks.keys()),
            "conversation": conversation_stats,
            "context": {
                "current_tokens": context_status["current_tokens"],
                "max_tokens": context_status["max_tokens"],
                "usage_percentage": context_status["percentage"],
                "needs_extraction": context_status["action_needed"]
            },
            "registered_tools": [tool.name for tool in self._tool_registry.get_all_tools()],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "max_heartbeat_steps": self.config.max_heartbeat_steps
            }
        }
        
    # ========================================================================
    # State Persistence - Serialization
    # ========================================================================

    def _to_state_dict(self) -> Dict[str, Any]:
        """
        Convert agent state to dictionary for serialization.
        
        Returns:
            Dictionary containing complete agent state
            
        Note:
            This includes:
            - Agent metadata (id, timestamps)
            - Configuration
            - Memory blocks (content + metadata)
            - Conversation history (FULL history)
            - Registered custom tools (names only)
            - Qdrant info (collection name, vector count estimate)
        """
        state = {
            "agent_id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            
            # Configuration
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "max_heartbeat_steps": self.config.max_heartbeat_steps,
                
                # Storage config
                "storage": {
                    "qdrant_url": self.config.storage.qdrant_url,
                    "qdrant_collection_name": self.config.storage.qdrant_collection_name,
                    "vector_size": self.config.storage.vector_size,
                    "state_backend": self.config.storage.state_backend,
                    "state_dir": self.config.storage.state_dir
                },
                
                # Memory config
                "memory": {
                    "eviction_strategy": self.config.memory.eviction_strategy,
                    "summary_token_limit": self.config.memory.summary_token_limit,
                    "context_window_threshold": self.config.memory.context_window_threshold,
                    "retrieval_k": self.config.memory.retrieval_k,
                    "score_threshold": self.config.memory.score_threshold
                }
            },
            
            # Memory blocks
            "blocks": {
                name: block.to_dict()
                for name, block in self.blocks.items()
            },
            
            # FULL conversation history
            "conversation_history": self.conversation_history.copy(),
            
            # Custom registered tools (names only, not the functions)
            "registered_tools": [
                tool.name for tool in self._tool_registry.get_all_tools()
                if tool.name not in {
                    "core_memory_append",
                    "core_memory_replace",
                    "archival_memory_insert",
                    "archival_memory_search",
                    "conversation_search"
                }
            ],
            
            # Qdrant info
            "qdrant_info": {
                "collection_name": f"asterix_memory_{self.id}",
                "vector_count": None  # Will be populated if needed
            }
        }
        
        logger.debug(f"Serialized agent state: {len(state['conversation_history'])} messages, {len(state['blocks'])} blocks")
        
        return state

    @classmethod
    def _from_state_dict(cls, state: Dict[str, Any]) -> 'Agent':
        """
        Create agent from state dictionary.
        
        Args:
            state: State dictionary from _to_state_dict()
            
        Returns:
            Reconstructed Agent instance
            
        Note:
            This recreates the agent with:
            - All configuration restored
            - Memory blocks restored
            - Conversation history restored
            - Custom tools NOT restored (user must re-register)
        """
        # Rebuild configuration
        config_data = state["config"]
        
        # Create BlockConfig objects
        blocks = {}
        for block_name, block_data in state["blocks"].items():
            blocks[block_name] = BlockConfig(
                size=block_data["size_limit"],
                priority=block_data["priority"],
                description=block_data.get("description", ""),
                initial_value=""  # Will be overridden below
            )
        
        # Create AgentConfig
        agent_config = AgentConfig(
            agent_id=state["agent_id"],
            blocks=blocks,
            model=config_data["model"],
            temperature=config_data["temperature"],
            max_tokens=config_data["max_tokens"],
            max_heartbeat_steps=config_data["max_heartbeat_steps"],
            
            memory=MemoryConfig(
                eviction_strategy=config_data["memory"]["eviction_strategy"],
                summary_token_limit=config_data["memory"]["summary_token_limit"],
                context_window_threshold=config_data["memory"]["context_window_threshold"],
                retrieval_k=config_data["memory"]["retrieval_k"],
                score_threshold=config_data["memory"]["score_threshold"]
            ),
            
            storage=StorageConfig(
                qdrant_url=config_data["storage"]["qdrant_url"],
                qdrant_collection_name=config_data["storage"]["qdrant_collection_name"],
                vector_size=config_data["storage"]["vector_size"],
                state_backend=config_data["storage"]["state_backend"],
                state_dir=config_data["storage"]["state_dir"]
            )
        )
        
        # Create agent
        agent = cls(config=agent_config)
        
        # Restore memory blocks (content + metadata)
        for block_name, block_data in state["blocks"].items():
            if block_name in agent.blocks:
                agent.blocks[block_name] = MemoryBlock.from_dict(block_data)
        
        # Restore conversation history
        agent.conversation_history = state["conversation_history"].copy()
        
        # Restore timestamps
        agent.created_at = datetime.fromisoformat(state["created_at"])
        agent.last_updated = datetime.fromisoformat(state["last_updated"])
        
        logger.info(
            f"Restored agent '{agent.id}' from state: "
            f"{len(agent.conversation_history)} messages, "
            f"{len(agent.blocks)} blocks"
        )
        
        # Log if custom tools need to be re-registered
        custom_tools = state.get("registered_tools", [])
        if custom_tools:
            logger.warning(
                f"Agent had {len(custom_tools)} custom tools registered: {custom_tools}. "
                f"These tools are NOT automatically restored - you must re-register them!"
            )
        
        return agent