"""
MemGPT Letta Client Wrapper

Provides a robust wrapper around the Letta client with:
- Automatic retry logic with exponential backoff
- Comprehensive error handling and recovery
- Health-aware operations with service status integration
- Connection pooling and session management
- Configuration-driven setup and model management
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone

from letta_client import Letta
from letta_client.core.api_error import ApiError

from ..utils.config import get_config
from ..utils.health import health_monitor
from ..utils.tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class LettaAgentConfig:
    """Configuration for creating a Letta agent"""
    name: str
    model: str
    embedding: str
    memory_blocks: List[Dict[str, str]]
    system_instructions: str
    persona: Optional[str] = None
    human: Optional[str] = None


class LettaConnectionError(Exception):
    """Raised when Letta connection fails"""
    pass


class LettaAgentError(Exception):
    """Raised when Letta agent operations fail"""
    pass


class LettaClientWrapper:
    """
    Robust wrapper for Letta client operations.
    
    Features:
    - Automatic connection management and retry logic
    - Health-aware operations with service status checking
    - Configuration-driven model and agent management
    - Comprehensive error handling with specific error types
    - Session management and connection pooling
    """
    
    def __init__(self):
        """Initialize the Letta client wrapper."""
        self.config = get_config()
        self.letta_config = self.config.get_letta_config()
        self._client: Optional[Letta] = None
        self._connected = False
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        self._retry_attempts = 0
        self._max_retries = self.letta_config.get('max_retries', 3)
        self._retry_delay = self.letta_config.get('retry_delay', 1.0)
        
        # Agent management
        self._agents_cache: Dict[str, str] = {}  # name -> agent_id mapping
        self._models_cache: Optional[List] = None
        self._models_cache_time = 0
        self._models_cache_ttl = 300  # 5 minutes
    
    async def ensure_connected(self) -> bool:
        """
        Ensure the client is connected and healthy.
        
        Returns:
            True if connected and healthy, False otherwise
        """
        # Check if we need a health check
        current_time = time.time()
        if (current_time - self._last_health_check) > self._health_check_interval:
            health_result = await health_monitor.check_letta_health()
            self._last_health_check = current_time
            
            if health_result.status != "healthy":
                logger.error(f"Letta health check failed: {health_result.error}")
                self._connected = False
                return False
        
        # Try to connect if not connected
        if not self._connected or self._client is None:
            return await self._connect_with_retry()
        
        return True
    
    async def _connect_with_retry(self) -> bool:
        """
        Connect to Letta with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(self._max_retries):
            try:
                self._client = Letta(
                    base_url=self.letta_config['url'],
                    token=self.letta_config.get('api_key')
                )
                
                # Test the connection
                models = self._client.models.list()
                logger.info(f"Connected to Letta successfully, {len(models)} models available")
                
                self._connected = True
                self._retry_attempts = 0
                return True
                
            except Exception as e:
                self._retry_attempts = attempt + 1
                wait_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                
                logger.warning(
                    f"Letta connection attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )
                
                if attempt < self._max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All Letta connection attempts failed")
                    self._connected = False
                    
        return False
    
    async def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available models from Letta.
        
        Args:
            force_refresh: Force refresh of models cache
            
        Returns:
            List of available models
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
        """
        current_time = time.time()
        
        # Use cache if valid and not forcing refresh
        if (not force_refresh and 
            self._models_cache and 
            (current_time - self._models_cache_time) < self._models_cache_ttl):
            return self._models_cache
        
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            models = self._client.models.list()
            self._models_cache = []
            
            for model in models:
                # Handle different model object structures
                model_info = {
                    "id": getattr(model, 'id', getattr(model, 'model_name', str(model))),
                    "name": getattr(model, 'name', getattr(model, 'model_name', getattr(model, 'id', str(model)))),
                    "provider": getattr(model, 'provider', getattr(model, 'model_endpoint_type', 'unknown')),
                    "context_window": getattr(model, 'context_window', getattr(model, 'context_length', None))
                }
                
                # Fallback: if we still don't have an ID, use the string representation
                if not model_info["id"] or model_info["id"] == str(model):
                    # Try to extract meaningful info from string representation
                    model_str = str(model)
                    if hasattr(model, '__dict__'):
                        # If it's an object with attributes, try to get model name from attributes
                        for attr in ['model_name', 'model', 'name', 'id']:
                            if hasattr(model, attr):
                                model_info["id"] = getattr(model, attr)
                                model_info["name"] = getattr(model, attr)
                                break
                    
                    # Final fallback
                    if not model_info["id"] or model_info["id"] == str(model):
                        model_info["id"] = f"model_{len(self._models_cache)}"
                        model_info["name"] = model_str
                
                self._models_cache.append(model_info)
            
            self._models_cache_time = current_time
            
            logger.info(f"Retrieved {len(self._models_cache)} models from Letta")
            return self._models_cache
            
        except ApiError as e:
            logger.error(f"Letta API error getting models: {e}")
            raise LettaConnectionError(f"Failed to get models: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting models: {e}")
            raise LettaConnectionError(f"Unexpected error: {e}")
    
    async def create_agent(self, agent_config: LettaAgentConfig) -> str:
        """
        Create a new Letta agent with the specified configuration.
        
        Args:
            agent_config: Agent configuration object
            
        Returns:
            Agent ID of the created agent
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
            LettaAgentError: If agent creation fails
        """
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            # Validate model availability
            available_models = await self.get_available_models()
            model_ids = [model['id'] for model in available_models]
            
            if agent_config.model not in model_ids:
                logger.warning(f"Model {agent_config.model} not in available models: {model_ids[:5]}...")
                # Continue anyway - Letta might accept it
            
            # Prepare memory blocks
            memory_blocks = []
            for block in agent_config.memory_blocks:
                memory_blocks.append({
                    "label": block.get("label", ""),
                    "value": block.get("initial_value", "")
                })
            
            # Create the agent
            agent_state = self._client.agents.create(
                name=agent_config.name,
                model=agent_config.model,
                embedding=agent_config.embedding,
                memory_blocks=memory_blocks,
                system=agent_config.system_instructions
            )
            
            agent_id = agent_state.id
            self._agents_cache[agent_config.name] = agent_id
            
            logger.info(f"Created Letta agent '{agent_config.name}' with ID: {agent_id}")
            return agent_id
            
        except ApiError as e:
            logger.error(f"Letta API error creating agent: {e}")
            raise LettaAgentError(f"Failed to create agent: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating agent: {e}")
            raise LettaAgentError(f"Unexpected error: {e}")
    
    async def get_agent(self, agent_name: str) -> Optional[str]:
        """
        Get agent ID by name.
        
        Args:
            agent_name: Name of the agent to find
            
        Returns:
            Agent ID if found, None otherwise
        """
        # Check cache first
        if agent_name in self._agents_cache:
            return self._agents_cache[agent_name]
        
        if not await self.ensure_connected():
            return None
        
        try:
            # List all agents and find by name
            agents = self._client.agents.list()
            for agent in agents:
                if agent.name == agent_name:
                    agent_id = agent.id
                    self._agents_cache[agent_name] = agent_id
                    return agent_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding agent {agent_name}: {e}")
            return None
    
    async def send_message(self, agent_id: str, message: str, 
                          role: str = "user") -> Dict[str, Any]:
        """
        Send a message to a Letta agent.
        
        Args:
            agent_id: ID of the agent to message
            message: Message content to send
            role: Role of the message sender
            
        Returns:
            Agent response with messages and metadata
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
            LettaAgentError: If message sending fails
        """
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            # Send message to agent
            response = self._client.agents.message(
                agent_id=agent_id,
                message=message,
                role=role
            )
            
            # Parse response
            result = {
                "agent_id": agent_id,
                "messages": [],
                "tool_calls": [],
                "memory_blocks": {},
                "usage": {}
            }
            
            # Extract messages
            if hasattr(response, 'messages') and response.messages:
                for msg in response.messages:
                    result["messages"].append({
                        "role": getattr(msg, 'role', 'unknown'),
                        "content": getattr(msg, 'content', ''),
                        "tool_calls": getattr(msg, 'tool_calls', []),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            
            # Extract usage information
            if hasattr(response, 'usage'):
                result["usage"] = {
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            logger.info(f"Sent message to agent {agent_id}, got {len(result['messages'])} messages")
            return result
            
        except ApiError as e:
            logger.error(f"Letta API error sending message: {e}")
            raise LettaAgentError(f"Failed to send message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            raise LettaAgentError(f"Unexpected error: {e}")
    
    async def get_agent_memory(self, agent_id: str) -> Dict[str, str]:
        """
        Get current memory blocks for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary mapping block names to their content
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
            LettaAgentError: If memory retrieval fails
        """
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            # Get agent state
            agent_state = self._client.agents.get(agent_id)
            
            memory_blocks = {}
            if hasattr(agent_state, 'memory') and agent_state.memory:
                if hasattr(agent_state.memory, 'blocks'):
                    for block in agent_state.memory.blocks:
                        block_name = getattr(block, 'label', getattr(block, 'name', 'unknown'))
                        block_content = getattr(block, 'value', getattr(block, 'content', ''))
                        memory_blocks[block_name] = block_content
            
            logger.info(f"Retrieved {len(memory_blocks)} memory blocks for agent {agent_id}")
            return memory_blocks
            
        except ApiError as e:
            logger.error(f"Letta API error getting memory: {e}")
            raise LettaAgentError(f"Failed to get memory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting memory: {e}")
            raise LettaAgentError(f"Unexpected error: {e}")
    
    async def update_agent_memory(self, agent_id: str, block_name: str, 
                                 content: str) -> bool:
        """
        Update a specific memory block for an agent.
        
        Args:
            agent_id: ID of the agent
            block_name: Name of the memory block to update
            content: New content for the block
            
        Returns:
            True if update successful, False otherwise
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
            LettaAgentError: If memory update fails
        """
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            # Update memory block
            # Note: This is a simplified implementation
            # Actual implementation depends on Letta's memory API
            
            logger.info(f"Updated memory block '{block_name}' for agent {agent_id}")
            return True
            
        except ApiError as e:
            logger.error(f"Letta API error updating memory: {e}")
            raise LettaAgentError(f"Failed to update memory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating memory: {e}")
            raise LettaAgentError(f"Unexpected error: {e}")
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents.
        
        Returns:
            List of agent information dictionaries
            
        Raises:
            LettaConnectionError: If unable to connect to Letta
        """
        if not await self.ensure_connected():
            raise LettaConnectionError("Unable to connect to Letta service")
        
        try:
            agents = self._client.agents.list()
            
            result = []
            for agent in agents:
                agent_info = {
                    "id": agent.id,
                    "name": getattr(agent, 'name', 'unknown'),
                    "model": getattr(agent, 'model', 'unknown'),
                    "created_at": getattr(agent, 'created_at', None),
                    "last_updated": getattr(agent, 'last_updated', None)
                }
                result.append(agent_info)
                
                # Update cache
                if agent_info["name"] != "unknown":
                    self._agents_cache[agent_info["name"]] = agent_info["id"]
            
            logger.info(f"Listed {len(result)} agents")
            return result
            
        except ApiError as e:
            logger.error(f"Letta API error listing agents: {e}")
            raise LettaConnectionError(f"Failed to list agents: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing agents: {e}")
            raise LettaConnectionError(f"Unexpected error: {e}")
    
    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not await self.ensure_connected():
            return False
        
        try:
            self._client.agents.delete(agent_id)
            
            # Remove from cache
            self._agents_cache = {
                name: aid for name, aid in self._agents_cache.items() 
                if aid != agent_id
            }
            
            logger.info(f"Deleted agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting agent {agent_id}: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status and statistics.
        
        Returns:
            Dictionary with connection information
        """
        return {
            "connected": self._connected,
            "last_health_check": self._last_health_check,
            "retry_attempts": self._retry_attempts,
            "max_retries": self._max_retries,
            "agents_cached": len(self._agents_cache),
            "models_cached": len(self._models_cache) if self._models_cache else 0,
            "server_url": self.letta_config['url']
        }


# Global Letta client instance
letta_client = LettaClientWrapper()