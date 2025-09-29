"""
MemGPT FastAPI Controller

Main controller implementing the heartbeat loop and tool handling:
- Service health integration with automatic failover
- Agent conversation orchestration with memory management
- Tool call execution and response handling
- Error handling and graceful degradation
- Structured logging and performance monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..utils.config import get_config
from ..utils.health import ensure_required_services, get_service_status_summary
from ..utils.tokens import count_tokens, analyze_memory_tokens
from ..services import (
    letta_client, qdrant_client, embedding_service, llm_manager,
    LettaAgentConfig, ArchivalRecord, LLMMessage,
    LettaConnectionError, LettaAgentError, QdrantOperationError, EmbeddingError, LLMError
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MemGPT Controller",
    description="Stateful ReAct Agent with Persistent Memory",
    version="1.0.0"
)

# Global configuration
config = get_config()
controller_config = config.get_controller_config()
memory_config = config.get_memory_config()


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    agent_id: str = Field(..., description="ID of the agent to chat with")
    user_id: str = Field(..., description="ID of the user")
    text: str = Field(..., description="Message text from user")
    max_heartbeat_steps: Optional[int] = Field(None, description="Override default heartbeat steps")
    include_raw: bool = Field(True, description="Include raw agent response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    assistant: str = Field(..., description="Final assistant response")
    agent_id: str = Field(..., description="Agent ID that processed the request")
    user_id: str = Field(..., description="User ID from request")
    heartbeat_steps: int = Field(..., description="Number of heartbeat steps taken")
    memory_ops: List[Dict[str, Any]] = Field(default_factory=list, description="Memory operations performed")
    service_status: Dict[str, Any] = Field(..., description="Service health status")
    processing_time: float = Field(..., description="Total processing time in seconds")
    raw: Optional[Dict[str, Any]] = Field(None, description="Raw agent response for debugging")


class ServiceHealthResponse(BaseModel):
    """Response model for service health endpoint"""
    services: Dict[str, Any]
    summary: Dict[str, Any]
    timestamp: str


class AgentCreateRequest(BaseModel):
    """Request model for creating an agent"""
    name: str = Field(..., description="Agent name")
    user_id: str = Field(..., description="User ID who owns the agent")
    model: Optional[str] = Field(None, description="Override default model")
    embedding: Optional[str] = Field(None, description="Override default embedding model")
    system_instructions: Optional[str] = Field(None, description="Override default system instructions")


# Middleware Configuration
def setup_middleware():
    """Configure FastAPI middleware"""
    # CORS middleware
    cors_config = config.get_yaml_config("service_config.yaml", "controller.cors", {})
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["http://localhost:3000"]),
        allow_credentials=True,
        allow_methods=cors_config.get("allow_methods", ["GET", "POST", "PUT", "DELETE"]),
        allow_headers=cors_config.get("allow_headers", ["*"])
    )


# Dependency Functions
async def verify_services() -> Dict[str, Any]:
    """Verify required services are healthy"""
    required_services = ["letta", "qdrant", "openai_embeddings"]
    
    all_healthy, errors = await ensure_required_services(required_services)
    
    if not all_healthy:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Required services unavailable",
                "service_errors": errors,
                "required_services": required_services
            }
        )
    
    return get_service_status_summary()


# Core Controller Classes
class MemoryOperationTracker:
    """Tracks memory operations during conversation processing"""
    
    def __init__(self):
        self.operations: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def log_operation(self, operation_type: str, details: Dict[str, Any], success: bool = True, error: str = None):
        """Log a memory operation"""
        operation = {
            "operation": operation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "success": success,
            "error": error,
            "latency_ms": round((time.time() - self.start_time) * 1000, 2)
        }
        self.operations.append(operation)
        
        # Log to structured logger
        log_data = {
            "operation_type": operation_type,
            "success": success,
            "latency_ms": operation.get("latency_ms"),
            "details": details
        }
        
        if success:
            logger.info(f"Memory operation: {operation_type}", extra=log_data)
        else:
            logger.error(f"Memory operation failed: {operation_type} - {error}", extra=log_data)


class HeartbeatController:
    """
    Modern Heartbeat Controller that works without request_heartbeat.
    
    The new approach:
    1. Send message to agent
    2. Check response for tool calls or reasoning that indicates "thinking"
    3. If agent is still thinking, continue the loop
    4. If agent provides assistant_message, return to user
    """
    
    def __init__(self, agent_id: str, user_id: str, max_steps: int = None):
        self.agent_id = agent_id
        self.user_id = user_id
        self.max_steps = max_steps or controller_config.heartbeat_max_steps
        self.current_step = 0
        self.memory_tracker = MemoryOperationTracker()
        self.start_time = time.time()
        self.conversation_history = []  # Track full conversation
    
    async def execute_heartbeat_loop(self, initial_message: str) -> Dict[str, Any]:
        """
        Execute the modern heartbeat loop without request_heartbeat.
        
        The new flow:
        1. Service health check
        2. Memory management
        3. Send message and analyze response
        4. Continue loop if agent is still "thinking"
        5. Return when agent provides final response
        """
        try:
            # Step 1: Service Health Check
            await self._verify_service_health()
            
            # Step 2: Get agent memory and prepare context
            memory_blocks = await self._get_agent_memory()
            
            # Step 3: Check for memory eviction needs
            await self._check_memory_eviction(memory_blocks)
            
            # Step 4: Execute the modern heartbeat loop
            final_response = await self._run_modern_heartbeat_loop(initial_message)
            
            # Step 5: Post-processing memory management
            await self._post_process_memory()
            
            return {
                "assistant": final_response,
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "heartbeat_steps": self.current_step,
                "memory_ops": self.memory_tracker.operations,
                "processing_time": time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Heartbeat loop failed for agent {self.agent_id}: {e}")
            self.memory_tracker.log_operation(
                "heartbeat_error",
                {"agent_id": self.agent_id, "error": str(e)},
                success=False,
                error=str(e)
            )
            raise
    
    async def _run_modern_heartbeat_loop(self, initial_message: str) -> str:
        """
        Modern heartbeat loop that works with memgpt_v2_agent architecture.
        
        Key insight: Instead of relying on request_heartbeat, we analyze
        the agent's response to determine if it needs more thinking time.
        """
        current_message = initial_message
        assistant_response = None
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            
            try:
                logger.info(f"Heartbeat step {self.current_step}: Processing message")
                
                # Send message to agent
                response = await letta_client.send_message(
                    self.agent_id,
                    current_message,
                    role="user" if self.current_step == 1 else "system"
                )
                
                # Analyze the response to determine next action
                action = self._analyze_response(response)
                
                if action["type"] == "assistant_response":
                    # Agent provided a final response
                    assistant_response = action["content"]
                    logger.info(f"Agent provided final response after {self.current_step} steps")
                    break
                    
                elif action["type"] == "tool_execution_needed":
                    # Agent wants to execute tools
                    tool_results = await self._execute_tools(action["tools"])
                    
                    # Prepare continuation message based on tool results
                    current_message = self._format_tool_continuation(tool_results)
                    
                elif action["type"] == "thinking_continues":
                    # Agent is still thinking, continue with a continuation prompt
                    current_message = "Continue your analysis and provide your response."
                    
                else:
                    # Unknown response type, try to extract any available response
                    assistant_response = action.get("content", "I need more time to process your request.")
                    break
                
            except Exception as e:
                logger.error(f"Heartbeat step {self.current_step} failed: {e}")
                if self.current_step == 1:
                    # If first step fails, raise the error
                    raise
                else:
                    # Return partial response for later steps
                    assistant_response = "I encountered an issue processing your request. Please try again."
                    break
        
        # Handle max steps reached
        if assistant_response is None:
            logger.warning(f"Reached max heartbeat steps ({self.max_steps}) for agent {self.agent_id}")
            assistant_response = "I need more time to process your request. Please try asking in a different way."
        
        return assistant_response
    
    def _analyze_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze agent response to determine what action to take next.
        
        Modern Letta agents respond with:
        - reasoning_message: Internal thinking (continue loop)
        - assistant_message: Final response to user (end loop)
        - tool_call_message: Tool execution needed (execute and continue)
        """
        messages = response.get("messages", [])
        
        assistant_content = None
        tool_calls = []
        has_reasoning = False
        
        for message in messages:
            msg_type = message.get("message_type", "")
            
            if msg_type == "assistant_message":
                assistant_content = message.get("content", "")
                
            elif msg_type == "reasoning_message":
                has_reasoning = True
                
            elif msg_type == "tool_call_message":
                if message.get("tool_call"):
                    tool_calls.append(message["tool_call"])
        
        # Decision logic for modern Letta
        if assistant_content and not tool_calls:
            # Agent provided final response
            return {
                "type": "assistant_response",
                "content": assistant_content
            }
            
        elif tool_calls:
            # Agent wants to execute tools
            return {
                "type": "tool_execution_needed",
                "tools": tool_calls
            }
            
        elif has_reasoning and not assistant_content:
            # Agent is still thinking
            return {
                "type": "thinking_continues",
                "content": "Agent is processing..."
            }
        
        else:
            # Fallback: try to extract any available content
            fallback_content = ""
            for message in messages:
                if message.get("content"):
                    fallback_content = message["content"]
                    break
            
            return {
                "type": "assistant_response",
                "content": fallback_content or "I'm still processing your request."
            }
    
    async def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tools that the agent requested.
        
        Since the modern agent doesn't use request_heartbeat,
        we execute tools and format the results for continuation.
        """
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            
            # Parse arguments if they're a string
            if isinstance(tool_args, str):
                try:
                    import json
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {tool_args}")
                    continue
            
            # Execute the tool
            result = await self._execute_single_tool(tool_name, tool_args)
            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result
            })
        
        return tool_results
    
    def _format_tool_continuation(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Format tool execution results for agent continuation.
        
        This replaces the old request_heartbeat=true mechanism.
        """
        if not tool_results:
            return "Continue processing."
        
        continuation_parts = ["Tool execution completed:"]
        
        for result in tool_results:
            tool_name = result["tool"]
            tool_result = result["result"]
            continuation_parts.append(f"- {tool_name}: {tool_result}")
        
        continuation_parts.append("Please continue your analysis based on these results.")
        return "\n".join(continuation_parts)
    
    async def _verify_service_health(self):
        """Verify all required services are healthy"""
        required_services = ["letta", "qdrant", "openai_embeddings"]
        all_healthy, errors = await ensure_required_services(required_services)
        
        self.memory_tracker.log_operation(
            "service_check",
            {"required_services": required_services, "errors": errors},
            success=all_healthy
        )
        
        if not all_healthy:
            raise HTTPException(
                status_code=503,
                detail=f"Services unavailable: {errors}"
            )
    
    async def _get_agent_memory(self) -> Dict[str, str]:
        """Get current agent memory blocks"""
        try:
            memory_blocks = await letta_client.get_agent_memory(self.agent_id)
            
            self.memory_tracker.log_operation(
                "memory_read",
                {
                    "agent_id": self.agent_id,
                    "blocks_count": len(memory_blocks),
                    "blocks": list(memory_blocks.keys())
                }
            )
            
            return memory_blocks
            
        except Exception as e:
            self.memory_tracker.log_operation(
                "memory_read",
                {"agent_id": self.agent_id},
                success=False,
                error=str(e)
            )
            logger.warning(f"Could not retrieve memory for agent {self.agent_id}: {e}")
            return {}
    
    async def _check_memory_eviction(self, memory_blocks: Dict[str, str]):
        """Check if any memory blocks need eviction"""
        token_analysis = analyze_memory_tokens(memory_blocks)
        
        blocks_over_limit = [
            block_name for block_name, info in token_analysis["blocks"].items()
            if info["exceeds_limit"]
        ]
        
        if blocks_over_limit:
            logger.info(f"Memory blocks over limit: {blocks_over_limit}")
            
            for block_name in blocks_over_limit:
                await self._evict_memory_block(block_name, memory_blocks[block_name])
        
        self.memory_tracker.log_operation(
            "eviction_check",
            {
                "total_tokens": token_analysis["total"]["current_tokens"],
                "blocks_over_limit": len(blocks_over_limit),
                "blocks_needing_eviction": blocks_over_limit
            }
        )
        
        # For now, just log - implement eviction logic later if needed
        if blocks_over_limit:
            logger.info(f"Memory blocks over limit detected: {blocks_over_limit}")
    
    async def _evict_memory_block(self, block_name: str, content: str):
        """Evict a memory block using summarize + archive strategy"""
        try:
            # Step 1: Summarize the content
            summary_response = await llm_manager.summarize_text(
                content,
                max_tokens=memory_config.summary_token_limit
            )
            summary = summary_response.content
            
            # Step 2: Extract key bullet points
            keywords = await llm_manager.extract_keywords(content, max_keywords=3)
            
            # Step 3: Create archival record
            embedding_result = await embedding_service.embed_texts(content)
            
            archival_record = ArchivalRecord(
                text=content,
                summary=summary,
                embedding=embedding_result.embeddings[0],
                metadata={
                    "source": f"evicted_core:{self.agent_id}",
                    "type": "memory_block",
                    "block_name": block_name,
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "keywords": keywords,
                    "original_tokens": count_tokens(content).tokens,
                    "summary_tokens": count_tokens(summary).tokens
                }
            )
            
            # Step 4: Insert into Qdrant
            point_ids = await qdrant_client.insert_vectors([archival_record])
            
            # Step 5: Replace memory block with summary
            await letta_client.update_agent_memory(self.agent_id, block_name, summary)
            
            self.memory_tracker.log_operation(
                "eviction",
                {
                    "block_name": block_name,
                    "original_tokens": count_tokens(content).tokens,
                    "summary_tokens": count_tokens(summary).tokens,
                    "archival_ids": point_ids,
                    "keywords": keywords
                }
            )
            
            logger.info(f"Evicted memory block '{block_name}' for agent {self.agent_id}")
            
        except Exception as e:
            self.memory_tracker.log_operation(
                "eviction",
                {"block_name": block_name},
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to evict memory block '{block_name}': {e}")
    
    async def _run_heartbeat_loop(self, message: str, memory_blocks: Dict[str, str]) -> str:
        """Run the main heartbeat loop with tool handling"""
        current_messages = [{"role": "user", "content": message}]
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            
            try:
                # Send message to Letta agent
                response = await letta_client.send_message(
                    self.agent_id,
                    message if self.current_step == 1 else "continue"
                )
                
                # Check if we have tool calls to execute
                if self._has_tool_calls(response):
                    tool_results = await self._execute_tool_calls(response)
                    # Continue the loop with tool results
                    continue
                else:
                    # We have a final assistant response
                    assistant_text = self._extract_assistant_text(response)
                    if assistant_text:
                        return assistant_text
                
            except Exception as e:
                logger.error(f"Heartbeat step {self.current_step} failed: {e}")
                if self.current_step == 1:
                    # If first step fails, raise the error
                    raise
                else:
                    # Return partial response
                    return f"I encountered an issue processing your request. Please try again."
        
        # If we reach max steps without a final response
        logger.warning(f"Reached max heartbeat steps ({self.max_steps}) for agent {self.agent_id}")
        return "I need more time to process your request. Please try asking in a different way."
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """Check if the response contains tool calls"""
        messages = response.get("messages", [])
        for message in messages:
            if message.get("tool_calls"):
                return True
        return False
    
    def _extract_assistant_text(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract assistant text from response"""
        messages = response.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]
        return None
    
    async def _execute_tool_calls(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool calls from agent response"""
        messages = response.get("messages", [])
        
        for message in messages:
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool arguments: {tool_args}")
                        continue
                
                # Execute the appropriate tool
                await self._execute_single_tool(tool_name, tool_args)
        
        return {"status": "tools_executed"}
    
    async def _execute_single_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Execute a single tool call"""
        try:
            if tool_name == "archival_memory_search":
                await self._tool_archival_search(tool_args)
            elif tool_name == "archival_memory_insert":
                await self._tool_archival_insert(tool_args)
            elif tool_name == "core_memory_append":
                await self._tool_core_memory_append(tool_args)
            elif tool_name == "core_memory_replace":
                await self._tool_core_memory_replace(tool_args)
            elif tool_name == "memory_summarize":
                await self._tool_memory_summarize(tool_args)
            elif tool_name == "service_health_check":
                await self._tool_service_health_check(tool_args)
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            self.memory_tracker.log_operation(
                f"tool_{tool_name}",
                tool_args,
                success=False,
                error=str(e)
            )
    
    async def _tool_archival_search(self, args: Dict[str, Any]):
        """Execute archival memory search"""
        query = args.get("query", "")
        k = args.get("k", memory_config.retrieval_k)
        
        # Generate embedding for query
        embedding_result = await embedding_service.embed_texts(query)
        
        # Search in Qdrant
        search_results = await qdrant_client.search_vectors(
            query_vector=embedding_result.embeddings[0],
            limit=k,
            filter_conditions={
                "user_id": self.user_id,
                "agent_id": self.agent_id
            }
        )
        
        # Format results for agent
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "summary": result.payload.get("summary", ""),
                "text_excerpt": result.payload.get("text", "")[:200] + "...",
                "created_at": result.payload.get("created_at"),
                "source": result.payload.get("source"),
                "type": result.payload.get("type")
            })
        
        self.memory_tracker.log_operation(
            "archival_search",
            {
                "query": query,
                "results_count": len(formatted_results),
                "k": k
            }
        )
    
    async def _tool_archival_insert(self, args: Dict[str, Any]):
        """Execute archival memory insert"""
        text = args.get("text", "")
        summary = args.get("summary", "")
        meta = args.get("meta", {})
        
        if not summary:
            # Generate summary if not provided
            summary_response = await llm_manager.summarize_text(text, max_tokens=220)
            summary = summary_response.content
        
        # Generate embedding
        embedding_result = await embedding_service.embed_texts(text)
        
        # Create archival record
        archival_record = ArchivalRecord(
            text=text,
            summary=summary,
            embedding=embedding_result.embeddings[0],
            metadata={
                "source": f"agent_insert:{self.agent_id}",
                "type": meta.get("type", "note"),
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                **meta
            }
        )
        
        # Insert into Qdrant
        point_ids = await qdrant_client.insert_vectors([archival_record])
        
        self.memory_tracker.log_operation(
            "archival_insert",
            {
                "text_tokens": count_tokens(text).tokens,
                "summary_tokens": count_tokens(summary).tokens,
                "archival_ids": point_ids
            }
        )
    
    async def _tool_core_memory_append(self, args: Dict[str, Any]):
        """Execute core memory append"""
        block = args.get("block", "")
        value = args.get("value", "")
        
        # Get current memory
        current_memory = await letta_client.get_agent_memory(self.agent_id)
        current_content = current_memory.get(block, "")
        
        # Append new content
        new_content = current_content + "\n" + value if current_content else value
        
        # Update memory
        await letta_client.update_agent_memory(self.agent_id, block, new_content)
        
        self.memory_tracker.log_operation(
            "core_memory_append",
            {
                "block": block,
                "added_tokens": count_tokens(value).tokens,
                "total_tokens": count_tokens(new_content).tokens
            }
        )
    
    async def _tool_core_memory_replace(self, args: Dict[str, Any]):
        """Execute core memory replace"""
        block = args.get("block", "")
        value = args.get("value", "")
        
        # Update memory
        await letta_client.update_agent_memory(self.agent_id, block, value)
        
        self.memory_tracker.log_operation(
            "core_memory_replace",
            {
                "block": block,
                "new_tokens": count_tokens(value).tokens
            }
        )
    
    async def _tool_memory_summarize(self, args: Dict[str, Any]):
        """Execute memory summarization"""
        text = args.get("text", "")
        max_tokens = args.get("max_tokens", memory_config.summary_token_limit)
        
        summary_response = await llm_manager.summarize_text(text, max_tokens=max_tokens)
        
        self.memory_tracker.log_operation(
            "memory_summarize",
            {
                "original_tokens": count_tokens(text).tokens,
                "summary_tokens": count_tokens(summary_response.content).tokens,
                "compression_ratio": count_tokens(summary_response.content).tokens / count_tokens(text).tokens
            }
        )
    
    async def _tool_service_health_check(self, args: Dict[str, Any]):
        """Execute service health check"""
        services = args.get("services", ["letta", "qdrant", "embeddings"])
        
        # This would return service health status
        service_status = get_service_status_summary()
        
        self.memory_tracker.log_operation(
            "service_health_check",
            {
                "requested_services": services,
                "health_summary": service_status
            }
        )
    
    async def _post_process_memory(self):
        """Post-processing memory management after conversation"""
        # Check if any additional memory operations are needed
        memory_blocks = await self._get_agent_memory()
        await self._check_memory_eviction(memory_blocks)


# Initialize middleware
setup_middleware()


# API Endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/services", response_model=ServiceHealthResponse)
async def service_health(background_tasks: BackgroundTasks):
    """Detailed service health check"""
    service_status = get_service_status_summary()
    
    return ServiceHealthResponse(
        services=service_status.get("services", {}),
        summary={
            "total_services": service_status.get("total_services", 0),
            "healthy_services": service_status.get("healthy_services", 0),
            "health_percentage": service_status.get("health_percentage", 0)
        },
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    service_status: Dict[str, Any] = Depends(verify_services)
):
    """
    Main chat endpoint implementing the heartbeat loop.
    
    This endpoint orchestrates the entire conversation flow:
    1. Service health verification
    2. Agent memory management
    3. Heartbeat loop with tool execution
    4. Memory eviction and archival
    """
    start_time = time.time()
    
    # Validate request parameters before processing
    if not request.agent_id or not request.agent_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid agent ID",
                "message": "Agent ID cannot be empty or null",
                "agent_id": request.agent_id
            }
        )
    
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid user ID", 
                "message": "User ID cannot be empty or null",
                "user_id": request.user_id
            }
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid message",
                "message": "Message text cannot be empty or null",
                "text": request.text
            }
        )
    
    # ADD AGENT EXISTENCE CHECK
    try:
        # Verify agent exists before processing
        agents = await letta_client.list_agents()
        agent_exists = any(agent["id"] == request.agent_id for agent in agents)
        
        if not agent_exists:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Agent not found",
                    "message": f"Agent with ID '{request.agent_id}' does not exist",
                    "agent_id": request.agent_id,
                    "available_agents": [agent["id"] for agent in agents[:5]]  # Show first 5
                }
            )
    except LettaConnectionError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Letta service unavailable",
                "message": "Cannot verify agent existence - Letta service is not available"
            }
        )
    
    try:
        # Initialize heartbeat controller
        heartbeat_controller = HeartbeatController(
            agent_id=request.agent_id,
            user_id=request.user_id,
            max_steps=request.max_heartbeat_steps
        )
        
        # Execute heartbeat loop
        result = await heartbeat_controller.execute_heartbeat_loop(request.text)
        
        # Build response
        response = ChatResponse(
            assistant=result["assistant"],
            agent_id=result["agent_id"],
            user_id=result["user_id"],
            heartbeat_steps=result["heartbeat_steps"],
            memory_ops=result["memory_ops"],
            service_status=service_status,
            processing_time=result["processing_time"],
            raw=result if request.include_raw else None
        )
        
        logger.info(
            f"Chat completed: agent={request.agent_id}, "
            f"steps={result['heartbeat_steps']}, "
            f"time={result['processing_time']:.3f}s"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Chat request failed: {e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during chat processing",
                "message": str(e),
                "processing_time": processing_time,
                "service_status": service_status
            }
        )


@app.post("/agents")
async def create_agent(
    request: AgentCreateRequest,
    service_status: Dict[str, Any] = Depends(verify_services)
):
    """Create a new agent with default configuration"""
    try:
        # Get core memory blocks configuration
        memory_blocks_config = config.get_core_memory_blocks()
        
        # Prepare memory blocks
        memory_blocks = []
        for block_name, block_config in memory_blocks_config.items():
            memory_blocks.append({
                "label": block_config.get("label", block_name),
                "initial_value": block_config.get("initial_value", "")
            })
        
        # Get default configurations
        default_model = request.model or config.get_yaml_config(
            "service_config.yaml", "letta.agent.default_model", "groq/llama-3.3-70b-versatile"
        )
        default_embedding = request.embedding or config.get_yaml_config(
            "service_config.yaml", "letta.agent.default_embedding", "openai/text-embedding-3-small"
        )
        default_instructions = request.system_instructions or config.get_yaml_config(
            "agent_config.yaml", "system_instructions", "You are a helpful AI assistant."
        )
        
        # Create agent configuration
        agent_config = LettaAgentConfig(
            name=request.name,
            model=default_model,
            embedding=default_embedding,
            memory_blocks=memory_blocks,
            system_instructions=default_instructions
        )
        
        # Create agent
        agent_id = await letta_client.create_agent(agent_config)
        
        return {
            "agent_id": agent_id,
            "name": request.name,
            "user_id": request.user_id,
            "model": default_model,
            "embedding": default_embedding,
            "memory_blocks": len(memory_blocks),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {e}"
        )


@app.get("/agents")
async def list_agents():
    """List all available agents"""
    try:
        agents = await letta_client.list_agents()
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.get_logging_config()["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(
        app,
        host=controller_config.host,
        port=controller_config.port,
        log_level=config.get_logging_config()["level"].lower()
    )