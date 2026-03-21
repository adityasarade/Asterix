"""
Asterix LLM Provider Manager

Provides intelligent routing and management for LLM providers with:
- Health-aware routing between Groq and OpenAI
- Automatic fallback and retry logic
- Configuration-driven provider switching
- Comprehensive error handling and monitoring
- Usage tracking and performance metrics
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone

import openai
import httpx
from groq import Groq
import google.generativeai as genai

from .config import get_config_manager
from ..utils.tokens import count_tokens

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    processing_time: float
    finish_reason: str
    raw_response: Optional[Dict] = None


@dataclass
class LLMMessage:
    """Message for LLM conversation"""
    role: str
    content: str


class LLMError(Exception):
    """Raised when LLM operations fail"""
    pass


class LLMProviderManager:
    """
    Intelligent manager for LLM providers with health-aware routing.
    
    Features:
    - Automatic provider selection based on health status
    - Intelligent fallback between Groq and OpenAI
    - Configuration-driven model selection and parameters
    - Comprehensive error handling and retry logic
    - Usage tracking and performance monitoring
    """
    
    def __init__(self):
        """Initialize the LLM provider manager."""
        self.config = get_config_manager()
        
        # Provider clients - initialize to None
        self._groq_client = None
        self._openai_client = None
        self._gemini_configured = False
        
        # Get API keys from environment
        import os
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Provider selection (simplified - no health checks for now)
        self._primary_provider = "groq"
        self._fallback_provider = "openai"
        
        # Performance tracking
        self._operation_count = {"groq": 0, "openai": 0, "gemini": 0}
        self._total_processing_time = {"groq": 0.0, "openai": 0.0, "gemini": 0.0}
        self._total_tokens = {"groq": 0, "openai": 0, "gemini": 0}
        self._error_count = {"groq": 0, "openai": 0, "gemini": 0}
        self._provider_failures = {"groq": 0, "openai": 0, "gemini": 0}
        self._provider_health = {"groq": True, "openai": True, "gemini": True}
        self._max_failures = 3
    
    async def _ensure_clients_initialized(self):
        """Ensure LLM provider clients are initialized."""
        # Initialize Groq client
        if self._groq_client is None and self._groq_api_key:
            self._groq_client = Groq(api_key=self._groq_api_key)
        
        # Initialize OpenAI client
        if self._openai_client is None and self._openai_api_key:
            self._openai_client = openai.OpenAI(api_key=self._openai_api_key)
        
        # Initialize Gemini client
        if not self._gemini_configured and self._gemini_api_key:
            genai.configure(api_key=self._gemini_api_key)
            self._gemini_configured = True
    
    async def _select_provider(self, force_provider: Optional[str] = None) -> str:
        """
        Select provider (simplified - no health checks).
        
        Args:
            force_provider: Force use of specific provider
            
        Returns:
            Name of the selected provider
        """
        if force_provider:
            return force_provider
        
        # Check failure counts
        if self._provider_failures[self._primary_provider] < self._max_failures:
            return self._primary_provider
        
        # Fall back to fallback provider
        if self._provider_failures[self._fallback_provider] < self._max_failures:
            logger.warning(f"Primary provider {self._primary_provider} has failed too many times, using fallback")
            return self._fallback_provider
        
        # Reset and try again
        logger.warning("Both providers have failed multiple times, resetting failure counts")
        self._provider_failures = {"groq": 0, "openai": 0, "gemini": 0}
        return self._primary_provider
    
    async def _call_groq(self, messages: List[LLMMessage],
                        model: str = "llama-3.3-70b-versatile",
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None,
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[Union[str, Dict]] = None) -> LLMResponse:
        """
        Call Groq API for completion.
        
        Args:
            messages: List of conversation messages
            model: Model name to use
            temperature: Temperature override
            max_tokens: Max tokens override
            
        Returns:
            LLM response object
        """
        if not self._groq_client:
            raise LLMError("Groq client not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare messages
            groq_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Already a dict, use as-is
                    groq_messages.append(msg)
                else:
                    # LLMMessage object, convert to dict
                    groq_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Make API call
            api_params = {
                "model": model,
                "messages": groq_messages,
                "temperature": temperature or 0.1,
                "max_tokens": max_tokens or 1000,
                "stream": False
            }
            
            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                # Use provided tool_choice or default to "auto"
                if tool_choice:
                    api_params["tool_choice"] = tool_choice
                else:
                    api_params["tool_choice"] = "auto"
            
            response = self._groq_client.chat.completions.create(**api_params)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"
            
            # Build usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Update metrics
            self._operation_count["groq"] += 1
            self._total_processing_time["groq"] += processing_time
            self._total_tokens["groq"] += usage["total_tokens"]
            
            # Reset failure count on success
            self._provider_failures["groq"] = 0
            
            result = LLMResponse(
                content=content,
                model=model,
                provider="groq",
                usage=usage,
                processing_time=processing_time,
                finish_reason=finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
            
            logger.info(f"Groq completion: {usage['total_tokens']} tokens in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self._error_count["groq"] += 1
            self._provider_failures["groq"] += 1
            logger.error(f"Groq API error: {e}")
            raise LLMError(f"Groq error: {e}")
    
    async def _call_openai(self, messages: List[LLMMessage],
                          model: str = "gpt-4o",
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          tool_choice: Optional[Union[str, Dict]] = None) -> LLMResponse:
        """
        Call OpenAI API for completion.
        
        Args:
            messages: List of conversation messages
            model: Model name to use
            temperature: Temperature override
            max_tokens: Max tokens override
            
        Returns:
            LLM response object
        """
        if not self._openai_client:
            raise LLMError("OpenAI client not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare messages
            openai_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Already a dict, use as-is
                    openai_messages.append(msg)
                else:
                    # LLMMessage object, convert to dict
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Get model name from parameter
            model_name = model

            # Build base API parameters
            api_params = {
                "model": model_name,
                "messages": openai_messages,
                "stream": False
            }

            # Don't send temperature for models that don't support it
            models_without_temperature = ["gpt-5-mini", "o1-preview", "o1-mini"]

            if not any(model_str in model_name for model_str in models_without_temperature):
                api_params["temperature"] = temperature or 0.1

            # Use max_completion_tokens for newer models
            newer_models = ["gpt-4o", "o1-preview", "o1-mini", "gpt-5-mini"]

            if any(model_str in model_name for model_str in newer_models):
                api_params["max_completion_tokens"] = max_tokens or 1000
            else:
                api_params["max_tokens"] = max_tokens or 1000
            
            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                # Use provided tool_choice or default to "auto"
                if tool_choice:
                    api_params["tool_choice"] = tool_choice
                else:
                    api_params["tool_choice"] = "auto"
            
            response = self._openai_client.chat.completions.create(**api_params)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"
            
            # Build usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Update metrics
            self._operation_count["openai"] += 1
            self._total_processing_time["openai"] += processing_time
            self._total_tokens["openai"] += usage["total_tokens"]
            
            # Reset failure count on success
            self._provider_failures["openai"] = 0
            
            result = LLMResponse(
                content=content,
                model=model_name,
                provider="openai",
                usage=usage,
                processing_time=processing_time,
                finish_reason=finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
            
            logger.info(f"OpenAI completion: {usage['total_tokens']} tokens in {processing_time:.3f}s")
            return result
            
        except openai.RateLimitError as e:
            self._error_count["openai"] += 1
            self._provider_failures["openai"] += 1
            logger.error(f"OpenAI rate limit: {e}")
            raise LLMError(f"OpenAI rate limit: {e}")
        except openai.AuthenticationError as e:
            self._error_count["openai"] += 1
            self._provider_failures["openai"] += 1
            logger.error(f"OpenAI authentication failed: {e}")
            raise LLMError(f"OpenAI authentication failed: {e}")
        except Exception as e:
            self._error_count["openai"] += 1
            self._provider_failures["openai"] += 1
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI error: {e}")
    
    async def _call_gemini(self, messages: List[LLMMessage],
                          model: str = "gemini-2.5-flash",
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          tool_choice: Optional[Union[str, Dict]] = None) -> LLMResponse:
        """
        Call Gemini API for completion.
        
        Args:
            messages: List of conversation messages
            model: Gemini model to use
            temperature: Temperature override
            max_tokens: Max tokens override
            tools: Tool definitions for function calling
            tool_choice: Tool choice setting
            
        Returns:
            LLM response object
        """
        if not self._gemini_configured:
            raise LLMError("Gemini client not initialized")
        
        start_time = time.time()
        
        try:
            # Get the generative model
            gemini_model = genai.GenerativeModel(model)
            
            # Convert messages to Gemini format
            # Gemini uses a different format: system instruction separate, then contents
            system_instruction = None
            gemini_contents = []
            
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    role = msg.role
                    content = msg.content
                
                if role == "system":
                    system_instruction = content
                elif role == "assistant":
                    gemini_contents.append({"role": "model", "parts": [content]})
                else:
                    gemini_contents.append({"role": "user", "parts": [content]})
            
            # Configure generation settings
            generation_config = {
                "temperature": temperature or 0.1,
                "max_output_tokens": max_tokens or 1000,
            }
            
            # Create model with system instruction if provided
            if system_instruction:
                gemini_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction
                )
            
            # Handle tools if provided (convert OpenAI format to Gemini format)
            gemini_tools = None
            if tools:
                gemini_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool.get("function", {})
                        gemini_tools.append({
                            "function_declarations": [{
                                "name": func.get("name"),
                                "description": func.get("description", ""),
                                "parameters": func.get("parameters", {})
                            }]
                        })
            
            # Make API call
            response = gemini_model.generate_content(
                gemini_contents,
                generation_config=generation_config,
                tools=gemini_tools if gemini_tools else None
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract response data
            content = ""
            finish_reason = "stop"
            raw_response = None
            
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    # Find the first text part (may not be index 0 if tool calls present)
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content = part.text
                            break
                finish_reason = str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else "stop"
            
            # Build usage stats (Gemini provides token counts)
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
            }
            
            # Update metrics
            self._operation_count["gemini"] += 1
            self._total_processing_time["gemini"] += processing_time
            self._total_tokens["gemini"] += usage["total_tokens"]
            
            # Reset failure count on success
            self._provider_failures["gemini"] = 0
            
            # Build raw response for tool calls
            if response.candidates and hasattr(response.candidates[0].content, 'parts'):
                parts = response.candidates[0].content.parts
                tool_calls = []
                if parts:
                    for i, part in enumerate(parts):
                        # Protobuf parts always have function_call attr; check name to confirm it's real
                        if hasattr(part, 'function_call') and part.function_call and part.function_call.name:
                            args_dict = dict(part.function_call.args) if part.function_call.args else {}
                            tool_calls.append({
                                "id": f"call_{part.function_call.name}_{i}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(args_dict)
                                }
                            })
                
                if tool_calls:
                    finish_reason = "tool_calls"
                    raw_response = {
                        "choices": [{
                            "message": {
                                "content": content,
                                "tool_calls": tool_calls
                            }
                        }]
                    }
            
            result = LLMResponse(
                content=content,
                model=model,
                provider="gemini",
                usage=usage,
                processing_time=processing_time,
                finish_reason=finish_reason,
                raw_response=raw_response
            )
            
            logger.info(f"Gemini completion: {usage['total_tokens']} tokens in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self._error_count["gemini"] += 1
            self._provider_failures["gemini"] += 1
            logger.error(f"Gemini API error: {e}")
            raise LLMError(f"Gemini error: {e}")
    
    async def complete(self, messages: Union[str, List[LLMMessage]], 
                   provider: Optional[str] = None,
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   tool_choice: Optional[Union[str, Dict]] = None,
                   retry_on_failure: bool = True) -> LLMResponse:
        """
        Generate completion using the best available provider.
        
        Args:
            messages: Single message string or list of conversation messages
            provider: Force specific provider (optional)
            temperature: Temperature override
            max_tokens: Max tokens override
            retry_on_failure: Whether to retry with fallback provider on failure
            
        Returns:
            LLM response object
            
        Raises:
            LLMError: If completion fails
        """
        # Normalize input
        if isinstance(messages, str):
            messages = [LLMMessage(role="user", content=messages)]
        
        if not messages:
            raise LLMError("No messages provided for completion")
        
        # Ensure clients are initialized
        await self._ensure_clients_initialized()
        
        # Select provider
        selected_provider = await self._select_provider(provider)
        
        try:
            # Call the appropriate provider with model
            if selected_provider == "groq":
                return await self._call_groq(messages, model or "llama-3.3-70b-versatile", temperature, max_tokens, tools, tool_choice)
            elif selected_provider == "openai":
                return await self._call_openai(messages, model or "gpt-4o", temperature, max_tokens, tools, tool_choice)
            elif selected_provider == "gemini":
                return await self._call_gemini(messages, model or "gemini-2.5-flash", temperature, max_tokens, tools, tool_choice)
            else:
                raise LLMError(f"Unknown provider: {selected_provider}")
                
        except LLMError as e:
            # If retry is enabled and we haven't tried the fallback yet
            if (retry_on_failure and 
                not provider and  # Don't retry if specific provider was requested
                selected_provider != self._fallback_provider):
                
                logger.warning(f"Provider {selected_provider} failed, trying fallback {self._fallback_provider}")
                return await self.complete(
                    messages, 
                    provider=self._fallback_provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    retry_on_failure=False  # Prevent infinite recursion
                )
            else:
                raise
    
    async def summarize_text(self, text: str, 
                           max_tokens: int = 220,
                           provider: Optional[str] = None) -> LLMResponse:
        """
        Summarize text using the configured summarization prompt.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens for summary
            provider: Force specific provider
            
        Returns:
            LLM response with summary
        """
        # Get summarization prompt from config
        summary_prompt = self.config.get_yaml_config(
            "memory_config.yaml",
            "summarization.prompts.default",
            "Summarize the following text in {max_tokens} tokens or less:\n\n{content}"
        )
        
        # Format prompt
        formatted_prompt = summary_prompt.format(
            max_tokens=max_tokens,
            content=text,
            original_tokens=count_tokens(text).tokens
        )
        
        messages = [LLMMessage(role="user", content=formatted_prompt)]
        
        return await self.complete(
            messages,
            provider=provider,
            temperature=0.1,  # Low temperature for consistent summaries
            max_tokens=max_tokens + 50  # Small buffer for formatting
        )
    
    async def extract_keywords(self, text: str,
                             max_keywords: int = 10,
                             provider: Optional[str] = None) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords
            provider: Force specific provider
            
        Returns:
            List of extracted keywords
        """
        prompt = f"""Extract the {max_keywords} most important keywords from the following text.
Return only the keywords separated by commas, no other text.

Text: {text}

Keywords:"""
        
        messages = [LLMMessage(role="user", content=prompt)]
        
        response = await self.complete(
            messages,
            provider=provider,
            temperature=0.1,
            max_tokens=100
        )
        
        # Parse keywords
        keywords = [
            keyword.strip() 
            for keyword in response.content.split(",")
            if keyword.strip()
        ]
        
        return keywords[:max_keywords]
    
    def get_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration for all providers.
        
        Returns:
            Dictionary with provider configurations
        """
        return {
            "groq": {
                "model": self._groq_config.model,
                "temperature": self._groq_config.temperature,
                "max_tokens": self._groq_config.max_tokens,
                "timeout": self._groq_config.timeout,
                "api_key_configured": bool(self._groq_config.api_key)
            },
            "openai": {
                "model": self._openai_config.model,
                "temperature": self._openai_config.temperature,
                "max_tokens": self._openai_config.max_tokens,
                "timeout": self._openai_config.timeout,
                "api_key_configured": bool(self._openai_config.api_key)
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all providers.
        
        Returns:
            Dictionary with performance information
        """
        metrics = {
            "primary_provider": self._primary_provider,
            "fallback_provider": self._fallback_provider,
            "provider_health": self._provider_health,
            "provider_failures": self._provider_failures,
            "providers": {}
        }
        
        for provider in ["groq", "openai", "gemini"]:
            operation_count = self._operation_count[provider]
            avg_processing_time = (
                self._total_processing_time[provider] / operation_count
                if operation_count > 0 else 0
            )
            
            metrics["providers"][provider] = {
                "operation_count": operation_count,
                "error_count": self._error_count[provider],
                "total_tokens": self._total_tokens[provider],
                "total_processing_time_ms": round(self._total_processing_time[provider] * 1000, 2),
                "average_processing_time_ms": round(avg_processing_time * 1000, 2),
                "failure_count": self._provider_failures[provider],
                "success_rate": round(
                    (operation_count - self._error_count[provider]) / operation_count * 100, 2
                ) if operation_count > 0 else 0
            }
        
        return metrics

# Global LLM manager instance
llm_manager = LLMProviderManager()