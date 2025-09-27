"""
MemGPT Token Counting Utilities

Provides token counting functionality for memory management and eviction decisions.
Supports tiktoken for accurate counting and fallback methods.
"""

import re
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, using approximate token counting")

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class TokenCount:
    """Token count result with metadata"""
    tokens: int
    characters: int
    words: int
    method: str  # "tiktoken", "approximate", "character_based"
    model: Optional[str] = None


class TokenCounter:
    """
    Token counting utility supporting multiple counting methods.
    
    Provides accurate token counting for memory management and eviction decisions.
    """
    
    def __init__(self):
        self.config = get_config()
        self._encoders: Dict[str, any] = {}
        self._load_encoders()
    
    def _load_encoders(self):
        """Load tiktoken encoders for different models"""
        if not TIKTOKEN_AVAILABLE:
            logger.warning("tiktoken not available, using fallback counting methods")
            return
        
        try:
            # Common encoders
            self._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")  # GPT-4, text-embedding-3
            self._encoders["p50k_base"] = tiktoken.get_encoding("p50k_base")      # GPT-3.5, text-davinci-003
            self._encoders["r50k_base"] = tiktoken.get_encoding("r50k_base")      # GPT-3, text-davinci-002
            
            logger.info("Loaded tiktoken encoders for accurate token counting")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoders: {e}")
            self._encoders.clear()
    
    def get_encoder_for_model(self, model: str) -> Optional[any]:
        """
        Get appropriate encoder for a given model.
        
        Args:
            model: Model name
            
        Returns:
            Tiktoken encoder or None if not available
        """
        if not TIKTOKEN_AVAILABLE or not self._encoders:
            return None
        
        # Model to encoder mapping
        model_encoders = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-4-turbo-preview": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-3-large": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            
            # Legacy models
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "r50k_base",
            "davinci": "r50k_base",
            "curie": "r50k_base",
            "babbage": "r50k_base",
            "ada": "r50k_base"
        }
        
        # Find encoder for model
        for model_name, encoder_name in model_encoders.items():
            if model_name in model.lower():
                return self._encoders.get(encoder_name)
        
        # Default to cl100k_base for unknown models
        return self._encoders.get("cl100k_base")
    
    def count_tokens_tiktoken(self, text: str, model: Optional[str] = None) -> TokenCount:
        """
        Count tokens using tiktoken for maximum accuracy.
        
        Args:
            text: Text to count tokens for
            model: Model name for encoder selection
            
        Returns:
            Token count result
        """
        if not TIKTOKEN_AVAILABLE or not self._encoders:
            return self.count_tokens_approximate(text)
        
        try:
            # Get encoder
            if model:
                encoder = self.get_encoder_for_model(model)
            else:
                # Use default encoder from config
                encoder_name = self.config.get_yaml_config(
                    "memory_config.yaml", 
                    "tokens.tiktoken_model", 
                    "cl100k_base"
                )
                encoder = self._encoders.get(encoder_name)
            
            if not encoder:
                logger.warning(f"No encoder found for model {model}, using approximate counting")
                return self.count_tokens_approximate(text)
            
            # Count tokens
            tokens = encoder.encode(text)
            
            return TokenCount(
                tokens=len(tokens),
                characters=len(text),
                words=len(text.split()),
                method="tiktoken",
                model=model
            )
            
        except Exception as e:
            logger.error(f"tiktoken counting failed: {e}, falling back to approximate")
            return self.count_tokens_approximate(text)
    
    def count_tokens_approximate(self, text: str) -> TokenCount:
        """
        Count tokens using approximate character-based method.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count result
        """
        # Get ratio from config
        char_per_token = self.config.get_yaml_config(
            "memory_config.yaml",
            "tokens.approximate_ratio",
            4
        )
        
        characters = len(text)
        words = len(text.split())
        tokens = max(1, characters // char_per_token)
        
        return TokenCount(
            tokens=tokens,
            characters=characters,
            words=words,
            method="approximate"
        )
    
    def count_tokens_character_based(self, text: str) -> TokenCount:
        """
        Count tokens using simple character-based method.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count result
        """
        characters = len(text)
        words = len(text.split())
        tokens = characters  # 1:1 ratio for conservative estimation
        
        return TokenCount(
            tokens=tokens,
            characters=characters,
            words=words,
            method="character_based"
        )
    
    def count_tokens(self, text: str, model: Optional[str] = None, method: Optional[str] = None) -> TokenCount:
        """
        Count tokens using specified or configured method.
        
        Args:
            text: Text to count tokens for
            model: Model name for encoder selection
            method: Counting method ("tiktoken", "approximate", "character_based")
            
        Returns:
            Token count result
        """
        if not text:
            return TokenCount(tokens=0, characters=0, words=0, method="empty")
        
        # Determine method
        if not method:
            method = self.config.get_yaml_config(
                "memory_config.yaml",
                "tokens.counter",
                "tiktoken"
            )
        
        # Count tokens based on method
        if method == "tiktoken":
            return self.count_tokens_tiktoken(text, model)
        elif method == "approximate":
            return self.count_tokens_approximate(text)
        elif method == "character_based":
            return self.count_tokens_character_based(text)
        else:
            logger.warning(f"Unknown counting method {method}, using tiktoken")
            return self.count_tokens_tiktoken(text, model)
    
    def estimate_tokens_for_messages(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> TokenCount:
        """
        Estimate tokens for a list of chat messages.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model name for accurate counting
            
        Returns:
            Total token count for all messages
        """
        total_tokens = 0
        total_chars = 0
        total_words = 0
        
        for message in messages:
            # Count content tokens
            content = message.get("content", "")
            role = message.get("role", "")
            
            content_count = self.count_tokens(content, model)
            role_count = self.count_tokens(role, model)
            
            # Add message formatting overhead (estimated)
            message_tokens = content_count.tokens + role_count.tokens + 4  # Overhead for message structure
            
            total_tokens += message_tokens
            total_chars += content_count.characters + role_count.characters
            total_words += content_count.words + role_count.words
        
        return TokenCount(
            tokens=total_tokens,
            characters=total_chars,
            words=total_words,
            method=content_count.method if messages else "empty",
            model=model
        )
    
    def truncate_text_to_tokens(self, text: str, max_tokens: int, model: Optional[str] = None, preserve_end: bool = False) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            model: Model name for accurate counting
            preserve_end: Whether to preserve the end of the text instead of beginning
            
        Returns:
            Truncated text
        """
        current_count = self.count_tokens(text, model)
        
        if current_count.tokens <= max_tokens:
            return text
        
        # Binary search for optimal truncation point
        if preserve_end:
            # Preserve end of text
            start = 0
            end = len(text)
            
            while start < end:
                mid = (start + end) // 2
                truncated = text[mid:]
                
                if self.count_tokens(truncated, model).tokens <= max_tokens:
                    end = mid
                else:
                    start = mid + 1
            
            return text[start:]
        else:
            # Preserve beginning of text
            start = 0
            end = len(text)
            
            while start < end:
                mid = (start + end + 1) // 2
                truncated = text[:mid]
                
                if self.count_tokens(truncated, model).tokens <= max_tokens:
                    start = mid
                else:
                    end = mid - 1
            
            return text[:start]
    
    def split_text_by_tokens(self, text: str, chunk_size: int, overlap: int = 0, model: Optional[str] = None) -> List[str]:
        """
        Split text into chunks of specified token size.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            model: Model name for accurate counting
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple sentence-based splitting with token counting
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self.count_tokens(test_chunk, model)
            
            if token_count.tokens <= chunk_size:
                current_chunk = test_chunk
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if overlap > 0 and chunks:
                    overlap_text = self.truncate_text_to_tokens(
                        current_chunk, 
                        overlap, 
                        model, 
                        preserve_end=True
                    )
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_token_budget_info(self, memory_blocks: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get token budget information for memory blocks.
        
        Args:
            memory_blocks: Dictionary mapping block names to their content
            model: Model name for accurate counting
            
        Returns:
            Dictionary with token budget analysis
        """
        memory_config = self.config.get_memory_config()
        
        block_info = {}
        total_tokens = 0
        
        for block_name, content in memory_blocks.items():
            token_count = self.count_tokens(content, model)
            limit = memory_config.core_block_token_limit
            
            block_info[block_name] = {
                "current_tokens": token_count.tokens,
                "limit_tokens": limit,
                "utilization": round((token_count.tokens / limit) * 100, 1) if limit > 0 else 0,
                "characters": token_count.characters,
                "words": token_count.words,
                "exceeds_limit": token_count.tokens > limit,
                "tokens_over_limit": max(0, token_count.tokens - limit)
            }
            
            total_tokens += token_count.tokens
        
        return {
            "blocks": block_info,
            "total": {
                "current_tokens": total_tokens,
                "limit_tokens": memory_config.core_block_token_limit * len(memory_blocks),
                "utilization": round((total_tokens / (memory_config.core_block_token_limit * len(memory_blocks))) * 100, 1) if memory_blocks else 0,
                "blocks_over_limit": sum(1 for info in block_info.values() if info["exceeds_limit"])
            },
            "method": token_count.method if memory_blocks else "none",
            "model": model
        }


# Global token counter instance
token_counter = TokenCounter()


def count_tokens(text: str, model: Optional[str] = None, method: Optional[str] = None) -> TokenCount:
    """
    Convenience function to count tokens in text.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoder selection
        method: Counting method override
        
    Returns:
        Token count result
    """
    return token_counter.count_tokens(text, model, method)


def estimate_message_tokens(messages: List[Dict[str, str]], model: Optional[str] = None) -> TokenCount:
    """
    Convenience function to estimate tokens for chat messages.
    
    Args:
        messages: List of message dictionaries
        model: Model name for accurate counting
        
    Returns:
        Total token count for messages
    """
    return token_counter.estimate_tokens_for_messages(messages, model)


def truncate_to_tokens(text: str, max_tokens: int, model: Optional[str] = None, preserve_end: bool = False) -> str:
    """
    Convenience function to truncate text to token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        model: Model name for accurate counting
        preserve_end: Whether to preserve end instead of beginning
        
    Returns:
        Truncated text
    """
    return token_counter.truncate_text_to_tokens(text, max_tokens, model, preserve_end)


def split_by_tokens(text: str, chunk_size: int, overlap: int = 0, model: Optional[str] = None) -> List[str]:
    """
    Convenience function to split text by token chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum tokens per chunk
        overlap: Overlap tokens between chunks
        model: Model name for accurate counting
        
    Returns:
        List of text chunks
    """
    return token_counter.split_text_by_tokens(text, chunk_size, overlap, model)


def analyze_memory_tokens(memory_blocks: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze token usage in memory blocks.
    
    Args:
        memory_blocks: Dictionary of memory block content
        model: Model name for accurate counting
        
    Returns:
        Token budget analysis
    """
    return token_counter.get_token_budget_info(memory_blocks, model)