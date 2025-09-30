import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memgpt.utils.config import get_config

config = get_config()

# Test configuration loading
print("=== Configuration Test ===")
print(f"LLM Config: {config.get_llm_config()}")
print(f"Embedding Config: {config.get_embedding_config()}")
print(f"Qdrant Config: {config.get_qdrant_config()}")
print(f"Memory Config: {config.get_memory_config()}")

# Test validation
validation = config.validate_configuration()
print(f"Validation Results: {validation}")