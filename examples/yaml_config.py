#!/usr/bin/env python3
"""
YAML Configuration Example
=========================

Demonstrates how to configure Asterix agents using YAML files.

This example shows:
1. Loading agent configuration from YAML files
2. Overriding YAML settings with Python code
3. Environment variable integration
4. Multiple configuration patterns

Run:
    python examples/yaml_config.py
"""

import os
from pathlib import Path
from asterix import Agent, BlockConfig

# ============================================================================
# Setup
# ============================================================================

print("=" * 70)
print("YAML CONFIGURATION EXAMPLE")
print("=" * 70)

# Get the examples directory
examples_dir = Path(__file__).parent
yaml_file = examples_dir / "sample_agent_config.yaml"

print(f"\nUsing YAML configuration file: {yaml_file}")
print()

# ============================================================================
# Pattern 1: Load Directly from YAML
# ============================================================================

print("1. Loading agent directly from YAML file")
print("-" * 70)

try:
    # Load agent with all settings from YAML
    agent1 = Agent.from_yaml(yaml_file)
    
    print(f"✓ Agent created: {agent1.id}")
    print(f"  Model: {agent1.config.model}")
    print(f"  Temperature: {agent1.config.temperature}")
    print(f"  Memory blocks: {list(agent1.blocks.keys())}")
    print(f"  Block sizes: {[f'{name}={block.config.size}' for name, block in agent1.blocks.items()]}")
    
    # Test conversation
    print("\n  Testing conversation...")
    response = agent1.chat("Hello! What are you configured to do?")
    print(f"  Agent: {response[:150]}...")
    
except Exception as e:
    print(f"✗ Error loading from YAML: {e}")
    print("  Make sure the YAML file exists and is properly formatted.")

# ============================================================================
# Pattern 2: YAML + Python Overrides
# ============================================================================

print("\n2. Loading from YAML with Python overrides")
print("-" * 70)

try:
    # Load from YAML but override specific settings
    agent2 = Agent.from_yaml(
        yaml_file,
        agent_id="yaml_plus_overrides",
        model="openai/gpt-4o-mini",    # Override model
        temperature=0.1,               # Override temperature
        max_heartbeat_steps=5          # Override heartbeat steps
    )
    
    print(f"✓ Agent created: {agent2.id}")
    print(f"  Model (overridden): {agent2.config.model}")
    print(f"  Temperature (overridden): {agent2.config.temperature}")
    print(f"  Memory blocks (from YAML): {list(agent2.blocks.keys())}")
    
    # Test conversation
    print("\n  Testing conversation...")
    response = agent2.chat("What model are you using?")
    print(f"  Agent: {response[:150]}...")
    
except Exception as e:
    print(f"✗ Error with overrides: {e}")

# ============================================================================
# Pattern 3: Creating Agent from Scratch (Python Only)
# ============================================================================

print("\n3. Creating agent from scratch with Python configuration")
print("-" * 70)

# No YAML file needed - pure Python configuration
agent3 = Agent(
    agent_id="python_configured",
    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    },
    model="openai/gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
    max_heartbeat_steps=10
)

print(f"✓ Agent created: {agent3.id}")
print(f"  Model: {agent3.config.model}")
print(f"  Memory blocks: {list(agent3.blocks.keys())}")

# ============================================================================
# Pattern 4: Environment Variables in YAML
# ============================================================================

print("\n4. Environment variable integration")
print("-" * 70)

# Check if environment variables are set
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_url and qdrant_api_key:
    print("✓ Qdrant credentials found in environment")
    print(f"  URL: {qdrant_url[:30]}...")
    print("  API Key: [REDACTED]")
    print("\n  YAML file uses ${QDRANT_URL} and ${QDRANT_API_KEY}")
    print("  These are automatically resolved when loading the agent")
else:
    print("⚠ Qdrant credentials not found in environment")
    print("  Set QDRANT_URL and QDRANT_API_KEY to use Qdrant features")
    print("\n  In YAML, you can reference env vars like:")
    print("    qdrant_url: \"${QDRANT_URL}\"")
    print("    qdrant_api_key: \"${QDRANT_API_KEY}\"")

# ============================================================================
# Pattern 5: Comparing Configurations
# ============================================================================

print("\n5. Configuration comparison")
print("-" * 70)

print("\nYAML-configured agent:")
print(f"  ID: {agent1.id}")
print(f"  Model: {agent1.config.model}")
print(f"  Blocks: {len(agent1.blocks)}")

print("\nYAML + overrides agent:")
print(f"  ID: {agent2.id}")
print(f"  Model: {agent2.config.model}")
print(f"  Blocks: {len(agent2.blocks)}")

print("\nPython-configured agent:")
print(f"  ID: {agent3.id}")
print(f"  Model: {agent3.config.model}")
print(f"  Blocks: {len(agent3.blocks)}")

# ============================================================================
# Demonstration: YAML Makes Configuration Management Easy
# ============================================================================

print("\n6. Why use YAML configuration?")
print("-" * 70)

print("""
YAML configuration provides several benefits:

✓ Separation of Concerns
  - Keep configuration separate from code
  - Easy to update settings without changing code
  - Different configs for dev/staging/prod

✓ Version Control
  - Track configuration changes in git
  - Review config changes in pull requests
  - Rollback to previous configurations

✓ Team Collaboration
  - Non-developers can modify settings
  - Clear documentation of agent behavior
  - Share configurations across team

✓ Environment-Specific Settings
  - Use different YAML files for different environments
  - Override settings per deployment
  - Secrets via environment variables

✓ Reusability
  - Create templates for different agent types
  - Share configurations across projects
  - Easy to duplicate and customize
""")

# ============================================================================
# Best Practices
# ============================================================================

print("\n7. Best practices for YAML configuration")
print("-" * 70)

print("""
📋 Configuration Best Practices:

1. Use descriptive agent_id values
   ✓ Good: "customer_support_agent"
   ✗ Bad: "agent1"

2. Document blocks with descriptions
   - Helps understand memory structure
   - Useful for team members

3. Keep secrets in environment variables
   - Never commit API keys to git
   - Use ${VAR_NAME} syntax in YAML

4. Set appropriate priorities
   - High (10): Persona, critical context
   - Medium (5): User data, preferences
   - Low (1-2): Temporary task context

5. Use reasonable token limits!
   - persona: 800-1200 tokens
   - user_info: 500-1000 tokens
   - task_context: 1000-2000 tokens
   - Increase limits for larger models

6. Test configurations locally first
   - Verify YAML syntax
   - Test with sample conversations
   - Check memory behavior
""")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("✅ YAML Configuration Example Complete!")
print("=" * 70)

print("\nKey takeaways:")
print("  • Use YAML for clean, maintainable configurations")
print("  • Override settings with Python when needed")
print("  • Reference environment variables with ${VAR_NAME}")
print("  • Choose the pattern that fits your use case")
print("  • Keep secrets out of YAML files")

print("\nNext steps:")
print("  • Copy sample_agent_config.yaml and customize it")
print("  • Create different configs for different agent types")
print("  • Use environment variables for sensitive data")
print("  • Version control your YAML configurations")

print("=" * 70)