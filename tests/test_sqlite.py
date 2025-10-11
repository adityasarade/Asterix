"""
Test SQLite backend for state persistence
"""

from asterix import Agent, BlockConfig, StorageConfig

print("=" * 60)
print("Testing SQLite Backend (Step 3.3)")
print("=" * 60)

# Create agent with SQLite backend
print("\n1. Creating agent with SQLite backend...")
agent = Agent(
    agent_id="sqlite_test",
    blocks={
        "task": BlockConfig(size=1500, priority=1)
    },
    storage=StorageConfig(
        state_backend="sqlite",
        state_db="test_agents.db"
    )
)
agent.blocks["task"].content = "Testing SQLite persistence"
print(f"   ✓ Agent created: {agent.id}")

# Save to SQLite
print("\n2. Saving to SQLite...")
agent.save_state()
print("   ✓ Saved successfully")

# Load from SQLite
print("\n3. Loading from SQLite...")
loaded_agent = Agent.load_state(
    "sqlite_test", 
    state_backend="sqlite",
    state_db="test_agents.db"
)
print(f"   ✓ Loaded: {loaded_agent.id}")
print(f"   ✓ Content: {loaded_agent.blocks['task'].content}")

# Verify
if agent.blocks["task"].content == loaded_agent.blocks["task"].content:
    print("\n✅ SQLite backend works!")
else:
    print("\n❌ Content mismatch")

print("=" * 60)