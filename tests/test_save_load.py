"""
Test save_state and load_state functionality
"""

from asterix import Agent, BlockConfig
from pathlib import Path

print("=" * 60)
print("Testing Agent Save/Load (Step 3.2)")
print("=" * 60)

# Create agent
print("\n1. Creating and configuring agent...")
agent = Agent(
    agent_id="persistence_test",
    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    }
)

# Add some state
agent.blocks["task"].content = "Building state persistence for Asterix"
agent.blocks["notes"].content = "User wants clean, modular code"
agent.conversation_history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
]
print(f"   ✓ Agent configured with {len(agent.blocks)} blocks")
print(f"   ✓ Added {len(agent.conversation_history)} messages")

# Save state
print("\n2. Saving agent state...")
agent.save_state()
save_path = Path("./agent_states/persistence_test.json")
print(f"   ✓ Saved to: {save_path}")
print(f"   ✓ File exists: {save_path.exists()}")

# Load state
print("\n3. Loading agent state...")
loaded_agent = Agent.load_state("persistence_test")
print(f"   ✓ Loaded agent: {loaded_agent.id}")
print(f"   ✓ Blocks: {list(loaded_agent.blocks.keys())}")
print(f"   ✓ Messages: {len(loaded_agent.conversation_history)}")

# Verify
print("\n4. Verifying integrity...")
checks = [
    ("Agent ID", agent.id == loaded_agent.id),
    ("Task content", agent.blocks["task"].content == loaded_agent.blocks["task"].content),
    ("Notes content", agent.blocks["notes"].content == loaded_agent.blocks["notes"].content),
    ("Conversation", len(agent.conversation_history) == len(loaded_agent.conversation_history)),
    ("Model config", agent.config.model == loaded_agent.config.model)
]

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"   {status} {check_name}")

all_passed = all(passed for _, passed in checks)

print("\n" + "=" * 60)
if all_passed:
    print("✅ All save/load tests PASSED!")
else:
    print("❌ Some tests FAILED")
print("=" * 60)

# Cleanup
print("\nCleaning up test file...")
if save_path.exists():
    save_path.unlink()
    print(f"   ✓ Deleted {save_path}")