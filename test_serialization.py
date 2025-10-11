"""
Test state serialization without requiring chat to work
"""

from asterix import Agent, BlockConfig
from datetime import datetime

print("=" * 60)
print("Testing Agent State Serialization (Step 3)")
print("=" * 60)

# Create agent
print("\n1. Creating agent...")
agent = Agent(
    agent_id="test_agent_1",
    blocks={
        "task": BlockConfig(size=1500, priority=1),
        "notes": BlockConfig(size=1000, priority=2)
    }
)
print(f"   ✓ Agent created: {agent.id}")

# Manually add some state (bypass chat)
print("\n2. Adding state manually...")
agent.blocks["task"].content = "Working on Asterix library state persistence"
agent.blocks["notes"].content = "User prefers clean code without redundancies"
agent.conversation_history = [
    {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
    {"role": "assistant", "content": "Hi there!", "timestamp": datetime.now().isoformat()}
]
print(f"   ✓ Added content to {len(agent.blocks)} blocks")
print(f"   ✓ Added {len(agent.conversation_history)} conversation messages")

# Serialize
print("\n3. Serializing to dict...")
state_dict = agent._to_state_dict()
print(f"   ✓ State keys: {list(state_dict.keys())}")
print(f"   ✓ Conversation messages: {len(state_dict['conversation_history'])}")
print(f"   ✓ Memory blocks: {list(state_dict['blocks'].keys())}")

# Deserialize
print("\n4. Deserializing from dict...")
restored_agent = Agent._from_state_dict(state_dict)
print(f"   ✓ Restored agent ID: {restored_agent.id}")
print(f"   ✓ Restored messages: {len(restored_agent.conversation_history)}")
print(f"   ✓ Restored blocks: {list(restored_agent.blocks.keys())}")

# Verify
print("\n5. Verifying data integrity...")
checks = []
checks.append(("Agent ID", agent.id == restored_agent.id))
checks.append(("Conversation length", len(agent.conversation_history) == len(restored_agent.conversation_history)))
checks.append(("Block count", len(agent.blocks) == len(restored_agent.blocks)))
checks.append(("Task content", agent.blocks["task"].content == restored_agent.blocks["task"].content))
checks.append(("Notes content", agent.blocks["notes"].content == restored_agent.blocks["notes"].content))

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"   {status} {check_name}")

all_passed = all(passed for _, passed in checks)

print("\n" + "=" * 60)
if all_passed:
    print("✅ All serialization tests PASSED!")
else:
    print("❌ Some tests FAILED")
print("=" * 60)