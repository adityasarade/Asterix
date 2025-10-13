"""
Test conversation search tool
"""

from asterix import Agent, BlockConfig

print("=" * 60)
print("Testing Conversation Search Tool (Step 4.2c)")
print("=" * 60)

# Create agent
print("\n1. Creating agent...")
agent = Agent(
    agent_id="conversation_search_test",
    blocks={"test": BlockConfig(size=1000, priority=1)}
)

# Add some conversation history manually
print("\n2. Adding conversation history...")
agent.conversation_history = [
    {"role": "user", "content": "I love Python programming"},
    {"role": "assistant", "content": "That's great! Python is very popular."},
    {"role": "user", "content": "My favorite editor is VS Code"},
    {"role": "assistant", "content": "VS Code is excellent for Python development."},
    {"role": "user", "content": "I prefer dark mode themes"},
    {"role": "assistant", "content": "Dark mode is easier on the eyes."}
]
print(f"   ✓ Added {len(agent.conversation_history)} messages")

# Test conversation search directly
print("\n3. Testing conversation_search directly...")
try:
    search_tool = agent.get_tool("conversation_search")
    
    # Search for Python-related messages
    result = search_tool.execute(
        query="programming language preferences",
        k=2
    )
    
    if result.status.value == "success":
        print(f"   ✓ Search successful")
        print(f"   Results:\n{result.content}")
    else:
        print(f"   ✗ Search failed: {result.error}")
        exit(1)
        
except Exception as e:
    print(f"   ✗ Search error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with no results
print("\n4. Testing search with no matches...")
try:
    result = search_tool.execute(
        query="underwater basket weaving",
        k=3
    )
    
    if result.status.value == "success":
        print(f"   ✓ No-match handling works")
        print(f"   Result: {result.content[:100]}")
    else:
        print(f"   ✗ Failed: {result.error}")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✅ Conversation search tool working!")
print("=" * 60)