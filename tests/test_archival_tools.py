"""
Test archival memory tools (insert and search)
"""

from asterix import Agent, BlockConfig, StorageConfig
import os

print("=" * 60)
print("Testing Archival Memory Tools (Step 4.2b)")
print("=" * 60)

# Check environment variables
if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
    print("\n⚠️  WARNING: QDRANT_URL and QDRANT_API_KEY must be set!")
    print("   Set these environment variables and try again.")
    exit(1)

if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    print("\n⚠️  WARNING: Need OPENAI_API_KEY or GROQ_API_KEY!")
    print("   Set at least one API key and try again.")
    exit(1)

# Create agent
print("\n1. Creating agent with archival memory enabled...")
agent = Agent(
    agent_id="archival_test",
    blocks={"test": BlockConfig(size=1000, priority=1)}
)

# Check that archival tools are registered
print("\n2. Checking registered tools...")
all_tools = agent.get_all_tools()
tool_names = [tool.name for tool in all_tools]
print(f"   Registered tools: {tool_names}")

archival_insert = "archival_memory_insert" in tool_names
archival_search = "archival_memory_search" in tool_names
conversation_search = "conversation_search" in tool_names

print(f"   ✓ archival_memory_insert: {'✓' if archival_insert else '✗'}")
print(f"   ✓ archival_memory_search: {'✓' if archival_search else '✗'}")
print(f"   ✓ conversation_search: {'✓' if conversation_search else '✗'}")

if not (archival_insert and archival_search):
    print("\n❌ Archival tools not registered properly!")
    exit(1)

# Test archival insert (manually, not via chat)
print("\n3. Testing archival_memory_insert directly...")
try:
    insert_tool = agent.get_tool("archival_memory_insert")
    result = insert_tool.execute(
        content="User prefers Python over JavaScript and uses VS Code",
        summary="User programming preferences",
        importance=0.9
    )
    
    if result.status.value == "success":
        print(f"   ✓ Insert successful: {result.content[:100]}")
    else:
        print(f"   ✗ Insert failed: {result.error}")
        exit(1)
except Exception as e:
    print(f"   ✗ Insert error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test archival search
print("\n4. Testing archival_memory_search directly...")
try:
    search_tool = agent.get_tool("archival_memory_search")
    result = search_tool.execute(
        query="programming preferences",
        k=3
    )
    
    if result.status.value == "success":
        print(f"   ✓ Search successful")
        print(f"   Results preview: {result.content[:200]}...")
    else:
        print(f"   ✗ Search failed: {result.error}")
        exit(1)
except Exception as e:
    print(f"   ✗ Search error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ All archival memory tools working!")
print("=" * 60)