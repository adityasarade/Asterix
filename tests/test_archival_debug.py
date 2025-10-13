"""
Debug archival memory insert and search
"""

import os
from asterix import Agent, BlockConfig

print("=" * 60)
print("Debugging Archival Memory (Step 4.3c)")
print("=" * 60)

# Check environment
required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✓ {var}: {value[:20]}..." if len(value) > 20 else f"✓ {var}: set")
    else:
        print(f"✗ {var}: NOT SET")
        exit(1)

# Create agent
print("\n1. Creating agent...")
agent = Agent(
    agent_id="debug_archival",
    blocks={"test": BlockConfig(size=1000, priority=1)}
)
print(f"   Agent ID: {agent.id}")
print(f"   Collection: asterix_memory_{agent.id}")

# Test insert with very simple content
print("\n2. Inserting test memory...")
insert_tool = agent.get_tool("archival_memory_insert")

result = insert_tool.execute(
    content="Python is my favorite programming language",
    summary="Programming preference",
    importance=0.9
)

print(f"   Status: {result.status.value}")
print(f"   Response: {result.content}")
if result.metadata:
    print(f"   Memory ID: {result.metadata.get('memory_id', 'N/A')}")
    print(f"   Embedding provider: {result.metadata.get('embedding_provider', 'N/A')}")
    print(f"   Embedding dims: {result.metadata.get('embedding_dimensions', 'N/A')}")

if result.status.value != "success":
    print(f"\n✗ Insert failed: {result.error}")
    exit(1)

# Test search with exact same words
print("\n3. Searching with exact match...")
search_tool = agent.get_tool("archival_memory_search")

result = search_tool.execute(
    query="Python programming language",
    k=5,
    score_threshold=0.3  # Lower threshold for testing
)

print(f"   Status: {result.status.value}")
print(f"   Response:\n{result.content}")

if result.metadata:
    results = result.metadata.get('results', [])
    print(f"\n   Found {len(results)} results:")
    for i, res in enumerate(results, 1):
        print(f"   {i}. Score: {res['score']:.3f}, Summary: {res['summary']}")

# Try with different query
print("\n4. Searching with related query...")
result = search_tool.execute(
    query="favorite coding language",
    k=5,
    score_threshold=0.3
)

print(f"   Status: {result.status.value}")
if result.metadata:
    results = result.metadata.get('results', [])
    print(f"   Found {len(results)} results")
    if results:
        for i, res in enumerate(results, 1):
            print(f"   {i}. Score: {res['score']:.3f}, Summary: {res['summary']}")

# Check Qdrant collection directly
print("\n5. Checking Qdrant collection status...")
try:
    from asterix.storage.qdrant_client import qdrant_client
    import asyncio
    
    # Ensure connected
    connected = asyncio.run(qdrant_client.ensure_connected())
    print(f"   Connected: {connected}")
    
    if connected:
        # Get collection info
        info = asyncio.run(qdrant_client.get_collection_info())
        print(f"   Collection: {info.get('name', 'N/A')}")
        print(f"   Points count: {info.get('points_count', 0)}")
        print(f"   Vectors count: {info.get('vectors_count', 0)}")
        
        # Count vectors with agent filter
        count = asyncio.run(qdrant_client.count_vectors({"agent_id": agent.id}))
        print(f"   Vectors for this agent: {count}")
        
except Exception as e:
    print(f"   ✗ Error checking Qdrant: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)