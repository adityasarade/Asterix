"""
Diagnostic script to identify exact issues with archival memory and context extraction
"""

import os
import asyncio
from asterix import Agent, BlockConfig

print("=" * 70)
print("DIAGNOSTIC TEST - Identifying Memory Management Issues")
print("=" * 70)

# Check environment
required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        print(f"❌ Missing: {var}")
        exit(1)
print("✅ Environment variables set\n")

# ============================================================================
# DIAGNOSTIC 1: Archival Insert/Search Flow
# ============================================================================
print("=" * 70)
print("DIAGNOSTIC 1: Archival Memory Insert → Search Flow")
print("=" * 70)

agent = Agent(
    agent_id="diagnostic_agent_1",
    blocks={"test": BlockConfig(size=1000, priority=1)}
)

print(f"\n1. Agent created: {agent.id}")
print(f"   Collection should be: asterix_memory_{agent.id}")

# Step 1: Insert a memory
print("\n2. Inserting test memory...")
insert_tool = agent.get_tool("archival_memory_insert")

result = insert_tool.execute(
    content="Python is the best programming language for AI development",
    summary="Python preference for AI",
    importance=0.9
)

print(f"   Status: {result.status.value}")
if result.status.value == "success":
    memory_id = result.metadata.get('memory_id')
    print(f"   ✅ Inserted with ID: {memory_id}")
    print(f"   Embedding provider: {result.metadata.get('embedding_provider')}")
    print(f"   Embedding dims: {result.metadata.get('embedding_dimensions')}")
else:
    print(f"   ❌ Insert failed: {result.error}")
    exit(1)

# Step 2: Verify storage in Qdrant directly
print("\n3. Verifying storage in Qdrant...")
try:
    from asterix.storage.qdrant_client import qdrant_client
    
    connected = asyncio.run(qdrant_client.ensure_connected())
    print(f"   Connected: {connected}")
    
    if not connected:
        print("   ❌ Cannot connect to Qdrant")
        exit(1)
    
    # Count vectors for this agent
    count = asyncio.run(qdrant_client.count_vectors({"agent_id": agent.id}))
    print(f"   ✅ Vectors for agent '{agent.id}': {count}")
    
    if count == 0:
        print("   ❌ PROBLEM: Memory not stored in Qdrant despite success message!")
        print("   This suggests archival_memory_insert isn't actually storing.")
    
except Exception as e:
    print(f"   ❌ Error checking Qdrant: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Search immediately after insert
print("\n4. Searching for inserted memory...")
search_tool = agent.get_tool("archival_memory_search")

result = search_tool.execute(
    query="Python programming language AI",
    k=5,
    score_threshold=0.3  # Low threshold
)

print(f"   Status: {result.status.value}")
if result.status.value == "success":
    results = result.metadata.get('results', [])
    print(f"   Results found: {len(results)}")
    
    if len(results) == 0:
        print("   ❌ PROBLEM: Search found nothing immediately after insert!")
        print("   Possible causes:")
        print("      - Agent ID filter mismatch")
        print("      - Collection name mismatch")
        print("      - Embedding not stored properly")
    else:
        print(f"   ✅ Search found {len(results)} results")
        for i, res in enumerate(results, 1):
            print(f"      {i}. Score: {res['score']:.3f}, Summary: {res['summary']}")
else:
    print(f"   ❌ Search failed: {result.error}")

# ============================================================================
# DIAGNOSTIC 2: Context Extraction & Fact Storage
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Context Extraction → Fact Storage")
print("=" * 70)

agent2 = Agent(
    agent_id="diagnostic_agent_2",
    blocks={"notes": BlockConfig(size=500, priority=1)}
)

print(f"\n1. Agent created: {agent2.id}")

# Add MORE THAN 10 messages with obvious facts
print("2. Adding messages with obvious facts...")
agent2.conversation_history = [
    {"role": "user", "content": "I prefer Python for AI development"},
    {"role": "assistant", "content": "Python is great for AI!"},
    {"role": "user", "content": "I use VS Code as my editor"},
    {"role": "assistant", "content": "VS Code is excellent!"},
    {"role": "user", "content": "My company is Microsoft"},
    {"role": "assistant", "content": "Microsoft has great tools!"},
    {"role": "user", "content": "I work on machine learning projects"},
    {"role": "assistant", "content": "ML is fascinating!"},
    {"role": "user", "content": "I prefer TensorFlow over PyTorch"},
    {"role": "assistant", "content": "Both are solid frameworks!"},
    {"role": "user", "content": "I'm based in Seattle"},
    {"role": "assistant", "content": "Great tech hub!"},
    {"role": "user", "content": "I have 5 years of experience"},
    {"role": "assistant", "content": "That's solid experience!"},
    {"role": "user", "content": "I specialize in NLP"},
    {"role": "assistant", "content": "Natural language processing is exciting!"},
    {"role": "user", "content": "I prefer GPT models"},
    {"role": "assistant", "content": "GPT models are powerful!"},
    {"role": "user", "content": "I drink coffee while coding"},
    {"role": "assistant", "content": "Coffee and code go well together!"},
    {"role": "user", "content": "My favorite color is blue"},
    {"role": "assistant", "content": "Blue is calming!"},
    {"role": "user", "content": "What should I work on today?"},  # Message 23-24
    {"role": "assistant", "content": "Based on your preferences, continue with your ML projects!"}
]

print(f"   Added {len(agent2.conversation_history)} messages with clear facts")
print(f"   (Need >10 to trigger extraction - have {len(agent2.conversation_history)})")

# Manually trigger extraction
print("\n3. Triggering context extraction...")
try:
    # Check count before extraction
    count_before = asyncio.run(qdrant_client.count_vectors({"agent_id": agent2.id}))
    print(f"   Vectors before extraction: {count_before}")
    
    agent2._extract_and_archive_context()
    
    # Check count after extraction
    count_after = asyncio.run(qdrant_client.count_vectors({"agent_id": agent2.id}))
    print(f"   Vectors after extraction: {count_after}")
    
    stored_count = count_after - count_before
    if stored_count == 0:
        print(f"   ❌ PROBLEM: No facts were stored in Qdrant!")
        print(f"   This suggests:")
        print(f"      - LLM fact extraction failed (returned empty list)")
        print(f"      - JSON parsing failed")
        print(f"      - Facts extracted but archival_memory_insert failed")
    else:
        print(f"   ✅ Stored {stored_count} new facts")
    
    print(f"   Conversation trimmed to: {len(agent2.conversation_history)} messages")
    
except Exception as e:
    print(f"   ❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# DIAGNOSTIC 3: Check Agent ID in Stored Vectors
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Verify Agent ID Filtering")
print("=" * 70)

print("\n1. Checking all vectors in collection...")
try:
    # Scroll through all vectors
    all_vectors, _ = asyncio.run(qdrant_client.scroll_vectors(limit=20))
    print(f"   Total vectors in collection: {len(all_vectors)}")
    
    if len(all_vectors) > 0:
        print("\n2. Agent IDs found:")
        agent_ids = set()
        for vec in all_vectors:
            agent_id = vec.payload.get('agent_id', 'MISSING')
            agent_ids.add(agent_id)
        
        for aid in agent_ids:
            count = sum(1 for v in all_vectors if v.payload.get('agent_id') == aid)
            print(f"      - '{aid}': {count} vectors")
        
        # Check if our agents' IDs match what's stored
        print("\n3. ID match verification:")
        print(f"   Agent 1 ID: {agent.id}")
        print(f"   Agent 2 ID: {agent2.id}")
        
        if agent.id not in agent_ids:
            print(f"   ❌ Agent 1 ID not found in stored vectors!")
        else:
            print(f"   ✅ Agent 1 ID found")
        
        if agent2.id not in agent_ids:
            print(f"   ❌ Agent 2 ID not found in stored vectors!")
        else:
            print(f"   ✅ Agent 2 ID found")
    else:
        print("   ❌ No vectors found in collection!")

except Exception as e:
    print(f"   ❌ Error scrolling vectors: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("\nRun this script and examine the output to identify:")
print("1. Whether archival_memory_insert actually stores in Qdrant")
print("2. Whether agent_id filtering is working correctly")
print("3. Whether context extraction produces valid facts")
print("4. Whether fact storage is working")
print("\nLook for ❌ PROBLEM markers in the output above.")
print("=" * 70)