"""
MemGPT Service Wrappers Test

Tests all service wrappers to ensure they're working correctly:
- Letta client wrapper
- Qdrant Cloud wrapper  
- Embedding service wrapper
- LLM provider manager
"""

import asyncio
import logging
from src.memgpt.services import (
    letta_client, qdrant_client, embedding_service, llm_manager,
    LettaAgentConfig, ArchivalRecord, LLMMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_letta_client():
    """Test Letta client wrapper functionality."""
    print("\n=== Testing Letta Client Wrapper ===")
    
    try:
        # Test connection
        print("Testing Letta connection...")
        connected = await letta_client.ensure_connected()
        if connected:
            print("✅ Letta connection: SUCCESS")
        else:
            print("❌ Letta connection: FAILED")
            return False
        
        # Test getting models
        print("Getting available models...")
        models = await letta_client.get_available_models()
        print(f"✅ Found {len(models)} models available")
        
        # Test listing agents
        print("Listing existing agents...")
        agents = await letta_client.list_agents()
        print(f"✅ Found {len(agents)} existing agents")
        
        # Get connection status
        status = letta_client.get_connection_status()
        print(f"✅ Connection status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Letta client test failed: {e}")
        return False


async def test_qdrant_client():
    """Test Qdrant Cloud wrapper functionality."""
    print("\n=== Testing Qdrant Cloud Wrapper ===")
    
    try:
        # Test connection
        print("Testing Qdrant connection...")
        connected = await qdrant_client.ensure_connected()
        if connected:
            print("✅ Qdrant connection: SUCCESS")
        else:
            print("❌ Qdrant connection: FAILED")
            return False
        
        # Test collection info
        print("Getting collection information...")
        collection_info = await qdrant_client.get_collection_info()
        if collection_info:
            print(f"✅ Collection info: {collection_info['name']} with {collection_info.get('points_count', 0)} points")
        else:
            print("⚠️  No collection info available (collection may not exist yet)")
        
        # Test vector count
        print("Counting vectors...")
        vector_count = await qdrant_client.count_vectors()
        print(f"✅ Vector count: {vector_count}")
        
        # Get performance metrics
        metrics = qdrant_client.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant client test failed: {e}")
        return False


async def test_embedding_service():
    """Test embedding service wrapper functionality."""
    print("\n=== Testing Embedding Service Wrapper ===")
    
    try:
        # Test single text embedding
        print("Testing single text embedding...")
        test_text = "This is a test sentence for embedding generation."
        result = await embedding_service.embed_texts(test_text)
        print(f"✅ Single embedding: {result.dimensions}D using {result.provider} ({result.model})")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Usage: {result.usage}")
        
        # Test batch embedding
        print("Testing batch embedding...")
        test_texts = [
            "First test sentence for batch processing.",
            "Second test sentence for batch processing.",
            "Third test sentence for batch processing."
        ]
        batch_result = await embedding_service.embed_batch(test_texts)
        print(f"✅ Batch embedding: {len(batch_result.embeddings)} vectors, {batch_result.dimensions}D")
        print(f"   Processing time: {batch_result.processing_time:.3f}s")
        print(f"   Usage: {batch_result.usage}")
        
        # Test provider switching
        print("Testing provider switching...")
        for provider in ["openai", "sentence_transformers"]:
            try:
                provider_result = await embedding_service.embed_texts(test_text, provider=provider)
                print(f"✅ {provider}: {provider_result.dimensions}D in {provider_result.processing_time:.3f}s")
            except Exception as e:
                print(f"⚠️  {provider}: {e}")
        
        # Get performance metrics
        metrics = embedding_service.get_performance_metrics()
        print(f"✅ Performance metrics: Cache hit rate: {metrics['cache_hit_rate']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


async def test_llm_manager():
    """Test LLM provider manager functionality."""
    print("\n=== Testing LLM Provider Manager ===")
    
    try:
        # Test simple completion
        print("Testing simple completion...")
        response = await llm_manager.complete("Hello, please respond with 'Test successful'")
        print(f"✅ Simple completion: {response.provider} ({response.model})")
        print(f"   Response: {response.content[:100]}...")
        print(f"   Processing time: {response.processing_time:.3f}s")
        print(f"   Usage: {response.usage}")
        
        # Test conversation completion
        print("Testing conversation completion...")
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="What is 2+2?"),
        ]
        conv_response = await llm_manager.complete(messages)
        print(f"✅ Conversation completion: {conv_response.provider}")
        print(f"   Response: {conv_response.content[:100]}...")
        
        # Test summarization
        print("Testing text summarization...")
        test_text = """
        This is a longer text that needs to be summarized. It contains multiple sentences
        and discusses various topics. The purpose is to test the summarization capability
        of the LLM provider manager. It should produce a concise summary that captures
        the main points while being much shorter than the original text.
        """
        summary_response = await llm_manager.summarize_text(test_text, max_tokens=50)
        print(f"✅ Summarization: {summary_response.provider}")
        print(f"   Summary: {summary_response.content}")
        
        # Test keyword extraction
        print("Testing keyword extraction...")
        keywords = await llm_manager.extract_keywords(test_text, max_keywords=5)
        print(f"✅ Keywords: {keywords}")
        
        # Test provider switching
        print("Testing provider switching...")
        for provider in ["groq", "openai"]:
            try:
                provider_response = await llm_manager.complete(
                    "Hello from " + provider, 
                    provider=provider
                )
                print(f"✅ {provider}: {provider_response.content[:50]}...")
            except Exception as e:
                print(f"⚠️  {provider}: {e}")
        
        # Get performance metrics
        metrics = llm_manager.get_performance_metrics()
        print(f"✅ Performance metrics:")
        print(f"   Primary: {metrics['primary_provider']}, Fallback: {metrics['fallback_provider']}")
        for provider, stats in metrics['providers'].items():
            print(f"   {provider}: {stats['operation_count']} ops, {stats['success_rate']}% success")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM manager test failed: {e}")
        return False


async def test_integration():
    """Test integration between services."""
    print("\n=== Testing Service Integration ===")
    
    try:
        # Test embedding + Qdrant integration
        print("Testing embedding + Qdrant integration...")
        
        # Generate embeddings for test data
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
        
        embedding_result = await embedding_service.embed_batch(test_texts)
        print(f"✅ Generated {len(embedding_result.embeddings)} embeddings")
        
        # Create archival records
        records = []
        for i, (text, embedding) in enumerate(zip(test_texts, embedding_result.embeddings)):
            record = ArchivalRecord(
                text=text,
                summary=f"Summary of text {i+1}",
                embedding=embedding,
                metadata={
                    "source": "integration_test",
                    "type": "test_document",
                    "user_id": "test_user",
                    "agent_id": "test_agent",
                    "index": i
                }
            )
            records.append(record)
        
        # Insert into Qdrant (if connected)
        if await qdrant_client.ensure_connected():
            print("Inserting test vectors into Qdrant...")
            point_ids = await qdrant_client.insert_vectors(records)
            print(f"✅ Inserted {len(point_ids)} vectors into Qdrant")
            
            # Test semantic search
            print("Testing semantic search...")
            query_text = "What is machine learning?"
            query_embedding = await embedding_service.embed_texts(query_text)
            
            search_results = await qdrant_client.search_vectors(
                query_vector=query_embedding.embeddings[0],
                limit=2,
                score_threshold=0.5
            )
            
            print(f"✅ Found {len(search_results)} similar vectors")
            for i, result in enumerate(search_results):
                print(f"   {i+1}. Score: {result.score:.3f} - {result.payload.get('text', '')[:60]}...")
        else:
            print("⚠️  Qdrant not available, skipping vector insertion test")
        
        # Test LLM + embedding integration
        print("Testing LLM + embedding integration...")
        
        # Use LLM to generate text, then embed it
        llm_response = await llm_manager.complete(
            "Write a brief sentence about artificial intelligence."
        )
        
        ai_embedding = await embedding_service.embed_texts(llm_response.content)
        print(f"✅ Generated text and embedded it: {ai_embedding.dimensions}D vector")
        print(f"   Generated text: {llm_response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Run all service wrapper tests."""
    print("🧪 MemGPT Service Wrappers Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run individual service tests
    test_results.append(("Letta Client", await test_letta_client()))
    test_results.append(("Qdrant Client", await test_qdrant_client()))
    test_results.append(("Embedding Service", await test_embedding_service()))
    test_results.append(("LLM Manager", await test_llm_manager()))
    test_results.append(("Integration", await test_integration()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All service wrappers are working correctly!")
        print("Ready to proceed to Step 4: Controller and Heartbeat Loop")
    else:
        print("⚠️  Some tests failed. Please check the service configurations and try again.")
    
    return passed == total


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())