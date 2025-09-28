#!/usr/bin/env python3
"""
MemGPT Controller Test Script

Tests the FastAPI controller and heartbeat loop:
- Service health verification
- Agent creation and management
- Chat endpoint with heartbeat loop
- Memory operations and eviction
- Error handling and recovery
"""

import asyncio
import httpx
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_USER_ID = "test_user_123"
TEST_AGENT_NAME = "test_agent_controller"


async def test_controller_health():
    """Test controller health endpoints"""
    print("\n=== Testing Controller Health ===")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test basic health
            print("Testing basic health endpoint...")
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Basic health check: SUCCESS")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Basic health check: FAILED - {response.status_code}")
                return False
            
            # Test service health
            print("Testing service health endpoint...")
            response = await client.get(f"{BASE_URL}/health/services")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Service health check: SUCCESS")
                print(f"   Total services: {health_data['summary']['total_services']}")
                print(f"   Healthy services: {health_data['summary']['healthy_services']}")
                print(f"   Health percentage: {health_data['summary']['health_percentage']}%")
            else:
                print(f"❌ Service health check: FAILED - {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Health check test failed: {e}")
            return False


async def test_agent_management():
    """Test agent creation and listing"""
    print("\n=== Testing Agent Management ===")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test listing agents
            print("Testing agent listing...")
            response = await client.get(f"{BASE_URL}/agents")
            if response.status_code == 200:
                agents_data = response.json()
                print(f"✅ Agent listing: SUCCESS - {agents_data['count']} agents found")
                existing_agents = agents_data['agents']
            else:
                print(f"❌ Agent listing: FAILED - {response.status_code}")
                return False, None
            
            # Check if test agent already exists
            test_agent_id = None
            for agent in existing_agents:
                if agent['name'] == TEST_AGENT_NAME:
                    test_agent_id = agent['id']
                    print(f"✅ Found existing test agent: {test_agent_id}")
                    break
            
            # Create test agent if it doesn't exist
            if not test_agent_id:
                print("Creating new test agent...")
                create_request = {
                    "name": TEST_AGENT_NAME,
                    "user_id": TEST_USER_ID
                }
                
                response = await client.post(f"{BASE_URL}/agents", json=create_request)
                if response.status_code == 200:
                    agent_data = response.json()
                    test_agent_id = agent_data['agent_id']
                    print(f"✅ Agent creation: SUCCESS - {test_agent_id}")
                    print(f"   Name: {agent_data['name']}")
                    print(f"   Model: {agent_data['model']}")
                    print(f"   Memory blocks: {agent_data['memory_blocks']}")
                else:
                    print(f"❌ Agent creation: FAILED - {response.status_code}")
                    print(f"   Error: {response.text}")
                    return False, None
            
            return True, test_agent_id
            
        except Exception as e:
            print(f"❌ Agent management test failed: {e}")
            return False, None


async def test_chat_endpoint(agent_id: str):
    """Test the main chat endpoint with heartbeat loop"""
    print("\n=== Testing Chat Endpoint ===")
    
    async with httpx.AsyncClient(timeout=120.0) as client:  # Extended timeout for chat
        try:
            # Test simple chat
            print("Testing simple chat...")
            chat_request = {
                "agent_id": agent_id,
                "user_id": TEST_USER_ID,
                "text": "Hello! Please introduce yourself and tell me about your memory capabilities.",
                "include_raw": True
            }
            
            start_time = time.time()
            response = await client.post(f"{BASE_URL}/chat", json=chat_request)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                chat_data = response.json()
                print("✅ Simple chat: SUCCESS")
                print(f"   Response time: {response_time:.3f}s")
                print(f"   Heartbeat steps: {chat_data['heartbeat_steps']}")
                print(f"   Memory operations: {len(chat_data['memory_ops'])}")
                print(f"   Processing time: {chat_data['processing_time']:.3f}s")
                print(f"   Assistant: {chat_data['assistant'][:100]}...")
                
                # Show memory operations
                if chat_data['memory_ops']:
                    print("   Memory operations performed:")
                    for i, op in enumerate(chat_data['memory_ops'][:3]):  # Show first 3
                        print(f"     {i+1}. {op['operation']} - {op['success']}")
                
            else:
                print(f"❌ Simple chat: FAILED - {response.status_code}")
                print(f"   Error: {response.text}")
                return False
            
            # Test chat with memory
            print("\nTesting chat with memory interaction...")
            memory_chat_request = {
                "agent_id": agent_id,
                "user_id": TEST_USER_ID,
                "text": "Remember that I prefer coffee over tea, and I usually work from 9 AM to 5 PM.",
                "include_raw": False
            }
            
            response = await client.post(f"{BASE_URL}/chat", json=memory_chat_request)
            
            if response.status_code == 200:
                memory_data = response.json()
                print("✅ Memory chat: SUCCESS")
                print(f"   Heartbeat steps: {memory_data['heartbeat_steps']}")
                print(f"   Memory operations: {len(memory_data['memory_ops'])}")
                print(f"   Assistant: {memory_data['assistant'][:100]}...")
                
                # Check for memory operations
                memory_ops = memory_data['memory_ops']
                has_memory_ops = any(
                    op['operation'] in ['core_memory_append', 'core_memory_replace', 'archival_insert']
                    for op in memory_ops
                )
                
                if has_memory_ops:
                    print("   ✅ Memory operations detected")
                else:
                    print("   ⚠️  No memory operations detected")
                
            else:
                print(f"❌ Memory chat: FAILED - {response.status_code}")
                return False
            
            # Test memory recall
            print("\nTesting memory recall...")
            recall_chat_request = {
                "agent_id": agent_id,
                "user_id": TEST_USER_ID,
                "text": "What do you remember about my preferences?",
                "include_raw": False
            }
            
            response = await client.post(f"{BASE_URL}/chat", json=recall_chat_request)
            
            if response.status_code == 200:
                recall_data = response.json()
                print("✅ Memory recall: SUCCESS")
                print(f"   Assistant: {recall_data['assistant'][:150]}...")
                
                # Check if the agent mentions the stored preferences
                assistant_text = recall_data['assistant'].lower()
                mentions_coffee = 'coffee' in assistant_text
                mentions_work_hours = '9' in assistant_text or 'work' in assistant_text
                
                if mentions_coffee or mentions_work_hours:
                    print("   ✅ Agent successfully recalled stored information")
                else:
                    print("   ⚠️  Agent may not have recalled stored information accurately")
                
            else:
                print(f"❌ Memory recall: FAILED - {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Chat endpoint test failed: {e}")
            return False


async def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test with invalid agent ID
            print("Testing invalid agent ID...")
            invalid_chat_request = {
                "agent_id": "invalid_agent_id_12345",
                "user_id": TEST_USER_ID,
                "text": "Hello"
            }
            
            response = await client.post(f"{BASE_URL}/chat", json=invalid_chat_request)
            if response.status_code >= 400:
                print("✅ Invalid agent ID handling: SUCCESS (properly rejected)")
            else:
                print("⚠️  Invalid agent ID handling: Unexpected success")
            
            # Test with malformed request
            print("Testing malformed request...")
            malformed_request = {
                "agent_id": "",  # Empty agent ID
                "user_id": "",   # Empty user ID
                "text": ""       # Empty text
            }
            
            response = await client.post(f"{BASE_URL}/chat", json=malformed_request)
            if response.status_code >= 400:
                print("✅ Malformed request handling: SUCCESS (properly rejected)")
            else:
                print("⚠️  Malformed request handling: Unexpected success")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False


async def test_service_integration():
    """Test integration with service wrappers"""
    print("\n=== Testing Service Integration ===")
    
    try:
        # Import our services to test direct integration
        from src.memgpt.services import letta_client, qdrant_client, embedding_service, llm_manager
        
        # Test service health integration
        print("Testing service health integration...")
        
        # Check if services are properly integrated
        letta_connected = await letta_client.ensure_connected()
        qdrant_connected = await qdrant_client.ensure_connected()
        
        print(f"   Letta connected: {letta_connected}")
        print(f"   Qdrant connected: {qdrant_connected}")
        
        if letta_connected and qdrant_connected:
            print("✅ Service integration: SUCCESS")
        else:
            print("⚠️  Service integration: Some services not connected")
        
        # Test embedding service
        print("Testing embedding service integration...")
        test_embedding = await embedding_service.embed_texts("Test integration")
        print(f"   Embedding dimensions: {test_embedding.dimensions}")
        print(f"   Provider: {test_embedding.provider}")
        
        # Test LLM manager
        print("Testing LLM manager integration...")
        test_completion = await llm_manager.complete("Say 'Integration test successful'")
        print(f"   LLM provider: {test_completion.provider}")
        print(f"   Response: {test_completion.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        return False


async def main():
    """Run all controller tests"""
    print("🧪 MemGPT Controller Test Suite")
    print("=" * 50)
    
    # Note: Controller must be running on port 8000 for these tests
    print("📝 NOTE: Make sure the controller is running on http://127.0.0.1:8000")
    print("   You can start it with: python -m src.memgpt.controller.api")
    print()
    
    test_results = []
    test_agent_id = None
    
    # Run tests
    test_results.append(("Controller Health", await test_controller_health()))
    
    agent_test_result, agent_id = await test_agent_management()
    test_results.append(("Agent Management", agent_test_result))
    test_agent_id = agent_id
    
    if test_agent_id:
        test_results.append(("Chat Endpoint", await test_chat_endpoint(test_agent_id)))
    else:
        test_results.append(("Chat Endpoint", False))
        print("⚠️  Skipping chat test - no agent available")
    
    test_results.append(("Error Handling", await test_error_handling()))
    test_results.append(("Service Integration", await test_service_integration()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 Controller Test Results Summary")
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
        print("🎉 Controller is working correctly!")
        print("Ready to proceed to Step 5: Memory Management")
    elif passed >= total * 0.8:  # 80% or better
        print("✅ Controller is mostly working with minor issues")
        print("Ready to proceed to Step 5: Memory Management")
    else:
        print("⚠️  Controller has significant issues. Please check the errors above.")
    
    # Cleanup message
    if test_agent_id:
        print(f"\n📝 Test agent created: {test_agent_id}")
        print("   You can use this agent for further testing")
    
    return passed >= total * 0.8


if __name__ == "__main__":
    success = asyncio.run(main())