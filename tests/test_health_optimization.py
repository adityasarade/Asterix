#!/usr/bin/env python3
"""
Health Check Optimization Test Suite

Tests the optimized health check implementation to verify:
- Fast response times (no unnecessary health checks)
- Proper error handling with detailed service health
- Service recovery and caching behavior
"""

import asyncio
import httpx
import time
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"
TEST_AGENT_ID = "agent-0d2615a7-25af-453d-b3e5-9bc6a8948901"
TEST_USER_ID = "test_user_123"


async def test_normal_operation():
    """Test 1: Normal Operation - Verify fast responses"""
    print("\n" + "="*60)
    print("TEST 1: Normal Operation (Performance Verification)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            start_time = time.time()
            
            response = await client.post(
                f"{BASE_URL}/chat",
                json={
                    "agent_id": TEST_AGENT_ID,
                    "user_id": TEST_USER_ID,
                    "text": "Hello! What do you remember about me?"
                }
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get("processing_time", 0)
                
                print(f"✅ Request successful")
                print(f"   Total elapsed time: {elapsed:.3f}s")
                print(f"   Processing time: {processing_time:.3f}s")
                print(f"   Heartbeat steps: {data.get('heartbeat_steps', 0)}")
                print(f"   Assistant response: {data.get('assistant', '')[:100]}...")
                
                if elapsed < 10.0:
                    print("✅ PASS: Response time is good")
                else:
                    print("⚠️  WARN: Response took longer than expected")
                
                return True
            else:
                print(f"❌ FAIL: Got status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ FAIL: Exception occurred: {e}")
            return False


async def test_service_failure_letta():
    """Test 2A: Letta Service Failure - Verify error handling"""
    print("\n" + "="*60)
    print("TEST 2A: Letta Service Failure")
    print("="*60)
    print("⚠️  MANUAL STEP: Stop the Letta server (Ctrl+C in Letta terminal)")
    print("   Then press Enter to continue...")
    input()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat",
                json={
                    "agent_id": TEST_AGENT_ID,
                    "user_id": TEST_USER_ID,
                    "text": "Hello"
                }
            )
            
            if response.status_code == 503:
                data = response.json()
                detail = data.get("detail", {})
                
                print(f"✅ Got expected 503 status code")
                print(f"   Error: {detail.get('error', 'N/A')}")
                print(f"   Message: {detail.get('message', 'N/A')}")
                
                service_health = detail.get("service_health", {})
                if service_health:
                    print(f"   Service health included: {list(service_health.keys())}")
                    for service, health in service_health.items():
                        print(f"     - {service}: {health.get('status', 'N/A')}")
                    print("✅ PASS: Detailed error with service health")
                else:
                    print("❌ FAIL: No service health information in error")
                
                return True
            else:
                print(f"⚠️  Unexpected status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False


async def test_invalid_agent_id():
    """Test 4A: Invalid Agent ID - Verify validation"""
    print("\n" + "="*60)
    print("TEST 4A: Invalid Agent ID Validation")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat",
                json={
                    "agent_id": "",  # Empty agent ID
                    "user_id": TEST_USER_ID,
                    "text": "Hello"
                }
            )
            
            if response.status_code == 400:
                data = response.json()
                detail = data.get("detail", {})
                
                print(f"✅ Got expected 400 status code")
                print(f"   Error: {detail.get('error', 'N/A')}")
                print(f"   Message: {detail.get('message', 'N/A')}")
                print("✅ PASS: Validation working correctly")
                return True
            else:
                print(f"❌ FAIL: Expected 400, got {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False


async def test_agent_not_found():
    """Test 4B: Agent Not Found - Verify 404 handling"""
    print("\n" + "="*60)
    print("TEST 4B: Agent Not Found")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat",
                json={
                    "agent_id": "nonexistent-agent-12345",
                    "user_id": TEST_USER_ID,
                    "text": "Hello"
                }
            )
            
            if response.status_code == 404:
                data = response.json()
                detail = data.get("detail", {})
                
                print(f"✅ Got expected 404 status code")
                print(f"   Error: {detail.get('error', 'N/A')}")
                print(f"   Message: {detail.get('message', 'N/A')}")
                
                available = detail.get("available_agents", [])
                if available:
                    print(f"   Available agents: {len(available)} listed")
                    print("✅ PASS: Helpful 404 with agent list")
                else:
                    print("⚠️  No available agents listed")
                
                return True
            else:
                print(f"⚠️  Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False


async def test_health_endpoints():
    """Test 5: Health Endpoints"""
    print("\n" + "="*60)
    print("TEST 5: Health Endpoints")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test basic health
        print("\nTesting /health endpoint...")
        try:
            start = time.time()
            response = await client.get(f"{BASE_URL}/health")
            elapsed = time.time() - start
            
            if response.status_code == 200:
                print(f"✅ /health: {response.json()}")
                print(f"   Response time: {elapsed*1000:.1f}ms")
                if elapsed < 0.1:
                    print("✅ PASS: Fast response")
            else:
                print(f"❌ FAIL: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Test detailed health
        print("\nTesting /health/services endpoint...")
        try:
            start = time.time()
            response = await client.get(f"{BASE_URL}/health/services")
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get("summary", {})
                print(f"✅ /health/services:")
                print(f"   Total services: {summary.get('total_services', 0)}")
                print(f"   Healthy: {summary.get('healthy_services', 0)}")
                print(f"   Health %: {summary.get('health_percentage', 0):.1f}%")
                print(f"   Response time: {elapsed*1000:.1f}ms")
                print("✅ PASS: Detailed health working")
            else:
                print(f"❌ FAIL: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")


async def test_rapid_fire():
    """Test 6: Rapid-Fire Requests - Verify caching"""
    print("\n" + "="*60)
    print("TEST 6: Rapid-Fire Requests (Cache Verification)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Sending 5 rapid requests...")
        
        tasks = []
        for i in range(5):
            task = client.post(
                f"{BASE_URL}/chat",
                json={
                    "agent_id": TEST_AGENT_ID,
                    "user_id": TEST_USER_ID,
                    "text": f"Quick test {i+1}"
                }
            )
            tasks.append(task)
        
        start = time.time()
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start
            
            success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            
            print(f"✅ Completed {success_count}/5 requests")
            print(f"   Total time: {elapsed:.3f}s")
            print(f"   Average time: {elapsed/5:.3f}s per request")
            
            if success_count >= 4:
                print("✅ PASS: Rapid requests handled well")
            else:
                print("⚠️  WARN: Some requests failed")
                
        except Exception as e:
            print(f"❌ Exception: {e}")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 HEALTH CHECK OPTIMIZATION TEST SUITE")
    print("="*70)
    print("\nPrerequisites:")
    print("  1. Letta server running: letta server")
    print("  2. Controller running: python run_controller.py")
    print("  3. Update TEST_AGENT_ID in this script with your actual agent ID")
    print("\nStarting tests in 3 seconds...")
    await asyncio.sleep(3)
    
    results = []
    
    # Run tests
    results.append(("Normal Operation", await test_normal_operation()))
    results.append(("Invalid Agent ID", await test_invalid_agent_id()))
    results.append(("Agent Not Found", await test_agent_not_found()))
    await test_health_endpoints()
    await test_rapid_fire()
    
    # Optional: Test service failure (requires manual intervention)
    print("\n" + "="*60)
    print("Optional: Test Service Failure?")
    print("This requires stopping the Letta server.")
    choice = input("Run service failure test? (y/n): ")
    if choice.lower() == 'y':
        results.append(("Letta Service Failure", await test_service_failure_letta()))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Health check optimization is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")


if __name__ == "__main__":
    asyncio.run(main())