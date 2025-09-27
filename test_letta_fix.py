#!/usr/bin/env python3
"""
Quick test for Letta client fix
"""

import asyncio
from src.memgpt.services import letta_client

async def test_letta_fix():
    print("🔧 Testing Letta Client Fix...")
    
    try:
        # Test connection
        print("Testing connection...")
        connected = await letta_client.ensure_connected()
        if not connected:
            print("❌ Connection failed")
            return False
        print("✅ Connection successful")
        
        # Test getting models (this was the failing part)
        print("Testing model listing...")
        models = await letta_client.get_available_models(force_refresh=True)
        print(f"✅ Successfully retrieved {len(models)} models")
        
        # Show first few models
        print("Sample models:")
        for i, model in enumerate(models[:3]):
            print(f"  {i+1}. ID: {model['id']}, Name: {model['name']}, Provider: {model['provider']}")
        
        # Test other operations
        print("Testing agent listing...")
        agents = await letta_client.list_agents()
        print(f"✅ Found {len(agents)} existing agents")
        
        print("✅ All Letta client operations working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_letta_fix())
    if success:
        print("\n🎉 Letta client fix successful! Ready to re-run full test suite.")
    else:
        print("\n⚠️  Fix needs more work. Check the error above.")