#!/usr/bin/env python3
"""
Direct Letta API Test
"""

import asyncio
from letta_client import Letta, MessageCreate

async def test_direct_letta():
    print("🔍 Testing Direct Letta API Call")
    print("=" * 50)
    
    try:
        # Connect to Letta
        client = Letta(base_url="http://localhost:8283")
        print("✅ Connected to Letta")
        
        # List agents
        agents = client.agents.list()
        if not agents:
            print("❌ No agents found")
            return
        
        agent = agents[0]
        agent_id = agent.id
        print(f"✅ Using agent: {agent.name} (ID: {agent_id})")
        
        # Try to send a simple message
        print("📤 Sending test message...")
        
        message_obj = MessageCreate(
            role="user",
            content="Hello, just say 'hi' back."
        )
        
        response = client.agents.messages.create(
            agent_id=agent_id,
            messages=[message_obj]
        )
        
        print("✅ Message sent successfully!")
        print(f"   Response type: {type(response)}")
        print(f"   Messages received: {len(response.messages) if hasattr(response, 'messages') else 0}")
        
        # Show response messages
        if hasattr(response, 'messages'):
            for i, msg in enumerate(response.messages):
                print(f"   Message {i+1}: {getattr(msg, 'message_type', 'unknown')} - {getattr(msg, 'content', '')[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_letta())