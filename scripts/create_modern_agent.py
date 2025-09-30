#!/usr/bin/env python3
"""
Create a new agent with modern system instructions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx

async def create_modern_agent():
    print("🚀 Creating Modern MemGPT Agent")
    print("=" * 40)
    
    try:
        async with httpx.AsyncClient() as client:
            # Create new agent
            create_request = {
                "name": "modern_memgpt_agent",
                "user_id": "test_user_123"
            }
            
            response = await client.post(
                "http://127.0.0.1:8000/agents",
                json=create_request
            )
            
            if response.status_code == 200:
                agent_data = response.json()
                print(f"✅ Created new agent: {agent_data['name']}")
                print(f"   Agent ID: {agent_data['agent_id']}")
                print(f"   Model: {agent_data['model']}")
                print("   This agent uses the modern system instructions!")
                return agent_data['agent_id']
            else:
                print(f"❌ Failed to create agent: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(create_modern_agent())