#!/usr/bin/env python3
"""
Delete an agent by name
"""

import asyncio
import httpx

async def delete_agent_by_name(agent_name: str):
    print(f"🗑️  Deleting Agent: {agent_name}")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            # Get list of agents
            response = await client.get("http://127.0.0.1:8000/agents")
            if response.status_code != 200:
                print(f"❌ Failed to list agents: {response.status_code}")
                return False
            
            agents_data = response.json()
            agents = agents_data.get("agents", [])
            
            # Find the agent
            target_agent = None
            for agent in agents:
                if agent["name"] == agent_name:
                    target_agent = agent
                    break
            
            if not target_agent:
                print(f"❌ Agent '{agent_name}' not found")
                print(f"Available agents: {[a['name'] for a in agents]}")
                return False
            
            agent_id = target_agent["id"]
            print(f"Found agent: {agent_name}")
            print(f"Agent ID: {agent_id}")
            
            # Delete using Letta client directly
            from letta_client import Letta
            letta = Letta(base_url="http://localhost:8283")
            letta.agents.delete(agent_id)
            
            print(f"✅ Successfully deleted agent: {agent_name}")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(delete_agent_by_name("modern_memgpt_agent"))