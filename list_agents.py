"""
List all available agents and their IDs
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.memgpt.services import letta_client


async def list_all_agents():
    """List all agents with their details"""
    print("🔍 Fetching all agents from Letta...")
    print("=" * 60)
    
    try:
        # Connect to Letta and list agents
        agents = await letta_client.list_agents()
        
        if not agents:
            print("❌ No agents found!")
            print("\nYou can create one using:")
            print("  python create_modern_agent.py")
            return
        
        print(f"✅ Found {len(agents)} agent(s):\n")
        
        for i, agent in enumerate(agents, 1):
            print(f"{i}. Agent Name: {agent['name']}")
            print(f"   Agent ID: {agent['id']}")
            print(f"   Model: {agent.get('model', 'N/A')}")
            print(f"   Created: {agent.get('created_at', 'N/A')}")
            print("-" * 60)
        
        print(f"\n💡 Copy the Agent ID you want to use for testing")
        print(f"   Update TEST_AGENT_ID in test_health_optimization.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Letta server is running: letta server")
        print("  2. Environment variables are configured in .env")


if __name__ == "__main__":
    asyncio.run(list_all_agents())