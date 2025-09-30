#!/usr/bin/env python3
"""
Inspect agent configuration to see actual system instructions
"""

from letta_client import Letta

def inspect_agent():
    print("🔍 Inspecting Agent Configuration")
    print("=" * 60)
    
    try:
        client = Letta(base_url="http://localhost:8283")
        print("✅ Connected to Letta\n")
        
        # List agents
        agents = client.agents.list()
        
        if not agents:
            print("No agents found")
            return
        
        # Find the modern agent
        target_agent = None
        for agent in agents:
            if "modern" in agent.name.lower():
                target_agent = agent
                break
        
        if not target_agent:
            target_agent = agents[-1]  # Use most recent
        
        print(f"Inspecting Agent: {target_agent.name}")
        print(f"Agent ID: {target_agent.id}\n")
        
        # Show system instructions
        system_instructions = getattr(target_agent, 'system', None)
        
        if system_instructions:
            print("System Instructions:")
            print("-" * 60)
            print(system_instructions[:500])  # First 500 chars
            print("...")
            print("-" * 60)
            
            # Check for problematic content
            if "request_heartbeat" in system_instructions:
                print("\n❌ FOUND 'request_heartbeat' in system instructions!")
                print("   This is the problem - agent has old instructions")
            else:
                print("\n✅ No 'request_heartbeat' found in system instructions")
        else:
            print("No system instructions found")
        
        # Show tools
        if hasattr(target_agent, 'tools'):
            print(f"\nAttached Tools: {len(target_agent.tools)}")
            for tool in target_agent.tools[:5]:
                print(f"  - {tool}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_agent()