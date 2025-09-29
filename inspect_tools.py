#!/usr/bin/env python3
"""
Inspect available tools in Letta
"""

from letta_client import Letta

def inspect_tools():
    print("🔍 Inspecting Letta Tools")
    print("=" * 50)
    
    try:
        client = Letta(base_url="http://localhost:8283")
        print("✅ Connected to Letta\n")
        
        # List all tools
        tools = client.tools.list()
        print(f"Found {len(tools)} tools:\n")
        
        for i, tool in enumerate(tools, 1):
            print(f"{i}. Tool:")
            print(f"   Name: {getattr(tool, 'name', 'N/A')}")
            print(f"   ID: {getattr(tool, 'id', 'N/A')}")
            print(f"   Description: {getattr(tool, 'description', 'N/A')[:80]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_tools()