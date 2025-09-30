#!/usr/bin/env python3
"""
Letta Diagnostic and Tool Schema Inspector

This script investigates:
1. Available tools and their schemas
2. Agent architecture details
3. Heartbeat system configuration
4. Model configurations
"""

import asyncio
import json
from letta_client import Letta
from pprint import pprint

async def main():
    print("🔍 Letta Diagnostic and Tool Schema Inspector")
    print("=" * 60)
    
    try:
        # Connect to Letta
        client = Letta(base_url="http://localhost:8283")
        print("✅ Connected to Letta server")
        
        # 1. Inspect available models
        print("\n📊 Available Models:")
        print("-" * 30)
        models = client.models.list()
        for i, model in enumerate(models[:5]):  # Show first 5
            print(f"  {i+1}. {model}")
        print(f"  ... and {len(models)-5} more models" if len(models) > 5 else "")
        
        # 2. List all agents
        print("\n👥 Current Agents:")
        print("-" * 30)
        agents = client.agents.list()
        for agent in agents:
            print(f"  - {agent.name} (ID: {agent.id})")
            print(f"    Model: {getattr(agent, 'model', 'unknown')}")
            print(f"    Created: {getattr(agent, 'created_at', 'unknown')}")
        
        if not agents:
            print("  No agents found.")
            return
        
        # 3. Inspect first agent in detail
        agent = agents[0]
        print(f"\n🔬 Detailed Agent Analysis: {agent.name}")
        print("-" * 50)
        
        # Check agent attributes
        print("Agent Attributes:")
        agent_attrs = [attr for attr in dir(agent) if not attr.startswith('_')]
        for attr in agent_attrs[:10]:  # Show first 10 attributes
            try:
                value = getattr(agent, attr)
                if not callable(value):
                    print(f"  {attr}: {str(value)[:100]}")
            except:
                print(f"  {attr}: <unable to access>")
        
        # 4. List available tools
        print(f"\n🛠️  Available Tools:")
        print("-" * 30)
        try:
            tools = client.tools.list()
            print(f"Found {len(tools)} tools:")
            
            for tool in tools:
                print(f"\n  📝 Tool: {tool.name}")
                print(f"     Description: {getattr(tool, 'description', 'N/A')[:100]}...")
                
                # Try to get tool schema
                if hasattr(tool, 'json_schema') and tool.json_schema:
                    schema = tool.json_schema
                    if isinstance(schema, dict) and 'parameters' in schema:
                        params = schema['parameters']
                        if 'properties' in params:
                            print(f"     Parameters: {list(params['properties'].keys())}")
                            
                            # Check specifically for request_heartbeat
                            if 'request_heartbeat' in params['properties']:
                                print(f"     ✅ Supports request_heartbeat!")
                            else:
                                print(f"     ❌ No request_heartbeat parameter")
                    else:
                        print(f"     Schema: {str(schema)[:200]}...")
                else:
                    print(f"     Schema: Not available")
                    
        except Exception as e:
            print(f"❌ Error listing tools: {e}")
        
        # 5. Test a simple message to understand tool usage
        print(f"\n💬 Testing Simple Message Exchange:")
        print("-" * 40)
        
        try:
            from letta_client import MessageCreate
            
            # Send a very simple message to see what tools are called
            response = client.agents.messages.create(
                agent_id=agent.id,
                messages=[
                    MessageCreate(
                        role="user",
                        content="Just say 'hello' - keep it simple."
                    )
                ]
            )
            
            print(f"Response received with {len(response.messages)} messages:")
            
            for i, msg in enumerate(response.messages):
                print(f"\n  Message {i+1}:")
                print(f"    Type: {getattr(msg, 'message_type', 'unknown')}")
                
                if hasattr(msg, 'content') and msg.content:
                    print(f"    Content: {msg.content[:150]}...")
                
                if hasattr(msg, 'tool_call') and msg.tool_call:
                    print(f"    Tool Called: {msg.tool_call.name}")
                    print(f"    Tool Args: {msg.tool_call.arguments}")
                    
                    # Parse arguments to see if request_heartbeat is used
                    try:
                        args = json.loads(msg.tool_call.arguments)
                        if 'request_heartbeat' in args:
                            print(f"    🎯 request_heartbeat: {args['request_heartbeat']}")
                        else:
                            print(f"    📝 Available args: {list(args.keys())}")
                    except:
                        print(f"    📝 Raw args: {msg.tool_call.arguments}")
                
                if hasattr(msg, 'reasoning') and msg.reasoning:
                    print(f"    Reasoning: {msg.reasoning[:100]}...")
                    
        except Exception as e:
            print(f"❌ Error testing message: {e}")
            print(f"   Error type: {type(e).__name__}")
        
        # 6. Check agent memory blocks
        print(f"\n🧠 Agent Memory Structure:")
        print("-" * 40)
        
        try:
            if hasattr(agent, 'memory_blocks') and agent.memory_blocks:
                print(f"Memory blocks found: {len(agent.memory_blocks)}")
                for block in agent.memory_blocks:
                    print(f"  - {getattr(block, 'label', 'unlabeled')}: {getattr(block, 'value', '')[:100]}...")
            else:
                print("No memory blocks visible in agent object")
                
        except Exception as e:
            print(f"Error accessing memory: {e}")
        
        # 7. Summary and recommendations
        print(f"\n📋 DIAGNOSTIC SUMMARY:")
        print("=" * 60)
        print(f"✅ Letta server: Connected")
        print(f"✅ Agents found: {len(agents)}")
        print(f"✅ Tools available: {len(tools) if 'tools' in locals() else 'Unknown'}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        print("1. Check tool schemas above for request_heartbeat support")
        print("2. If send_message doesn't support request_heartbeat, use other tools")
        print("3. Consider agent architecture for heartbeat compatibility")
        print("4. Verify system instructions match actual tool capabilities")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(main())