"""
Basic Chat Example - Asterix

Demonstrates:
- Creating a simple agent with memory blocks
- Having a conversation
- Viewing agent memory
- Agent self-editing memory during chat

This is the simplest possible usage of asterix-agent.
"""

from asterix import Agent, BlockConfig

def main():
    print("=" * 60)
    print("Asterix - Basic Chat Example")
    print("=" * 60)
    
    # Create an agent with two memory blocks
    print("\n1. Creating agent with memory blocks...")
    agent = Agent(
        agent_id="basic_chat_demo",
        blocks={
            # Task context - what the agent is currently working on
            "task": BlockConfig(
                size=1000,  # Max 1000 tokens before eviction
                priority=1,  # Lower priority - evicted first
                description="Current task and objectives"
            ),
            # User preferences - important information to remember
            "user": BlockConfig(
                size=500,   # Max 500 tokens
                priority=5,  # High priority - rarely evicted
                description="User preferences and information"
            )
        },
        model="openai/gpt-5-mini"
    )
    print(f"✓ Agent '{agent.id}' created")
    
    # View initial memory state (empty)
    print("\n2. Initial memory state:")
    memory = agent.get_memory()
    for block_name, content in memory.items():
        print(f"   [{block_name}]: {content if content else '(empty)'}")
    
    # Have a conversation
    print("\n3. Starting conversation...")
    print("-" * 60)
    
    # First message - agent learns about user
    user_msg = "Hi! My name is Aditya and I'm an AI Engineer."
    print(f"\nUser: {user_msg}")
    response = agent.chat(user_msg)
    print(f"Agent: {response}")
    
    # Second message - agent uses memory
    user_msg = "What's my name and what do I do?"
    print(f"\nUser: {user_msg}")
    response = agent.chat(user_msg)
    print(f"Agent: {response}")
    
    # View updated memory
    print("\n" + "-" * 60)
    print("\n4. Memory after conversation:")
    memory = agent.get_memory()
    for block_name, content in memory.items():
        print(f"\n   [{block_name}]:")
        if content:
            # Show first 150 chars
            display = content[:150] + "..." if len(content) > 150 else content
            print(f"   {display}")
        else:
            print("   (empty)")
    
    # Show conversation statistics
    print("\n5. Conversation statistics:")
    stats = agent.get_conversation_stats()
    print(f"   Messages: {stats['message_count']}")
    print(f"   Turns: {stats['turn_count']}")
    print(f"   Token usage: {stats['total_tokens']} tokens")
    
    print("\n" + "=" * 60)
    print("✅ Basic chat example complete!")
    print("\nKey takeaways:")
    print("• Agent automatically manages its own memory")
    print("• Memory persists across conversation turns")
    print("• Agent decides when to update memory blocks")
    print("=" * 60)


if __name__ == "__main__":
    main()