"""
Persistent Agent Example - Asterix

Demonstrates:
- Creating an agent and having a conversation
- Saving agent state to disk
- Loading agent state in a new session
- Verifying memory persistence across sessions

This shows how agents can maintain context between program runs.
"""

from asterix import Agent, BlockConfig
import os
import time

def session_1_save_agent():
    """
    Session 1: Create agent, chat, and save state
    """
    print("=" * 70)
    print("SESSION 1: Creating Agent and Saving State")
    print("=" * 70)
    
    # Create agent with memory blocks
    print("\n1. Creating agent...")
    agent = Agent(
        agent_id="persistent_demo",
        blocks={
            "user_profile": BlockConfig(
                size=800,
                priority=5,  # High priority - important to keep
                description="User information and preferences"
            ),
            "conversation_context": BlockConfig(
                size=1200,
                priority=3,  # Medium priority
                description="Recent conversation topics"
            )
        },
        model="openai/gpt-4o-mini"
    )
    print(f"   ✓ Agent '{agent.id}' created")
    
    # Have a conversation to build up memory
    print("\n2. Having a conversation...")
    print("-" * 70)
    
    messages = [
        "Hi! I'm Aditya, an AI Engineer working on AI Agents.",
        "I'm building a CLI tools for day to day use. I prefer Python over JavaScript.",
        "Can you remember these preferences for our future conversations?"
    ]
    
    for msg in messages:
        print(f"\n   User: {msg}")
        response = agent.chat(msg)
        print(f"   Agent: {response[:100]}..." if len(response) > 100 else f"   Agent: {response}")
        time.sleep(0.5)  # Brief pause between messages
    
    # Show memory state before saving
    print("\n" + "-" * 70)
    print("\n3. Agent memory before saving:")
    memory = agent.get_memory()
    for block_name, content in memory.items():
        print(f"\n   [{block_name}]:")
        if content:
            # Show first 200 chars
            display = content[:200] + "..." if len(content) > 200 else content
            print(f"   {display}")
        else:
            print("   (empty)")
    
    # Save state
    print("\n4. Saving agent state...")
    agent.save_state()
    
    # Get save location
    state_file = f"./agent_states/{agent.id}_state.json"
    file_exists = os.path.exists(state_file)
    file_size = os.path.getsize(state_file) if file_exists else 0
    
    print(f"   ✓ State saved to: {state_file}")
    print(f"   ✓ File size: {file_size:,} bytes")
    
    # Show stats
    stats = agent.get_conversation_stats()
    print(f"\n5. Session statistics:")
    print(f"   • Messages: {stats['message_count']}")
    print(f"   • Turns: {stats['turn_count']}")
    print(f"   • Total tokens: {stats['total_tokens']}")
    
    print("\n" + "=" * 70)
    print("✅ Session 1 complete - Agent state saved!")
    print("=" * 70)
    
    return agent.id


def session_2_load_agent(agent_id: str):
    """
    Session 2: Load saved agent and verify persistence
    """
    print("\n\n")
    print("=" * 70)
    print("SESSION 2: Loading Agent from Saved State")
    print("=" * 70)
    
    # Load agent from disk
    print("\n1. Loading agent from saved state...")
    agent = Agent.load_state(agent_id)
    print(f"   ✓ Agent '{agent.id}' loaded successfully")
    
    # Show restored memory
    print("\n2. Agent memory after loading:")
    memory = agent.get_memory()
    for block_name, content in memory.items():
        print(f"\n   [{block_name}]:")
        if content:
            display = content[:200] + "..." if len(content) > 200 else content
            print(f"   {display}")
        else:
            print("   (empty)")
    
    # Verify agent remembers
    print("\n3. Testing memory persistence...")
    print("-" * 70)
    
    test_questions = [
        "What's my name?",
        "What programming language do I prefer?",
        "What project am I working on?"
    ]
    
    for question in test_questions:
        print(f"\n   User: {question}")
        response = agent.chat(question)
        print(f"   Agent: {response}")
        time.sleep(0.5)
    
    print("\n" + "-" * 70)
    
    # Show updated stats
    stats = agent.get_conversation_stats()
    print(f"\n4. Updated session statistics:")
    print(f"   • Messages: {stats['message_count']}")
    print(f"   • Turns: {stats['turn_count']}")
    print(f"   • Total tokens: {stats['total_tokens']}")
    
    print("\n" + "=" * 70)
    print("✅ Session 2 complete - Memory successfully persisted!")
    print("=" * 70)


def main():
    """
    Run both sessions to demonstrate persistence
    """
    print("\n🔹 Asterix - Persistent Agent Example\n")
    
    # Session 1: Create and save
    agent_id = session_1_save_agent()
    
    # Session 2: Load and verify
    session_2_load_agent(agent_id)
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Agents can save their complete state to disk")
    print("  • State includes conversation history and memory blocks")
    print("  • Loaded agents remember everything from previous sessions")
    print("  • Perfect for long-running assistants and resumable workflows")
    print("\nState file location: ./agent_states/persistent_demo_state.json")
    print("=" * 70)


if __name__ == "__main__":
    main()