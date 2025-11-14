"""
Persistent Agent with SQLite Backend - Asterix

Demonstrates:
- Using SQLite backend instead of JSON for state persistence
- Benefits: better for managing many agents, querying, atomic updates
- Creating and saving agent state to SQLite database
- Loading agent state from SQLite database
- Querying agent metadata without loading full state

This example shows the same persistent agent functionality as persistent_agent.py,
but using SQLite for improved performance and querying capabilities.
"""

from asterix import Agent, BlockConfig, StorageConfig
import os
import time

def session_1_save_agent():
    """
    Session 1: Create agent with SQLite backend, chat, and save state
    """
    print("=" * 70)
    print("SESSION 1: Creating Agent with SQLite Backend")
    print("=" * 70)

    # Create agent with SQLite backend
    print("\n1. Creating agent with SQLite backend...")
    agent = Agent(
        agent_id="persistent_demo_sqlite",
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
        model="openai/gpt-4o-mini",
        storage=StorageConfig(
            state_backend="sqlite",  # Use SQLite instead of JSON
            state_db="./agent_states/agents.db"  # Database file path
        )
    )
    print(f"   ✓ Agent '{agent.id}' created with SQLite backend")

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

    # Save state to SQLite
    print("\n4. Saving agent state to SQLite...")
    agent.save_state()

    # Get save location
    db_file = "./agent_states/agents.db"
    file_exists = os.path.exists(db_file)
    file_size = os.path.getsize(db_file) if file_exists else 0

    print(f"   ✓ State saved to SQLite database: {db_file}")
    print(f"   ✓ Database size: {file_size:,} bytes")

    # Show stats
    stats = agent.get_conversation_stats()
    print(f"\n5. Session statistics:")
    print(f"   • Messages: {stats['message_count']}")
    print(f"   • Turns: {stats['turn_count']}")
    print(f"   • Total tokens: {stats['total_tokens']}")

    print("\n" + "=" * 70)
    print("✅ Session 1 complete - Agent state saved to SQLite!")
    print("=" * 70)

    return agent.id


def session_2_load_agent(agent_id: str):
    """
    Session 2: Load saved agent from SQLite and verify persistence
    """
    print("\n\n")
    print("=" * 70)
    print("SESSION 2: Loading Agent from SQLite Database")
    print("=" * 70)

    # Load agent from SQLite database
    print("\n1. Loading agent from SQLite database...")
    agent = Agent.load_state(
        agent_id,
        state_backend="sqlite",
        state_db="./agent_states/agents.db"
    )
    print(f"   ✓ Agent '{agent.id}' loaded successfully from SQLite")

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


def demo_sqlite_querying():
    """
    Bonus: Demonstrate SQLite backend querying capabilities
    """
    print("\n\n")
    print("=" * 70)
    print("BONUS: SQLite Backend Querying")
    print("=" * 70)

    from asterix.storage import SQLiteStateBackend

    backend = SQLiteStateBackend("./agent_states/agents.db")

    # List all agents
    print("\n1. Listing all agents in database:")
    agents = backend.list_agents()
    for agent_id in agents:
        print(f"   • {agent_id}")

    # Get agent metadata without loading full state
    print("\n2. Getting agent metadata (without loading full state):")
    if agents:
        info = backend.get_agent_info(agents[0])
        if info:
            print(f"   Agent ID: {info['agent_id']}")
            print(f"   Model: {info['model']}")
            print(f"   Block count: {info['block_count']}")
            print(f"   Message count: {info['message_count']}")
            print(f"   Created: {info['created_at']}")
            print(f"   Last updated: {info['last_updated']}")

    print("\n" + "=" * 70)


def main():
    """
    Run both sessions to demonstrate SQLite persistence
    """
    print("\n🔹 Asterix - Persistent Agent with SQLite Backend\n")

    # Session 1: Create and save
    agent_id = session_1_save_agent()

    # Session 2: Load and verify
    session_2_load_agent(agent_id)

    # Bonus: Demonstrate querying
    demo_sqlite_querying()

    # Final summary
    print("\n\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • SQLite backend is better for managing multiple agents")
    print("  • Provides atomic updates and better query performance")
    print("  • Can query agent metadata without loading full state")
    print("  • All agents stored in single database file")
    print("\nSQLite database location: ./agent_states/agents.db")
    print("\nChoosing between JSON and SQLite:")
    print("  • JSON: Simple, human-readable, good for 1-10 agents")
    print("  • SQLite: Better for 10+ agents, querying, production use")
    print("=" * 70)


if __name__ == "__main__":
    main()
