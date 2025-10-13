from asterix import Agent, BlockConfig

# Create agent with small block limit
agent = Agent(
    agent_id="eviction_test",
    blocks={"test": BlockConfig(size=100, priority=1)}
)

# Add content that exceeds limit
long_text = "This is a test sentence. " * 50  # ~250 tokens
agent.update_memory("test", long_text)

# Check the result
print(f"Block tokens: {agent.blocks['test'].tokens}")  # Should be ~100
print(f"Block content: {agent.blocks['test'].content[:200]}...")  # Should be summarized