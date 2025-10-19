"""
Custom Tools Example - Asterix

Demonstrates:
- Registering custom tools with @agent.tool() decorator
- Tools with parameters and type hints
- Error handling in tools
- Agent using custom tools during conversation
- Multiple tools working together

This shows how to extend agents with custom capabilities.
"""

from asterix import Agent, BlockConfig
import json
import os
from datetime import datetime
from typing import Dict, List

def main():
    print("=" * 70)
    print("Asterix - Custom Tools Example")
    print("=" * 70)
    
    # Create agent
    print("\n1. Creating agent with custom tools...")
    agent = Agent(
        agent_id="custom_tools_demo",
        blocks={
            "task": BlockConfig(size=1500, priority=2),
            "notes": BlockConfig(size=1000, priority=3)
        },
        model="openai/gpt-4o-mini"
    )
    print(f"   ✓ Agent '{agent.id}' created")
    
    # ========================================================================
    # Tool 1: Simple tool with no parameters
    # ========================================================================
    
    @agent.tool(
        name="get_current_time",
        description="Get the current date and time"
    )
    def get_current_time() -> str:
        """Returns the current date and time in human-readable format."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S %A")
    
    print("   ✓ Registered tool: get_current_time")
    
    # ========================================================================
    # Tool 2: Tool with parameters and type hints
    # ========================================================================
    
    @agent.tool(
        name="calculate",
        description="Perform basic arithmetic operations (add, subtract, multiply, divide)"
    )
    def calculate(operation: str, a: float, b: float) -> str:
        """
        Perform a calculation between two numbers.
        
        Args:
            operation: Operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            Result of the calculation as a string
        """
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return "Error: Cannot divide by zero"
                result = a / b
            else:
                return f"Error: Unknown operation '{operation}'"
            
            return f"{a} {operation} {b} = {result}"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    print("   ✓ Registered tool: calculate")
    
    # ========================================================================
    # Tool 3: Tool that works with data structures
    # ========================================================================
    
    @agent.tool(
        name="analyze_list",
        description="Analyze a list of numbers and return statistics"
    )
    def analyze_list(numbers: List[float]) -> str:
        """
        Analyze a list of numbers and return basic statistics.
        
        Args:
            numbers: List of numbers to analyze
            
        Returns:
            JSON string with statistics
        """
        if not numbers:
            return json.dumps({"error": "Empty list provided"})
        
        stats = {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers)
        }
        
        return json.dumps(stats, indent=2)
    
    print("   ✓ Registered tool: analyze_list")
    
    # ========================================================================
    # Tool 4: Tool with file system interaction (with error handling)
    # ========================================================================
    
    @agent.tool(
        name="list_directory",
        description="List files and directories in a given path"
    )
    def list_directory(path: str = ".") -> str:
        """
        List contents of a directory.
        
        Args:
            path: Directory path (defaults to current directory)
            
        Returns:
            Formatted list of directory contents
        """
        try:
            if not os.path.exists(path):
                return f"Error: Path '{path}' does not exist"
            
            if not os.path.isdir(path):
                return f"Error: '{path}' is not a directory"
            
            items = os.listdir(path)
            
            if not items:
                return f"Directory '{path}' is empty"
            
            # Separate files and directories
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            
            result = f"Contents of '{path}':\n\n"
            
            if dirs:
                result += f"Directories ({len(dirs)}):\n"
                for d in sorted(dirs):
                    result += f"  📁 {d}/\n"
                result += "\n"
            
            if files:
                result += f"Files ({len(files)}):\n"
                for f in sorted(files):
                    result += f"  📄 {f}\n"
            
            return result
        
        except PermissionError:
            return f"Error: Permission denied for '{path}'"
        except Exception as e:
            return f"Error: {str(e)}"
    
    print("   ✓ Registered tool: list_directory")
    
    # Show registered tools
    print("\n2. Available custom tools:")
    all_tools = agent._tool_registry.list_tools()
    custom_tools = [t for t in all_tools if not t['name'].startswith("core_memory_") 
                    and not t['name'].startswith("archival_memory_") 
                    and t['name'] != "conversation_search"]
    
    for tool_info in custom_tools:
        print(f"   • {tool_info['name']}: {tool_info['description']}")
    
    # ========================================================================
    # Test tools in conversation
    # ========================================================================
    
    print("\n3. Testing tools in conversation...")
    print("=" * 70)
    
    # Give initial instruction
    print("\n   Instructing agent to use tools...")
    agent.chat("You have custom tools available. Always use them when relevant instead of answering directly.")
    
    test_queries = [
    "Use the get_current_time tool to tell me what time it is right now",
    "Use the calculate tool to multiply 25 by 4",
    "Use the analyze_list tool to analyze these numbers: [10, 25, 30, 15, 40, 20]",
    "Use the list_directory tool to list files in the current directory"
    ]
    
    for query in test_queries:
        print(f"\n💬 User: {query}")
        print("-" * 70)
        response = agent.chat(query)
        print(f"🤖 Agent: {response}\n")
    
    # Show memory after tool usage
    print("=" * 70)
    print("\n4. Agent memory after using tools:")
    memory = agent.get_memory()
    for block_name, content in memory.items():
        if content:
            print(f"\n   [{block_name}]:")
            display = content[:200] + "..." if len(content) > 200 else content
            print(f"   {display}")
    
    # Show conversation statistics
    print("\n5. Session statistics:")
    stats = agent.get_conversation_stats()
    print(f"   • Messages: {stats['message_count']}")
    print(f"   • Turns: {stats['turn_count']}")
    
    print("\n" + "=" * 70)
    print("✅ Custom tools example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Use @agent.tool() decorator to register custom functions")
    print("  • Tools can have parameters with type hints")
    print("  • Error handling in tools prevents crashes")
    print("  • Agent automatically calls tools when needed")
    print("  • Multiple tools can work together in conversations")
    print("=" * 70)


if __name__ == "__main__":
    main()