#!/usr/bin/env python3
"""
CLI Agent Example
=================

A full-featured CLI agent with file operations and code review capabilities.

This example demonstrates:
1. File system operations (read, write, list, search)
2. Code review and analysis
3. Multi-tool coordination
4. Real-world CLI assistant use case
5. Memory management during complex tasks

Run:
    python examples/cli_agent.py
"""

import os
from pathlib import Path
from asterix import Agent, BlockConfig

# ============================================================================
# Setup
# ============================================================================

print("=" * 70)
print("CLI AGENT WITH FILE OPERATIONS")
print("=" * 70)
print()

# Create agent with appropriate memory blocks for file operations
agent = Agent(
    agent_id="cli_assistant",
    blocks={
        "current_task": BlockConfig(
            size=2000,
            priority=1,
            description="Current task and context"
        ),
        "file_context": BlockConfig(
            size=3000,
            priority=2,
            description="Content of files being worked with"
        ),
        "findings": BlockConfig(
            size=1500,
            priority=3,
            description="Code review findings and notes"
        )
    },
    model="openai/gpt-4o-mini",
    temperature=0.1,  # Low temperature for consistent file operations
    max_heartbeat_steps=15  # Allow more steps for complex tasks
)

print(f"✓ Created CLI agent: {agent.id}")
print(f"  Model: {agent.config.model}")
print(f"  Memory blocks: {list(agent.blocks.keys())}")
print()

# ============================================================================
# Register File Operation Tools
# ============================================================================

print("Registering file operation tools...")
print("-" * 70)

@agent.tool(name="read_file", description="Read contents of a file")
def read_file(filepath: str) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        File contents or error message
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return f"Error: File '{filepath}' does not exist"
        
        if not path.is_file():
            return f"Error: '{filepath}' is not a file"
        
        # Read file with UTF-8 encoding
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Truncate if too large (>10KB)
        if len(content) > 10000:
            return (
                f"File '{filepath}' ({len(content)} chars):\n\n"
                f"{content[:10000]}\n\n"
                f"... [truncated, file is too large]"
            )
        
        return f"File '{filepath}' ({len(content)} chars):\n\n{content}"
    
    except UnicodeDecodeError:
        return f"Error: File '{filepath}' is not a text file (binary content)"
    except PermissionError:
        return f"Error: Permission denied for '{filepath}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"

print("  ✓ Registered: read_file")


@agent.tool(name="write_file", description="Write content to a file")
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file (creates or overwrites).
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message or error
    """
    try:
        path = Path(filepath)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✓ Successfully wrote {len(content)} characters to '{filepath}'"
    
    except PermissionError:
        return f"Error: Permission denied for '{filepath}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"

print("  ✓ Registered: write_file")


@agent.tool(name="list_directory", description="List contents of a directory")
def list_directory(path: str = ".") -> str:
    """
    List files and directories in a path.
    
    Args:
        path: Directory path (defaults to current directory)
        
    Returns:
        Formatted directory listing
    """
    try:
        dir_path = Path(path)
        
        if not dir_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        if not dir_path.is_dir():
            return f"Error: '{path}' is not a directory"
        
        items = list(dir_path.iterdir())
        
        if not items:
            return f"Directory '{path}' is empty"
        
        # Separate files and directories
        files = [f for f in items if f.is_file()]
        dirs = [d for d in items if d.is_dir()]
        
        result = f"Contents of '{path}':\n\n"
        
        if dirs:
            result += f"📁 Directories ({len(dirs)}):\n"
            for d in sorted(dirs):
                result += f"  {d.name}/\n"
            result += "\n"
        
        if files:
            result += f"📄 Files ({len(files)}):\n"
            for f in sorted(files):
                # Get file size
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                result += f"  {f.name} ({size_str})\n"
        
        return result
    
    except PermissionError:
        return f"Error: Permission denied for '{path}'"
    except Exception as e:
        return f"Error: {str(e)}"

print("  ✓ Registered: list_directory")


@agent.tool(name="search_files", description="Search for files matching a pattern")
def search_files(pattern: str, directory: str = ".") -> str:
    """
    Search for files matching a pattern in a directory.
    
    Args:
        pattern: Glob pattern to search for (e.g., "*.py", "test_*.txt")
        directory: Directory to search in (defaults to current)
        
    Returns:
        List of matching files
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        # Search for matching files
        matches = list(dir_path.glob(pattern))
        
        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'"
        
        result = f"Found {len(matches)} file(s) matching '{pattern}':\n\n"
        
        for match in sorted(matches):
            if match.is_file():
                size = match.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                result += f"  📄 {match.name} ({size_str})\n"
        
        return result
    
    except Exception as e:
        return f"Error searching files: {str(e)}"

print("  ✓ Registered: search_files")


@agent.tool(name="get_file_info", description="Get detailed information about a file")
def get_file_info(filepath: str) -> str:
    """
    Get detailed information about a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File metadata and information
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return f"Error: File '{filepath}' does not exist"
        
        stat = path.stat()
        
        result = f"File information for '{filepath}':\n\n"
        result += f"  Type: {'File' if path.is_file() else 'Directory'}\n"
        result += f"  Size: {stat.st_size:,} bytes ({stat.st_size/1024:.2f} KB)\n"
        result += f"  Created: {stat.st_ctime}\n"
        result += f"  Modified: {stat.st_mtime}\n"
        result += f"  Accessed: {stat.st_atime}\n"
        
        if path.is_file():
            # Try to count lines for text files
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                result += f"  Lines: {lines:,}\n"
            except:
                pass
        
        return result
    
    except Exception as e:
        return f"Error getting file info: {str(e)}"

print("  ✓ Registered: get_file_info")

print()
print(f"Total tools registered: {len(agent.get_all_tools())}")
print()

# ============================================================================
# Demonstration 1: Simple File Operations
# ============================================================================

print("=" * 70)
print("DEMONSTRATION 1: Simple File Operations")
print("=" * 70)
print()

# Create a test file
test_content = """# Test Python Script
def hello():
    print("Hello from CLI agent!")
    
def add(a, b):
    return a + b

if __name__ == "__main__":
    hello()
    result = add(5, 3)
    print(f"5 + 3 = {result}")
"""

print("Creating a test file...")
agent.chat("Use write_file to create 'test_script.py' with some sample Python code")

print("\n" + "-" * 70)
print("User: List the current directory")
print("-" * 70)
response = agent.chat("Use list_directory to show me what's in the current directory")
print(f"Agent: {response}")

print("\n" + "-" * 70)
print("User: Read the test file")
print("-" * 70)
response = agent.chat("Use read_file to read 'test_script.py' and tell me what it does")
print(f"Agent: {response}")

# ============================================================================
# Demonstration 2: Code Review
# ============================================================================

print("\n" + "=" * 70)
print("DEMONSTRATION 2: Code Review")
print("=" * 70)
print()

print("User: Review this code for issues")
print("-" * 70)
response = agent.chat(
    "Review the code in 'test_script.py'. Look for potential issues, "
    "suggest improvements, and check for best practices. "
    "Store your findings in the 'findings' memory block."
)
print(f"Agent: {response}")

# ============================================================================
# Demonstration 3: Multi-Step Task
# ============================================================================

print("\n" + "=" * 70)
print("DEMONSTRATION 3: Multi-Step File Task")
print("=" * 70)
print()

print("User: Find all Python files and create a summary")
print("-" * 70)
response = agent.chat(
    "Use search_files to find all .py files in the current directory, "
    "then read each one and create a summary document called 'code_summary.txt' "
    "with information about each file."
)
print(f"Agent: {response}")

# ============================================================================
# View Memory State
# ============================================================================

print("\n" + "=" * 70)
print("AGENT MEMORY STATE")
print("=" * 70)
print()

memory = agent.get_memory()
for block_name, content in memory.items():
    if content and content.strip():
        print(f"[{block_name}]:")
        # Show first 300 chars
        display = content[:300] + "..." if len(content) > 300 else content
        print(f"{display}")
        print()

# ============================================================================
# Statistics
# ============================================================================

print("=" * 70)
print("SESSION STATISTICS")
print("=" * 70)
print()

stats = agent.get_conversation_stats()
print(f"Messages: {stats['message_count']}")
print(f"Turns: {stats['turn_count']}")
print(f"Tokens used: {stats['total_tokens']:,}")

# Clean up test files
print("\nCleaning up test files...")
try:
    if Path("test_script.py").exists():
        Path("test_script.py").unlink()
        print("  ✓ Removed test_script.py")
    if Path("code_summary.txt").exists():
        Path("code_summary.txt").unlink()
        print("  ✓ Removed code_summary.txt")
except:
    pass

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("✅ CLI AGENT EXAMPLE COMPLETE!")
print("=" * 70)

print("""
Key Takeaways:
  • CLI agents can perform real file operations safely
  • Multiple tools can coordinate for complex tasks
  • Memory blocks store context during multi-step operations
  • Low temperature (0.1) gives consistent, reliable behavior
  • Code review and analysis work well with proper context

Use Cases for CLI Agents:
  • File management and organization
  • Code review and refactoring assistance
  • Documentation generation
  • Log file analysis
  • Project scaffolding and setup
  • Automated testing and validation

Next Steps:
  • Add more specialized tools for your domain
  • Implement safety checks for destructive operations
  • Add undo/rollback functionality
  • Integrate with git for version control
  • Create domain-specific CLI assistants
""")

print("=" * 70)