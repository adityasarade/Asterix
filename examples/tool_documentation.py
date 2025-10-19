#!/usr/bin/env python3
"""
Tool Documentation Example
==========================

Demonstrates automatic documentation generation for custom tools.

This example shows:
1. Auto-generating markdown documentation
2. Exporting tool catalogs as JSON
3. Creating quick reference guides
4. Documenting tool parameters and constraints
5. Building API reference for tool libraries

Run:
    python examples/tool_documentation.py
"""

from pathlib import Path
from asterix import Agent, BlockConfig

# ============================================================================
# Setup
# ============================================================================

print("=" * 70)
print("TOOL DOCUMENTATION GENERATOR")
print("=" * 70)
print()

# Create agent for demonstration
agent = Agent(
    agent_id="doc_generator",
    blocks={
        "task": BlockConfig(size=1000, priority=1)
    },
    model="openai/gpt-4o-mini"
)

print(f"✓ Created agent: {agent.id}")
print()

# ============================================================================
# Register Example Tools with Rich Documentation
# ============================================================================

print("Registering example tools with documentation...")
print("-" * 70)

@agent.tool(
    name="create_user",
    description="Create a new user account with validation"
)
def create_user(username: str, email: str, age: int) -> str:
    """
    Create a new user account.
    
    This tool validates user input and creates a user account with
    the provided information.
    
    Args:
        username: User's desired username (3-20 chars, alphanumeric)
        email: User's email address (must be valid format)
        age: User's age (must be 13 or older)
        
    Returns:
        Success message with user ID or error message
        
    Example:
        create_user(username="john_doe", email="john@example.com", age=25)
        # Returns: "Created user john_doe (ID: 12345)"
    """
    # Validation
    if len(username) < 3 or len(username) > 20:
        return f"Error: Username must be 3-20 characters, got {len(username)}"
    
    if "@" not in email or "." not in email:
        return f"Error: Invalid email format: {email}"
    
    if age < 13:
        return f"Error: User must be at least 13 years old, got {age}"
    
    # Simulate user creation
    user_id = hash(username) % 100000
    return f"✓ Created user '{username}' (ID: {user_id}, Email: {email}, Age: {age})"

print("  ✓ Registered: create_user")


@agent.tool(
    name="calculate_discount",
    description="Calculate discounted price with tax"
)
def calculate_discount(price: float, discount_percent: float, tax_rate: float = 0.0) -> str:
    """
    Calculate the final price after discount and tax.
    
    Args:
        price: Original price in dollars
        discount_percent: Discount percentage (0-100)
        tax_rate: Tax rate as decimal (default: 0.0)
        
    Returns:
        Formatted price breakdown
        
    Example:
        calculate_discount(price=100.0, discount_percent=20, tax_rate=0.08)
        # Returns price breakdown with discount and tax applied
    """
    if price < 0:
        return "Error: Price cannot be negative"
    
    if discount_percent < 0 or discount_percent > 100:
        return f"Error: Discount must be 0-100%, got {discount_percent}%"
    
    if tax_rate < 0 or tax_rate > 1:
        return f"Error: Tax rate must be 0.0-1.0, got {tax_rate}"
    
    # Calculate
    discount_amount = price * (discount_percent / 100)
    discounted_price = price - discount_amount
    tax_amount = discounted_price * tax_rate
    final_price = discounted_price + tax_amount
    
    result = f"""Price Breakdown:
  Original price: ${price:.2f}
  Discount ({discount_percent}%): -${discount_amount:.2f}
  Subtotal: ${discounted_price:.2f}
  Tax ({tax_rate*100:.1f}%): +${tax_amount:.2f}
  Final price: ${final_price:.2f}"""
    
    return result

print("  ✓ Registered: calculate_discount")


@agent.tool(
    name="format_text",
    description="Format text with various styles"
)
def format_text(text: str, style: str = "uppercase") -> str:
    """
    Format text according to specified style.
    
    Args:
        text: Text to format
        style: Formatting style - "uppercase", "lowercase", "title", "reverse"
        
    Returns:
        Formatted text
        
    Example:
        format_text(text="hello world", style="title")
        # Returns: "Hello World"
    """
    valid_styles = ["uppercase", "lowercase", "title", "reverse"]
    
    if style not in valid_styles:
        return f"Error: Style must be one of {valid_styles}, got '{style}'"
    
    if style == "uppercase":
        return text.upper()
    elif style == "lowercase":
        return text.lower()
    elif style == "title":
        return text.title()
    elif style == "reverse":
        return text[::-1]
    
    return text

print("  ✓ Registered: format_text")

print()

# ============================================================================
# 1. List All Available Tools
# ============================================================================

print("=" * 70)
print("1. LISTING ALL AVAILABLE TOOLS")
print("=" * 70)
print()

all_tools = agent._tool_registry.list_tools()

print(f"Total tools available: {len(all_tools)}")
print()

# Separate by category
core_tools = [t for t in all_tools if t['name'].startswith('core_memory_')]
archival_tools = [t for t in all_tools if t['name'].startswith('archival_memory_')]
conversation_tools = [t for t in all_tools if t['name'] == 'conversation_search']
custom_tools = [t for t in all_tools if not t['name'].startswith('core_memory_') 
                and not t['name'].startswith('archival_memory_') 
                and t['name'] != 'conversation_search']

print("📦 Core Memory Tools:")
for tool in core_tools:
    print(f"  • {tool['name']}: {tool['description']}")

print("\n📚 Archival Memory Tools:")
for tool in archival_tools:
    print(f"  • {tool['name']}: {tool['description']}")

print("\n🔍 Conversation Tools:")
for tool in conversation_tools:
    print(f"  • {tool['name']}: {tool['description']}")

print("\n🛠️  Custom Tools:")
for tool in custom_tools:
    print(f"  • {tool['name']}: {tool['description']}")

# ============================================================================
# 2. Generate Markdown Documentation for Single Tool
# ============================================================================

print("\n" + "=" * 70)
print("2. MARKDOWN DOCUMENTATION FOR SINGLE TOOL")
print("=" * 70)
print()

print("Generating documentation for 'create_user' tool...")
print("-" * 70)

try:
    create_user_docs = agent._tool_registry.generate_tool_docs(
        tool_name="create_user",
        format="markdown"
    )
    print(create_user_docs)
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: If generate_tool_docs is not available, the tool system may need")
    print("to be updated with documentation generation methods.")

# ============================================================================
# 3. Generate Complete Registry Documentation
# ============================================================================

print("\n" + "=" * 70)
print("3. COMPLETE TOOL REGISTRY DOCUMENTATION")
print("=" * 70)
print()

print("Generating full registry documentation...")
print("-" * 70)

try:
    full_docs = agent._tool_registry.generate_registry_docs(
        format="markdown",
        group_by_category=True
    )
    
    # Show first 1000 chars as preview
    print("Preview (first 1000 characters):")
    print(full_docs[:1000] + "...\n")
    
    # Save to file
    docs_file = Path("TOOL_REFERENCE.md")
    with open(docs_file, 'w', encoding='utf-8') as f:
        f.write(full_docs)
    
    print(f"✓ Full documentation saved to: {docs_file}")
    print(f"  Total size: {len(full_docs):,} characters")
    
except AttributeError:
    print("Note: generate_registry_docs method not available.")
    print("Creating manual documentation...\n")
    
    # Manual documentation generation
    manual_docs = "# Tool Registry Documentation\n\n"
    manual_docs += f"Total tools: {len(all_tools)}\n\n"
    
    manual_docs += "## Custom Tools\n\n"
    for tool_info in custom_tools:
        manual_docs += f"### {tool_info['name']}\n\n"
        manual_docs += f"**Description:** {tool_info['description']}\n\n"
        manual_docs += f"**Parameters:** {', '.join(tool_info['parameters'])}\n\n"
        manual_docs += "---\n\n"
    
    docs_file = Path("TOOL_REFERENCE.md")
    with open(docs_file, 'w', encoding='utf-8') as f:
        f.write(manual_docs)
    
    print(f"✓ Manual documentation saved to: {docs_file}")

# ============================================================================
# 4. Export Tool Catalog as JSON
# ============================================================================

print("\n" + "=" * 70)
print("4. EXPORT TOOL CATALOG (JSON)")
print("=" * 70)
print()

print("Exporting tool catalog to JSON...")
print("-" * 70)

import json

# Create catalog manually (works with current API)
catalog_data = {
    "metadata": {
        "total_tools": len(all_tools),
        "custom_tools": len(custom_tools),
        "core_tools": len(core_tools) + len(archival_tools) + len(conversation_tools)
    },
    "tools": all_tools
}

catalog = json.dumps(catalog_data, indent=2)

# Save to file
catalog_file = Path("tool_catalog.json")
with open(catalog_file, 'w', encoding='utf-8') as f:
    f.write(catalog)

print(f"✓ Tool catalog exported to: {catalog_file}")
print(f"  Total tools: {catalog_data['metadata']['total_tools']}")
print(f"  Custom tools: {catalog_data['metadata']['custom_tools']}")
print(f"  Core tools: {catalog_data['metadata']['core_tools']}")
print()

# Show JSON preview
print("Preview (first 500 characters):")
print(catalog[:500] + "...")

# ============================================================================
# 5. Generate Quick Reference Guide
# ============================================================================

print("\n" + "=" * 70)
print("5. QUICK REFERENCE GUIDE")
print("=" * 70)
print()

try:
    quick_ref = agent._tool_registry.generate_quick_reference()
    print(quick_ref)
except AttributeError:
    # Create manual quick reference
    print("QUICK REFERENCE - Custom Tools\n")
    print("=" * 50)
    
    for tool_info in custom_tools:
        print(f"\n{tool_info['name']}")
        print("-" * len(tool_info['name']))
        print(f"Description: {tool_info['description']}")
        print(f"Parameters: {', '.join(tool_info['parameters'])}")

# ============================================================================
# 6. Tool Discovery and Filtering
# ============================================================================

print("\n" + "=" * 70)
print("6. TOOL DISCOVERY AND FILTERING")
print("=" * 70)
print()

print("Searching for tools by name pattern...")
print("-" * 70)

# Find tools with "memory" in the name
memory_tools = [t for t in all_tools if "memory" in t['name'].lower()]
print(f"\nTools with 'memory' in name ({len(memory_tools)}):")
for tool in memory_tools:
    print(f"  • {tool['name']}")

# Find tools with "calculate" in the name
calc_tools = [t for t in all_tools if "calculate" in t['name'].lower()]
print(f"\nTools with 'calculate' in name ({len(calc_tools)}):")
for tool in calc_tools:
    print(f"  • {tool['name']}")

# ============================================================================
# 7. Documenting Tool Usage Examples
# ============================================================================

print("\n" + "=" * 70)
print("7. TOOL USAGE EXAMPLES")
print("=" * 70)
print()

print("Testing custom tools with examples...")
print("-" * 70)

# Test create_user
print("\nExample 1: create_user")
print("  Input: username='alice', email='alice@example.com', age=28")
result = create_user(username="alice", email="alice@example.com", age=28)
print(f"  Output: {result}")

# Test calculate_discount
print("\nExample 2: calculate_discount")
print("  Input: price=100.0, discount_percent=20, tax_rate=0.08")
result = calculate_discount(price=100.0, discount_percent=20, tax_rate=0.08)
print(f"  Output:\n{result}")

# Test format_text
print("\nExample 3: format_text")
print("  Input: text='hello world', style='title'")
result = format_text(text="hello world", style="title")
print(f"  Output: {result}")

# ============================================================================
# Summary and Best Practices
# ============================================================================

print("\n" + "=" * 70)
print("✅ TOOL DOCUMENTATION EXAMPLE COMPLETE!")
print("=" * 70)

print("""
Generated Documentation Files:
  • TOOL_REFERENCE.md - Full markdown documentation
  • tool_catalog.json - JSON catalog for programmatic access

Key Takeaways:
  • Tools are self-documenting via docstrings
  • Markdown and JSON formats available
  • Quick reference guides for users
  • Parameter validation is documented
  • Examples help users understand usage

Best Practices for Tool Documentation:

1. Write Clear Docstrings
   ✓ Describe what the tool does
   ✓ Document all parameters
   ✓ Include usage examples
   ✓ Specify return values

2. Use Descriptive Names
   ✓ create_user (clear purpose)
   ✗ make_thing (vague)

3. Document Constraints
   ✓ "age: User's age (must be 13+)"
   ✗ "age: User's age"

4. Provide Examples
   ✓ Show typical usage patterns
   ✓ Include edge cases
   ✓ Demonstrate error handling

5. Keep Docs Updated
   ✓ Update when changing tools
   ✓ Version your tool libraries
   ✓ Share docs with team

Use Cases:
  • Team tool libraries - Share custom tools across projects
  • API documentation - Auto-generate API references
  • User guides - Help users discover available tools
  • Integration docs - Document third-party integrations
  • Training materials - Teach new team members

Next Steps:
  • Document your custom tool libraries
  • Share tool catalogs with your team
  • Build domain-specific tool collections
  • Version control your tool documentation
  • Create interactive tool explorers
""")

print("=" * 70)

# Cleanup
print("\nCleaning up generated files...")
try:
    if Path("TOOL_REFERENCE.md").exists():
        Path("TOOL_REFERENCE.md").unlink()
        print("  ✓ Removed TOOL_REFERENCE.md")
    if Path("tool_catalog.json").exists():
        Path("tool_catalog.json").unlink()
        print("  ✓ Removed tool_catalog.json")
except:
    pass

print("\nNote: This example works with the current tool registry API.")
print("Some advanced features may require additional implementation.")
print("=" * 70)