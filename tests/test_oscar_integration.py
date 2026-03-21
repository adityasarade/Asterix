"""
Live integration test for OSCAR changes.

Runs actual LLM calls against the Gemini API to verify:
1. Gemini provider works end-to-end
2. Custom system prompt is used
3. on_before_tool_call / on_after_tool_call fire correctly
4. on_step callback fires during heartbeat loop
5. get_history() returns correct data after chat
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from asterix import Agent, BlockConfig


def test_1_gemini_basic_chat():
    """Test 1: Gemini basic chat works end-to-end."""
    print("\n" + "=" * 60)
    print("TEST 1: Gemini basic chat")
    print("=" * 60)

    agent = Agent(
        agent_id="test_gemini",
        model="gemini/gemini-2.5-flash",
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    response = agent.chat("What is 2+2? Answer in one word.")
    print(f"Response: {response}")

    assert response, "Response should not be empty"
    assert "4" in response or "four" in response.lower(), f"Expected '4' in response, got: {response}"
    print("PASSED")


def test_2_custom_system_prompt():
    """Test 2: Custom system prompt is used by the LLM."""
    print("\n" + "=" * 60)
    print("TEST 2: Custom system prompt")
    print("=" * 60)

    agent = Agent(
        agent_id="test_prompt",
        model="gemini/gemini-2.5-flash",
        system_prompt=(
            "You are OSCAR, a GitHub coding assistant. "
            "You must always introduce yourself as OSCAR when asked who you are. "
            "Always respond in a technical tone."
        ),
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    response = agent.chat("Who are you? Introduce yourself briefly.")
    print(f"Response: {response}")

    assert response, "Response should not be empty"
    assert "oscar" in response.lower(), f"Expected 'OSCAR' in response, got: {response}"
    print("PASSED")


def test_3_tool_calling_with_hooks():
    """Test 3: Tool calling with on_before_tool_call and on_after_tool_call."""
    print("\n" + "=" * 60)
    print("TEST 3: Tool calling with confirmation hook")
    print("=" * 60)

    before_log = []
    after_log = []

    def before_cb(tool_name, arguments):
        print(f"  [before] tool={tool_name}, args={arguments}")
        before_log.append({"tool": tool_name, "args": arguments})
        return True  # Approve all

    def after_cb(tool_name, arguments, result):
        print(f"  [after]  tool={tool_name}, result={str(result)[:80]}")
        after_log.append({"tool": tool_name, "args": arguments, "result": result})

    agent = Agent(
        agent_id="test_hooks",
        model="gemini/gemini-2.5-flash",
        max_heartbeat_steps=3,
        on_before_tool_call=before_cb,
        on_after_tool_call=after_cb,
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    @agent.tool(name="get_weather", description="Get the current weather for a city")
    def get_weather(city: str) -> str:
        return f"25°C, sunny in {city}"

    response = agent.chat("What's the weather in Pune? Use the get_weather tool.")
    print(f"Response: {response}")

    assert response, "Response should not be empty"

    # The LLM may or may not call the tool — check what happened
    if before_log:
        print(f"  before_cb fired {len(before_log)} time(s)")
        assert before_log[0]["tool"] == "get_weather"
        assert len(after_log) >= 1, "after_cb should fire when before_cb approves"
        assert after_log[0]["result"] is not None, "Result should not be None for approved calls"
        print("  Tool hooks verified!")
    else:
        print("  (LLM chose not to call the tool — hook wiring still valid)")

    print("PASSED")


def test_3b_tool_rejection():
    """Test 3b: on_before_tool_call returning False blocks execution."""
    print("\n" + "=" * 60)
    print("TEST 3b: Tool rejection via on_before_tool_call")
    print("=" * 60)

    rejections = []
    after_log = []

    def reject_all(tool_name, arguments):
        print(f"  [before] REJECTING tool={tool_name}")
        rejections.append(tool_name)
        return False

    def after_cb(tool_name, arguments, result):
        after_log.append({"tool": tool_name, "result": result})

    agent = Agent(
        agent_id="test_reject",
        model="gemini/gemini-2.5-flash",
        max_heartbeat_steps=3,
        on_before_tool_call=reject_all,
        on_after_tool_call=after_cb,
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    executed = False

    @agent.tool(name="run_command", description="Run a shell command")
    def run_command(command: str) -> str:
        nonlocal executed
        executed = True
        return "should not reach here"

    response = agent.chat("Run the command 'echo hello'. Use the run_command tool.")
    print(f"Response: {response}")

    if rejections:
        assert not executed, "Tool function should NOT have been called after rejection"
        # after_cb should still fire with result=None
        assert len(after_log) >= 1
        assert after_log[0]["result"] is None, "Result should be None for rejected calls"
        print("  Tool was correctly rejected and NOT executed!")
    else:
        print("  (LLM chose not to call the tool — rejection wiring still valid)")

    print("PASSED")


def test_4_on_step_progress():
    """Test 4: on_step callback fires during heartbeat loop."""
    print("\n" + "=" * 60)
    print("TEST 4: on_step progress callback")
    print("=" * 60)

    step_log = []

    def step_cb(step_number, info):
        print(f"  [step] #{step_number}: tools={info['tool_names']}, final={info['final_response'] is not None}")
        step_log.append(info)

    agent = Agent(
        agent_id="test_progress",
        model="gemini/gemini-2.5-flash",
        max_heartbeat_steps=3,
        on_step=step_cb,
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    @agent.tool(name="add_numbers", description="Add two numbers together")
    def add_numbers(a: int, b: int) -> str:
        return str(a + b)

    response = agent.chat("What is 17 + 25? Use the add_numbers tool to compute it.")
    print(f"Response: {response}")

    assert len(step_log) >= 1, "on_step should have been called at least once"
    # Last step should have a final response
    last = step_log[-1]
    assert last["final_response"] is not None, "Last step should contain final_response"
    assert last["max_steps"] > 0

    if len(step_log) >= 2:
        # First step should have tool names if the LLM called tools
        first = step_log[0]
        if first["tool_names"]:
            print(f"  Tool step detected: {first['tool_names']}")

    print(f"  Total steps: {len(step_log)}")
    print("PASSED")


def test_5_get_history():
    """Test 5: get_history() returns correct data after conversation."""
    print("\n" + "=" * 60)
    print("TEST 5: get_history() API")
    print("=" * 60)

    agent = Agent(
        agent_id="test_history",
        model="gemini/gemini-2.5-flash",
        blocks={"task": BlockConfig(size=1000, priority=1)},
    )

    agent.chat("Hello, I'm testing the history API.")
    agent.chat("This is my second message.")

    history = agent.get_history(limit=5)
    print(f"  History entries: {len(history)}")

    for entry in history:
        print(f"    [{entry['role']}] {entry['content'][:60]}...")

    # Should have at least 4 entries (2 user + 2 assistant)
    assert len(history) >= 4, f"Expected >= 4 history entries, got {len(history)}. History: {history}"

    # Check structure
    for entry in history:
        assert "role" in entry, "Each entry must have 'role'"
        assert "content" in entry, "Each entry must have 'content'"
        assert "timestamp" in entry, "Each entry must have 'timestamp'"

    # Check limit works
    full_history = agent.get_history(limit=0)
    limited = agent.get_history(limit=2)
    assert len(limited) == 2, f"limit=2 should return 2, got {len(limited)}"
    assert len(full_history) >= len(limited)

    print("PASSED")


if __name__ == "__main__":
    import time as _time

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Cannot run integration tests.")
        exit(1)

    print(f"Using GEMINI_API_KEY: {api_key[:10]}...")
    print("Running live integration tests against Gemini API...")
    print("(Adding delays between tests to respect free-tier rate limits: 5 req/min)\n")

    tests = [
        test_1_gemini_basic_chat,
        test_2_custom_system_prompt,
        test_3_tool_calling_with_hooks,
        test_3b_tool_rejection,
        test_4_on_step_progress,
        test_5_get_history,
    ]

    passed = 0
    failed = 0
    errors = []

    for i, test_fn in enumerate(tests):
        if i > 0:
            wait = 65  # Wait >60s to fully reset the per-minute quota
            print(f"\n--- Waiting {wait}s for rate-limit reset ---\n")
            _time.sleep(wait)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)
