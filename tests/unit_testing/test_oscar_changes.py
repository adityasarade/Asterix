"""
Unit tests for OSCAR-related changes to Asterix.

Covers the 6 changes implemented for OSCAR:
- Change 1.5: Gemini in _select_provider() failure reset
- Change 1.8: Gemini in metrics loop + _provider_health init
- Change 3.1: Custom system prompt
- Changes 4.1/4.2: on_before_tool_call / on_after_tool_call callbacks
- Change 5.1: on_step callback in heartbeat loop
- Change 6.1: get_history() method
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from asterix.agent import Agent, MemoryBlock
from asterix.core.config import BlockConfig, AgentConfig
from asterix.core.llm_manager import LLMProviderManager, LLMResponse


# ========================================================================
# Change 1.5 — Gemini in _select_provider() failure reset
# ========================================================================

class TestGeminiProviderFailureReset:
    """Test that _select_provider() resets Gemini failures properly."""

    def test_failure_reset_includes_gemini(self):
        """When both primary and fallback exceed max failures, reset dict must include gemini."""
        mgr = LLMProviderManager()
        # Exhaust both primary and fallback
        mgr._provider_failures["groq"] = 10
        mgr._provider_failures["openai"] = 10
        mgr._provider_failures["gemini"] = 10

        import asyncio
        asyncio.run(mgr._select_provider())

        # After reset, all three should be 0
        assert mgr._provider_failures["gemini"] == 0
        assert mgr._provider_failures["groq"] == 0
        assert mgr._provider_failures["openai"] == 0

    def test_gemini_in_initial_failures(self):
        """__init__ should include gemini in _provider_failures."""
        mgr = LLMProviderManager()
        assert "gemini" in mgr._provider_failures
        assert mgr._provider_failures["gemini"] == 0


# ========================================================================
# Change 1.8 — Gemini in metrics loop + _provider_health
# ========================================================================

class TestGeminiMetrics:
    """Test Gemini is included in performance metrics."""

    def test_provider_health_initialized(self):
        """_provider_health must exist and include gemini."""
        mgr = LLMProviderManager()
        assert hasattr(mgr, "_provider_health")
        assert "gemini" in mgr._provider_health
        assert mgr._provider_health["gemini"] is True

    def test_metrics_include_gemini(self):
        """get_performance_metrics() must include gemini in provider breakdown."""
        mgr = LLMProviderManager()
        metrics = mgr.get_performance_metrics()

        assert "gemini" in metrics["providers"]
        gemini_metrics = metrics["providers"]["gemini"]
        assert "operation_count" in gemini_metrics
        assert "error_count" in gemini_metrics
        assert "total_tokens" in gemini_metrics

    def test_metrics_no_attribute_error(self):
        """get_performance_metrics() must not raise AttributeError for _provider_health."""
        mgr = LLMProviderManager()
        # This would raise AttributeError before the fix
        metrics = mgr.get_performance_metrics()
        assert "provider_health" in metrics


# ========================================================================
# Change 3.1 — Custom system prompt
# ========================================================================

class TestCustomSystemPrompt:
    """Test custom system prompt support."""

    def test_default_prompt_when_none(self):
        """Without system_prompt, default prompt is used."""
        agent = Agent(agent_id="test_default_prompt")
        prompt = agent._build_system_prompt()
        assert "You are a helpful AI assistant" in prompt
        assert "Memory Blocks" in prompt

    def test_custom_prompt_replaces_default(self):
        """Custom system_prompt replaces the default first line."""
        custom = "You are OSCAR, a GitHub-specialized coding assistant."
        agent = Agent(agent_id="test_custom_prompt", system_prompt=custom)
        prompt = agent._build_system_prompt()

        assert custom in prompt
        assert "You are a helpful AI assistant" not in prompt
        # Memory blocks and tool guidelines should still be appended
        assert "Memory Blocks" in prompt
        assert "Tool Usage Guidelines" in prompt

    def test_system_prompt_stored(self):
        """system_prompt is stored as instance attribute."""
        custom = "Custom prompt"
        agent = Agent(agent_id="test_stored", system_prompt=custom)
        assert agent.system_prompt == custom

    def test_system_prompt_default_none(self):
        """system_prompt defaults to None."""
        agent = Agent(agent_id="test_none")
        assert agent.system_prompt is None


# ========================================================================
# Changes 4.1 / 4.2 — on_before_tool_call / on_after_tool_call
# ========================================================================

class TestToolCallCallbacks:
    """Test on_before_tool_call and on_after_tool_call callbacks."""

    def test_callbacks_default_none(self):
        """Callbacks default to None for backward compat."""
        agent = Agent(agent_id="test_cb_default")
        assert agent.on_before_tool_call is None
        assert agent.on_after_tool_call is None

    def test_callbacks_stored(self):
        """Callbacks are stored when provided."""
        before = lambda name, args: True
        after = lambda name, args, result: None
        agent = Agent(
            agent_id="test_cb_stored",
            on_before_tool_call=before,
            on_after_tool_call=after
        )
        assert agent.on_before_tool_call is before
        assert agent.on_after_tool_call is after

    def test_before_callback_approve(self):
        """on_before_tool_call returning True allows execution."""
        before_calls = []
        after_calls = []

        def before_cb(name, args):
            before_calls.append((name, args))
            return True

        def after_cb(name, args, result):
            after_calls.append((name, args, result))

        agent = Agent(
            agent_id="test_approve",
            on_before_tool_call=before_cb,
            on_after_tool_call=after_cb,
        )

        # Register a simple tool
        @agent.tool(name="echo", description="Echo input")
        def echo(text: str) -> str:
            return f"echoed: {text}"

        # Simulate tool calls list
        tool_calls = [{
            "id": "call_1",
            "name": "echo",
            "arguments": json.dumps({"text": "hello"})
        }]

        results = agent._execute_tool_calls(tool_calls)

        # Before callback was called
        assert len(before_calls) == 1
        assert before_calls[0] == ("echo", {"text": "hello"})

        # After callback was called with result
        assert len(after_calls) == 1
        assert after_calls[0][0] == "echo"
        assert after_calls[0][2] is not None  # result is not None

        # Tool result returned
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "echoed: hello" in results[0]["content"]

    def test_before_callback_reject(self):
        """on_before_tool_call returning False skips execution."""
        before_calls = []
        after_calls = []

        def before_cb(name, args):
            before_calls.append((name, args))
            return False  # Reject

        def after_cb(name, args, result):
            after_calls.append((name, args, result))

        agent = Agent(
            agent_id="test_reject",
            on_before_tool_call=before_cb,
            on_after_tool_call=after_cb,
        )

        @agent.tool(name="dangerous", description="Dangerous tool")
        def dangerous(cmd: str) -> str:
            raise RuntimeError("Should not be called")

        tool_calls = [{
            "id": "call_2",
            "name": "dangerous",
            "arguments": json.dumps({"cmd": "rm -rf /"})
        }]

        results = agent._execute_tool_calls(tool_calls)

        # Before callback was called
        assert len(before_calls) == 1

        # After callback fires with result=None for rejected calls
        assert len(after_calls) == 1
        assert after_calls[0][2] is None

        # Result is a rejection message, not an error
        assert len(results) == 1
        assert "rejected" in results[0]["content"].lower()

    def test_after_callback_exception_swallowed(self):
        """on_after_tool_call exceptions are logged but don't crash."""
        def after_cb(name, args, result):
            raise ValueError("Callback error!")

        agent = Agent(
            agent_id="test_after_err",
            on_after_tool_call=after_cb,
        )

        @agent.tool(name="safe", description="Safe tool")
        def safe(x: str) -> str:
            return f"done: {x}"

        tool_calls = [{
            "id": "call_3",
            "name": "safe",
            "arguments": json.dumps({"x": "test"})
        }]

        # Should NOT raise despite callback error
        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert "done: test" in results[0]["content"]

    def test_before_callback_exception_swallowed(self):
        """on_before_tool_call exceptions are logged but don't block execution."""
        def before_cb(name, args):
            raise RuntimeError("Before callback crash!")

        agent = Agent(
            agent_id="test_before_err",
            on_before_tool_call=before_cb,
        )

        @agent.tool(name="resilient", description="Resilient tool")
        def resilient(val: str) -> str:
            return f"ok: {val}"

        tool_calls = [{
            "id": "call_4",
            "name": "resilient",
            "arguments": json.dumps({"val": "abc"})
        }]

        # Should NOT raise; tool should still execute
        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert "ok: abc" in results[0]["content"]

    def test_after_callback_on_json_error(self):
        """on_after_tool_call fires with result=None on JSON parse errors."""
        after_calls = []

        def after_cb(name, args, result):
            after_calls.append((name, args, result))

        agent = Agent(
            agent_id="test_json_err",
            on_after_tool_call=after_cb,
        )

        tool_calls = [{
            "id": "call_5",
            "name": "nonexistent",
            "arguments": "not valid json {{{"
        }]

        results = agent._execute_tool_calls(tool_calls)

        # After callback fired with result=None
        assert len(after_calls) == 1
        assert after_calls[0][2] is None

        # Error result returned
        assert "Error" in results[0]["content"]


# ========================================================================
# Change 5.1 — on_step callback
# ========================================================================

class TestOnStepCallback:
    """Test on_step callback in heartbeat loop."""

    def test_on_step_default_none(self):
        """on_step defaults to None."""
        agent = Agent(agent_id="test_step_none")
        assert agent.on_step is None

    def test_on_step_stored(self):
        """on_step is stored as instance attribute."""
        cb = lambda n, info: None
        agent = Agent(agent_id="test_step_stored", on_step=cb)
        assert agent.on_step is cb

    def test_on_step_called_on_final_response(self):
        """on_step fires on the final-response path with correct info."""
        step_calls = []

        def step_cb(step_num, info):
            step_calls.append((step_num, info))

        agent = Agent(agent_id="test_step_final", on_step=step_cb)

        # Mock LLM to return a direct response (no tool calls)
        mock_response = LLMResponse(
            content="Hello!",
            model="test",
            provider="groq",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            processing_time=0.1,
            finish_reason="stop",
            raw_response=None  # No tool calls
        )

        with patch.object(agent, '_ensure_llm_manager'):
            agent._llm_manager = Mock()
            # Make complete() return our mock response
            import asyncio

            async def mock_complete(**kwargs):
                return mock_response
            agent._llm_manager.complete = mock_complete

            response = agent.chat("Hi")

        assert len(step_calls) == 1
        step_num, info = step_calls[0]
        assert step_num == 1
        assert info["step_number"] == 1
        assert info["tool_names"] == []
        assert info["final_response"] == "Hello!"

    def test_on_step_called_on_tool_path(self):
        """on_step fires on the tool-calls path with tool names."""
        step_calls = []

        def step_cb(step_num, info):
            step_calls.append((step_num, info))

        agent = Agent(agent_id="test_step_tools", on_step=step_cb)

        @agent.tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hi {name}!"

        # First response: tool call. Second response: final text.
        tool_response = LLMResponse(
            content="",
            model="test",
            provider="groq",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            processing_time=0.1,
            finish_reason="tool_calls",
            raw_response={
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "greet",
                                "arguments": json.dumps({"name": "World"})
                            }
                        }]
                    }
                }]
            }
        )

        final_response = LLMResponse(
            content="I greeted World for you!",
            model="test",
            provider="groq",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            processing_time=0.1,
            finish_reason="stop",
            raw_response=None
        )

        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_response
            return final_response

        with patch.object(agent, '_ensure_llm_manager'):
            agent._llm_manager = Mock()
            agent._llm_manager.complete = mock_complete
            response = agent.chat("Greet World")

        # Two steps: one tool-call, one final-response
        assert len(step_calls) == 2

        # Step 1: tool call
        assert step_calls[0][0] == 1
        assert step_calls[0][1]["tool_names"] == ["greet"]
        assert step_calls[0][1]["final_response"] is None

        # Step 2: final response
        assert step_calls[1][0] == 2
        assert step_calls[1][1]["tool_names"] == []
        assert step_calls[1][1]["final_response"] is not None


# ========================================================================
# Change 6.1 — get_history()
# ========================================================================

class TestGetHistory:
    """Test get_history() method."""

    def test_empty_history(self):
        """get_history returns empty list when no messages."""
        agent = Agent(agent_id="test_hist_empty")
        assert agent.get_history() == []

    def test_history_returns_copies(self):
        """get_history returns copies, not references to internal data."""
        agent = Agent(agent_id="test_hist_copies")
        agent._add_to_conversation_history("user", "Hello")
        agent._add_to_conversation_history("assistant", "Hi there!")

        history = agent.get_history()
        assert len(history) == 2

        # Modifying returned data should not affect internal state
        history[0]["content"] = "MODIFIED"
        assert agent.conversation_history[0]["content"] == "Hello"

    def test_history_limit(self):
        """get_history(limit=N) returns only the last N messages."""
        agent = Agent(agent_id="test_hist_limit")
        for i in range(10):
            agent._add_to_conversation_history("user", f"Message {i}")

        history = agent.get_history(limit=3)
        assert len(history) == 3
        assert history[0]["content"] == "Message 7"
        assert history[2]["content"] == "Message 9"

    def test_history_limit_zero_returns_all(self):
        """get_history(limit=0) returns all messages."""
        agent = Agent(agent_id="test_hist_all")
        for i in range(5):
            agent._add_to_conversation_history("user", f"Msg {i}")

        history = agent.get_history(limit=0)
        assert len(history) == 5

    def test_history_limit_negative_returns_all(self):
        """get_history(limit=-1) returns all messages."""
        agent = Agent(agent_id="test_hist_neg")
        for i in range(5):
            agent._add_to_conversation_history("user", f"Msg {i}")

        history = agent.get_history(limit=-1)
        assert len(history) == 5

    def test_history_default_limit(self):
        """Default limit is 20."""
        agent = Agent(agent_id="test_hist_default")
        for i in range(30):
            agent._add_to_conversation_history("user", f"Msg {i}")

        history = agent.get_history()
        assert len(history) == 20

    def test_history_structure(self):
        """Each history entry has role, content, and timestamp."""
        agent = Agent(agent_id="test_hist_struct")
        agent._add_to_conversation_history("user", "Hello")

        history = agent.get_history()
        assert len(history) == 1

        entry = history[0]
        assert "role" in entry
        assert "content" in entry
        assert "timestamp" in entry
        assert entry["role"] == "user"
        assert entry["content"] == "Hello"
        assert entry["timestamp"] is not None

    def test_history_limit_larger_than_history(self):
        """If limit > len(history), return all messages."""
        agent = Agent(agent_id="test_hist_large")
        agent._add_to_conversation_history("user", "Only one")

        history = agent.get_history(limit=100)
        assert len(history) == 1


# ========================================================================
# Backward Compatibility
# ========================================================================

class TestBackwardCompatibility:
    """Verify existing Agent usage still works without new params."""

    def test_init_without_new_params(self):
        """Agent() with only original params still works."""
        agent = Agent(agent_id="compat_test", model="openai/gpt-5-mini")
        assert agent.system_prompt is None
        assert agent.on_before_tool_call is None
        assert agent.on_after_tool_call is None
        assert agent.on_step is None

    def test_tool_execution_without_callbacks(self):
        """_execute_tool_calls works normally without callbacks."""
        agent = Agent(agent_id="compat_tools")

        @agent.tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        tool_calls = [{
            "id": "call_1",
            "name": "add",
            "arguments": json.dumps({"a": 2, "b": 3})
        }]

        results = agent._execute_tool_calls(tool_calls)
        assert len(results) == 1
        assert "5" in results[0]["content"]

    def test_system_prompt_has_memory_and_tools(self):
        """System prompt always includes memory blocks and tool guidelines."""
        agent = Agent(
            agent_id="compat_prompt",
            blocks={"task": BlockConfig(size=1000, priority=1, description="Current task")}
        )
        prompt = agent._build_system_prompt()
        assert "Memory Blocks" in prompt
        assert "Tool Usage Guidelines" in prompt
        assert "task" in prompt
