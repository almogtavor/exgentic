# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the repeated-tool-call guard in the LiteLLM tool-calling agent."""

from __future__ import annotations

from unittest.mock import patch

from exgentic.agents.litellm_tool_calling import instance as mod
from exgentic.core.types import ModelSettings

Inst = mod.LiteLLMToolCallingAgentInstance


def _make_agent(max_repeated_tool_calls):
    with patch.object(mod, "check_model_accessible_sync"):
        return Inst(session_id="s1", model="m", model_settings=ModelSettings(),
                    max_repeated_tool_calls=max_repeated_tool_calls)


def _call(arguments='{"command": "ls"}'):
    return {"name": "bash", "arguments": arguments, "id": "call_1"}


def test_signature_is_argument_order_independent():
    assert Inst._tool_call_signature(_call('{"a": 1, "b": 2}')) == Inst._tool_call_signature(_call('{"b": 2, "a": 1}'))


def test_guard_disabled_never_blocks():
    agent = _make_agent(0)
    assert all(agent._filter_repeated_tool_calls([_call()]) == ([_call()], []) for _ in range(3))


def test_nth_consecutive_identical_call_is_blocked():
    agent = _make_agent(3)
    assert agent._filter_repeated_tool_calls([_call()]) == ([_call()], [])
    assert agent._filter_repeated_tool_calls([_call()]) == ([_call()], [])
    assert agent._filter_repeated_tool_calls([_call()]) == ([], [_call()])


def test_streak_resets_for_unissued_signature():
    agent = _make_agent(2)
    agent._filter_repeated_tool_calls([_call('{"command": "a"}')])
    agent._filter_repeated_tool_calls([_call('{"command": "b"}')])  # resets "a"'s streak
    assert agent._filter_repeated_tool_calls([_call('{"command": "a"}')]) == ([_call('{"command": "a"}')], [])
