# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the tool-result size cap in the LiteLLM tool-calling agent."""

from __future__ import annotations

from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance as Inst


def test_short_result_is_unchanged():
    content = "ok" * 10
    assert Inst._truncate_tool_result(content) == content


def test_long_result_is_truncated_and_marked():
    content = "x" * (Inst._TOOL_RESULT_MAX_CHARS + 5000)
    out = Inst._truncate_tool_result(content)

    assert len(out) < len(content)
    assert "tool output truncated" in out
    # Head and tail of the original are preserved around the marker.
    assert out.startswith("x")
    assert out.endswith("x")
