# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for system-prompt resolution in the LiteLLM tool-calling agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exgentic.agents.litellm_tool_calling import instance as mod
from exgentic.core.types import ModelSettings


def _make_agent(system_prompt_file):
    with patch.object(mod, "check_model_accessible_sync"):
        return mod.LiteLLMToolCallingAgentInstance(
            session_id="s1",
            model="openai/gpt-4o-mini",
            model_settings=ModelSettings(),
            system_prompt_file=system_prompt_file,
        )


def test_explicit_empty_string_disables_system_prompt(monkeypatch):
    monkeypatch.setenv("EXGENTIC_AGENT_SYSTEM_PROMPT", "from-env")
    assert _make_agent("")._resolve_system_prompt() == ""


def test_path_is_read(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("  be concise  ")
    assert _make_agent(str(f))._resolve_system_prompt() == "be concise"


def test_none_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("EXGENTIC_AGENT_SYSTEM_PROMPT", "  from-env  ")
    assert _make_agent(None)._resolve_system_prompt() == "from-env"


@pytest.mark.usefixtures("tmp_path")
def test_none_without_env_or_global_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("EXGENTIC_AGENT_SYSTEM_PROMPT", raising=False)
    monkeypatch.setattr(mod.os.path, "expanduser", lambda p: str(tmp_path / "missing.txt"))
    assert _make_agent(None)._resolve_system_prompt() == ""
