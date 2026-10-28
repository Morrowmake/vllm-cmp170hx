# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(autouse=True)
def _sm80_plugin_registered(monkeypatch):
    """Mark the fork's plugin as registered for these tests.

    These tests replace vLLM's plugin discovery on purpose (a patched
    ``importlib.metadata.entry_points`` or a ``VLLM_PLUGINS`` list of test
    plugins), so the fork's plugin never runs in them. The startup guard in
    ``vllm_sm80/guard.py`` checks real discovery and would refuse the configs
    these tests build, as would its check of the platform plugin.
    """
    monkeypatch.setattr("vllm_sm80.plugin._registered", True)
    monkeypatch.setattr("vllm_sm80.guard.check_platform", lambda: None)
