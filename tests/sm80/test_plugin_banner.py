# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""register() logs one INFO line per process, the first time it runs."""

import vllm.logger
from vllm_sm80 import guard, plugin


class _Recorder:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


def test_register_logs_once_per_process(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(vllm.logger, "init_logger", lambda name: recorder)
    for name in (
        "register_models",
        "_register_exl3",
        "register_triton_mla_sparse_backend",
        "register_moe_layers",
    ):
        monkeypatch.setattr(plugin, name, lambda: None)
    monkeypatch.setattr(plugin, "_registered", False)

    plugin.register()
    plugin.register()

    assert len(recorder.lines) == 1
    assert recorder.lines[0].startswith(
        f"sm80 plugin registered ({guard.DISTRIBUTION} "
    )
    assert plugin.is_registered()
