# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the GPU timing of CUDA-graph forwards
(``vllm_sm80.worker.light_timing``, VLLM_SM80_LIGHT_TIMING): the stamp
kernel is replaced by a recorder, graph capture by a switch."""

import json
import os

import numpy as np
import pytest
import torch
from torch import nn

from vllm_sm80 import plugin
from vllm_sm80.worker import light_timing as LT

FLAG = LT.FLAG


class Attn(nn.Module):
    def forward(self, x):
        return x + 1


class Ffn(nn.Module):
    def forward(self, x):
        return x * 2


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.ffn = Ffn()

    def forward(self, x):
        return self.ffn(self.attn(x))


class ToyModel(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList(ToyLayer() for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ToyDraft(ToyModel):
    pass


class Wrapper(nn.Module):
    """A causal-LM shell: the target root runs inside it."""

    def __init__(self):
        super().__init__()
        self.model = ToyModel()

    def forward(self, x):
        return self.model(x)


@pytest.fixture
def timing(monkeypatch, tmp_path):
    """install() with a toy configuration; the launches recorded as
    (column, tag), the capture state in state['capturing']."""
    calls, state = [], {"capturing": False}

    def allocate(self, root):
        self.device = torch.device("cpu")
        self.ring = torch.zeros((self.rows, self.width), dtype=torch.int64)
        self.counter = torch.zeros(1, dtype=torch.int64)

    monkeypatch.setattr(LT.LightTimer, "allocate", allocate)
    monkeypatch.setattr(LT, "_launch", lambda timer, column, tag: calls.append((column, tag)))
    monkeypatch.setattr(LT, "_capturing", lambda: state["capturing"])
    monkeypatch.setattr(LT, "_timer", None)
    cfg = {"dir": str(tmp_path), "roots": {"target": "ToyModel", "draft": "ToyDraft"},
           "layer": "ToyLayer", "scopes": ["attn", "ffn"], "width": 16, "rows": 8}
    monkeypatch.setenv(FLAG, json.dumps(cfg))
    assert LT.install()
    yield LT._timer, calls, state
    if LT._timer.handle is not None:
        LT._timer.handle.remove()


def test_off_by_default_installs_nothing(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(LT, "_timer", None)
    before = dict(nn.modules.module._global_forward_pre_hooks)
    plugin.register_light_timing()
    assert LT._timer is None
    assert dict(nn.modules.module._global_forward_pre_hooks) == before


@pytest.mark.parametrize("cfg, why", [
    ("[1]", "JSON object"),
    ('{"dir": "rel", "roots": {"t": "M"}, "layer": "L"}', "absolute"),
    ('{"dir": "/x", "roots": {}, "layer": "L"}', "roots"),
    ('{"dir": "/x", "roots": {"a": "M", "b": "M"}, "layer": "L"}', "distinct"),
    ('{"dir": "/x", "roots": {"t": "M"}}', "layer"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": []}', "layer"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": ["L", 1]}', "layer"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": "L", "scopes": "attn"}', "scopes"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": "L", "width": 2}', "width"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": "L", "rows": 0}', "rows"),
    ('{"dir": "/x", "roots": {"t": "M"}, "layer": "L", "extra": 1}', "unknown"),
])
def test_parse_refuses(cfg, why):
    with pytest.raises(ValueError, match=why):
        LT.parse(cfg)


def test_eager_forward_opens_a_row_and_stamps_nothing(timing):
    timer, calls, _ = timing
    Wrapper()(torch.zeros(1))
    # The first call attaches inside the global hook and still opens its row.
    assert calls == [(0, -1)]
    assert timer.depth == 0 and timer.current is None
    Wrapper().model(torch.zeros(1))
    assert calls[-1] == (0, -1)


def test_captured_forward_stamps_layers_scopes_and_exit(timing):
    timer, calls, state = timing
    w = Wrapper()
    w(torch.zeros(1))                     # eager warmup: attaches
    calls.clear()
    state["capturing"] = True
    w(torch.zeros(1))
    cols = {k: c for k, c in timer.columns.items()}
    order = [k for k, _ in sorted(cols.items(), key=lambda kv: kv[1])]
    assert order == [
        ("target", "layers.0", "entry"),
        ("target", "layers.0.attn", "entry"), ("target", "layers.0.attn", "exit"),
        ("target", "layers.0.ffn", "entry"), ("target", "layers.0.ffn", "exit"),
        ("target", "layers.1", "entry"),
        ("target", "layers.1.attn", "entry"), ("target", "layers.1.attn", "exit"),
        ("target", "layers.1.ffn", "entry"), ("target", "layers.1.ffn", "exit"),
        ("target", "", "exit"),
    ]
    assert calls == [(0, 1)] + [(cols[k], 0) for k in order]
    # A second capture (another batch size) writes the same columns.
    calls.clear()
    w(torch.zeros(3))
    assert calls == [(0, 1)] + [(cols[k], 0) for k in order]
    assert timer.columns == cols


def test_roots_keep_their_own_tags_and_the_finder_goes(timing):
    timer, calls, state = timing
    target, draft = Wrapper(), ToyDraft(layers=1)
    target(torch.zeros(1))
    assert timer.handle is not None       # the draft class is still unseen
    draft(torch.zeros(1))
    assert timer.handle is None
    calls.clear()
    state["capturing"] = True
    draft(torch.zeros(1))
    assert calls[0] == (0, 2)
    assert {k[0] for k in timer.columns} == {"draft"}


def test_finder_goes_at_the_next_forward_when_a_root_is_missing(timing):
    """A rank that holds the target only: the finder is gone once the target
    starts its second forward, and a root first called after that is not
    timed."""
    timer, calls, state = timing
    target = Wrapper()
    target(torch.zeros(1))
    assert timer.handle is not None
    target(torch.zeros(1))
    assert timer.handle is None
    calls.clear()
    ToyDraft(layers=1)(torch.zeros(1))
    assert calls == []


def test_drafter_layers_of_another_class_are_stamped(timing):
    """A layer list times a drafter whose layers are not the target's class;
    a scope its layers lack is skipped."""
    timer, calls, state = timing
    timer.cfg["layer"] = ["ToyLayer", "DraftLayer"]

    class DraftLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

        def forward(self, x):
            return self.attn(x)

    draft = ToyDraft(layers=0)
    draft.layers.append(DraftLayer())
    draft(torch.zeros(1))
    state["capturing"] = True
    draft(torch.zeros(1))
    assert sorted(timer.columns, key=timer.columns.get) == [
        ("draft", "layers.0", "entry"),
        ("draft", "layers.0.attn", "entry"), ("draft", "layers.0.attn", "exit"),
        ("draft", "", "exit"),
    ]


def test_nested_root_is_part_of_the_outer_forward(timing):
    timer, calls, state = timing

    class Outer(ToyDraft):
        def __init__(self):
            super().__init__(layers=1)
            self.inner = ToyModel(layers=1)

        def forward(self, x):
            return self.inner(super().forward(x))

    Outer.__name__ = "ToyDraft"           # a draft root holding a target root
    o = Outer()
    o(torch.zeros(1))
    calls.clear()
    state["capturing"] = True
    o(torch.zeros(1))
    opens = [c for c in calls if c[0] == 0]
    assert opens == [(0, 2)]
    assert ("target", "layers.0", "entry") not in timer.columns


def test_width_too_small_fails_the_capture(timing):
    timer, calls, state = timing
    timer.width = 6                       # four stamps per forward
    w = Wrapper()
    w(torch.zeros(1))
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="width 6 holds 4 stamps"):
        w(torch.zeros(1))


def test_first_call_inside_a_capture_is_refused(timing):
    _, _, state = timing
    state["capturing"] = True
    with pytest.raises(RuntimeError, match="inside a CUDA graph capture"):
        Wrapper()(torch.zeros(1))


def _fill(timer, n):
    """Rows as the kernel would leave them after n opens: row k holds tag 1
    and entry time 1000 + k."""
    for k in range(n):
        row = timer.ring[k % timer.rows]
        row.zero_()
        row[0], row[1] = 1, 1000 + k
    timer.counter[0] = n


def test_dump_writes_rows_since_the_previous_dump(timing):
    timer, _, _ = timing
    Wrapper()(torch.zeros(1))
    _fill(timer, 3)
    with np.load(timer.dump("1")) as d:
        assert int(d["counter"]) == 3 and int(d["first"]) == 0 and int(d["lost"]) == 0
        assert d["rows"][:, 1].tolist() == [1000, 1001, 1002]
        meta = json.loads(str(d["meta"]))
    assert meta["tags"] == ["target", "draft"] and meta["rank"] == 0
    _fill(timer, 13)                      # 10 more opens into 8 rows
    with np.load(timer.dump("2")) as d:
        assert int(d["first"]) == 5 and int(d["lost"]) == 2
        assert d["rows"][:, 1].tolist() == list(range(1005, 1013))
    assert sorted(os.listdir(timer.cfg["dir"])) == ["rank0.1.npz", "rank0.2.npz"]


def test_requests_present_at_start_are_not_served(timing, monkeypatch):
    timer, _, _ = timing
    Wrapper()(torch.zeros(1))
    served = []
    monkeypatch.setattr(timer, "dump", lambda name: served.append(name))
    reads = iter(["7", "7", None, "8", "8"])

    def read():
        try:
            return next(reads)
        except StopIteration:
            raise SystemExit from None

    monkeypatch.setattr(timer, "read_request", read)
    monkeypatch.setattr(LT, "POLL_S", 0)
    with pytest.raises(SystemExit):
        timer.serve_dumps()
    assert served == ["8"]
