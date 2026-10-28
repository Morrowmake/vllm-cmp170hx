# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of VLLM_SM80_PREFILL_REDUCE_OVERLAP
(vllm_sm80/models/deepseek_v41/sm80_split_reduce.py): where a step splits,
that a split decoder layer returns the unsplit layer's tensors bit for bit
with row-wise sums, the order its work and sums are issued in, and the
side-stream hand-offs.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_split_reduce.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode
from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm_sm80.models.deepseek_v41 import sm80_model
from vllm_sm80.models.deepseek_v41 import sm80_moe
from vllm_sm80.models.deepseek_v41 import sm80_split_reduce as split
from vllm_sm80.models.deepseek_v41.ops import sm80_mhc

from .test_sm80_attention_wiring import _piece_comm, _PieceCustomAllreduce

BF16 = torch.bfloat16
HC, HIDDEN, ITERS, EPS, NORM_EPS = 4, 64, 20, 1e-6, 1e-20
MIX = (2 + HC) * HC


@pytest.mark.parametrize(
    "rows,cut",
    [
        (2044, 1024),
        (2048, 1024),
        (4096, 2048),
        (1536, 1024),
        (1100, 512),
        (1024, 512),
        (1023, None),
        (600, None),
        (6, None),
    ],
)
def test_cut_rows(rows, cut) -> None:
    """Slice A is the multiple of 512 nearest half the rows; both slices hold
    at least 512 rows, else the step runs whole."""
    assert split.cut_rows(rows) == cut
    if cut is not None:
        assert cut % split.SLICE_ALIGN == 0
        assert rows - cut >= split.SLICE_ALIGN


class _CudaLike:
    """The attention input as split_point sees it: a CUDA tensor's flags, a
    CPU tensor's rows (the communicator stand-ins only read sizes)."""

    is_cuda = True

    def __init__(self, rows: int, cols: int):
        self._t = torch.zeros(rows, cols, dtype=BF16)
        self.shape = self._t.shape

    def __getitem__(self, index):
        return self._t[index]


def _eager(monkeypatch, mode=CUDAGraphMode.NONE, capturing=False, max_size=1024):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(split, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        split,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    monkeypatch.setattr(split.capture, "active", False)
    monkeypatch.setattr(split, "_workspace_locked", lambda: True)
    ca = _PieceCustomAllreduce(max_size)
    return ca


def _tp(monkeypatch, ca, host_takes=False):
    comm = _piece_comm(ca, host_takes=host_takes)
    monkeypatch.setattr(
        sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )


@pytest.mark.parametrize(
    "case,cut",
    [
        ("eager_prefill", 1024),
        ("custom_takes_a_slice", None),
        ("host_takes_a_slice", None),
        ("graph_replay", None),
        ("capturing", None),
        ("mega_gate", None),
        ("kernel_capture", None),
        ("too_few_rows", None),
        ("cpu", None),
        ("workspace_unlocked", None),
    ],
)
def test_split_point(monkeypatch, case, cut) -> None:
    """An eager step splits when both slices are too large for the host-staged
    and the custom all-reduce; graph capture or replay, the kernel capture,
    MegaMoE metadata, short steps and the profile/warm-up runs (workspace not
    yet locked) run whole."""
    ca = _eager(
        monkeypatch,
        mode=CUDAGraphMode.FULL if case == "graph_replay" else CUDAGraphMode.NONE,
        capturing=case == "capturing",
        # 1024 rows x 64 BF16 = 128 KiB per slice.
        max_size=(1 << 20) if case == "custom_takes_a_slice" else 1024,
    )
    _tp(monkeypatch, ca, host_takes=case == "host_takes_a_slice")
    if case == "kernel_capture":
        monkeypatch.setattr(split.capture, "active", True)
    if case == "workspace_unlocked":
        monkeypatch.setattr(split, "_workspace_locked", lambda: False)
    rows = 1000 if case == "too_few_rows" else 2044
    x = torch.zeros(rows, 64, dtype=BF16) if case == "cpu" else _CudaLike(rows, 64)
    meta = object() if case == "mega_gate" else None
    assert split.split_point(x, meta) == cut


HIDDEN_WS = 5120
# The EXL3 routed experts' workspace2: a fixed BF16 buffer sized for the row
# chunk (quant.py workspace_shapes); its size does not depend on the call.
WS2_ELEMENTS = 1 << 20


def _moe_call(rows: int) -> None:
    """The routed-experts call's workspace request, as modular_kernel
    _allocate_buffers makes it for the EXL3 experts: fused output (M, K) and
    workspace2, in BF16."""
    from vllm.v1.worker.workspace import current_workspace_manager

    current_workspace_manager().get_simultaneous(
        ((rows * HIDDEN_WS,), BF16), ((2 * WS2_ELEMENTS,), BF16)
    )


def _layer_moe_calls(rows: int) -> None:
    """The MoE calls one split-reduce layer makes for a step of ``rows``."""
    cut = split.split_point(_CudaLike(rows, HIDDEN_WS), None)
    for part in ((rows,) if cut is None else (cut, rows - cut)):
        _moe_call(part)


@pytest.mark.parametrize("profile_splits", [True, False])
def test_locked_workspace_holds_every_later_step(monkeypatch, profile_splits) -> None:
    """The profile run (2048 rows) sizes the workspace, then it locks. A later
    step of 1056 rows runs whole because the custom all-reduce takes its
    512/544-row slices. If the profile run splits (the gate ignoring the
    lock: ``profile_splits``), its two 1024-row calls size the workspace for
    1024 rows only and that step hits the served refusal ("Workspace is locked
    but allocation from ... _allocate_buffers requires ..."). With the gate,
    the profile run is whole and every later step fits."""
    from vllm.v1.worker import workspace as ws

    _eager(monkeypatch)
    if not profile_splits:
        monkeypatch.setattr(
            split, "_workspace_locked", lambda: ws.current_workspace_manager().is_locked()
        )
    # The custom all-reduce takes slices below 8 MiB, as served.
    _tp(monkeypatch, _PieceCustomAllreduce(8 << 20))
    ws.init_workspace_manager(torch.device("cpu"))
    try:
        _layer_moe_calls(2048)  # the profile run
        ws.lock_workspace()
        assert split.split_point(_CudaLike(2048, HIDDEN_WS), None) == 1024
        assert split.split_point(_CudaLike(1056, HIDDEN_WS), None) is None
        if profile_splits:
            with pytest.raises(AssertionError, match="Workspace is locked but allocation"):
                _layer_moe_calls(1056)
            return
        for rows in (2048, 2044, 1056, 1639, 1024, 128, 6):
            _layer_moe_calls(rows)
    finally:
        ws.reset_workspace_manager()


# --- a decoder layer on CPU -------------------------------------------------


class _Attn:
    """The attention's split API with row-wise stand-ins; ``__call__`` is the
    unsplit forward (``_o_proj`` of ``attend``). ``log`` records the calls."""

    def __init__(self, generator, log):
        self.w = torch.randn(HIDDEN, HIDDEN, generator=generator) * 0.1
        self.wb = (torch.randn(HIDDEN, HIDDEN, generator=generator) * 0.1).to(BF16)
        self.log = log

    def attend(self, positions, x):
        self.log.append(("attend", x.shape[0]))
        return torch.tanh(x.float() @ self.w + positions[:, None].float() * 1e-3).to(BF16)

    def wo_b_partial(self, attn_out):
        self.log.append(("wo_b_partial", attn_out.shape[0]))
        return attn_out @ self.wb

    def reduce_wo_b(self, partial):
        self.log.append(("reduce_wo_b", partial.shape[0]))
        return (partial.float() * 4).to(BF16)

    def __call__(self, positions, x, _):
        return self.reduce_wo_b(self.wo_b_partial(self.attend(positions, x)))


class _Moe:
    """The adopted MoE's split API with a row-wise stand-in."""

    def __init__(self, generator, log):
        self.w = torch.randn(HIDDEN, HIDDEN, generator=generator) * 0.1
        self.log = log

    def forward_partial(self, x, input_ids=None, mega_gate_metadata=None):
        self.log.append(("ffn_partial", x.shape[0]))
        ids = input_ids.float()[:, None] * 1e-4
        return (torch.tanh(x.float() @ self.w) + ids).to(BF16)

    def reduce(self, out):
        self.log.append(("reduce_ffn", out.shape[0]))
        return (out.float() * 4).to(BF16)

    def __call__(self, x, input_ids, mega_gate_metadata):
        return self.reduce(self.forward_partial(x, input_ids, mega_gate_metadata))


def _mix_params(generator):
    fn = torch.randn(MIX, HC * HIDDEN, generator=generator) * 0.02
    scale = torch.rand(3, generator=generator) + 0.5
    base = torch.randn(MIX, generator=generator) * 0.1
    return fn, scale, base


def _layer(generator, log):
    lay = sm80_model.DeepseekV4DecoderLayer.__new__(sm80_model.DeepseekV4DecoderLayer)
    nn.Module.__init__(lay)
    lay.hc_attn_fn, lay.hc_attn_scale, lay.hc_attn_base = _mix_params(generator)
    lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base = _mix_params(generator)
    lay.rms_norm_eps, lay.hc_eps, lay.hc_post_alpha = NORM_EPS, EPS, 2.0
    lay.hc_sinkhorn_iters, lay.hc_mult, lay.engram = ITERS, HC, None
    lay.attn_norm = SimpleNamespace(
        weight=(torch.rand(HIDDEN, generator=generator) + 0.5).to(BF16),
        variance_epsilon=NORM_EPS,
    )
    lay.ffn_norm = SimpleNamespace(
        weight=(torch.rand(HIDDEN, generator=generator) + 0.5).to(BF16),
        variance_epsilon=NORM_EPS,
    )
    lay._mhc_ops = sm80_mhc
    lay.attn = _Attn(generator, log)
    lay.ffn = _Moe(generator, log)
    return lay


def _run(monkeypatch, layers, embed, positions, ids, cut):
    """Two layers (the model entry, then a fused boundary) from ``embed``;
    ``cut`` None runs them whole."""
    monkeypatch.setattr(split, "split_point", lambda x, meta: cut)
    x, residual, post, comb, pre = embed, None, None, None, None
    outs = []
    for lay in layers:
        lay._split_reduce = cut is not None
        x, residual, post, comb, pre, _ = lay(
            x, positions, ids, pre, post, comb, residual
        )
        outs.append((x, residual, post, comb, pre))
    return outs


@pytest.mark.parametrize("rows,cut", [(1100, 512), (2044, 1024)])
def test_split_layers_match_whole_layers(monkeypatch, rows, cut) -> None:
    """With row-wise sublayers and sums, two split decoder layers (the entry
    from the embedding and a fused post + pre boundary) return the unsplit
    layers' FFN output, streams, mixes and next pre-mix bit for bit."""
    g = torch.Generator().manual_seed(7)
    log: list = []
    layers = [_layer(g, log) for _ in range(2)]
    embed = torch.randn(rows, HIDDEN, generator=g).to(BF16)
    positions = torch.arange(rows)
    ids = torch.randint(0, 1000, (rows,), generator=g)
    whole = _run(monkeypatch, layers, embed, positions, ids, None)
    log.clear()
    sliced = _run(monkeypatch, layers, embed, positions, ids, cut)
    for w, s in zip(whole, sliced):
        for a, b in zip(w, s):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.equal(a, b)
    # Per layer: the attention on every row, both wo_b partials issued before
    # either FFN slice, each FFN slice right after its own wo_b sum.
    a, b = cut, rows - cut
    per_layer = [
        ("attend", rows),
        ("wo_b_partial", a), ("reduce_wo_b", a),
        ("wo_b_partial", b), ("reduce_wo_b", b),
        ("ffn_partial", a), ("reduce_ffn", a),
        ("ffn_partial", b), ("reduce_ffn", b),
    ]
    assert log == per_layer * 2


# --- the side stream ----------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.events: list = []


class _FakeStream:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def wait_stream(self, other):
        self.rec.events.append(("wait_stream", self.name, other.name))

    def wait_event(self, event):
        self.rec.events.append(("wait_event", self.name, event.name))


class _FakeEvent:
    count = 0

    def __init__(self, rec):
        _FakeEvent.count += 1
        self.rec, self.name = rec, f"e{_FakeEvent.count}"

    def record(self, stream):
        self.rec.events.append(("record", self.name, stream.name))


class _FakeTensor:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def record_stream(self, stream):
        self.rec.events.append(("record_stream", self.name, stream.name))


def test_side_stream_hand_offs(monkeypatch) -> None:
    """issue: the side stream waits for the current stream, runs fn under the
    side stream, records its event there and keeps the input alive on it;
    wait: the current stream waits for that event and keeps the output alive
    on itself."""
    rec = _Recorder()
    main, side = _FakeStream(rec, "main"), _FakeStream(rec, "side")
    current = {"s": main}

    class _Ctx:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.prev, current["s"] = current["s"], self.stream

        def __exit__(self, *exc):
            current["s"] = self.prev

    monkeypatch.setattr(split, "current_stream", lambda: current["s"])
    monkeypatch.setattr(torch.cuda, "stream", _Ctx)
    monkeypatch.setattr(torch.cuda, "Event", lambda: _FakeEvent(rec))
    stream = object.__new__(split.SideStream)
    stream.stream = side

    def fn(t):
        rec.events.append(("fn", t.name, current["s"].name))
        return _FakeTensor(rec, "out")

    pending = stream.issue(fn, _FakeTensor(rec, "in"))
    assert current["s"] is main
    out = pending.wait()
    assert out.name == "out"
    ev = pending.done.name
    assert rec.events == [
        ("wait_stream", "side", "main"),
        ("fn", "in", "side"),
        ("record", ev, "side"),
        ("record_stream", "in", "side"),
        ("wait_event", "main", ev),
        ("record_stream", "out", "main"),
    ]


def test_side_stream_is_high_priority(monkeypatch) -> None:
    """The sums' stream outranks the compute stream for block dispatch."""
    seen = {}

    def stream(**kw):
        seen.update(kw)
        return "side"

    monkeypatch.setattr(torch.cuda, "Stream", stream)
    device = torch.device("cuda", 0)
    assert split.SideStream(device).stream == "side"
    assert seen == {"device": device, "priority": -1}


def test_cpu_issue_runs_inline() -> None:
    side = split.side_stream(torch.device("cpu"))
    t = torch.ones(3)
    assert torch.equal(side.issue(lambda x: x * 2, t).wait(), t * 2)


# --- adoption -----------------------------------------------------------------


def _bare_moe(cls):
    ffn = object.__new__(cls)
    nn.Module.__init__(ffn)
    return ffn


def test_adopt_marks_layers_that_issue_their_own_sums() -> None:
    """Only layers whose attention all-reduces its own wo_b partial and whose
    MoE is the adopted layer-reduced class may split."""
    layers = []
    for reduces, cls in [
        (True, sm80_moe.Sm80DeepseekV41MoE),
        (False, sm80_moe.Sm80DeepseekV41MoE),
        (True, sm80_moe.upstream_model.DeepseekV4MoE),
    ]:
        layers.append(
            SimpleNamespace(
                ffn=_bare_moe(cls),
                attn=SimpleNamespace(reduces_wo_b=lambda r=reduces: r),
            )
        )
    split.adopt_split_reduce(layers)
    assert [getattr(layer, "_split_reduce", False) for layer in layers] == [
        True,
        False,
        False,
    ]


def test_prefill_reduce_overlap_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    name = "VLLM_SM80_PREFILL_REDUCE_OVERLAP"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


def test_prefill_reduce_overlap_banner(monkeypatch) -> None:
    """Adopting the split says so once at start, naming the switch."""
    said = []
    monkeypatch.setattr(
        split.logger, "info_once", lambda msg, *args, **_: said.append(msg % args)
    )
    layers = [
        SimpleNamespace(
            ffn=_bare_moe(sm80_moe.Sm80DeepseekV41MoE),
            attn=SimpleNamespace(reduces_wo_b=lambda: True),
        )
    ]
    split.adopt_split_reduce(layers)
    assert len(said) == 1
    assert "1 decoder layers" in said[0]
    assert "VLLM_SM80_PREFILL_REDUCE_OVERLAP=1" in said[0]


def test_unsplit_layer_class_default() -> None:
    assert sm80_model.DeepseekV4DecoderLayer._split_reduce is False
