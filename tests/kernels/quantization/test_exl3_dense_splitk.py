# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense split-K (VLLM_EXL3_DENSE_SPLITK, csrc/exl3/dense_splitk.cu).

CPU: the hardware gate. Native checks cover the split rule and invalid geometry.

GPU (SM 8.0 only): with the switch off _linear is linear_out bit for bit;
launches outside the split-K path are linear_out bit for bit; on the split-K
path the result is held to linear_out's error against an FP64 reference
(pooled mean <= 1.10x, max <= 1.25x, BF16 rounding ties forgiven, exact where
linear_out is exact), is deterministic, and a captured call replays bitwise
like the eager call without any allocator event.
"""

import numpy as np
import pytest
import torch

from vllm.sm80.kernels.exl3 import ops as exl3_ops

needs_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="EXL3 kernels are built for SM 8.0",
)

SERVED_FORMATS = [(10, 2), (12, 2), (14, 2)]
SMS = 74  # the multiprocessor count the split rule was measured on


def _extension_or_skip():
    try:
        exl3_ops.ensure_extension()
    except RuntimeError as exc:
        pytest.skip(f"vllm._exl3_C unavailable: {exc}")


# ---------------------------------------------------------------- CPU tests


def test_gate_opens_only_on_sm80():
    assert exl3_ops.dense_splitk_closed_reason((8, 0)) is None
    assert exl3_ops.dense_splitk_closed_reason((8, 6)) is not None
    assert exl3_ops.dense_splitk_closed_reason(None) is not None


def test_split_rule_and_invalid_geometry():
    _extension_or_skip()
    splits = torch.ops._exl3_C.linear_splitk_splits
    override = torch.ops._exl3_C.set_gemm_shape_override
    try:
        # Forced block shapes: tiny (4 x 32) and small (16 x 32) split while
        # the grid has at most 1024 blocks; medium and large never split.
        for rows in (1, 3, 4, 5, 16, 17, 64, 128, 129, 3200):
            for n in (128, 256, 640, 1024, 5120, 32768, 124160):
                for shape, bm in ((0, 4), (1, 16)):
                    override(shape)
                    blocks = -(-rows // bm) * (n // 32)
                    want = 2 if blocks <= 1024 else 1
                    for bits_x2, codebook in SERVED_FORMATS:
                        assert splits(rows, n, SMS, bits_x2, codebook) == want
                    # Any other format keeps linear_out.
                    for bits_x2, codebook in ((8, 2), (16, 2), (14, 1), (10, 1)):
                        assert splits(rows, n, SMS, bits_x2, codebook) == 1
                for shape in (2, 3):
                    override(shape)
                    assert splits(rows, n, SMS, 14, 2) == 1
        override(-1)
        # The row-count heuristic at 74 SMs on served classes: decode launches
        # split, the LM head (N 124160) and prefill chunks do not.
        for rows in (1, 2, 4, 8, 16, 32):
            for n in (256, 640, 2560, 3072, 5120):
                assert splits(rows, n, SMS, 14, 2) == 2, (rows, n)
            assert splits(rows, 124160, SMS, 12, 2) == 1, rows
        assert splits(3200, 2560, SMS, 14, 2) == 1
        assert splits(963, 640, SMS, 14, 2) == 1
        assert splits(0, 640, SMS, 14, 2) == 1
        with pytest.raises(RuntimeError, match="multiprocessor count"):
            splits(1, 640, 0, 14, 2)
        with pytest.raises(RuntimeError, match="H128 multiples"):
            splits(1, 600, SMS, 14, 2)
    finally:
        override(-1)


# ---------------------------------------------------------------- GPU tests


def _matrix(seed, padded_in, padded_out, bits_x2, unaligned=False):
    rng = np.random.default_rng(seed)
    trellis = rng.integers(
        -(2**15), 2**15, (padded_in // 16, padded_out // 16, 8 * bits_x2)
    ).astype(np.int16)
    suh = (
        rng.choice([-1.0, 1.0], padded_in) * rng.uniform(0.5, 1.5, padded_in)
    ).astype(np.float16)
    svh = (
        rng.choice([-1.0, 1.0], padded_out) * rng.uniform(0.5, 1.5, padded_out) / 64
    ).astype(np.float16)
    device = torch.device("cuda")
    # An odd int16 offset leaves the packed words only 2-byte aligned.
    offset = 1 if unaligned else 0
    base = torch.empty(trellis.size + offset, dtype=torch.int16, device=device)
    t = base[offset:].view(trellis.shape)
    t.copy_(torch.from_numpy(trellis))
    return dict(
        numpy=(trellis, suh, svh),
        trellis=t,
        owner=base,
        suh=torch.from_numpy(suh).to(device),
        svh=torch.from_numpy(svh).to(device),
    )


def _inputs(seed, rows, logical_in):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, logical_in, generator=g).to(torch.bfloat16).cuda()


def _run(operator, m, x, bits_x2, logical_out, codebook=2):
    return _run_core(operator, m, x, bits_x2, logical_out, codebook)[0]


def _run_core(operator, m, x, bits_x2, logical_out, codebook=2):
    """(BF16 output, FP32 GEMM result rows): both operators leave the GEMM's
    rows x padded_out result after the transformed input in the workspace."""
    trellis = m["trellis"]
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.full(
        (x.shape[0], logical_out), float("nan"), dtype=torch.bfloat16, device=x.device
    )
    workspace = torch.full(
        (exl3_ops.linear_workspace_size(x.shape[0], padded_in, padded_out),),
        float("nan"),
        device=x.device,
    )
    operator(
        x,
        trellis,
        m["suh"],
        m["svh"],
        bits_x2,
        codebook,
        x.shape[1],
        logical_out,
        out,
        workspace,
    )
    torch.accelerator.synchronize()
    return out, workspace[x.shape[0] * padded_in :].clone()


def _bitwise(actual, expected, name):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        f"{name}: {int((actual.view(torch.int16) != expected.view(torch.int16)).sum())}"
        f" of {actual.numel()} elements differ"
    )


def _sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _bf16_ulp(y):
    a = y.abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


class _Pool:
    """validate.py's BF16-level gate: pooled mean <= 1.10x and max <= 1.25x of
    linear_out's error against FP64, genuine rounding ties forgiven (at most
    0.1 % of elements), exact where linear_out is exact, all finite. check()
    returns (mean ratio, max ratio after ties, ties)."""

    def __init__(self):
        self.parts = []

    def add(self, cand, inc, r64):
        self.parts.append((cand.double(), inc.double(), r64))

    def check(self, name):
        ec = [(c - r).abs() for c, _, r in self.parts]
        ei = [(i - r).abs() for _, i, r in self.parts]
        mean_ratio = sum(float(e.mean()) for e in ec) / sum(float(e.mean()) for e in ei)
        inc_max = max(float(e.max()) for e in ei)
        budget = 1.25 * inc_max
        ties = elements = 0
        worst = 0.0
        for (c, i, r), e, f in zip(self.parts, ec, ei):
            assert torch.isfinite(c).all(), f"{name}: non-finite output"
            assert not ((f == 0) & (e != 0)).any(), f"{name}: exact element broken"
            ulp = _bf16_ulp(i)
            mid = (c + i) / 2
            tie = (
                (e > budget)
                & ((c - i).abs() <= ulp * 1.0000001)
                & ((r - mid).abs() <= ulp / 16)
                & (c != i)
            )
            ties += int(tie.sum())
            elements += e.numel()
            rest = e[~tie]
            if rest.numel():
                worst = max(worst, float(rest.max()))
        assert mean_ratio <= 1.10, f"{name}: mean ratio {mean_ratio:.4f}"
        assert worst <= budget, f"{name}: max ratio {worst / inc_max:.4f}"
        assert ties <= 0.001 * elements, f"{name}: {ties} ties of {elements}"
        return mean_ratio, worst / inc_max, ties


@needs_sm80
@pytest.mark.parametrize("bits_x2,codebook", SERVED_FORMATS)
@pytest.mark.parametrize(
    "padded_in,padded_out,logical_in,logical_out,unaligned",
    [
        (2560, 640, 2560, 640, False),  # shared expert gate/up, indexer q/k
        (2560, 256, 2560, 256, False),  # k/v projections
        (640, 384, 600, 300, True),  # cropped edges, 2-byte aligned words
    ],
)
def test_splitk_holds_linear_outs_error_against_fp64(
    bits_x2, codebook, padded_in, padded_out, logical_in, logical_out, unaligned
):
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _extension_or_skip()
    m = _matrix(1000 * bits_x2 + padded_out, padded_in, padded_out, bits_x2, unaligned)
    trellis_np, suh_np, svh_np = m["numpy"]
    weight = torch.from_numpy(
        effective_weight(trellis_np, suh_np, svh_np, bits_x2, codebook)
    ).cuda()
    weight = weight[:logical_in, :logical_out]
    for rows in (1, 3, 4, 8, 17, 32):
        assert (
            exl3_ops.linear_splitk_splits(rows, padded_out, _sms(), bits_x2, codebook)
            == 2
        ), rows
        pool = _Pool()
        split_seeds = 0
        for seed in range(3):
            x = _inputs(100 * rows + seed, rows, logical_in)
            cand, cand_core = _run_core(
                exl3_ops.linear_out_splitk, m, x, bits_x2, logical_out, codebook
            )
            again = _run(
                exl3_ops.linear_out_splitk, m, x, bits_x2, logical_out, codebook
            )
            _bitwise(again, cand, f"rows {rows}: repeat call")
            inc, inc_core = _run_core(
                exl3_ops.linear_out, m, x, bits_x2, logical_out, codebook
            )
            # The split path ran: its FP32 GEMM sums (blocked, two halves) differ
            # from linear_out's single chain. BF16 outputs rarely differ at
            # these sizes, so the FP32 result rows are the witness.
            split_seeds += int(not torch.equal(cand_core, inc_core))
            pool.add(cand, inc, x.double() @ weight)
        assert split_seeds > 0, f"rows {rows}: split-K path never ran"
        pool.check(f"bits_x2={bits_x2} {padded_in}x{padded_out} rows {rows}")


@needs_sm80
@pytest.mark.parametrize(
    "rows,padded_out,bits_x2",
    [
        (963, 640, 14),  # a prefill-sized launch
        (1, 40960, 12),  # a tiny grid above 1024 blocks (LM-head sized)
        (4, 640, 8),  # a format outside the served set
    ],
)
def test_launches_outside_the_split_path_are_linear_out(rows, padded_out, bits_x2):
    _extension_or_skip()
    assert exl3_ops.linear_splitk_splits(rows, padded_out, _sms(), bits_x2, 2) == 1
    m = _matrix(7 + rows, 256, padded_out, bits_x2)
    x = _inputs(rows, rows, 256)
    _bitwise(
        _run(exl3_ops.linear_out_splitk, m, x, bits_x2, padded_out),
        _run(exl3_ops.linear_out, m, x, bits_x2, padded_out),
        "fallback",
    )


@needs_sm80
@pytest.mark.parametrize("switch", [False, True])
def test_linear_output_is_the_selected_operator_bitwise(monkeypatch, switch):
    _extension_or_skip()
    m = _matrix(11, 2560, 640, 14)
    x = _inputs(12, 37, 2560)  # chunks of 16, 16 and 5 rows
    scratch = torch.empty(exl3_ops.linear_scratch_elements(), device=x.device)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", switch)
    out = exl3_ops._linear(x, m["trellis"], m["suh"], m["svh"], 14, 2, 640, 16)
    operator = exl3_ops.linear_out_splitk if switch else exl3_ops.linear_out
    for start in (0, 16, 32):
        _bitwise(
            out[start : start + 16],
            _run(operator, m, x[start : start + 16], 14, 640),
            f"rows {start}..",
        )
    if not switch:
        reference = _run(torch.ops._exl3_C.linear_out_reference, m, x, 14, 640)
        _bitwise(out, reference, "off path vs the scalar reference")


def _allocations():
    return torch.accelerator.memory_stats()["allocation.all.allocated"]


@needs_sm80
@pytest.mark.parametrize("rows", [4, 32])  # tiny and small blocks
def test_splitk_graph_replay_is_bitwise_eager_without_allocations(rows):
    _extension_or_skip()
    # The captured call must be the split-K path, not linear_out.
    assert exl3_ops.linear_splitk_splits(rows, 640, _sms(), 14, 2) == 2
    m = _matrix(21, 2560, 640, 14)
    x = _inputs(22, rows, 2560)
    out = torch.empty(rows, 640, dtype=torch.bfloat16, device=x.device)
    workspace = torch.empty(
        exl3_ops.linear_workspace_size(rows, 2560, 640), device=x.device
    )

    def call():
        exl3_ops.linear_out_splitk(
            x, m["trellis"], m["suh"], m["svh"], 14, 2, 2560, 640, out, workspace
        )

    torch.accelerator.synchronize()
    before = _allocations()
    call()
    torch.accelerator.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = _allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert _allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(out, eager, "replay vs eager")
    # New inputs in the captured buffer: the replay recomputes them.
    x2 = _inputs(23, rows, 2560)
    x.copy_(x2)
    graph.replay()
    torch.accelerator.synchronize()
    _bitwise(out, _run(exl3_ops.linear_out_splitk, m, x2, 14, 640), "new inputs")
