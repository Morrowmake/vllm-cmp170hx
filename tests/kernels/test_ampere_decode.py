# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side tests for the opt-in sm_80 decode kernels (vllm_sm80/kernels/ and
the GLM KDA decode step in vllm_sm80/models/glm5next/ops/kda_decode.py).

Everything here runs without a GPU: the dispatch gates are pure host
logic, and so is the block-alignment size arithmetic the fused MoE routing
kernel has to reproduce bit-for-bit. The GPU correctness of the kernels
themselves is covered by the standalone GPU correctness suite.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_decode.py
"""

try:
    import pytest
except ImportError:  # the vllm-dev venv has no pytest; see __main__ below
    pytest = None
import torch

if pytest is None:  # minimal stand-ins so the module still imports and runs
    class _Mark:
        def __getattr__(self, _name):
            return lambda *a, **k: (lambda f: f)

    class _Pytest:
        mark = _Mark()

        @staticmethod
        def fixture(f):
            return f

    pytest = _Pytest()

from tests.sm80.tuning_rows import use_rows
from vllm_sm80.kernels import _platform
from vllm_sm80.kernels.mhc import use_ampere_mhc_decode_v3
from vllm_sm80.kernels.moe import use_ampere_moe_routing
from vllm_sm80.models.glm5next.ops.kda_decode import use_ampere_kda_decode_v2

# The production decode shapes. M = num_seqs * (1 + num_spec); DFlash k=3 so
# T = 4, i.e. M=4 at concurrency 1 and M=16 at concurrency 4.
HIDDEN = 4096
HC = 4
EXPERTS = 288
TOPK = 8
KDA_HEADS = 16
KDA_DIM = 128
PREFILL_CHUNK = 1152


def _all_on(monkeypatch, sm80=True):
    """Master + per-family flags on (KDA has none), and a fake sm_80 answer.

    The capability lookup is cached in a module global precisely so the 166
    gate evaluations per decode step do not re-enter the platform layer; the
    tests set that global rather than a device.
    """
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")
    monkeypatch.setenv("VLLM_GLM5_DECODE_MOE_ROUTING", "1")
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", sm80)
    return monkeypatch


@pytest.fixture
def on(monkeypatch):
    return _all_on(monkeypatch)


def _gates(num_tokens, num_seqs=1):
    return (
        use_ampere_mhc_decode_v3(num_tokens, HC, HIDDEN),
        use_ampere_moe_routing(num_tokens, EXPERTS, TOPK),
        use_ampere_kda_decode_v2(num_seqs, num_tokens, KDA_HEADS, KDA_DIM),
    )


# ----------------------------------------------------------------- gates OFF

@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64, 1152])
def test_gates_are_off_by_default(monkeypatch, num_tokens):
    """With VLLM_AMPERE_DECODE_KERNELS unset nothing dispatches -- ever.

    Includes M=4, the production concurrency-1 decode shape, and M=16, the
    concurrency-4 one.
    """
    monkeypatch.delenv("VLLM_AMPERE_DECODE_KERNELS", raising=False)
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", True)
    assert _gates(num_tokens) == (False, False, False)
    # and the flag check comes first, so no device was touched
    assert torch.cuda.is_initialized() is False


def test_master_flag_zero_is_off(monkeypatch):
    monkeypatch.setenv("VLLM_AMPERE_DECODE_KERNELS", "0")
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", True)
    assert _gates(4) == (False, False, False)


def test_per_family_flags(on):
    """mHC and MoE routing can be turned off on their own with the master
    flag on (the KDA decode step has no family switch: the master flag is it).

    This is how the GPU validation bisects a regression without a rebuild.
    """
    assert _gates(4) == (True, True, True)

    on.setenv("VLLM_GLM5_DECODE_MHC_V2", "0")
    assert _gates(4) == (False, True, True)
    on.setenv("VLLM_GLM5_DECODE_MHC_V2", "1")

    on.setenv("VLLM_GLM5_DECODE_MOE_ROUTING", "0")
    assert _gates(4) == (True, False, True)
    on.setenv("VLLM_GLM5_DECODE_MOE_ROUTING", "1")


def test_gates_off_on_non_sm80(monkeypatch):
    """Even with every flag on, a non-GA100 part keeps the upstream kernels."""
    _all_on(monkeypatch, sm80=False)
    assert _gates(4) == (False, False, False)


def test_capability_probe_is_safe_without_a_device(monkeypatch):
    """The lookup must return False, not raise, and not initialise CUDA.

    "No device" is simulated, so the check also holds on a GPU host.
    """
    monkeypatch.setattr("vllm_sm80.kernels._platform._SM80_CACHE", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    initialized = torch.cuda.is_initialized()
    assert _platform._is_sm80() is False
    assert torch.cuda.is_initialized() is initialized


# ------------------------------------------------------- per-family M bounds

def test_prefill_chunk_never_dispatches(on):
    """This is a decode port: the 1152-token prefill chunk must not fire."""
    assert _gates(PREFILL_CHUNK, num_seqs=1) == (False, False, False)
    assert _gates(2304, num_seqs=1) == (False, False, False)


def test_moe_bound_is_8(on):
    """At block_size 8 -- what marlin actually picks at every decode M -- the
    family is 1.56x at M=4 but 0.98x at M=16 and 0.80x at M=32.  The earlier result of
    1.06x at M=16 was measured at block_size 48, a prefill shape.  Measured
    GPU 2, --repeats 3, incumbent in the same run."""
    for m in (1, 2, 4, 8):
        assert use_ampere_moe_routing(m, EXPERTS, TOPK) is True, m
    for m in (9, 16, 17, 32, 64):
        assert use_ampere_moe_routing(m, EXPERTS, TOPK) is False, m


def test_kda_bound(on):
    """Up to 8 sequences of up to 8 tokens (5..8 sequences only up to 5 at
    16 heads); every shape is covered by tests/kernels/test_ampere_kda_dispatch.py."""
    for nseq in range(1, 9):
        for t in range(1, 6):
            assert use_ampere_kda_decode_v2(nseq, nseq * t, KDA_HEADS, KDA_DIM), (nseq, t)
    for m in (9, 65, 128, 1152):
        assert use_ampere_kda_decode_v2(1, m, KDA_HEADS, KDA_DIM) is False, m
    assert use_ampere_kda_decode_v2(9, 36, KDA_HEADS, KDA_DIM) is False


def test_bounds_are_tuning_rows(on):
    use_rows(on)
    assert use_ampere_mhc_decode_v3(16, HC, HIDDEN) is True
    assert use_ampere_moe_routing(8, EXPERTS, TOPK) is True
    use_rows(
        on,
        ("mhc", "decode_v2_max_rows", {"hidden": HIDDEN, "hc_mult": HC}, 8),
        ("moe_router", "fused_route_max_rows", {"top_k": TOPK}, 4),
    )
    assert use_ampere_mhc_decode_v3(16, HC, HIDDEN) is False
    assert use_ampere_mhc_decode_v3(8, HC, HIDDEN) is True
    assert use_ampere_moe_routing(8, EXPERTS, TOPK) is False
    assert use_ampere_moe_routing(4, EXPERTS, TOPK) is True


# ------------------------------------------------------- per-family contracts

def test_mhc_needs_a_norm_weight(on):
    """The kernel fuses the input RMSNorm, so a call without it stays on TileLang."""
    assert use_ampere_mhc_decode_v3(4, HC, HIDDEN, norm_weight=None) is False
    assert use_ampere_mhc_decode_v3(4, HC, HIDDEN, norm_weight=object()) is True
    # the default means "the caller has one"
    assert use_ampere_mhc_decode_v3(4, HC, HIDDEN) is True


def test_moe_routing_method_is_pinned(on):
    """The kernel asserts sigmoid, one group, renormalize, E=288."""
    assert use_ampere_moe_routing(4, EXPERTS, TOPK) is True
    assert use_ampere_moe_routing(4, EXPERTS, TOPK, scoring_func="softmax") is False
    assert use_ampere_moe_routing(4, EXPERTS, TOPK, scoring_func="") is False
    assert use_ampere_moe_routing(4, EXPERTS, TOPK, num_expert_group=8) is False
    assert use_ampere_moe_routing(4, EXPERTS, TOPK, topk_group=4) is False
    assert use_ampere_moe_routing(4, EXPERTS, TOPK, renormalize=False) is False
    assert use_ampere_moe_routing(4, 256, TOPK) is False
    assert use_ampere_moe_routing(4, 384, TOPK) is False
    assert use_ampere_moe_routing(4, EXPERTS, 6) is False  # not a power of two
    assert use_ampere_moe_routing(4, EXPERTS, 2) is False  # tl.topk needs k>=4


def test_kda_shape_is_pinned(on):
    """Only the TP4 and PP4 rank shapes (16 or 64 heads of 128) dispatch."""
    assert use_ampere_kda_decode_v2(1, 4, 16, 128) is True
    assert use_ampere_kda_decode_v2(1, 4, 64, 128) is True
    assert use_ampere_kda_decode_v2(1, 4, 32, 128) is False
    assert use_ampere_kda_decode_v2(1, 4, 16, 64) is False
    assert use_ampere_kda_decode_v2(0, 4, 16, 128) is False
    assert use_ampere_kda_decode_v2(8, 4, 16, 128) is False  # more seqs than tokens


# ------------------------------------------ moe_align_block_size arithmetic

def _upstream_align_sizes(numel, num_experts, block_size, pad_sorted_ids=False):
    """Transcribed from vllm/model_executor/layers/fused_moe/
    moe_align_block_size.py::moe_align_block_size -- the host-side part that
    sizes sorted_ids and expert_ids. The clamp ORDER matters: the
    numel < num_experts clamp is applied after the pad_sorted_ids round-up.
    """
    from vllm.triton_utils import triton
    from vllm.utils.math_utils import round_up

    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    return max_num_tokens_padded, triton.cdiv(max_num_tokens_padded, block_size)


# Hand-computed, E=288, topk=8. Every decode M has numel = M*8 < 288, so the
# clamp fires and mnp collapses to numel*block_size with exactly numel blocks.
# M=64 is above the clamp and is there to prove the other branch.
#
# block_size 8 FIRST because it is the one production decode actually uses:
# `fused_marlin_moe` picks block_size_m from
#     for bsm in (8, 16, 32, 48, 64): if M*topk/E/bsm < 0.9: break
# and at M=4 that is 4*8/288/8 = 0.014, so it breaks at the first iteration.
# 48 is the 1152-token prefill chunk's size. See marlin_block_size_m().
EXPECTED = {
    (1, 8): (64, 8),
    (4, 8): (256, 32),
    (16, 8): (1024, 128),
    (64, 8): (2528, 316),
    (1, 16): (128, 8),
    (1, 32): (256, 8),
    (1, 48): (384, 8),
    (1, 64): (512, 8),
    (4, 16): (512, 32),
    (4, 32): (1024, 32),
    (4, 48): (1536, 32),
    (4, 64): (2048, 32),
    (16, 16): (2048, 128),
    (16, 32): (4096, 128),
    (16, 48): (6144, 128),
    (16, 64): (8192, 128),
    (64, 16): (4832, 302),
    (64, 32): (9440, 295),
    (64, 48): (14048, 293),
    (64, 64): (18656, 292),
}


@pytest.mark.parametrize("m", [1, 4, 16, 64])
@pytest.mark.parametrize("block_size", [8, 16, 32, 48, 64])
def test_align_sizes_matches_upstream(m, block_size):
    """align_sizes() must be moe_align_block_size's host wrapper, verbatim.

    If it is not, the fused kernel writes sorted_ids/expert_ids at a different
    length than the Marlin GEMM reads and the failure is silent.
    """
    from vllm_sm80.kernels.moe.routing import align_sizes

    numel = m * TOPK
    got = align_sizes(numel, EXPERTS, block_size)
    assert got == EXPECTED[(m, block_size)], (m, block_size, got)
    assert got == _upstream_align_sizes(numel, EXPERTS, block_size)


def test_align_sizes_clamp_boundary():
    """The numel < num_experts clamp, either side of the boundary."""
    from vllm_sm80.kernels.moe.routing import align_sizes

    for bs in (8, 48):
        assert align_sizes(287, 288, bs) == _upstream_align_sizes(287, 288, bs)
        assert align_sizes(288, 288, bs) == _upstream_align_sizes(288, 288, bs)
    e, bs = 288, 48
    assert align_sizes(287, e, bs) == (287 * 48, 287)
    assert align_sizes(288, e, bs) == (288 + 288 * 47, (288 + 288 * 47 + 47) // 48)
    # the production decode block size, either side of the clamp
    assert align_sizes(287, e, 8) == (287 * 8, 287)
    assert align_sizes(288, e, 8) == (288 + 288 * 7, (288 + 288 * 7) // 8)


class _Launch:
    """Stands in for a Triton kernel: records the launch, runs nothing."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        return lambda *a, **k: self.log.append(self.name)


def test_launch_count_dispatch(monkeypatch):
    """FUSE_MAX=4: one fused launch up to 4 rows, routing then alignment
    (two launches) above, up to the gate's 8 rows."""
    from vllm_sm80.kernels.moe import routing as moe

    assert moe.FUSE_MAX == 4
    log = []
    mod = "vllm_sm80.kernels.moe.routing"
    monkeypatch.setattr(f"{mod}._SCRATCH", {})
    for name in ("_fused_route_align_kernel", "_route_kernel", "_align_kernel"):
        monkeypatch.setattr(f"{mod}.{name}", _Launch(name, log))
    for m in (1, 2, 3, 4, 5, 8):
        log.clear()
        moe.fused_route_align(torch.zeros(m, EXPERTS), torch.zeros(EXPERTS),
                              topk=TOPK, block_size=8, num_experts=EXPERTS)
        want = (["_fused_route_align_kernel"] if m <= 4
                else ["_route_kernel", "_align_kernel"])
        assert log == want, m


def test_shape_dispatch_constants_are_the_measured_ones():
    """These are the result of measured sweeps; a port must not retune them."""
    from vllm_sm80.models.glm5next.ops import kda_decode as kda
    from vllm_sm80.kernels.moe import routing as moe

    assert (moe.FUSE_MAX, moe.PAIRWISE_MAX) == (4, 32)
    assert (kda.TARGET_CTAS, kda.BV_MIN, kda.BV_MAX, kda.NUM_WARPS) == (
        512,
        8,
        32,
        2,
    )
    assert (kda.WIDE_HEADS, kda.WIDE_BV, kda.WIDE_NW) == (64, 64, 4)


# --------------------------------------------------------- import hygiene

def test_import_does_not_initialise_cuda():
    """The CPU tests, and any CPU-only vllm import, must stay off the device."""
    import importlib

    import vllm_sm80.kernels.mhc
    import vllm_sm80.kernels.moe

    for module in (_platform, vllm_sm80.kernels.mhc, vllm_sm80.kernels.moe):
        importlib.reload(module)
    assert torch.cuda.is_initialized() is False


def test_gate_signatures_match_the_artifact_script():
    """The integration artifact script calls these
    with exactly these positional arities; keep them."""
    assert use_ampere_mhc_decode_v3(4, 4, 4096) in (True, False)
    assert use_ampere_moe_routing(4, 288, 8) in (True, False)
    assert use_ampere_kda_decode_v2(1, 4, 16, 128) in (True, False)


# --------------------------------------------------------------------------
# Standalone runner: `python tests/kernels/test_ampere_decode.py`.
# The vllm-dev venv has no pytest and this suite must be runnable there, so
# this reimplements just enough of monkeypatch/parametrize to execute it.
# Under real pytest this block does not run.
# --------------------------------------------------------------------------
class _MonkeyPatch:
    def __init__(self):
        self._env = []
        self._attr = []

    def setenv(self, k, v):
        import os

        self._env.append((k, os.environ.get(k)))
        os.environ[k] = str(v)

    def delenv(self, k, raising=True):
        import os

        self._env.append((k, os.environ.get(k)))
        os.environ.pop(k, None)

    def setattr(self, target, value):
        import importlib

        mod, _, attr = target.rpartition(".")
        m = importlib.import_module(mod)
        self._attr.append((m, attr, getattr(m, attr)))
        setattr(m, attr, value)

    def undo(self):
        import os

        for k, v in reversed(self._env):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for m, a, v in reversed(self._attr):
            setattr(m, a, v)
        self._env, self._attr = [], []


def _main():
    import os
    import sys
    import traceback

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    cases = []
    for n in (1, 2, 4, 8, 16, 32, 64, 1152):
        cases.append(
            (
                f"test_gates_are_off_by_default[{n}]",
                lambda mp, n=n: test_gates_are_off_by_default(mp, n),
            )
        )
    for fn in (
        test_master_flag_zero_is_off,
        test_gates_off_on_non_sm80,
        test_capability_probe_is_safe_without_a_device,
    ):
        cases.append((fn.__name__, lambda mp, f=fn: f(mp)))
    for fn in (
        test_per_family_flags,
        test_prefill_chunk_never_dispatches,
        test_moe_bound_is_8,
        test_kda_bound,
        test_bounds_are_configurable,
        test_mhc_needs_a_norm_weight,
        test_moe_routing_method_is_pinned,
        test_kda_shape_is_pinned,
    ):
        cases.append((fn.__name__, lambda mp, f=fn: f(_all_on(mp))))
    for m in (1, 4, 16, 64):
        for bs in (8, 16, 32, 48, 64):
            cases.append(
                (
                    f"test_align_sizes_matches_upstream[{m},{bs}]",
                    lambda mp, m=m, bs=bs: test_align_sizes_matches_upstream(m, bs),
                )
            )
    for fn in (
        test_align_sizes_clamp_boundary,
        test_shape_dispatch_constants_are_the_measured_ones,
        test_import_does_not_initialise_cuda,
        test_gate_signatures_match_the_artifact_script,
    ):
        cases.append((fn.__name__, lambda mp, f=fn: f()))
    cases.append(("test_launch_count_dispatch", test_launch_count_dispatch))

    failed = 0
    for name, run in cases:
        mp = _MonkeyPatch()
        try:
            run(mp)
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
