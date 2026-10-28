# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the sm_80 GLM-5 decode features meet.

- KDA recover commits replay only the accepted prefix; unused records may hold
  NaN without changing the committed state or conv window.
- Graph padding x compiled Marlin decode: live rows keep the incumbent's
  accuracy, ignore stale scratch, and repeat bitwise.
- Flags-in-data all-reduce (VLLM_SM80_ALLREDUCE) x KDA recover x the
  state-index check (VLLM_STATE_INDEX_CHECK): the flags path rotates its
  stages per call and assumes every rank issues the same collectives in the
  same order; the recover commit and the debug check add host and device work
  between steps but no collective.

CPU parts run with ``CUDA_VISIBLE_DEVICES=""``; GPU parts skip without an
sm_80 GPU (and, for Marlin, without the optional prebuilt extension).
"""

import importlib.util

import pytest
import torch

from tests.kernels.kda_recover_reference import commit_reference
from vllm_sm80.kernels.kda import recover as kr

CASES = [
    ((0, 3, 7), (8, 8, 8)),
    ((7, 7, 7, 7), (8, 8, 8, 8)),
    ((2, 0, 5, 1, 6, 3, 4, 7), (8,) * 8),
    ((1, 4), (3, 6)),  # adaptive depth: different widths in one batch
]


@pytest.mark.parametrize("caps,query_lens", CASES)
def test_recover_commit_reference_ignores_dead_rows(caps, query_lens):
    """CPU reference of the commit: poisoning every row past the replayed
    prefix (NaN, as a dead row may hold) leaves the state bitwise unchanged."""
    H, D = 2, 8
    g = torch.Generator().manual_seed(len(caps))
    num_sampled = [min(cap, q - 1) + 1 for cap, q in zip(caps, query_lens)]
    states = torch.randn(len(caps) + 1, H, D, D, generator=g) * 0.5
    rec = torch.randn(3, len(caps), kr.WS_T, H, D, generator=g) * 0.3
    rec[2] = torch.exp(-torch.rand(len(caps), kr.WS_T, H, D, generator=g))
    poisoned = rec.clone()
    for r, n in enumerate(num_sampled):
        poisoned[:, r, n:] = float("nan")
    for r, n in enumerate(num_sampled):
        clean = commit_reference(states, rec, r + 1, n, r)
        dirty = commit_reference(states, poisoned, r + 1, n, r)
        assert torch.isfinite(dirty).all()
        assert torch.equal(clean.view(torch.int32), dirty.view(torch.int32))


_COLLECTIVES = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
    "reduce_scatter_tensor", "broadcast", "barrier", "all_to_all",
    "all_to_all_single", "send", "recv",
)


@pytest.fixture
def no_collectives(monkeypatch):
    """Fail on any collective: torch.distributed and vLLM's TP wrappers."""
    import torch.distributed as dist

    import vllm.distributed.communication_op as cop

    def boom(*a, **k):
        raise AssertionError("collective issued between steps")

    for name in _COLLECTIVES:
        if hasattr(dist, name):
            monkeypatch.setattr(dist, name, boom)
    for name in dir(cop):
        if name.startswith("tensor_model_parallel_"):
            monkeypatch.setattr(cop, name, boom)


def test_recover_and_state_index_check_issue_no_collective(no_collectives, monkeypatch):
    """With the flags all-reduce, KDA recover and the state-index check all on,
    the check of recover metadata (including its commit metadata) and the
    commit's host path run without any collective, so every rank keeps the
    same all-reduce sequence and the flags path's stage rotation stays in
    step."""
    from tests.v1.worker.test_state_index_check import _recover_md

    from vllm import envs
    from vllm_sm80.worker import state_index_check as sic
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    # KDA recover has no switch of its own: the recover metadata below is what
    # it builds.
    for flag in ("VLLM_SM80_ALLREDUCE", sic.FLAG):
        monkeypatch.setenv(flag, "1")
        assert envs.environment_variables[flag]() is True
    sic.check_kda_metadata("l", _recover_md([[5] + [0] * 7, [9] + [3] * 7],
                                            [1, 1], [0, 8, 16]), 415, 16)
    with pytest.raises(sic.StateIndexError):
        sic.check_kda_metadata("l", _recover_md([[5] + [0] * 7], [1], [0, 8],
                                                commit_state=[415]), 415, 8)
    state = RecoverSSMState()
    state.record_step({}, [], for_capture=True)
    state.commit_step(torch.ones(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
                      state_indices=None, num_accepted_tokens=torch.ones(1))
    state.commit_step(2, torch.zeros(1, dtype=torch.int32), state_indices=None,
                      num_accepted_tokens=torch.ones(1))


def test_flags_dispatch_is_rank_independent():
    """The flags path is chosen from dtype and size only, never from the rank
    or from state the recover commit or the check touch, so all ranks take the
    same path for every call of a step."""
    from vllm_sm80.distributed import custom_all_reduce_flags as fl

    class _F(fl.FlagsAllreduce):
        def __init__(self, rank, max_bytes):  # no buffers, no group
            self.rank, self.max_bytes = rank, max_bytes

    hidden = 6144
    sizes = [r * hidden for r in (1, 2, 4, 8, 16, 21, 32, 56, 64, 3463)]
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for n in sizes:
            x = torch.empty(n, dtype=dtype)
            got = {_F(rank, 256 * 1024).eligible(x) for rank in range(4)}
            assert len(got) == 1
            assert got.pop() == (dtype == torch.bfloat16 and x.nbytes <= 256 * 1024)


# ------------------------------------------------------------------ GPU tests

_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 GPU",
)
_marlin = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("vllm_sm80._ampere_marlin_C") is None,
    reason="requires CUDA and the optional prebuilt extension",
)

try:
    from tests.kernels import test_ampere_tp4_marlin_prefill as _tp_helpers

    decode_weights = _tp_helpers.hp.weights
except Exception:  # pragma: no cover - helpers unavailable
    _tp_helpers = None


@_sm80
@_marlin
@pytest.mark.parametrize("M,live_rows", [(16, 9), (24, 17), (32, 25)])
def test_marlin_decode_live_rows_with_graph_padding(
    decode_weights, monkeypatch, M, live_rows
):
    from vllm_sm80.kernels.marlin_moe import decode
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_sm80.kernels.moe.align import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_MOE_DETERMINISTIC_ALIGN", "1")
    deterministic_moe_align_mode.cache_clear()
    hp = _tp_helpers.hp
    wd = decode_weights
    dead = torch.arange(M, device="cuda") >= live_rows
    live = (~dead).nonzero().squeeze(1)
    layer = hp._gpu_layer(wd)
    x, tw, ids = hp._inputs(M, 7 * M + live_rows)
    masked = ids.clone()
    masked[dead] = -1
    decode.warmup(x.device, max_tokens=M)
    N = wd["w2"].size(1) * 16
    ws = decode._workspaces(x.device, N, M, create=False)

    def run(i):
        out = torch.empty_like(x)
        decode.run(layer, out, x, wd["w1"], wd["w2"], tw, i, MoEActivation.SILU)
        torch.cuda.synchronize()
        return out

    full = run(ids)
    for key in ("h", "c3", "part"):
        if torch.is_floating_point(ws[key]):
            ws[key].fill_(float("nan"))
    got = run(masked)
    assert torch.isfinite(got[live]).all()
    reference = hp.reference_fp64(x, wd, tw, ids, live)
    incumbent = hp._incumbent(wd, x, tw, masked, monkeypatch)
    cm, cx = hp._err(got[live], reference)
    im, ix = hp._err(incumbent[live], reference)
    assert cm <= 1.10 * im + 1e-12 and cx <= 1.25 * ix + 1e-12
    # the unmasked run is held to the same bound on the live rows
    fm, fx = hp._err(full[live], reference)
    assert fm <= 1.10 * im + 1e-12 and fx <= 1.25 * ix + 1e-12
    again = run(masked)
    assert torch.equal(again[live], got[live])
    deterministic_moe_align_mode.cache_clear()


@_sm80
@pytest.mark.parametrize("nseq,T", [(1, 8), (2, 8), (4, 5), (8, 8)])
def test_recover_commit_ignores_poisoned_dead_rows(nseq, T):
    """Verify once with records, then commit n tokens twice: from the records
    as written and with every record and conv row past n set to NaN. State
    and the next step's conv window are bitwise equal."""
    from tests.kernels import test_ampere_kda_recover as rt
    from vllm_sm80.models.glm5next.ops import kda_decode as k2

    k2.warmup(plans=((nseq, 8),), recover=True)
    nslot = 2 + nseq * 8
    p = rt._inputs(nseq, T, 31 * T + nseq, nslot)
    ssm = torch.as_tensor([[1 + s * 8] for s in range(nseq)], device="cuda",
                          dtype=torch.int32)
    ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
    pool = torch.full((1, 3, 8, kr.WS_T, rt.H, rt.D), float("nan"), device="cuda")
    rec, conv = p.rec.clone(), p.conv.clone()
    rt._v2(p, conv, rec, ssm, ones, T, records=kr.layer_records(pool, 0))
    torch.cuda.synchronize()
    src = ssm[:, 0].contiguous()
    history = rt.CONV_K - 1
    for n in range(1, T + 1):
        results = []
        for poison in (False, True):
            rc, cc, pl = rec.clone(), conv.clone(), pool.clone()
            if poison:
                pl[0, :, :nseq, n:] = float("nan")
                for s in range(nseq):
                    cc[1 + s * 8, history + n:] = float("nan")
            ctx = rt._context(cc, rc, pl)
            nsamp = torch.full((nseq,), n, device="cuda", dtype=torch.int32)
            ctx.commit(nsamp, src, p.qsl)
            torch.cuda.synchronize()
            results.append((rc, cc))
        (r0, c0), (r1, c1) = results
        for s in range(nseq):
            assert torch.equal(r0[1 + s * 8].view(torch.int32),
                               r1[1 + s * 8].view(torch.int32)), (n, s)
            assert torch.equal(c0[1 + s * 8, :history], c1[1 + s * 8, :history]), (n, s)
