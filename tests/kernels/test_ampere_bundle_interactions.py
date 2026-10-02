# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the sm_80 GLM-5 decode features meet.

- Draft skipping (VLLM_GLM5_DFLASH_SKIP) x KDA recover (VLLM_GLM5_KDA_RECOVER):
  the skip mask ends every request's accepted prefix at its last live row, so
  the recover commit replays live rows only; rows past the cap may hold any
  value (including NaN) without changing the committed state or conv window.
- Draft skipping x compiled Marlin decode (VLLM_GLM5_MARLIN_DECODE_CUDA): rows
  the skip mask routes to no expert sit between live rows of other requests;
  the live rows keep the incumbent's accuracy, ignore whatever the scratch
  held before the call, and repeat bitwise.
- Flags-in-data all-reduce (VLLM_CUSTOM_ALLREDUCE_FLAGS) x KDA recover x the
  state-index check (VLLM_GLM5_STATE_INDEX_CHECK): the flags path rotates its
  stages per call and assumes every rank issues the same collectives in the
  same order; the recover commit and the debug check add host and device work
  between steps but no collective.

CPU parts run with ``CUDA_VISIBLE_DEVICES=""``; GPU parts skip without an
sm_80 GPU (and, for Marlin, without the optional prebuilt extension).
"""

import importlib.util
import itertools

import pytest
import torch

from vllm.ampere_decode import kda_recover as kr
from vllm.v1.worker.gpu.spec_decode.draft_confidence import (
    DraftConfidence,
    FrozenCoefficients,
)

WIDTH = 7  # DFlash2 draft slots at TP4 (depth up to 7)
FIRST_POS = 10


def _skip_batch(caps, query_lens, device="cpu"):
    """A decode batch whose skip mask comes from the real DraftConfidence
    mask builder (CPU path), with the given per-request caps."""
    nreq = len(caps)
    predictor = DraftConfidence(
        nreq, WIDTH, device, FrozenCoefficients(1.0, (0.0,) * WIDTH, 0.3)
    )
    predictor.positions[:nreq, 0] = FIRST_POS + 1
    predictor.caps[:nreq] = torch.tensor(caps, dtype=torch.int32)
    qsl = torch.tensor([0, *itertools.accumulate(query_lens)], device=device)
    rows = int(qsl[-1])
    batch = type("Batch", (), {})()
    batch.num_reqs = nreq
    batch.query_start_loc = qsl
    batch.idx_mapping = torch.arange(nreq, device=device)
    batch.positions = torch.cat(
        [torch.arange(q, device=device) + FIRST_POS for q in query_lens]
    )
    batch.is_padding = torch.zeros(rows, dtype=torch.bool, device=device)
    batch.logits_indices = torch.arange(rows, device=device)
    predictor.apply_mask(batch, torch.zeros(nreq, dtype=torch.bool, device=device))
    return batch


def _greedy_num_sampled(batch, query_lens):
    """Greedy verification as the rejection sampler sees it: proposals are the
    input ids at the logits rows, with -1 on rows the skip mask killed. Every
    draft matches the target here (the most a request can accept)."""
    tokens = torch.arange(int(batch.query_start_loc[-1])) + 1000
    proposals = tokens.masked_fill(batch.draft_skip_mask, -1)
    out = []
    for r, q in enumerate(query_lens):
        s = int(batch.query_start_loc[r])
        n = 1
        for j in range(1, q):
            if int(proposals[s + j]) != int(tokens[s + j]):
                break
            n += 1
        out.append(n)
    return out


CASES = [
    ((0, 3, 7), (8, 8, 8)),
    ((7, 7, 7, 7), (8, 8, 8, 8)),
    ((2, 0, 5, 1, 6, 3, 4, 7), (8,) * 8),
    ((1, 4), (3, 6)),  # adaptive depth: different widths in one batch
]


@pytest.mark.parametrize("caps,query_lens", CASES)
def test_skip_mask_ends_acceptance_at_the_last_live_row(caps, query_lens):
    batch = _skip_batch(caps, query_lens)
    num_sampled = _greedy_num_sampled(batch, query_lens)
    for r, (cap, q) in enumerate(zip(caps, query_lens)):
        s = int(batch.query_start_loc[r])
        dead = batch.is_padding[s : s + q]
        # rows 0..cap live, the rest routed to no expert
        assert dead.tolist() == [j > cap for j in range(q)]
        assert num_sampled[r] == min(cap, q - 1) + 1
        # the recover commit replays rows [0, num_sampled): all live
        assert not dead[: num_sampled[r]].any()


@pytest.mark.parametrize("caps,query_lens", CASES)
def test_recover_commit_reference_ignores_dead_rows(caps, query_lens):
    """CPU reference of the commit: poisoning every row past the replayed
    prefix (NaN, as a dead row may hold) leaves the state bitwise unchanged."""
    H, D = 2, 8
    g = torch.Generator().manual_seed(len(caps))
    batch = _skip_batch(caps, query_lens)
    num_sampled = _greedy_num_sampled(batch, query_lens)
    states = torch.randn(len(caps) + 1, H, D, D, generator=g) * 0.5
    rec = torch.randn(3, len(caps), kr.WS_T, H, D, generator=g) * 0.3
    rec[2] = torch.exp(-torch.rand(len(caps), kr.WS_T, H, D, generator=g))
    poisoned = rec.clone()
    for r, n in enumerate(num_sampled):
        poisoned[:, r, n:] = float("nan")
    for r, n in enumerate(num_sampled):
        clean = kr.commit_reference(states, rec, r + 1, n, r)
        dirty = kr.commit_reference(states, poisoned, r + 1, n, r)
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
    from vllm.ampere_prefill import state_index_check as sic
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    for flag in ("VLLM_CUSTOM_ALLREDUCE_FLAGS", "VLLM_GLM5_KDA_RECOVER",
                 sic.FLAG):
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
    from vllm.distributed.device_communicators import custom_all_reduce_flags as fl

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
    or importlib.util.find_spec("vllm._ampere_marlin_C") is None,
    reason="requires CUDA and the optional prebuilt extension",
)

try:
    from tests.kernels import test_ampere_tp4_marlin_prefill as _tp_helpers

    decode_weights = _tp_helpers.hp.weights
except Exception:  # pragma: no cover - helpers unavailable
    _tp_helpers = None


@_sm80
@_marlin
@pytest.mark.parametrize("name", ["orig", "exact"])
@pytest.mark.parametrize("caps,query_lens", [c for c in CASES if sum(c[1]) <= 32])
def test_marlin_decode_live_rows_with_interleaved_skip_rows(
    decode_weights, monkeypatch, name, caps, query_lens
):
    from vllm.ampere_decode import marlin_moe as decode
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        deterministic_moe_align_mode,
    )

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    monkeypatch.setenv("VLLM_GLM5_MARLIN_DECODE_VARIANT", name)
    deterministic_moe_align_mode.cache_clear()
    hp = _tp_helpers.hp
    wd = decode_weights
    M = sum(query_lens)
    dead = _skip_batch(caps, query_lens).is_padding.cuda()
    live = (~dead).nonzero().squeeze(1)
    layer = hp._gpu_layer(wd)
    x, tw, ids = hp._inputs(M, 7 * M + len(caps))
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
    from vllm.ampere_decode import kda_decode_v2 as k2

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
