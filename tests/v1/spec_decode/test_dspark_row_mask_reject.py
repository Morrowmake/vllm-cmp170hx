# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark adaptive depth: the drafts on masked verify rows are rejected
(``vllm_sm80/spec_decode/dspark_row_mask_reject.py``).

``cap_drafts`` marks the logits rows whose token row the mask left without
routed experts; the rejection sampler verifies those drafts as -1 proposals.
The full ``RejectionSampler`` runs with its Triton kernels in the interpreter
(a subprocess with ``TRITON_INTERPRET=1``):

- greedy, at steps verifying k = 7 and 3 drafts of a 7-draft drafter: for
  every live_len L in 0..k and every unmasked acceptance a in 0..k, the
  capped step emits the unmasked step's tokens truncated to min(a, L)
  accepted drafts plus the target argmax at row min(a, L) (L = 0: exactly the
  target token of row 0), while the masked rows hold other logits;
  num_rejected counts the capped drafts; with every row live the step is the
  switch-off step;
- probabilistic (one-hot drafts): the capped step equals a step that
  proposed only the first L drafts (same seeds, positions and live rows);
- -1 placeholder drafts (a step starting inside the prefill) give the same
  output capped or not;
- the switch off, or no mask applied this step: ``cap_drafts`` returns its
  input unchanged and launches nothing.

    pytest -q tests/v1/spec_decode/test_dspark_row_mask_reject.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
K = 7
VOCAB = 64
PREFILL = 4
START = 40  # position of row 0 of a request past its prefill


def _state(num_tokens: int, max_num_reqs: int):
    from vllm_sm80.spec_decode.dspark_row_mask import RowMaskState

    return RowMaskState(max_num_reqs, num_tokens + 7, K, torch.device("cpu"))


def _apply_mask(state, live_len: list[int], rows_per_req: int) -> None:
    """Row j of request r is live iff j <= live_len[r] (padding rows stay 1)."""
    for r, n in enumerate(live_len):
        state.live_len[r] = n
        base = r * rows_per_req
        state.row_live[base + n + 1 : base + rows_per_req] = 0
    state.active = True


def _sampler(num_slots: int, temperature: float, seeds: torch.Tensor):
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    sampler = SimpleNamespace(
        compute_nans=False,
        logprobs_mode="raw_logprobs",
        return_sampling_mask=False,
        use_fp64_gumbel=False,
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(
                gpu=torch.full((num_slots,), PREFILL, dtype=torch.int32)
            )
        ),
        sampling_states=SimpleNamespace(
            max_num_logprobs=lambda idx_mapping_np: NO_LOGPROBS,
            temperature=SimpleNamespace(
                gpu=torch.full((num_slots,), temperature, dtype=torch.float32)
            ),
            seeds=SimpleNamespace(gpu=seeds),
        ),
        apply_sampling_params=lambda logits, *args, **kwargs: logits,
    )
    spec = SimpleNamespace(
        num_speculative_tokens=K,
        enable_adaptive_verification=False,
        rejection_sample_method="standard",
        fly_entropy_threshold=0.3,
        fly_entropy_top_k=3,
    )
    return RejectionSampler(sampler, spec, torch.device("cpu"))


def _batch(drafts: list[list[int]], starts: list[int]):
    """A decode step: request r verifies drafts[r] at positions starts[r]..;
    every row is a logits row."""
    n_rows = [len(d) + 1 for d in drafts]
    cu = np.zeros(len(drafts) + 1, dtype=np.int32)
    np.cumsum(n_rows, out=cu[1:])
    input_ids = torch.tensor([t for d in drafts for t in [7, *d]], dtype=torch.int32)
    positions = torch.tensor(
        [s + j for s, n in zip(starts, n_rows) for j in range(n)], dtype=torch.int64
    )
    num_reqs = len(drafts)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    seq_lens = torch.tensor([s + n for s, n in zip(starts, n_rows)], dtype=torch.int32)
    return SimpleNamespace(
        input_ids=input_ids,
        positions=positions,
        logits_indices=torch.arange(int(cu[-1]), dtype=torch.int64),
        expanded_idx_mapping=torch.repeat_interleave(
            idx_mapping, torch.tensor(n_rows)
        ),
        expanded_local_pos=torch.tensor(
            [j for n in n_rows for j in range(n)], dtype=torch.int32
        ),
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping.numpy(),
        cu_num_logits=torch.from_numpy(cu),
        cu_num_logits_np=cu,
        num_reqs=num_reqs,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
    )


def _run(sampler, logits, batch, invalid_drafts):
    out = sampler(logits, batch, None, invalid_drafts)
    sampled = out.sampled_token_ids.tolist()
    num = out.num_sampled.tolist()
    return [s[:n] for s, n in zip(sampled, num)], out.num_rejected.tolist()


def greedy_cases(k: int = K) -> dict:
    """Run in the Triton interpreter: a step verifying ``k`` of a ``K``-draft
    drafter's drafts."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.spec_decode import dspark_row_mask as rm
    from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts

    g = torch.Generator().manual_seed(0)
    # (a, L) for every unmasked acceptance a and live_len L, then two
    # requests whose step starts inside the prefill (all drafts -1).
    pairs = [(a, n) for a in range(k + 1) for n in range(k + 1)]
    placeholders = [0, 3]
    num_reqs = len(pairs) + len(placeholders)
    logits = torch.randn(num_reqs * (k + 1), VOCAB, generator=g)
    argmax = logits.argmax(-1).tolist()
    drafts, starts = [], []
    for r, (a, _) in enumerate(pairs):
        top = argmax[r * (k + 1) : (r + 1) * (k + 1)]
        d = [int(x) for x in torch.randint(0, VOCAB, (k,), generator=g)]
        d[:a] = top[:a]
        if a < k:
            d[a] = (top[a] + 1) % VOCAB
        drafts.append(d)
        starts.append(START + r)
    for _ in placeholders:
        drafts.append([int(x) for x in torch.randint(0, VOCAB, (k,), generator=g)])
        starts.append(PREFILL - 1)
    batch = _batch(drafts, starts)
    sampler = _sampler(num_reqs, 0.0, torch.arange(num_reqs, dtype=torch.int64))
    live = [n for _, n in pairs] + placeholders

    rm._reset_for_tests()
    off = cap_drafts(batch.logits_indices, None)
    uncapped, uncapped_rej = _run(sampler, logits, batch, off)

    state = _state(int(batch.cu_num_logits_np[-1]), num_reqs)
    _apply_mask(state, live, k + 1)
    rm._STATE = state
    try:
        invalid = cap_drafts(batch.logits_indices, None)
    finally:
        rm._reset_for_tests()
    masked_logits = logits.clone()
    rows = state.row_live[: logits.shape[0]] == 0
    # The masked rows took no routed expert: other logits, never read.
    masked_logits[rows] = torch.randn(int(rows.sum()), VOCAB, generator=g)
    capped, capped_rej = _run(sampler, masked_logits, batch, invalid)

    # Every row live (a policy keeping every draft): the mask is applied but
    # adds nothing, and the step is the switch-off step.
    every = _state(int(batch.cu_num_logits_np[-1]), num_reqs)
    every.active = True
    rm._STATE = every
    try:
        all_live = cap_drafts(batch.logits_indices, None)
    finally:
        rm._reset_for_tests()
    kept, kept_rej = _run(sampler, logits, batch, all_live)
    return {
        "k": k,
        "all_live_invalid": all_live.tolist(),
        "kept": kept,
        "kept_rej": kept_rej,
        "off_is_none": off is None,
        "pairs": pairs,
        "argmax": argmax,
        "uncapped": uncapped,
        "uncapped_rej": uncapped_rej,
        "capped": capped,
        "capped_rej": capped_rej,
        "placeholders": placeholders,
    }


def random_cases() -> dict:
    """Run in the Triton interpreter: probabilistic, one-hot drafts."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.spec_decode import dspark_row_mask as rm
    from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts

    g = torch.Generator().manual_seed(1)
    live = [n for n in range(K + 1) for _ in range(4)]
    num_reqs = len(live)
    # Peaked rows so that drafts equal to the row's argmax are often accepted.
    logits = 3.0 * torch.randn(num_reqs * (K + 1), VOCAB, generator=g)
    argmax = logits.argmax(-1).tolist()
    drafts = []
    for r in range(num_reqs):
        top = argmax[r * (K + 1) : (r + 1) * (K + 1)]
        drafts.append(
            [top[i] if i % 3 != 2 else (top[i] + 1) % VOCAB for i in range(K)]
        )
    starts = [START + 3 * r for r in range(num_reqs)]
    seeds = torch.randint(0, 2**31, (num_reqs,), generator=g, dtype=torch.int64)
    sampler = _sampler(num_reqs, 1.0, seeds)

    batch = _batch(drafts, starts)
    state = _state(int(batch.cu_num_logits_np[-1]), num_reqs)
    _apply_mask(state, live, K + 1)
    rm._STATE = state
    try:
        invalid = cap_drafts(batch.logits_indices, None)
    finally:
        rm._reset_for_tests()
    masked_logits = logits.clone()
    rows = state.row_live[: logits.shape[0]] == 0
    masked_logits[rows] = torch.randn(int(rows.sum()), VOCAB, generator=g)
    capped, capped_rej = _run(sampler, masked_logits, batch, invalid)

    # The same requests proposing only their first L drafts.
    short = _batch([d[:n] for d, n in zip(drafts, live)], starts)
    keep = torch.tensor(
        [r * (K + 1) + j for r, n in enumerate(live) for j in range(n + 1)]
    )
    reference, _ = _run(sampler, logits[keep].contiguous(), short, None)
    full, _ = _run(sampler, logits, batch, None)
    return {
        "live": live,
        "capped": capped,
        "capped_rej": capped_rej,
        "reference": reference,
        "full": full,
    }


def _interpret(name: str) -> dict:
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = (
        "import json; from tests.v1.spec_decode.test_dspark_row_mask_reject "
        f"import {name} as f; print('RESULT ' + json.dumps(f()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


@pytest.fixture(scope="module", params=(K, 3), ids=lambda k: f"k{k}")
def greedy(request):
    return _interpret("greedy_cases" if request.param == K else f"greedy_k{request.param}")


def greedy_k3() -> dict:
    return greedy_cases(3)


@pytest.fixture(scope="module")
def random():
    return _interpret("random_cases")


def test_greedy_capped_step_is_the_unmasked_step_truncated(greedy):
    assert greedy["off_is_none"]
    argmax = greedy["argmax"]
    K = greedy["k"]  # noqa: N806
    for r, (a, n) in enumerate(greedy["pairs"]):
        full = greedy["uncapped"][r]
        assert len(full) == a + 1  # the drafts were built to accept a
        kept = min(a, n)
        want = full[: kept + 1]
        assert want[-1] == argmax[r * (K + 1) + kept]
        assert greedy["capped"][r] == want, (a, n)
        assert greedy["capped_rej"][r] == K - kept, (a, n)
        assert greedy["uncapped_rej"][r] == K - a


def test_placeholder_drafts_are_unaffected(greedy):
    K = greedy["k"]  # noqa: N806
    argmax = greedy["argmax"]
    base = len(greedy["pairs"])
    for i, _ in enumerate(greedy["placeholders"]):
        r = base + i
        assert greedy["uncapped"][r] == [argmax[r * (K + 1)]]
        assert greedy["capped"][r] == greedy["uncapped"][r]
        assert greedy["capped_rej"][r] == greedy["uncapped_rej"][r] == K


def test_every_row_live_is_the_switch_off_step(greedy):
    assert greedy["all_live_invalid"] == [False] * len(greedy["all_live_invalid"])
    assert greedy["kept"] == greedy["uncapped"]
    assert greedy["kept_rej"] == greedy["uncapped_rej"]


def test_random_capped_draft_is_no_proposal(random):
    assert random["capped"] == random["reference"]
    for n, toks, rej in zip(random["live"], random["capped"], random["capped_rej"]):
        assert 1 <= len(toks) <= n + 1
        assert rej == K + 1 - len(toks)
    # The cases exercise acceptance and the cap.
    assert any(len(t) > 1 for t in random["capped"])
    assert random["capped"] != random["full"]


def test_switch_off_returns_the_input(monkeypatch):
    from vllm_sm80.spec_decode import dspark_row_mask as rm
    from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts

    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DSPARK_ADAPTIVE_DEPTH", raising=False)
    monkeypatch.setattr(envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "", raising=False)
    rm._reset_for_tests()
    assert rm.init_row_mask_state(
        max_num_reqs=2, max_num_tokens=12, max_draft=K, device="cpu", verify_drafts=[0, K, K]
    ) is None
    grammar = torch.tensor([False, True, False])
    indices = torch.arange(3)
    assert cap_drafts(indices, grammar) is grammar
    assert cap_drafts(indices, None) is None


def test_no_mask_this_step_returns_the_input(monkeypatch):
    from vllm_sm80.spec_decode import dspark_row_mask as rm
    from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts

    state = _state(12, 2)
    state.row_live[3] = 0  # stale, but the step applied no mask
    monkeypatch.setattr(rm, "_STATE", state)
    grammar = torch.tensor([False, True, False, False])
    indices = torch.arange(4)
    assert cap_drafts(indices, grammar) is grammar
    assert cap_drafts(indices, None) is None


def test_grammar_and_mask_combine(monkeypatch):
    from vllm_sm80.spec_decode import dspark_row_mask as rm
    from vllm_sm80.spec_decode.dspark_row_mask_reject import cap_drafts

    state = _state(12, 2)
    _apply_mask(state, [3, 1], 6)  # 5-draft rows: 4, 5 and 8..11 masked
    monkeypatch.setattr(rm, "_STATE", state)
    indices = torch.tensor([0, 2, 4, 6, 8, 9], dtype=torch.int32)
    grammar = torch.tensor([False, True, False, False, False, True])
    got = cap_drafts(indices, grammar)
    assert got.tolist() == [False, True, True, False, True, True]
    assert grammar.tolist() == [False, True, False, False, False, True]
    assert cap_drafts(indices, None).tolist() == [False, False, True, False, True, True]
