# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.draft_confidence import (
    DraftConfidence,
    FrozenCoefficients,
    log_odds_max_q,
    survival_cap,
)


def batch(slot=0, start_pos=10):
    return SimpleNamespace(
        idx_mapping_np=np.array([slot]),
        idx_mapping=torch.tensor([slot]),
        req_ids=["request"],
        num_reqs=1,
        is_prefilling_np=np.array([False]),
        query_start_loc_np=np.array([0, 4]),
        query_start_loc=torch.tensor([0, 4]),
        positions=torch.arange(start_pos, start_pos + 4),
        input_ids=torch.tensor([10, 11, 12, 13]),
        is_padding=torch.zeros(4, dtype=torch.bool),
        logits_indices=torch.arange(4),
    )


def test_feature_is_log_odds_of_max_distribution_not_drawn_token():
    scores = torch.tensor([[[0.0, 0.0], [0.0, -2.0]]])
    assert torch.allclose(
        log_odds_max_q(scores, torch.zeros(1)), torch.tensor([[0.0, 2.0]])
    )
    assert torch.allclose(
        log_odds_max_q(scores, torch.tensor([2.0])), torch.tensor([[0.0, 1.0]])
    )


def test_first_dead_input_row_keeps_rejection_and_bonus_row_live():
    coef = FrozenCoefficients(1.0, (0.0, 0.0, 0.0), 0.3)
    predictor = DraftConfidence(2, 3, "cpu", coef)
    predictor.predict(
        torch.zeros(1, 3, 2),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([11, 12, 13]),
        torch.zeros(2),
        3,
    )
    b = batch()
    predictor.apply_mask(b, torch.from_numpy(b.is_prefilling_np))
    assert predictor.caps[0] == 1
    assert b.is_padding.tolist() == [False, False, True, True]
    assert b.draft_skip_mask.tolist() == [False, False, True, True]


def test_stale_slot_or_new_prefix_never_masks_target_rows():
    predictor = DraftConfidence(2, 3, "cpu", FrozenCoefficients(1.0, (0.0,) * 3, 0.9))
    predictor.predict(
        torch.zeros(1, 3, 2),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([11, 12, 13]),
        torch.zeros(2),
        3,
    )
    b = batch(start_pos=20)
    predictor.apply_mask(b, torch.from_numpy(b.is_prefilling_np))
    assert not b.is_padding.any()


def test_measurement_labels_are_censored_and_do_not_change_mask(tmp_path):
    predictor = DraftConfidence(2, 3, "cpu", log_dir=str(tmp_path))
    predictor.predict(
        torch.zeros(1, 3, 2),
        torch.zeros(3, dtype=torch.int32),
        torch.tensor([11, 12, 13]),
        torch.zeros(2),
        3,
    )
    b = batch()
    predictor.apply_mask(b, torch.from_numpy(b.is_prefilling_np))
    predictor.observe(b, torch.tensor([2]), torch.tensor([2]))
    rows = [json.loads(x) for x in predictor.log_path.read_text().splitlines()]
    assert [r["accepted"] for r in rows] == [True, False, False]
    assert [r["observed"] for r in rows] == [True, True, False]
    assert not b.is_padding.any()


def test_unrelated_request_and_padded_capture_cannot_change_decision():
    p = DraftConfidence(2, 3, "cpu", FrozenCoefficients(1.0, (0.0,) * 3, 0.3))

    def predict(slot, value):
        p.predict(
            torch.full((1, 3, 2), value),
            torch.full((3,), slot),
            torch.tensor([11, 12, 13]),
            torch.zeros(2),
            3,
        )

    predict(0, 0.0)
    first = p.caps[0].clone()
    predict(1, 10.0)
    predict(-1, -10.0)
    assert p.caps[0] == first
    predict(0, 0.0)
    assert p.caps[0] == first


@pytest.mark.parametrize(
    "change",
    [
        {"calibrated": False},
        {"slope": float("nan")},
        {"threshold": 0},
        {"intercepts": [0]},
    ],
)
def test_invalid_or_unmeasured_coefficients_fail_closed(tmp_path, change):
    data = dict(
        version=1, calibrated=True, slope=0.5, intercepts=[0] * 3, threshold=0.3
    )
    data.update(change)
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        FrozenCoefficients.load(p, 3)


def test_cap_uses_survival_not_independent_position_thresholds():
    caps = survival_cap(torch.zeros(1, 3), 1.0, torch.zeros(3), 0.3)
    assert caps.tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpu_placeholder_boundary_uses_last_live_target_distribution(temperature):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    device = "cuda"
    logits = torch.full((4, 8), -float("inf"), device=device)
    logits[torch.arange(4, device=device), torch.arange(1, 5, device=device)] = 0
    proposals = torch.tensor([0, 1, -1, -1], device=device)
    args = dict(
        target_logits=logits,
        draft_logits=None,
        draft_sampled=proposals,
        cu_num_logits=torch.tensor([0, 4], dtype=torch.int32, device=device),
        pos=torch.arange(4, device=device),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.zeros(4, dtype=torch.int32, device=device),
        expanded_local_pos=torch.arange(4, dtype=torch.int32, device=device),
        temperature=torch.tensor([temperature], device=device),
        seed=torch.tensor([123], device=device),
        num_speculative_steps=3,
    )
    sampled, counts = rejection_sample(**args)
    assert counts.item() == 2
    assert sampled[0].tolist() == [1, 2, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_confidence_graph_replay_changes_mask_without_recapture():
    coef = FrozenCoefficients(1.0, (0.0,) * 3, 0.3)
    p = DraftConfidence(2, 3, "cuda", coef)
    scores = torch.zeros(1, 3, 2, device="cuda")
    idx = torch.zeros(3, dtype=torch.int32, device="cuda")
    pos = torch.tensor([11, 12, 13], device="cuda")
    temp = torch.zeros(2, device="cuda")
    b = batch()
    for name in (
        "idx_mapping",
        "query_start_loc",
        "positions",
        "is_padding",
        "logits_indices",
    ):
        setattr(b, name, getattr(b, name).to("cuda"))
    is_prefilling = torch.zeros(1, dtype=torch.bool, device="cuda")

    def step():
        b.is_padding.zero_()
        p.predict(scores, idx, pos, temp, 3)
        p.apply_mask(b, is_prefilling)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    graph.replay()
    torch.cuda.synchronize()
    assert p.caps[0].item() == 1
    assert b.is_padding.tolist() == [False, False, True, True]
    scores[:, :, 1] = -10
    graph.replay()
    torch.cuda.synchronize()
    assert p.caps[0].item() == 3
    assert not b.is_padding.any()


def test_kill_switch_default_allocates_no_predictor(monkeypatch):
    from vllm import envs
    from vllm.v1.worker.gpu.spec_decode.draft_confidence import create_draft_confidence

    for flag in ("VLLM_GLM5_DFLASH_SKIP", "VLLM_GLM5_DFLASH_CONFIDENCE_LOG"):
        monkeypatch.delenv(flag, raising=False)
    assert not envs.VLLM_GLM5_DFLASH_SKIP
    assert create_draft_confidence(None, 8, 7, "cpu") is None


def test_pp_skip_fails_explicitly_until_cap_transport_is_supported(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.draft_confidence import create_draft_confidence

    monkeypatch.setenv("VLLM_GLM5_DFLASH_SKIP", "1")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=4),
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="TP-only"):
        create_draft_confidence(config, 8, 7, "cpu")


@pytest.mark.parametrize("seed", range(20))
def test_vectorized_mask_matches_random_batch_reference(seed):
    rng = np.random.default_rng(seed)
    width, max_reqs = 7, 32
    n = int(rng.integers(1, max_reqs + 1))
    lengths = rng.integers(1, 20, size=n)
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    slots = rng.choice(max_reqs, size=n, replace=False)
    prefilling = rng.choice([False, True], size=n)
    total = int(boundaries[-1])
    capture_padding = int(rng.integers(0, 12))
    positions = torch.arange(total + capture_padding)
    initial_padding = torch.from_numpy(
        rng.choice([False, True], size=total + capture_padding)
    )
    p = DraftConfidence(
        max_reqs, width, "cpu", FrozenCoefficients(1.0, (0.0,) * width, 0.3)
    )
    p.caps[:] = torch.from_numpy(rng.integers(0, width + 1, size=max_reqs + 1))
    for r, slot in enumerate(slots):
        p.positions[slot, 0] = positions[boundaries[r]] + int(rng.choice([1, 2]))
    logits_indices = torch.from_numpy(rng.choice(total, size=n, replace=False))
    b = SimpleNamespace(
        num_reqs=n,
        query_start_loc=torch.from_numpy(np.concatenate((boundaries, [total] * 3))),
        idx_mapping=torch.from_numpy(slots),
        positions=positions,
        is_padding=initial_padding.clone(),
        logits_indices=logits_indices,
    )
    expected = torch.zeros_like(initial_padding)
    for r, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if prefilling[r] or end - start <= 1:
            continue
        slot = slots[r]
        valid = p.positions[slot, 0] == positions[start] + 1
        expected[start:end] = (torch.arange(end - start) > p.caps[slot]) & valid
    p.apply_mask(b, torch.from_numpy(prefilling))
    assert torch.equal(b.is_padding, initial_padding | expected)
    assert torch.equal(b.draft_skip_mask, expected[logits_indices])


def test_vectorized_mask_ignores_invalid_slot_and_empty_padded_requests():
    p = DraftConfidence(2, 3, "cpu", FrozenCoefficients(1.0, (0.0,) * 3, 0.3))
    b = batch(slot=-1)
    p.positions[p.sentinel, 0] = 11
    p.caps[p.sentinel] = 0
    p.apply_mask(b, torch.tensor([False]))
    assert not b.is_padding.any()
    b.num_reqs = 2
    b.idx_mapping = torch.tensor([0, -1])
    b.query_start_loc = torch.tensor([0, 4, 4])
    p.positions[0, 0] = 11
    p.caps[0] = 1
    p.apply_mask(b, torch.tensor([False, False]))
    assert b.draft_skip_mask.tolist() == [False, False, True, True]


def test_compiled_mask_matches_eager_without_device_to_host_reads():
    from vllm.v1.worker.gpu.spec_decode.draft_confidence import _apply_mask_compiled

    p = DraftConfidence(2, 3, "cpu", FrozenCoefficients(1.0, (0.0,) * 3, 0.3))
    p.positions[0, 0] = 11
    p.caps[0] = 1
    b = batch()
    p.apply_mask(b, torch.tensor([False]))
    padding = torch.zeros_like(b.is_padding)
    result = _apply_mask_compiled(
        b.query_start_loc,
        b.idx_mapping,
        b.positions,
        padding,
        b.logits_indices,
        p.positions,
        p.caps,
        torch.tensor([False]),
        p.sentinel,
    )
    assert torch.equal(result, b.draft_skip_mask)
    assert torch.equal(padding, b.is_padding)
