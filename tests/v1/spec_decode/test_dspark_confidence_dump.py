# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for VLLM_DSPARK_CONFIDENCE_DUMP (dspark_confidence.py)."""

import json
import os
import re
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.envs import FORK_ENV_VARS
from vllm_sm80.spec_decode import dspark_confidence as dump_module
from vllm_sm80.spec_decode import dspark_width as width_module
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as upstream

pytestmark = pytest.mark.cpu_test


def _config():
    spec = SimpleNamespace(
        num_speculative_tokens=5,
        num_speculative_tokens_per_batch_size=[(1, 3, 5), (4, 8, 3)],
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        uses_dynamic_speculative_decoding=lambda: True,
    )
    return SimpleNamespace(
        speculative_config=spec, scheduler_config=SimpleNamespace(max_num_seqs=8)
    )


def _init_cpu(self, config, device):
    self.vllm_config = config
    self.device = device
    self.max_num_reqs = 8
    self.max_width = self.width = self.num_speculative_steps = 5
    self.draft_token_confidence_probs = torch.full((8, 5), -1.0)
    self.use_confidence_head = False


def _head():
    return SimpleNamespace(proj=SimpleNamespace(weight=torch.ones(1, 8)))


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "")
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_CONFIDENCE_DUMP", str(tmp_path))
    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DUMP", str(tmp_path))

    def build(head=_head):
        monkeypatch.setattr(
            upstream.DSparkSpeculator,
            "load_draft_model",
            lambda self, target, names: SimpleNamespace(
                model=SimpleNamespace(confidence_head=head())
            ),
            raising=False,
        )
        return width_module.create_dspark_speculator(_config(), torch.device("cpu"))

    return build


def test_flag_defaults_off_and_keeps_the_original_class(monkeypatch):
    monkeypatch.delenv("VLLM_DSPARK_CONFIDENCE_DUMP", raising=False)
    assert FORK_ENV_VARS["VLLM_DSPARK_CONFIDENCE_DUMP"]() == ""
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "")
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_CONFIDENCE_DUMP", "")
    result = width_module.create_dspark_speculator(_config(), torch.device("cpu"))
    assert type(result) is upstream.DSparkSpeculator


def test_flag_wraps_the_plain_and_the_per_width_drafter(make, monkeypatch):
    drafter = make()
    assert isinstance(drafter, dump_module.ConfidenceDumpMixin)
    assert isinstance(drafter, upstream.DSparkSpeculator)
    cls = dump_module.confidence_dump_class(width_module.PerKDSparkSpeculator)
    assert issubclass(cls, width_module.PerKDSparkSpeculator)
    assert cls.__mro__[1] is dump_module.ConfidenceDumpMixin


def test_load_turns_the_confidence_head_on(make):
    drafter = make()
    assert drafter.use_confidence_head is False
    drafter.load_draft_model(None, set())
    assert drafter.use_confidence_head is True
    drafter._confidence_dump.close()


def test_load_refuses_a_drafter_without_a_head(make):
    drafter = make(head=lambda: None)
    with pytest.raises(ValueError, match="no confidence head"):
        drafter.load_draft_model(None, set())


def _propose_positional(drafter, req_ids, num_sampled, num_rejected, num_drafts=5):
    batch = SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))
    t = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    # The runner's positional call (model_runner.py propose).
    return drafter.propose(
        batch, {}, {}, None, None, t(num_sampled), t(num_rejected),
        None, None, None, None, num_speculative_tokens=num_drafts,
    )


def _rows(tmp_path):
    (path,) = tmp_path.glob("dspark-confidence.*.rank0.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def _fake_draft_step(monkeypatch):
    drafted = []

    def fake_propose(self, input_batch, *args, **kwargs):
        # What the draft step writes: a distinct confidence row per request.
        step = len(drafted) + 1
        for i in range(input_batch.num_reqs):
            self.draft_token_confidence_probs[i] = torch.tensor(
                [step + i / 10 + k / 100 for k in range(5)]
            )
        drafted.append(list(input_batch.req_ids))

    monkeypatch.setattr(upstream.DSparkSpeculator, "propose", fake_propose, raising=False)
    return drafted


def test_rows_pair_each_rounds_confidences_with_its_verification(make, monkeypatch, tmp_path):
    _fake_draft_step(monkeypatch)
    drafter = make()
    drafter.load_draft_model(None, set())
    # Step 1: prefill of a (no drafts verified yet, nothing to pair).
    _propose_positional(drafter, ["a"], [1], [0])
    # Step 2: a verified 4 drafts, accepted 2; b is new (no previous row).
    _propose_positional(drafter, ["a", "b"], [3, 1], [2, 0])
    # A dummy run (warm-up/capture) neither pairs nor replaces the previous batch.
    drafter.propose(
        SimpleNamespace(req_ids=["x"], num_reqs=1), {}, {}, None, None,
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=None, next_prefill_tokens=None, temperature=None, seeds=None,
        dummy_run=True,
    )
    # Step 3: reordered batch; b accepted all 5; a is mid-prefill (no sample).
    _propose_positional(drafter, ["b", "a"], [6, 0], [0, 0])
    drafter._confidence_dump.close()
    rows = _rows(tmp_path)
    assert [(r["req"], r["verified"], r["accepted"], r["width"], r["batch"]) for r in rows] == [
        ("a", 4, 2, 5, 2),
        ("b", 5, 5, 5, 2),
    ]
    # a's row carries step 1's confidences (row 0); b's carries step 2's row 1,
    # not the dummy run's.
    assert rows[0]["conf"] == pytest.approx([1.0, 1.01, 1.02, 1.03, 1.04])
    assert rows[1]["conf"] == pytest.approx([2.1, 2.11, 2.12, 2.13, 2.14])
    # Monotonic stage times: step 2's verification before step 3's.
    assert rows[0]["t"] < rows[1]["t"]
    assert rows[1]["step"] > rows[0]["step"]


def test_a_round_that_verified_nothing_writes_no_row(make, monkeypatch, tmp_path):
    # The last chunk of a chunked prefill samples one token from one logit:
    # nothing it carries was verified, whatever the buffer holds.
    _fake_draft_step(monkeypatch)
    drafter = make()
    drafter.load_draft_model(None, set())
    _propose_positional(drafter, ["a"], [1], [0])
    _propose_positional(drafter, ["a"], [1], [0])
    _propose_positional(drafter, ["a"], [2], [4])
    drafter._confidence_dump.close()
    assert [(r["verified"], r["accepted"]) for r in _rows(tmp_path)] == [(5, 1)]


def test_a_zero_draft_step_leaves_nothing_to_pair(make, monkeypatch, tmp_path):
    drafted = _fake_draft_step(monkeypatch)
    drafter = make()
    drafter.load_draft_model(None, set())
    _propose_positional(drafter, ["a", "b"], [1, 1], [0, 0])
    # Zero drafts: the draft step does not run (DFlashSpeculator.propose), so the
    # buffer still holds the previous batch's rows, in its order.
    monkeypatch.setattr(
        upstream.DSparkSpeculator, "propose", lambda self, *a, **k: None, raising=False
    )
    _propose_positional(drafter, ["b", "a"], [3, 3], [2, 2], num_drafts=0)
    # The verification after it (stale verified counts on purpose) pairs nothing.
    _propose_positional(drafter, ["b", "a"], [3, 3], [2, 2])
    drafter._confidence_dump.close()
    rows = _rows(tmp_path)
    assert [(r["req"], r["conf"][0]) for r in rows] == [
        ("b", pytest.approx(1.1)),
        ("a", pytest.approx(1.0)),
    ]
    assert len(drafted) == 1


def test_warmup_requests_write_no_row(make, monkeypatch, tmp_path):
    _fake_draft_step(monkeypatch)
    drafter = make()
    drafter.load_draft_model(None, set())
    ids = ["_warmup_0_", "_v2_mixed_warmup_1_", "real"]
    _propose_positional(drafter, ids, [1, 1, 1], [0, 0, 0])
    _propose_positional(drafter, ids, [2, 2, 2], [3, 3, 3])
    drafter._confidence_dump.close()
    assert [r["req"] for r in _rows(tmp_path)] == ["real"]


def test_staging_copies_whole_contiguous_rows(tmp_path):
    # A strided device-to-host copy goes through unpinned staging and waits
    # for the stream; the slot receives whole rows even at a narrow width.
    dump = dump_module.ConfidenceDump(
        str(tmp_path), max_num_reqs=2, max_width=5, device=torch.device("cpu"), rank=0
    )
    for slot in dump._slots:
        slot.conf.fill_(float("nan"))
    conf = torch.arange(10, dtype=torch.float32).view(2, 5)
    dump.drafted(["a", "b"], 3)
    t = torch.ones(2, dtype=torch.int32)
    dump.stage(conf, ["a", "b"], t + 1, t)
    staged = dump._slots[dump._pending[0].slot].conf if dump._pending else dump._slots[0].conf
    assert torch.equal(staged, conf)
    dump.close()
    assert [r["conf"] for r in _rows(tmp_path)] == [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]


def test_each_server_start_writes_its_own_file(tmp_path):
    dump = dump_module.ConfidenceDump(
        str(tmp_path), max_num_reqs=1, max_width=5, device=torch.device("cpu"), rank=0
    )
    dump.close()
    assert re.fullmatch(
        r"dspark-confidence\.\d{8}T\d{6}-%d\.rank0\.jsonl" % os.getpid(), dump.path.name
    )


def test_other_ranks_stage_nothing(tmp_path):
    dump = dump_module.ConfidenceDump(
        str(tmp_path), max_num_reqs=2, max_width=5, device=torch.device("cpu"), rank=1
    )
    dump.drafted(["a"], 5)
    t = torch.ones(1, dtype=torch.int32)
    dump.stage(torch.zeros(2, 5), ["a"], t, t)
    dump.close()
    assert not list(tmp_path.glob("dspark-confidence.*"))


def test_draft_stats_are_the_top1_probability_and_entropy_of_the_logits():
    logits = torch.tensor([[2.0, 0.0, -1.0, float("-inf")], [0.0, 0.0, 0.0, 0.0]])
    out = torch.full((3, 2, 5), float("nan"))
    dump_module.write_draft_stats(out, logits.to(torch.bfloat16), 2)
    p = torch.softmax(logits, dim=-1)
    assert out[:2, 0, 2].tolist() == pytest.approx(p.amax(dim=-1).tolist(), rel=1e-6)
    entropy = [-(q[q > 0] * q[q > 0].log()).sum().item() for q in p]
    assert out[:2, 1, 2].tolist() == pytest.approx(entropy, rel=1e-6)
    assert out[1, 1, 2].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    # Only column `step` of the given rows is written.
    assert torch.isnan(out[2]).all() and torch.isnan(out[:, :, [0, 1, 3, 4]]).all()


def test_rows_carry_the_draft_stats_of_the_positions_the_drafter_sampled(
    make, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        upstream.DSparkSpeculator,
        "_sample_logits",
        lambda self, logits, idx_map, sample_pos, step: logits.argmax(dim=-1),
        raising=False,
    )
    # Position i of request r: all mass on one token, or uniform over 4.
    sharp = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
    flat = torch.zeros(1, 4)

    def fake_propose(self, input_batch, *args, **kwargs):
        for i in range(5):
            rows = [sharp if (r + i) % 2 == 0 else flat for r in range(input_batch.num_reqs)]
            self._sample_logits(torch.cat(rows), None, None, i)

    monkeypatch.setattr(upstream.DSparkSpeculator, "propose", fake_propose, raising=False)
    drafter = make()
    drafter.load_draft_model(None, set())
    _propose_positional(drafter, ["a", "b"], [1, 1], [0, 0])
    _propose_positional(drafter, ["a", "b"], [2, 4], [3, 1])
    drafter._confidence_dump.close()
    rows = _rows(tmp_path)
    assert [r["req"] for r in rows] == ["a", "b"]
    ln4 = torch.log(torch.tensor(4.0)).item()
    assert rows[0]["top1"] == pytest.approx([1.0, 0.25, 1.0, 0.25, 1.0], abs=1e-6)
    assert rows[0]["ent"] == pytest.approx([0.0, ln4, 0.0, ln4, 0.0], abs=1e-5)
    assert rows[1]["top1"] == pytest.approx([0.25, 1.0, 0.25, 1.0, 0.25], abs=1e-6)


def test_rows_carry_no_draft_stats_when_the_drafter_never_sampled_logits(
    make, monkeypatch, tmp_path
):
    # Per-rank argmax (shard_argmax.py) drafts without _sample_logits.
    _fake_draft_step(monkeypatch)
    drafter = make()
    drafter.load_draft_model(None, set())
    _propose_positional(drafter, ["a"], [1], [0])
    _propose_positional(drafter, ["a"], [3], [2])
    drafter._confidence_dump.close()
    (row,) = _rows(tmp_path)
    assert "top1" not in row and "ent" not in row
