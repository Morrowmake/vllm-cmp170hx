# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the draft side of VLLM_DSPARK_ADAPTIVE_DEPTH
(dspark_row_mask_conf.py) and the policy file (dspark_row_mask.load_policy).

The publish kernel runs in the Triton interpreter (a subprocess with
``TRITON_INTERPRET=1``), driven through the mixin's ``_sample_sequential``
over the plain and the per-width drafter: each request's confidences (and
draft-logit statistics, when the policy uses them) land at its request-state
slot, graph padding requests write nothing, columns past a narrower draft
width are zero, and the drafted tokens are untouched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.envs import FORK_ENV_VARS
from vllm_sm80.spec_decode import dspark_row_mask as state_module
from vllm_sm80.spec_decode import dspark_row_mask_conf as conf_module
from vllm_sm80.spec_decode import dspark_width as width_module
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as upstream

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
MAX_REQS, MAX_TOKENS, MAX_DRAFT = 8, 64, 5


def _config(draft_sample_method="greedy"):
    spec = SimpleNamespace(
        num_speculative_tokens=MAX_DRAFT,
        num_speculative_tokens_per_batch_size=[(1, 3, 5), (4, 8, 3)],
        adaptive_k_config=None,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        uses_dynamic_speculative_decoding=lambda: True,
        draft_sample_method=draft_sample_method,
    )
    return SimpleNamespace(
        speculative_config=spec, scheduler_config=SimpleNamespace(max_num_seqs=8)
    )


def _init_cpu(self, config, device):
    self.vllm_config = config
    self.speculative_config = config.speculative_config
    self.device = device
    self.max_num_reqs = MAX_REQS
    self.max_num_tokens = MAX_TOKENS
    self.max_width = self.width = self.num_speculative_steps = MAX_DRAFT
    self.draft_token_confidence_probs = torch.full((MAX_REQS, MAX_DRAFT), -1.0)
    self.use_confidence_head = False
    self.dp_size = 1


def _head():
    return SimpleNamespace(proj=SimpleNamespace(weight=torch.ones(1, 8)))


def _policy(path: Path, **changes) -> str:
    policy = {
        "schema": "dspark-depth-policy/1",
        "max_depth": MAX_DRAFT,
        "features": ["bias", "logit_conf"],
        "weights": [[0.0, 1.0]] * MAX_DRAFT,
        "costs": {
            "1": {str(k): 1 + k for k in range(MAX_DRAFT + 1)},
            "8": {str(k): 2 + k for k in range(MAX_DRAFT + 1)},
        },
    }
    policy.update(changes)
    path.write_text(json.dumps(policy))
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_state():
    state_module._reset_for_tests()
    yield
    state_module._reset_for_tests()


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_CONFIDENCE_DUMP", "")
    monkeypatch.setattr(width_module.envs, "VLLM_SM80_SHARD_ARGMAX", False)
    monkeypatch.setattr(
        width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", _policy(tmp_path / "p.json")
    )

    def build(head=_head, draft_sample_method="greedy"):
        monkeypatch.setattr(
            upstream.DSparkSpeculator,
            "load_draft_model",
            lambda self, target, names: SimpleNamespace(
                model=SimpleNamespace(confidence_head=head())
            ),
            raising=False,
        )
        return width_module.create_dspark_speculator(
            _config(draft_sample_method), torch.device("cpu")
        )

    return build


def test_switch_off_keeps_the_class_and_creates_no_state(monkeypatch):
    monkeypatch.delenv("VLLM_DSPARK_ADAPTIVE_DEPTH", raising=False)
    assert FORK_ENV_VARS["VLLM_DSPARK_ADAPTIVE_DEPTH"]() == ""
    monkeypatch.setattr(upstream.DSparkSpeculator, "__init__", _init_cpu)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_PER_K_WIDTH", False)
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_CONFIDENCE_DUMP", "")
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", "")
    monkeypatch.setattr(
        upstream.DSparkSpeculator,
        "load_draft_model",
        lambda self, target, names: SimpleNamespace(
            model=SimpleNamespace(confidence_head=_head())
        ),
        raising=False,
    )
    drafter = width_module.create_dspark_speculator(_config(), torch.device("cpu"))
    assert type(drafter) is upstream.DSparkSpeculator
    drafter.load_draft_model(None, set())
    assert drafter.use_confidence_head is False
    assert state_module.get_row_mask_state() is None


def test_switch_wraps_the_plain_and_the_per_width_drafter(make):
    drafter = make()
    assert isinstance(drafter, conf_module.RowMaskConfidenceMixin)
    assert isinstance(drafter, upstream.DSparkSpeculator)
    cls = conf_module.row_mask_class(width_module.PerKDSparkSpeculator)
    assert issubclass(cls, width_module.PerKDSparkSpeculator)
    assert cls.__mro__[1] is conf_module.RowMaskConfidenceMixin


def test_refuses_the_confidence_dump(make, monkeypatch, tmp_path):
    # The dump would label drafts the mask cut as rejected by the target.
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_CONFIDENCE_DUMP", str(tmp_path))
    monkeypatch.setenv("VLLM_DSPARK_CONFIDENCE_DUMP", str(tmp_path))
    with pytest.raises(ValueError, match="VLLM_DSPARK_CONFIDENCE_DUMP"):
        make()


def test_load_refuses_probabilistic_drafts(make):
    # With sampled drafts the live length depends on the sampled tokens, which
    # would bias the verified distribution.
    drafter = make(draft_sample_method="probabilistic")
    with pytest.raises(ValueError, match="greedy"):
        drafter.load_draft_model(None, set())
    assert state_module.get_row_mask_state() is None


def test_load_turns_the_head_on_and_creates_the_state(make):
    drafter = make()
    assert drafter.use_confidence_head is False
    assert state_module.get_row_mask_state() is None
    drafter.load_draft_model(None, set())
    assert drafter.use_confidence_head is True
    state = state_module.get_row_mask_state()
    assert state is drafter._row_mask
    assert (state.max_num_reqs, state.max_num_tokens, state.max_draft) == (
        MAX_REQS, MAX_TOKENS, MAX_DRAFT
    )
    assert tuple(state.conf_by_slot.shape) == (MAX_REQS, MAX_DRAFT)


def test_load_refuses_a_drafter_without_a_head(make):
    drafter = make(head=lambda: None)
    with pytest.raises(ValueError, match="no confidence head"):
        drafter.load_draft_model(None, set())
    assert state_module.get_row_mask_state() is None


def test_load_checks_the_policy_against_the_verified_drafts(make, monkeypatch, tmp_path):
    # K(1..3) = 5: a batch-1 cost row of k = 0 .. 3 cannot price them.
    short = {"1": {str(k): 1 + k for k in range(4)}, "8": {str(k): 2 + k for k in range(6)}}
    monkeypatch.setattr(
        width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", _policy(tmp_path / "s.json", costs=short)
    )
    drafter = make()
    with pytest.raises(ValueError, match="verifies 5 drafts"):
        drafter.load_draft_model(None, set())
    # Batches 4 .. 8 verify 3: k = 0 .. 3 at batch 8 (and so, interpolated
    # from batch 3, at 4 .. 7) suffices.
    state_module._reset_for_tests()
    fits = {
        "1": {str(k): 1 + k for k in range(6)},
        "3": {str(k): 1.5 + k for k in range(6)},
        "8": {str(k): 2 + k for k in range(4)},
    }
    monkeypatch.setattr(
        width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", _policy(tmp_path / "f.json", costs=fits)
    )
    make().load_draft_model(None, set())
    cost = state_module.get_row_mask_state().cost
    assert cost[1].tolist() == [1, 2, 3, 4, 5, 6]
    assert cost[2].tolist() == [1.25 + k for k in range(6)]
    # Batch 4: 1/5 of the way from batch 3 to 8 on k = 0 .. 3; nothing past.
    assert cost[4, :4].tolist() == pytest.approx([1.6 + k for k in range(4)])
    assert bool(cost[4, 4:].isnan().all())


@pytest.mark.parametrize(
    "changes, match",
    [
        (dict(schema="dspark-row-mask-costs/1"), "schema"),
        (dict(max_depth=4), "max_depth"),
        (dict(features=["bias", "conf"]), "features"),
        (dict(weights=[[0.0, 1.0]] * 4), "weights needs"),
        (dict(weights=[[0.0]] * MAX_DRAFT), "weights row 0"),
        (dict(features=["bias", "hist_ema"]), "features"),
        (dict(tau={"2": 1.5}), "tau"),
        (dict(tau=[0.5]), "tau"),
        (dict(costs={"1": {"0": 1, "2": 2}}), "not 0 .. m"),
        (dict(costs={"1": {"0": 2, "1": 1}}), "nondecreasing"),
    ],
)
def test_malformed_policies_are_refused(tmp_path, changes, match):
    path = _policy(tmp_path / "bad.json", **changes)
    with pytest.raises(ValueError, match=match):
        state_module.load_policy(path, MAX_REQS, MAX_DRAFT, [0] + [3] * MAX_REQS)


def test_tau_is_interpolated_from_batch_two(tmp_path):
    path = _policy(tmp_path / "tau.json", tau={"1": 0.9, "2": 0.2, "6": 0.6})
    policy = state_module.load_policy(path, MAX_REQS, MAX_DRAFT, [0] + [3] * MAX_REQS)
    # b = 1 always takes the cost rule; 3 .. 5 between 2 and 6; past 6, 6's.
    assert policy.tau.tolist() == pytest.approx([0, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6])
    path = _policy(tmp_path / "none.json")
    assert state_module.load_policy(path, MAX_REQS, MAX_DRAFT, [0] + [3] * MAX_REQS).tau.tolist() == [0.0] * 9


def test_policy_weights_land_in_the_kernel_columns(tmp_path):
    path = _policy(
        tmp_path / "w.json",
        max_depth=7,
        features=["logit_top1", "logit_conf", "bias", "ent"],
        weights=[[i, 1.0, -i, 0.5] for i in range(7)],
    )
    policy = state_module.load_policy(path, MAX_REQS, MAX_DRAFT, [0] + [3] * MAX_REQS)
    cols = {name: state_module.FEATURES.index(name) for name in state_module.FEATURES}
    assert policy.weights.shape == (MAX_DRAFT, len(state_module.FEATURES))
    assert policy.weights[:, cols["logit_top1"]].tolist() == [0, 1, 2, 3, 4]
    assert policy.weights[:, cols["bias"]].tolist() == [0, -1, -2, -3, -4]
    assert policy.weights[:, cols["ent"]].tolist() == [0.5] * 5
    assert policy.weights[:, cols["logit_conf"]].tolist() == [1.0] * 5
    assert policy.uses_stats


def test_draft_logit_features_keep_statistics_and_refuse_draft_topk(make, monkeypatch, tmp_path):
    path = _policy(tmp_path / "t.json", features=["bias", "logit_top1"])
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", path)
    # With the per-rank argmax sampler too: its exchange carries the statistics.
    for shard_argmax in (False, True):
        state_module._reset_for_tests()
        monkeypatch.setattr(
            width_module.envs, "VLLM_SM80_SHARD_ARGMAX", shard_argmax
        )
        drafter = make()
        drafter._draft_topk = None
        drafter.load_draft_model(None, set())
        assert tuple(drafter._row_stats.shape) == (MAX_REQS, 2, MAX_DRAFT)
    state_module._reset_for_tests()
    drafter = make()
    drafter._draft_topk = 32
    with pytest.raises(ValueError, match="top-k"):
        drafter.load_draft_model(None, set())
    # Without them no statistics are kept.
    state_module._reset_for_tests()
    monkeypatch.setattr(width_module.envs, "VLLM_DSPARK_ADAPTIVE_DEPTH", _policy(tmp_path / "c.json"))
    drafter = make()
    drafter.load_draft_model(None, set())
    assert drafter._row_stats is None

def _propose(drafter, slots, num_speculative_tokens):
    batch = SimpleNamespace(
        num_reqs=len(slots), idx_mapping=torch.tensor(slots, dtype=torch.int64)
    )
    t = torch.ones(len(slots), dtype=torch.int32)
    return drafter.propose(
        batch, {}, {}, None, None, t, t, None, None, None, None,
        num_speculative_tokens=num_speculative_tokens,
    )


def test_a_step_that_drafts_nothing_resets_its_slots(make, monkeypatch):
    calls = []
    monkeypatch.setattr(
        upstream.DSparkSpeculator,
        "propose",
        lambda self, batch, *a, **k: calls.append(batch.num_reqs) or "drafts",
        raising=False,
    )
    drafter = make()
    drafter.load_draft_model(None, set())
    conf = drafter._row_mask.conf_by_slot
    stale = torch.arange(MAX_REQS * MAX_DRAFT, dtype=torch.float32).view(
        MAX_REQS, MAX_DRAFT
    ) / 100
    conf.copy_(stale)
    # A drafting step leaves the publish to the draft step itself.
    assert _propose(drafter, [6, 2], 3) == "drafts"
    assert _propose(drafter, [6, 2], None) == "drafts"
    assert torch.equal(conf, stale)
    # A zero-draft step resets its slots, under data parallelism too: propose
    # skips the draft step on every rank (the batch sync agrees on it).
    want = stale.clone()
    want[[6, 2]] = 1.0
    for dp_size in (2, 1):
        drafter.dp_size = dp_size
        conf.copy_(stale)
        assert _propose(drafter, [6, 2], 0) == "drafts"
        assert torch.equal(conf, want)
    # The width passed positionally.
    t = torch.ones(1, dtype=torch.int32)
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([4]))
    assert drafter.propose(
        batch, {}, {}, None, None, t, t, None, None, None, None,
        None, False, False, None, False, 0,
    ) == "drafts"
    want[4] = 1.0
    assert torch.equal(conf, want)
    assert calls == [2, 2, 2, 2, 1]


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("num_speculative_tokens", [0, 3, None])
def test_drafts_nothing_matches_the_zero_draft_return(dp_size, num_speculative_tokens):
    # DFlashSpeculator.propose returns before the draft step exactly when the
    # step schedules zero drafts, on every data-parallel rank.
    spec = SimpleNamespace(dp_size=dp_size)
    assert conf_module.drafts_nothing(spec, num_speculative_tokens) == (
        num_speculative_tokens == 0
    )


def kernel_cases() -> dict:
    """Run in the Triton interpreter; returns plain lists for the asserts."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    cpu = torch.device("cpu")
    out: dict = {}

    def sample(self, num_reqs, head_hidden):
        # Stands for the head: (requests, width) block of the current view;
        # with statistics, the sampler's per-position writes.
        width = self.num_speculative_steps
        self.draft_tokens[:num_reqs, :width] = self.next_tokens[:num_reqs, :width]
        self.draft_token_confidence_probs[:num_reqs] = self.next_conf[:num_reqs, :width]
        if self._row_stats is not None:
            self._row_stats[:num_reqs, :, :width] = self.next_stats[:num_reqs, :, :width]

    upstream.DSparkSpeculator._sample_sequential = sample

    def drafter(base):
        cls = conf_module.row_mask_class(base)
        d = cls.__new__(cls)
        d._row_mask = state_module.RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, cpu)
        full = torch.full((MAX_REQS, MAX_DRAFT), -1.0)
        d.draft_token_confidence_probs = full
        d.draft_tokens = torch.zeros(MAX_REQS, MAX_DRAFT, dtype=torch.int64)
        d.max_width = MAX_DRAFT
        d.sample_idx_mapping = torch.full((MAX_REQS * MAX_DRAFT,), -1, dtype=torch.int64)
        if base is width_module.PerKDSparkSpeculator:
            d._confidence_by_width = {w: full[:, :w] for w in (3, MAX_DRAFT)}
        return d

    def step(d, slots, width, padded):
        # prepare_dflash_inputs' layout: per request `width` sample rows
        # holding its slot, -1 from the batch's end to max_num_reqs.
        d.width = d.num_speculative_steps = width
        d.sample_idx_mapping.fill_(-1)
        for r, s in enumerate(slots):
            d.sample_idx_mapping[r * width : (r + 1) * width] = s
        g = torch.Generator().manual_seed(len(slots) * 10 + width)
        d.next_conf = torch.rand(MAX_REQS, MAX_DRAFT, generator=g) + 0.5
        d.next_conf[len(slots):] = 9.0  # what padding rows of the graph compute
        d.next_tokens = torch.randint(0, 1000, (MAX_REQS, MAX_DRAFT), generator=g)
        d.next_stats = torch.rand(MAX_REQS, 2, MAX_DRAFT, generator=g) + 0.5
        d.next_stats[len(slots):] = 9.0
        d._sample_sequential(padded, None)
        assert d.draft_token_confidence_probs.shape == (MAX_REQS, MAX_DRAFT)
        return {
            "conf_by_slot": d._row_mask.conf_by_slot.tolist(),
            "stats_by_slot": d._row_mask.stats_by_slot.tolist(),
            "stats": d.next_stats[:, :, :width].tolist(),
            "head": d.next_conf[:, :width].tolist(),
            "drafts": d.draft_tokens[:padded, :width].tolist(),
            "want_drafts": d.next_tokens[:padded, :width].tolist(),
        }

    plain = drafter(upstream.DSparkSpeculator)
    plain._row_mask.conf_by_slot[4] = 0.25  # a slot outside the batch
    out["plain"] = step(plain, [6, 1, 3], MAX_DRAFT, padded=4)

    per_k = drafter(width_module.PerKDSparkSpeculator)
    out["per_k_wide"] = step(per_k, [0, 5, 2, 7], MAX_DRAFT, padded=4)
    out["per_k_narrow"] = step(per_k, [5, 2], 3, padded=4)

    stats = drafter(width_module.PerKDSparkSpeculator)
    stats._row_stats = torch.full((MAX_REQS, 2, MAX_DRAFT), -1.0)
    out["stats_wide"] = step(stats, [3, 0], MAX_DRAFT, padded=3)
    out["stats_narrow"] = step(stats, [0, 6], 3, padded=3)
    return out


@pytest.fixture(scope="module")
def kernels():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = (
        "import json; from tests.v1.spec_decode.test_dspark_row_mask_conf "
        "import kernel_cases as f; print('RESULT ' + json.dumps(f()))"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_each_request_publishes_at_its_slot(kernels):
    case = kernels["plain"]
    conf, head = case["conf_by_slot"], case["head"]
    for r, slot in enumerate([6, 1, 3]):
        assert conf[slot] == pytest.approx(head[r])


def test_padding_requests_write_no_slot(kernels):
    conf = kernels["plain"]["conf_by_slot"]
    for slot in (0, 2, 5, 7):
        assert conf[slot] == [1.0] * MAX_DRAFT
    assert conf[4] == [0.25] * MAX_DRAFT
    assert all(v != 9.0 for row in conf for v in row)


def test_a_narrower_width_zero_fills_the_rest(kernels):
    wide, narrow = kernels["per_k_wide"], kernels["per_k_narrow"]
    for r, slot in enumerate([0, 5, 2, 7]):
        assert wide["conf_by_slot"][slot] == pytest.approx(wide["head"][r])
    conf = narrow["conf_by_slot"]
    for r, slot in enumerate([5, 2]):
        assert conf[slot] == pytest.approx(narrow["head"][r] + [0.0, 0.0])
    # Slots outside the narrow batch keep the wide step's rows.
    for slot, r in ((0, 0), (7, 3)):
        assert conf[slot] == pytest.approx(wide["head"][r])


def test_drafted_tokens_are_unchanged(kernels):
    for case in kernels.values():
        assert case["drafts"] == case["want_drafts"]


def test_statistics_publish_at_the_slot_and_zero_fill(kernels):
    wide, narrow = kernels["stats_wide"], kernels["stats_narrow"]
    for r, slot in enumerate([3, 0]):
        assert wide["stats_by_slot"][slot][0] == pytest.approx(wide["stats"][r][0])
        assert wide["stats_by_slot"][slot][1] == pytest.approx(wide["stats"][r][1])
    for r, slot in enumerate([0, 6]):
        for j in (0, 1):
            assert narrow["stats_by_slot"][slot][j] == pytest.approx(
                narrow["stats"][r][j] + [0.0, 0.0])
    # Slot 3 keeps the wide step's row; padding and untouched slots stay 0.
    assert narrow["stats_by_slot"][3][0] == pytest.approx(wide["stats"][0][0])
    for slot in (1, 2, 4, 5, 7):
        assert narrow["stats_by_slot"][slot] == [[0.0] * MAX_DRAFT] * 2
    # Without statistics nothing is written.
    assert kernels["plain"]["stats_by_slot"] == [[[0.0] * MAX_DRAFT] * 2] * MAX_REQS
