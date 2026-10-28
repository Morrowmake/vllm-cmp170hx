# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_SM80_SHARD_ARGMAX`` across models, on CPU.

- The switch is off by default and logs one startup line on the last
  pipeline stage, naming how the drafter takes its greedy drafts.
- ``shard_argmax.drafting`` per drafter kind: the DSpark Markov sampler (also
  through a subclass that overrides it) and a greedy ``sample_draft``
  drafter (MTP) are served or refused with their reason; a candidate drafter
  (DFlash2), probabilistic drafting and a drafter without a draft head have
  nothing to serve.
- The fork runner's ``load_model`` ends in ``target_argmax.load``: with the
  switch on it refuses at load and adopts a greedy ``sample_draft`` drafter,
  whose drafts then come from the draft model's ``compute_logits_local``
  through ``shard_tokens``; probabilistic and candidate drafters are left
  alone, and the switch off changes nothing.
- ``shard_tokens`` (Triton interpreter, one rank) is torch.argmax's over the
  row cut at the vocabulary, with ties, NaN, signed zeros and junk in the
  padding columns, in FP32 and BF16.
- The target classes the switch reaches (DeepSeek-V4.1, GLM-5.3, Qwen4Exp on
  sm_80) put ``ShardArgmaxLogits`` before their upstream class; the Qwen4Exp
  classes add this rank's vocabulary shard of the logits and describe
  themselves as the upstream classes they replace.

    python -m pytest -q tests/sm80/test_shard_argmax.py
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SWITCH = "VLLM_SM80_SHARD_ARGMAX"


def test_switch_is_off_by_default(monkeypatch):
    from vllm_sm80.envs import FORK_ENV_VARS

    monkeypatch.delenv(SWITCH, raising=False)
    assert FORK_ENV_VARS[SWITCH]() is False


def _draft_model(vocab=248320, tp_size=2):
    calls = []
    model = SimpleNamespace(
        compute_logits_local=lambda h, step=0: calls.append(step) or h,
        lm_head=SimpleNamespace(tp_rank=0, tp_size=tp_size),
        logits_processor=SimpleNamespace(org_vocab_size=vocab),
        calls=calls,
    )
    return model


def _mtp(**overrides):
    """vLLM's MTP speculator, greedy, over a draft model with the vocabulary
    shard logits."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    s = object.__new__(MTPSpeculator)
    s.draft_logits = None
    s.use_local_argmax_reduction = False
    s.acceptance_estimator = None
    s.model = _draft_model()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _runner(**overrides):
    """A runner whose target and head the switch serves."""
    lm = SimpleNamespace(
        lm_head=SimpleNamespace(tp_rank=0, tp_size=2),
        logits_processor=SimpleNamespace(org_vocab_size=248320, logits_as_input=False),
    )
    model = SimpleNamespace(compute_logits_local=lambda h: h, language_model=lm,
                            target_argmax_logits=True)
    fields = dict(model=model, lora_config=None, batch_sharder=None,
                  target_argmax_steps=True, speculator=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_startup_line_names_the_switch_and_the_drafts(monkeypatch):
    from vllm_sm80.worker import target_argmax as ta

    lines = []
    monkeypatch.setattr(ta.logger, "info_once", lambda msg, *a: lines.append(msg % a))
    runner = _runner(speculator=_mtp())
    ta.set_enabled(runner, True)
    assert runner._target_shard_argmax is True
    assert lines == [
        f"{SWITCH}=1: greedy target sampling takes a per-rank argmax, no "
        "all-gather of the target logits when every request of a step is "
        "greedy; greedy drafts: MTPSpeculator samples per rank (sample_draft)."
    ]
    # A pipeline stage before the last samples nothing: no line.
    lines.clear()
    ta.set_enabled(_runner(is_last_pp_rank=False), True)
    assert lines == []
    # Off: no line, nothing refused.
    ta.set_enabled(_runner(model=SimpleNamespace()), False)
    assert lines == []


def _dspark_classes():
    """A DSpark-like speculator whose Markov sampler carries the decorator,
    and a subclass that overrides it and calls it (as the per-width one
    does)."""
    from vllm_sm80.spec_decode import shard_argmax as sa

    class Markov:
        def __init__(self, **overrides):
            self.model = SimpleNamespace(
                compute_logits_local=lambda h: h,
                draft_id_to_target_id=None,
                logits_processor=SimpleNamespace(org_vocab_size=129280),
            )
            self.draft_logits = None
            self._draft_topk = None
            self.acceptance_estimator = None
            for k, v in overrides.items():
                setattr(self, k, v)

        @sa.shard_argmax_sampling
        def _sample_sequential(self, num_reqs, head_hidden):
            pass

        def sample_draft(self, *args):
            raise AssertionError("never sampled here")

    class PerWidth(Markov):
        def _sample_sequential(self, num_reqs, head_hidden):
            super()._sample_sequential(num_reqs, head_hidden)

    return Markov, PerWidth


class _DFlash2Like:
    candidate_sampler = object()
    draft_logits = None

    def sample_draft(self, *args):
        raise AssertionError("never sampled here")


def test_drafting_per_drafter_kind():
    from vllm_sm80.spec_decode import shard_argmax as sa

    Markov, PerWidth = _dspark_classes()
    for cls in (Markov, PerWidth):
        assert sa.drafting(cls()) == ("the DSpark Markov sampler's per-rank argmax", None)
        assert "top-k" in sa.drafting(cls(_draft_topk=8))[1]
    assert sa.drafting(None) == ("no drafter", None)
    drafts, reason = sa.drafting(_DFlash2Like())
    assert reason is None and "top-k candidates" in drafts
    drafts, reason = sa.drafting(SimpleNamespace(draft_logits=None))
    assert reason is None and "no draft head" in drafts
    drafts, reason = sa.drafting(_mtp(draft_logits=torch.zeros(1)))
    assert reason is None and "samples its drafts" in drafts
    assert sa.drafting(_mtp()) == ("MTPSpeculator samples per rank (sample_draft)", None)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(use_local_argmax_reduction=True), "use_local_argmax_reduction"),
        (dict(acceptance_estimator=object()), "acceptance estimator"),
        (dict(model=SimpleNamespace()), "compute_logits_local"),
        (dict(model=SimpleNamespace(compute_logits_local=None, draft_id_to_target_id=object())),
         "reduced draft vocabulary"),
        (dict(model=SimpleNamespace(compute_logits_local=None)), "vocabulary-parallel"),
        (dict(model=_draft_model(vocab=(1 << 24) + 1)), "vocabulary above"),
    ],
)
def test_sample_draft_refusals(overrides, reason):
    from vllm_sm80.spec_decode import shard_argmax as sa
    from vllm_sm80.worker import target_argmax as ta

    s = _mtp(**overrides)
    assert reason in sa.drafting(s)[1]
    assert reason in ta.refusal(_runner(speculator=s))
    with pytest.raises(ValueError, match=f"{SWITCH}: .*{reason}"):
        sa.adopt_sample_draft(s)


def test_sample_draft_refuses_its_own_draft_logits():
    from vllm_sm80.spec_decode import shard_argmax as sa
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    class OwnLogits(MTPSpeculator):
        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return hidden_states

    s = object.__new__(OwnLogits)
    s.__dict__.update(vars(_mtp()))
    assert "its own way" in sa.drafting(s)[1]
    other = SimpleNamespace(draft_logits=None, sample_draft=lambda *a: None)
    assert "outside vLLM's draft-model speculator" in sa.drafting(other)[1]


def test_runner_refuses_a_drafter_it_cannot_serve():
    from vllm_sm80.worker import target_argmax as ta

    Markov, _ = _dspark_classes()
    assert ta.refusal(_runner(speculator=Markov())) is None
    runner = _runner(speculator=Markov(acceptance_estimator=object()))
    assert "acceptance estimator" in ta.refusal(runner)
    with pytest.raises(ValueError, match=f"{SWITCH}: .*acceptance estimator"):
        ta.set_enabled(runner, True)


def test_load_adopts_a_greedy_sample_draft_drafter(monkeypatch):
    from vllm_sm80.spec_decode import shard_argmax as sa
    from vllm_sm80.worker import target_argmax as ta

    seen = []
    monkeypatch.setattr(sa, "shard_tokens",
                        lambda local, rank, tp, vocab: seen.append((local, rank, tp, vocab)) or "tokens")
    monkeypatch.setenv(SWITCH, "1")
    s = _mtp()
    runner = _runner(speculator=s)
    ta.load(runner)
    assert runner._target_shard_argmax is True
    h = torch.zeros(3, 4)
    args = (h, None, None, None, None, None, None)
    assert s.sample_draft(*args, spec_step_idx=2) == "tokens"
    assert s.sample_draft(*args) == "tokens"
    assert s.model.calls == [2, 0]
    assert seen == [(h, 0, 2, 248320)] * 2
    # Probabilistic and candidate drafters are left alone.
    for other in (_mtp(draft_logits=torch.zeros(1)), _DFlash2Like()):
        before = other.sample_draft
        ta.load(_runner(speculator=other))
        assert other.sample_draft == before
    # Switch off: nothing happens, not even a refusal.
    monkeypatch.setenv(SWITCH, "0")
    s = _mtp(use_local_argmax_reduction=True)
    runner = _runner(speculator=s)
    ta.load(runner)
    assert "sample_draft" not in vars(s) and not hasattr(runner, "_target_shard_argmax")
    # A refused drafter refuses at load.
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match=f"{SWITCH}: .*use_local_argmax_reduction"):
        ta.load(_runner(speculator=_mtp(use_local_argmax_reduction=True)))


def test_fork_runner_loads_through_target_argmax(monkeypatch):
    from vllm_sm80.worker import graph_mem_worker
    from vllm_sm80.worker import target_argmax as ta
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    seen = []
    monkeypatch.setattr(GPUModelRunner, "load_model",
                        lambda self, *a, **kw: seen.append(("load", a, kw)))
    monkeypatch.setattr(ta, "load", lambda r: seen.append(("shard", r)))
    cls = graph_mem_worker.runner_cls()
    runner = cls.__new__(cls)
    runner.load_model(True, model=None)
    assert seen == [("load", (True,), {"model": None}), ("shard", runner)]


def _shard_tokens_case() -> dict:
    """In the Triton interpreter: shard_tokens over one rank's rows of a
    padded head (junk in the padding), small chunks, FP32 and BF16."""
    from vllm_sm80.spec_decode import shard_argmax as sa

    sa.BLOCK = 8
    vocab, width = 97, 120
    nan, inf = float("nan"), float("inf")
    out = []
    for dtype in (torch.float32, torch.bfloat16):
        for seed in range(4):
            g = torch.Generator().manual_seed(seed)
            rows = torch.randint(-3, 4, (6, vocab), generator=g).float()
            if seed == 1:
                rows[torch.rand(rows.shape, generator=g) < 0.3] = -inf
            if seed == 2:
                rows = rows.clamp(max=0) * 0.0 * torch.where(
                    torch.rand(rows.shape, generator=g) < 0.5, -1.0, 1.0)
            if seed == 3:
                rows[torch.rand(rows.shape, generator=g) < 0.03] = nan
            junk = torch.tensor([nan, inf]).repeat(width)[: width - vocab]
            local = torch.cat([rows, junk.expand(6, -1)], 1).to(dtype)
            want = torch.argmax(local[:, :vocab].float(), -1).tolist()
            got = sa.shard_tokens(local, 0, 1, vocab).tolist()
            out.append({"want": want, "got": got})
    assert sa.shard_tokens(torch.zeros(0, width), 0, 1, vocab).shape == (0,)
    return out


def test_shard_tokens_is_torch_argmax():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.sm80.test_shard_argmax import _shard_tokens_case as f; "
            "print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    cases = json.loads(line[len("RESULT "):])
    assert len(cases) == 8
    for case in cases:
        assert case["got"] == case["want"]


def _load(qualified: str) -> type:
    module_name, _, class_name = qualified.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV41ForCausalLM", "Glm5NextForCausalLM",
     "Glm5NextForConditionalGeneration", "Qwen4ExpForConditionalGeneration"],
)
def test_reached_targets_serve_greedy_steps(architecture):
    from vllm_sm80 import plugin
    from vllm_sm80.worker.target_argmax import ShardArgmaxLogits

    cls = _load(plugin.MODEL_OVERRIDES[architecture])
    assert cls.__bases__[0] is ShardArgmaxLogits and cls.__name__ == architecture
    assert cls.compute_logits is ShardArgmaxLogits.compute_logits
    assert cls.target_argmax_logits and hasattr(cls, "compute_logits_local")


@pytest.mark.parametrize(
    "architecture, module",
    [("Qwen4ExpForConditionalGeneration", "model"), ("Qwen4ExpMTP", "mtp")],
)
def test_qwen_classes_match_upstream(architecture, module):
    from vllm.model_executor.models.registry import ModelRegistry, _ModelInfo
    from vllm_sm80 import plugin

    upstream = getattr(importlib.import_module(f"vllm.models.qwen4_exp.nvidia.{module}"),
                       architecture)
    plugin.register_models()
    cls = _load(plugin.MODEL_OVERRIDES[architecture])
    assert ModelRegistry._try_load_model_cls(architecture) is cls
    assert cls.__bases__[-1] is upstream and cls.__name__ == architecture
    assert _ModelInfo.from_model_cls(cls) == _ModelInfo.from_model_cls(upstream)


def test_qwen_classes_shard_the_logits(monkeypatch):
    from vllm_sm80 import plugin
    from vllm_sm80.worker import target_argmax as ta

    calls = []

    def processor(head, h, skip_gather=False):
        calls.append((head, skip_gather))
        return h

    head = object()
    target_cls = _load(plugin.MODEL_OVERRIDES["Qwen4ExpForConditionalGeneration"])
    model = target_cls.__new__(target_cls)
    torch.nn.Module.__init__(model)
    model.language_model = SimpleNamespace(
        lm_head=head, logits_processor=processor,
        compute_logits=lambda h: calls.append(("gather",)) or h,
    )
    h = torch.zeros(2, 4)
    assert model.compute_logits_local(h) is h
    assert calls == [(head, True)]
    # Outside a greedy-only step the gathered logits, inside the argmax rows.
    calls.clear()
    assert model.compute_logits(h) is h
    assert calls == [("gather",)]
    monkeypatch.setattr(ta, "_active", True)
    monkeypatch.setattr(ta, "argmax_rows", lambda m, x: ("rows", m, x))
    assert model.compute_logits(h) == ("rows", model, h)

    mtp_cls = _load(plugin.MODEL_OVERRIDES["Qwen4ExpMTP"])
    mtp = mtp_cls.__new__(mtp_cls)
    torch.nn.Module.__init__(mtp)
    mtp.lm_head, mtp.logits_processor = head, processor
    calls.clear()
    assert mtp.compute_logits_local(h, 1) is h
    assert calls == [(head, True)]
