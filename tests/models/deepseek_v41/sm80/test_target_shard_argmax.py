# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy target sampling with a per-rank argmax
(``vllm_sm80/worker/target_argmax.py``) on CPU.

- The fork runner's ``sample`` with the switch off and on, in four gloo
  ranks, with the sm_80 target's ``compute_logits`` over a stand-in head and
  the stock ``Sampler`` and ``RejectionSampler`` (their Triton kernels and
  the slot kernels in the Triton interpreter): on a greedy-only step the on
  arm never computes the gathered logits,
  hands the samplers rows holding only torch.argmax of the gathered row (random rows full of ties, ties across ranks and chunks, -inf,
  signed zeros, NaN, NaN and +inf in the padding columns, FP32 and BF16
  heads), and samples, accepts and rejects the same tokens as the off arm on
  every row without NaN; with NaN it follows torch.argmax.
- A step that is not greedy-only takes the gather: a sampled request,
  logprobs, penalties, logit bias, allowed tokens, bad words, a grammar
  bitmask, a NaN count, a sampling mask, a watermarking or custom sampler,
  a custom rejection sampler, draft logits narrower than the vocabulary,
  top-k/top-p in the sampling states (in the four-rank test too, ties
  included: both arms gather), the switch off.
- The seam: the fork runner samples inside ``target_argmax.step``, the sm_80
  target's ``compute_logits`` gathers outside a greedy-only step, the dummy
  sampler run warms the kernels, the switch selects the fork worker and
  refuses other workers and runners; a greedy-only step of a runner with
  batch-sharded sampling samples unsharded and gets its sharder back.
- The switch is off by default and refuses the targets it cannot serve, at
  the first step and at the boot warmup: among them an unsplit head (TP 1)
  and a ``sample_draft`` drafter with vLLM's ``use_local_argmax_reduction``
  set (two per-rank argmaxes).

    python -m pytest -q tests/models/deepseek_v41/sm80/test_target_shard_argmax.py
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

ROOT = Path(__file__).resolve().parents[4]
SWITCH = "VLLM_SM80_SHARD_ARGMAX"
NUM_SPEC = 3


def cpu_buffers() -> None:
    """Let the v2 sampling states live on a host without a GPU: host memory
    in place of the pinned UVA buffers, no FlashInfer probe, and vLLM's
    PyTorch top-k/top-p (the Triton one does not run in the interpreter; the
    GPU test runs it)."""
    from vllm.utils import torch_utils
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.sample import sampler

    class HostBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()

        def uva(self, n=None):
            return self.cpu[:n] if n is not None else self.cpu

    buffer_utils.UvaBuffer = HostBuffer
    torch_utils.is_pin_memory_available = lambda: False
    topk_topp_sampler.HAS_TRITON = False
    sampler.flashinfer_sampler_supported = lambda: False
    sampler.xpu_sampler_supported = lambda: False


def make_sampler(device, params: list, vocab: int, stage: bool = True,
                 filters: list | None = None):
    """The stock Sampler with one request per entry of ``params``, every
    request past its prefill; ``filters`` (top_k, top_p) per request are
    written into the sampling states after ``add_request`` (which takes
    SamplingParams' values: it resets them for a greedy request); ``stage``
    False leaves the device copies of the request states unwritten
    (host-state checks only, no kernel)."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    n = len(params)
    req_states = RequestState(
        max_num_reqs=n, max_model_len=256, max_num_batched_tokens=256,
        num_speculative_steps=NUM_SPEC, vocab_size=vocab, device=device,
    )
    sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=None), max_num_reqs=n,
        vocab_size=vocab, device=device, req_states=req_states,
        num_speculative_tokens=NUM_SPEC + 1,
    )
    for i, p in enumerate(params):
        req_states.prefill_len.np[i] = 2
        sampler.add_request(i, p)
        top_k, top_p = filters[i] if filters else (None, None)
        if top_k is not None:
            sampler.sampling_states.top_k.np[i] = top_k
        if top_p is not None:
            sampler.sampling_states.top_p.np[i] = top_p
    if stage:
        req_states.prefill_len.copy_to_uva()
        sampler.apply_staged_writes()
    return sampler


def make_rejection_sampler(sampler, device):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    config = SimpleNamespace(
        num_speculative_tokens=NUM_SPEC, enable_adaptive_verification=False,
        rejection_sample_method="standard", synthetic_acceptance_rates=None,
        fly_entropy_threshold=0.3, fly_entropy_top_k=3,
    )
    return RejectionSampler(sampler, config, device)


def make_batch(num_reqs: int, per_req: int, input_ids: torch.Tensor, device):
    """A decode batch of ``num_reqs`` requests with ``per_req`` logits rows
    each (1 + drafts), token i of the batch at logits row i."""
    rows = num_reqs * per_req
    local_pos = torch.arange(per_req).repeat(num_reqs)
    positions = 10 + local_pos
    i32 = dict(dtype=torch.int32, device=device)
    cu = np.arange(num_reqs + 1, dtype=np.int32) * per_req
    seq_lens = torch.full((num_reqs,), 10 + per_req, **i32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=num_reqs * (per_req - 1),
        logits_indices=torch.arange(rows, device=device),
        input_ids=input_ids.to(torch.int32).to(device),
        positions=positions.to(device),
        idx_mapping=torch.arange(num_reqs, **i32),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        expanded_idx_mapping=torch.arange(num_reqs).repeat_interleave(per_req).to(**i32),
        expanded_local_pos=local_pos.to(**i32),
        cu_num_logits=torch.from_numpy(cu).to(device),
        cu_num_logits_np=cu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
    )


def greedy_verify(argmax: list[int], input_ids: list[int], per_req: int) -> list[list[int]]:
    """The tokens greedy verification emits per request: the target argmax of
    each row while the next input token (the draft) equals it, then the
    argmax at the first mismatch, or the bonus token."""
    out = []
    for start in range(0, len(argmax), per_req):
        tokens = []
        for j in range(per_req):
            tokens.append(argmax[start + j])
            if j + 1 < per_req and input_ids[start + j + 1] != argmax[start + j]:
                break
        out.append(tokens)
    return out


def _rank_main(rank: int, tp: int, init_file: str, result_file: str,
               device_name: str, vocab: int, width: int, block: int | None) -> None:
    """One gloo rank: the fork runner's ``sample`` with the switch off and
    on, the sm_80 target's ``compute_logits`` on a stand-in head."""
    import torch.distributed as dist

    from tests.models.deepseek_v41.sm80.test_dspark_shard_argmax import named_rows
    from vllm_sm80.models.deepseek_v41.sm80_model import DeepseekV41ForCausalLM
    from vllm.sampling_params import SamplingParams
    from vllm_sm80.spec_decode import shard_argmax as sa
    from vllm_sm80.worker import target_argmax as ta
    from vllm_sm80.worker.graph_mem_worker import runner_cls

    device = torch.device(device_name)
    if device.type == "cpu":
        cpu_buffers()
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=tp)

    def all_reduce(t: torch.Tensor) -> torch.Tensor:
        f = t.float().cpu()
        dist.all_reduce(f)
        return f.to(device=t.device, dtype=t.dtype)

    def all_gather(t: torch.Tensor) -> torch.Tensor:
        parts = [torch.empty(t.shape, dtype=torch.float32) for _ in range(tp)]
        dist.all_gather(parts, t.float().cpu().contiguous())
        return torch.cat(parts, dim=-1).to(device=t.device, dtype=t.dtype)

    if block is not None:
        sa.BLOCK = block
    sa.tensor_model_parallel_all_reduce = all_reduce
    pad = tp * width - vocab
    # The served DeepSeek-V4.1 head has no padding (129280 = 4 x 32320); the
    # small CPU vocabularies pad, with NaN and +inf in the padding columns.
    assert 0 <= pad
    calls: list[str] = []

    def target(full: torch.Tensor) -> DeepseekV41ForCausalLM:
        """The sm_80 target over a stand-in text model whose head on this
        rank is its shard of ``full`` [rows, V], with NaN and +inf in the
        padding columns; the text model's compute_logits gathers the shards
        in rank order and cuts at V, as LogitsProcessor does."""
        junk = torch.tensor([float("nan"), float("inf")]).repeat(pad)[:pad]
        padded = torch.cat([full, junk.to(full.dtype).expand(full.shape[0], pad)], 1)
        padded = padded.to(device)

        def local(h):
            return padded[h[:, 0].long(), rank * width : (rank + 1) * width]

        def compute_logits(h):
            calls.append("gather")
            return all_gather(local(h).contiguous())[:, :vocab]

        def compute_logits_local(h):
            calls.append("local")
            return local(h)

        model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.language_model = SimpleNamespace(
            lm_head=SimpleNamespace(tp_rank=rank, tp_size=tp),
            logits_processor=SimpleNamespace(org_vocab_size=vocab, logits_as_input=False),
            compute_logits=compute_logits,
            compute_logits_local=compute_logits_local,
        )
        return model

    def make_runner(full, filters, per_req):
        sampler = make_sampler(device, [greedy] * len(filters), vocab, filters=filters)
        cls = runner_cls()
        runner = cls.__new__(cls)
        runner.__dict__.update(
            model=target(full),
            sampler=sampler,
            rejection_sampler=(make_rejection_sampler(sampler, device)
                               if per_req > 1 else None),
            speculator=SimpleNamespace(draft_logits=None),
            batch_sharder=None,
            structured_outputs_worker=None,
            lora_config=None,
            vocab_size=vocab,
            is_last_pp_rank=True,
        )
        return runner

    # Greedy requests, plain or (odd batches) with top-k / top-p in the
    # sampler's states: SamplingParams resets them for a greedy request, so
    # they are written here; Sampler.sample would apply them before its
    # argmax, and with ties at the maximum top-p may keep a later one, so a
    # step with any of them takes the gather.
    greedy = SamplingParams(temperature=0.0)
    variants = [(None, None), (None, 0.95), (1, None), (5, 0.5)]

    def run(full: torch.Tensor, input_ids: torch.Tensor, per_req: int, shift: int) -> dict:
        num_reqs = full.shape[0] // per_req
        filters = [variants[(i + shift) % len(variants)] if shift % 2 else (None, None)
                   for i in range(num_reqs)]
        hidden = torch.arange(full.shape[0], dtype=torch.float32, device=device)[:, None]
        gathered_ref = full.float()
        ref = torch.argmax(gathered_ref, dim=-1).tolist()
        nan_rows = torch.isnan(gathered_ref).any(-1).view(num_reqs, per_req)
        out = {"argmax": ref, "per_req": per_req,
               "nan": nan_rows.any(-1).tolist(),
               "verify": greedy_verify(ref, input_ids.tolist(), per_req)}
        for on in (False, True):
            os.environ[ta.SWITCH] = "1" if on else "0"
            runner = make_runner(full, filters, per_req)
            batch = make_batch(num_reqs, per_req, input_ids, device)
            top_k, top_p, _ = runner.sampler.sampling_states.get_top_k_top_p(
                batch.expanded_idx_mapping, batch.idx_mapping_np, full.shape[-1]
            )
            out["filtered"] = top_k is not None or top_p is not None
            calls.clear()
            output, num_sampled, num_rejected = runner.sample(hidden, batch, None)
            assert runner._target_shard_argmax is on and not ta.active()
            shard = on and not out["filtered"]
            assert calls == (["local"] if shard else ["gather"]), calls
            n = num_sampled.tolist()
            sampled = output.sampled_token_ids.cpu().tolist()
            out["on" if on else "off"] = {
                "tokens": [row[:k] for row, k in zip(sampled, n)],
                "num_sampled": n,
                "num_rejected": num_rejected.tolist(),
            }
        # The rows the on arm hands the samplers: 0.0 at the token, -inf
        # elsewhere, in the gathered logits' shape and dtype.
        model = target(full)
        rows = ta.argmax_rows(model, hidden)
        calls.clear()
        gathered = model.compute_logits(hidden)  # outside a step: the gather
        assert calls == ["gather"]
        assert rows.shape == gathered.shape and rows.dtype == gathered.dtype
        finite = torch.isfinite(rows)
        assert torch.equal(finite.sum(-1).cpu(), torch.ones(full.shape[0], dtype=torch.long))
        assert not rows[finite].any() and bool((torch.isneginf(rows) | finite).all())
        out["rows"] = torch.argmax(rows.float(), -1).tolist()
        return out

    results = []
    nan, inf = float("nan"), float("inf")
    # The served target logits are FP32; stock CUDA top-k/top-p (the gather
    # fallback's rows) asserts FP32, so BF16 logits run on CPU only.
    dtypes = (torch.float32,) if str(device).startswith("cuda") else (torch.float32, torch.bfloat16)
    for dtype in dtypes:
        # Adversarial rows, one request each (Sampler).
        rows = named_rows(vocab, width)
        full = torch.stack([r for r, _ in rows.values()]).to(dtype)
        res = run(full, torch.zeros(full.shape[0], dtype=torch.long), 1, 0)
        if dtype == torch.float32:  # BF16 rounds some of the rows' values
            res["want"] = [w for _, w in rows.values()]
        results.append(res)
        for seed in range(6):
            g = torch.Generator().manual_seed(seed)
            # Sampler: one row per request; small integers, so ties
            # everywhere, across ranks and chunks.
            full = torch.randint(-3, 4, (6, vocab), generator=g).float()
            if seed % 3 == 1:
                full[torch.rand(full.shape, generator=g) < 0.2] = -inf
            if seed % 3 == 2:
                full[torch.rand(full.shape, generator=g) < 0.03] = nan
            results.append(run(full.to(dtype), torch.zeros(6, dtype=torch.long), 1, seed))
            # RejectionSampler: 4 requests of 1 + NUM_SPEC rows; drafts follow
            # the argmax until a planted miss (or none: the bonus token).
            per_req = NUM_SPEC + 1
            full = torch.randint(-3, 4, (4 * per_req, vocab), generator=g).float()
            if seed % 3 == 1:
                sign = torch.where(torch.rand(full.shape, generator=g) < 0.5, -1.0, 1.0)
                full = full.clamp(max=0) * 0.0 * sign  # signed zeros everywhere
                full[torch.rand(full.shape, generator=g) < 0.1] = -inf
            if seed % 3 == 2:
                full[torch.rand(full.shape, generator=g) < 0.03] = nan
            ref = torch.argmax(full, -1)
            ids = torch.zeros(full.shape[0], dtype=torch.long)
            for r in range(4):
                miss = (seed + r) % per_req  # 0 = every draft accepted
                for j in range(1, per_req):
                    t = int(ref[r * per_req + j - 1])
                    ids[r * per_req + j] = (t + 1) % vocab if j == miss else t
            results.append(run(full.to(dtype), ids, per_req, seed))
    # No rows: no collective on any rank.
    empty = ta.argmax_rows(target(torch.zeros(0, vocab)), torch.zeros(0, 1, device=device))
    assert empty.shape == (0, vocab)
    # The boot warmup (the dummy sampler run's rows): the shard path once
    # with the switch on, nothing with it off.
    hidden = torch.zeros(3, 1, device=device)
    for on in (False, True):
        os.environ[ta.SWITCH] = "1" if on else "0"
        runner = make_runner(torch.zeros(3, vocab), [variants[0]] * 3, 1)
        calls.clear()
        ta.warm(runner, hidden)
        assert calls == (["local"] if on else []), calls
    with open(result_file, "w") as f:
        json.dump(results, f)
    dist.destroy_process_group()


def gloo_cases(tmp: str, device: str = "cpu", vocab: int = 97, width: int = 25,
               block: int | None = 8) -> list:
    """Four gloo ranks' results (see :func:`_rank_main`)."""
    import torch.multiprocessing as mp

    tp = 4
    files = [os.path.join(tmp, f"rank{r}.json") for r in range(tp)]
    init = os.path.join(tmp, "gloo-init")
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_rank_main,
                    args=(r, tp, init, files[r], device, vocab, width, block))
        for r in range(tp)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(1800)
        assert p.exitcode == 0, p.exitcode
    out = []
    for name in files:
        with open(name) as f:
            out.append(json.load(f))
    return out


def check_ranks(ranks: list) -> None:
    """Every rank samples the same; the on arm is torch.argmax's; the arms
    agree wherever the gathered rows hold no NaN."""
    # 13 cases per logits dtype: FP32 and BF16 on CPU, FP32 only on CUDA.
    assert len(ranks) == 4 and all(len(r) in (13, 26) and len(r) == len(ranks[0]) for r in ranks)
    for rank in ranks:
        assert rank == ranks[0]
    named = compared = with_nan = filtered = 0
    for case in ranks[0]:
        assert case["rows"] == case["argmax"]
        if "want" in case:
            named += 1
            assert case["argmax"] == case["want"]
        on, off = case["on"], case["off"]
        if case["filtered"]:
            # top-k/top-p in the step: both arms gather.
            filtered += 1
            assert on == off
            continue
        assert on["tokens"] == case["verify"]
        assert on["num_sampled"] == [len(t) for t in case["verify"]]
        assert all(a + b == case["per_req"]
                   for a, b in zip(on["num_sampled"], on["num_rejected"]))
        # Request by request: the arms agree wherever the request's gathered
        # rows hold no NaN.
        for r, nan in enumerate(case["nan"]):
            with_nan += nan
            if not nan:
                compared += 1
                assert [arm["tokens"][r] for arm in (on, off)] == [on["tokens"][r]] * 2
                assert off["num_sampled"][r] == on["num_sampled"][r]
                assert off["num_rejected"][r] == on["num_rejected"][r]
    dtypes = len(ranks[0]) // 13
    assert named == 1 and compared >= 25 * dtypes and with_nan >= 5 * dtypes and filtered == 6 * dtypes


def _interpret(code: str) -> object:
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_sample_matches_the_gathered_path_on_four_ranks(tmp_path):
    ranks = _interpret(
        "import json; from tests.models.deepseek_v41.sm80.test_target_shard_argmax "
        f"import gloo_cases as f; print('RESULT ' + json.dumps(f({str(tmp_path)!r})))"
    )
    check_ranks(ranks)


def test_greedy_verify_reference():
    # Request 0 accepts both drafts (bonus token), request 1 misses the first.
    assert greedy_verify([5, 6, 7, 1, 2, 3], [0, 5, 6, 0, 9, 2], 3) == [[5, 6, 7], [1]]


def test_switch_is_off_by_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(SWITCH, raising=False)
    assert getattr(envs, SWITCH) is False


class _Target:
    """A target whose head records which logits were asked for."""

    target_argmax_logits = True

    def __init__(self, vocab=97, tp_size=4):
        self.calls = []
        self.language_model = SimpleNamespace(
            lm_head=SimpleNamespace(tp_rank=0, tp_size=tp_size),
            logits_processor=SimpleNamespace(org_vocab_size=vocab, logits_as_input=False),
        )

    def compute_logits(self, h):
        self.calls.append("gather")
        return h

    def compute_logits_local(self, h):
        self.calls.append("local")
        return h


@pytest.fixture
def host_sampling():
    cpu_buffers()


def _runner(params, sampler_cls=None, rejection_cls=None, **overrides):
    """A runner as the fork V2 runner looks to target_argmax."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    sampler = make_sampler(torch.device("cpu"), params, 97, stage=False)
    if sampler_cls is not None:
        sampler.__class__ = sampler_cls
    rejection = None
    if rejection_cls is not None:
        rejection = make_rejection_sampler(sampler, torch.device("cpu"))
        rejection.__class__ = rejection_cls
    assert isinstance(sampler, Sampler)
    runner = SimpleNamespace(model=_Target(), sampler=sampler,
                             rejection_sampler=rejection, lora_config=None,
                             batch_sharder=None, target_argmax_steps=True,
                             speculator=SimpleNamespace(draft_logits=None))
    for k, v in overrides.items():
        setattr(runner, k, v)
    return runner


def _batch(n):
    return SimpleNamespace(idx_mapping_np=np.arange(n, dtype=np.int32))


def _step_active(runner, n, grammar=None) -> bool:
    from vllm_sm80.worker import target_argmax as ta

    with ta.step(runner, _batch(n), grammar):
        on = ta.active()
    assert not ta.active()
    return on


def _fallback_cases():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    greedy = SamplingParams(temperature=0.0)

    class WatermarkLike(Sampler):
        pass

    class CustomRejection(RejectionSampler):
        pass

    def one(**kw):
        return [greedy, SamplingParams(temperature=0.0, **kw), greedy]

    return {
        "a sampled request": dict(params=[greedy, SamplingParams(temperature=0.7)]),
        "logprobs": dict(params=one(logprobs=2)),
        "presence penalty": dict(params=one(presence_penalty=0.5)),
        "repetition penalty": dict(params=one(repetition_penalty=1.2)),
        "logit bias": dict(params=one(logit_bias={3: 2.0})),
        "allowed tokens": dict(params=one(allowed_token_ids=[1, 2, 3])),
        "bad words": dict(params=one(_bad_words_token_ids=[[4]])),
        "grammar bitmask": dict(params=[greedy], grammar=object()),
        "NaN count": dict(params=[greedy], attrs=dict(compute_nans=True)),
        "sampling mask": dict(params=[greedy], attrs=dict(return_sampling_mask=True)),
        "watermarking sampler": dict(params=[greedy], sampler_cls=WatermarkLike),
        "custom rejection sampler": dict(params=[greedy], rejection_cls=CustomRejection),
        "narrower draft logits": dict(
            params=[greedy],
            runner=dict(speculator=SimpleNamespace(draft_logits=torch.zeros(1, 3, 96))),
        ),
    }


@pytest.mark.parametrize("case", sorted(_fallback_cases()))
def test_steps_that_read_the_logits_take_the_gather(monkeypatch, host_sampling, case):
    spec = _fallback_cases()[case]
    monkeypatch.setenv(SWITCH, "1")
    runner = _runner(spec["params"], spec.get("sampler_cls"), spec.get("rejection_cls"),
                     **spec.get("runner", {}))
    for k, v in spec.get("attrs", {}).items():
        setattr(runner.sampler, k, v)
    assert not _step_active(runner, len(spec["params"]), spec.get("grammar"))
    assert runner._target_shard_argmax is True


def test_a_greedy_only_step_takes_the_shard_argmax(monkeypatch, host_sampling):
    from vllm.sampling_params import SamplingParams

    monkeypatch.setenv(SWITCH, "1")
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0, seed=7),
              SamplingParams(temperature=0.0, top_p=0.95),
              SamplingParams(temperature=0.0, top_k=1)]
    runner = _runner(params)
    assert _step_active(runner, 4)
    # top-k / top-p in the states (SamplingParams resets them for greedy).
    for field, value in (("top_k", 1), ("top_p", 0.95)):
        runner = _runner(params)
        getattr(runner.sampler.sampling_states, field).np[3] = value
        assert not _step_active(runner, 4)
        assert _step_active(runner, 3)
    runner = _runner(params)
    # Draft logits as wide as the target vocabulary do not cut the rows.
    runner.speculator = SimpleNamespace(draft_logits=torch.zeros(1, 3, 97))
    assert _step_active(runner, 4)
    # A greedy step of a server that also has a sampled request elsewhere:
    # only the step's requests count.
    runner = _runner(params + [SamplingParams(temperature=0.7)])
    assert _step_active(runner, 4)
    assert not _step_active(runner, 5)


def test_greedy_requests_carry_no_top_k_top_p():
    """SamplingParams resets top-k/top-p/min-p for a greedy request, so the
    four-rank test writes them into the sampler's states itself."""
    from vllm.sampling_params import SamplingParams

    p = SamplingParams(temperature=0.0, top_p=0.95, top_k=1, min_p=0.1)
    assert (p.top_p, p.top_k, p.min_p) == (1.0, 0, 0.0)


def test_a_greedy_only_step_samples_unsharded(monkeypatch, host_sampling):
    """Batch-sharded sampling: a greedy-only step samples every row from the
    argmax rows (no sharder while it runs); other steps keep the sharder."""
    from vllm.sampling_params import SamplingParams
    from vllm_sm80.worker import target_argmax as ta

    monkeypatch.setenv(SWITCH, "1")
    sharder = object()
    greedy = SamplingParams(temperature=0.0)
    runner = _runner([greedy, greedy], batch_sharder=sharder)
    with ta.step(runner, _batch(2), None):
        assert ta.active() and runner.batch_sharder is None
    assert runner.batch_sharder is sharder
    with pytest.raises(RuntimeError, match="boom"):
        with ta.step(runner, _batch(2), None):
            raise RuntimeError("boom")
    assert runner.batch_sharder is sharder
    runner = _runner([greedy, SamplingParams(temperature=0.7)], batch_sharder=sharder)
    with ta.step(runner, _batch(2), None):
        assert not ta.active() and runner.batch_sharder is sharder
    assert runner.batch_sharder is sharder


def test_step_ends_on_an_error(monkeypatch, host_sampling):
    from vllm.sampling_params import SamplingParams
    from vllm_sm80.worker import target_argmax as ta

    monkeypatch.setenv(SWITCH, "1")
    runner = _runner([SamplingParams(temperature=0.0)])
    with pytest.raises(RuntimeError, match="boom"):
        with ta.step(runner, _batch(1), None):
            assert ta.active()
            raise RuntimeError("boom")
    assert not ta.active()


def test_switch_off_keeps_the_gather(monkeypatch, host_sampling):
    from vllm.sampling_params import SamplingParams

    monkeypatch.setenv(SWITCH, "0")
    runner = _runner([SamplingParams(temperature=0.0)], model=SimpleNamespace())
    assert not _step_active(runner, 1)  # the model would be refused if on
    assert runner._target_shard_argmax is False


def _mtp_like(local: bool):
    """vLLM's MTP speculator, greedy, whose draft model has the vocabulary
    shard logits; ``local``: use_local_argmax_reduction."""
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    s = object.__new__(MTPSpeculator)
    s.draft_logits = None
    s.use_local_argmax_reduction = local
    s.acceptance_estimator = None
    s.model = SimpleNamespace(
        compute_logits_local=lambda h, step=0: h,
        lm_head=SimpleNamespace(tp_rank=0, tp_size=4),
        logits_processor=SimpleNamespace(org_vocab_size=97),
    )
    return s


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(target_argmax_steps=False), "does not sample in greedy steps"),
        (dict(model=SimpleNamespace(compute_logits_local=lambda h: h)), "argmax rows"),
        (dict(model=SimpleNamespace(target_argmax_logits=True)), "compute_logits_local"),
        (dict(model=SimpleNamespace(target_argmax_logits=True,
                                    compute_logits_local=lambda h: h)), "vocabulary-parallel"),
        (dict(lora_config=object()), "LoRA"),
        (dict(model=_Target(tp_size=1)), "TP 1"),
        (dict(speculator=_mtp_like(local=True)), "use_local_argmax_reduction"),
    ],
)
def test_switch_refuses_what_it_cannot_serve(monkeypatch, host_sampling, overrides, reason):
    from vllm.sampling_params import SamplingParams
    from vllm_sm80.worker import target_argmax as ta

    monkeypatch.setenv(SWITCH, "1")
    runner = _runner([SamplingParams(temperature=0.0)], **overrides)
    with pytest.raises(ValueError, match=f"{SWITCH}: .*{reason}"):
        _step_active(runner, 1)
    # The boot warmup refuses it the same way.
    runner = _runner([SamplingParams(temperature=0.0)], **overrides)
    with pytest.raises(ValueError, match=f"{SWITCH}: .*{reason}"):
        ta.warm(runner, torch.zeros(1, 97))


def test_switch_refuses_logits_as_input_and_a_large_vocabulary(monkeypatch):
    from vllm_sm80.worker import target_argmax as ta

    target = _Target()
    runner = SimpleNamespace(model=target, lora_config=None, batch_sharder=object(),
                             target_argmax_steps=True)
    # Batch-sharded sampling is served (a greedy-only step samples unsharded).
    assert ta.refusal(runner) is None
    # A greedy sample_draft drafter whose draft model has the shard logits.
    runner.speculator = _mtp_like(local=False)
    assert ta.refusal(runner) is None
    target.language_model.logits_processor.logits_as_input = True
    assert "logits as input" in ta.refusal(runner)
    target.language_model.logits_processor.logits_as_input = False
    target.language_model.logits_processor.org_vocab_size = (1 << 24) + 1
    assert "vocabulary above" in ta.refusal(runner)
    assert "does not sample in greedy steps" in ta.refusal(None)
    # A pipeline stage before the last samples nothing: nothing to refuse.
    assert ta.refusal(SimpleNamespace(target_argmax_steps=True, is_last_pp_rank=False)) is None


def test_target_gathers_outside_a_greedy_step(monkeypatch):
    """The sm_80 target: the gathered logits outside a greedy-only step
    (prompt logprobs, the dummy sampler run), the argmax rows inside."""
    from vllm_sm80.models.deepseek_v41.sm80_model import DeepseekV41ForCausalLM
    from vllm_sm80.worker import target_argmax as ta

    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.language_model = _Target()
    h = torch.zeros(2, 97)
    assert model.compute_logits(h) is h
    assert model.language_model.calls == ["gather"]
    monkeypatch.setattr(ta, "_active", True)
    monkeypatch.setattr(ta, "argmax_rows", lambda m, x: ("rows", m, x))
    assert model.compute_logits(h) == ("rows", model, h)
    assert model.language_model.calls == ["gather"]
    assert model.target_argmax_logits


def test_fork_runner_samples_in_a_step(monkeypatch):
    """The fork runner: sample() runs upstream's inside target_argmax.step;
    the dummy sampler run warms after upstream's; the captures stay
    upstream's without the estimate split."""
    from vllm_sm80.worker import cudagraph_mem, graph_mem_worker
    from vllm_sm80.worker import target_argmax as ta
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    seen = []
    monkeypatch.setattr(GPUModelRunner, "sample",
                        lambda self, h, b, g: seen.append(("sample", ta.active(), h, b, g)) or "out")
    monkeypatch.setattr(GPUModelRunner, "_dummy_sampler_run",
                        lambda self, h: seen.append(("dummy", h)))
    monkeypatch.setattr(GPUModelRunner, "capture_model",
                        lambda self, *, profile_only=False: seen.append(("capture", profile_only)) or 1)
    monkeypatch.setattr(GPUModelRunner, "profile_cudagraph_memory",
                        lambda self: seen.append(("profile",)) or 2)
    monkeypatch.setattr(ta, "greedy_only", lambda r, b, g: g == "greedy")
    monkeypatch.setattr(ta, "warm", lambda r, h: seen.append(("warm", h)))
    monkeypatch.delenv("VLLM_CUDAGRAPH_ESTIMATE_SPLIT", raising=False)
    cls = graph_mem_worker.runner_cls()
    runner = cls.__new__(cls)
    runner._target_shard_argmax = True
    assert cls.target_argmax_steps
    assert runner.sample("h", "b", "greedy") == "out"
    assert runner.sample("h", "b", None) == "out"
    assert not ta.active()
    runner._dummy_sampler_run("rows")
    assert runner.capture_model(profile_only=True) == 1
    assert runner.profile_cudagraph_memory() == 2
    assert not cudagraph_mem.enabled()
    assert seen == [("sample", True, "h", "b", "greedy"), ("sample", False, "h", "b", None),
                    ("dummy", "rows"), ("warm", "rows"), ("capture", True), ("profile",)]


@pytest.mark.parametrize("estimate", ["0", "1"])
def test_the_switch_selects_the_fork_worker(monkeypatch, estimate):
    from vllm_sm80.worker import cudagraph_mem

    monkeypatch.setenv("VLLM_CUDAGRAPH_ESTIMATE_SPLIT", estimate)
    monkeypatch.setenv(SWITCH, "1")
    parallel = SimpleNamespace(worker_cls=cudagraph_mem.UPSTREAM_WORKER)
    cudagraph_mem.select_worker(parallel)
    assert parallel.worker_cls == cudagraph_mem.WORKER
    # Another worker would never sample in greedy steps: refused at boot.
    with pytest.raises(ValueError, match=f"{SWITCH}: .*fork GPU worker"):
        cudagraph_mem.select_worker(SimpleNamespace(worker_cls="my.Worker"))


def test_the_fork_worker_refuses_a_runner_without_steps(monkeypatch):
    from vllm_sm80.worker import graph_mem_worker

    monkeypatch.setenv(SWITCH, "1")
    worker = graph_mem_worker.GraphMemWorker.__new__(graph_mem_worker.GraphMemWorker)
    worker.use_v2_model_runner = False
    worker.vllm_config = SimpleNamespace(is_mm_encoder_only=False)
    with pytest.raises(ValueError, match=f"{SWITCH}: .*V2 model runner"):
        worker._make_model_runner()
