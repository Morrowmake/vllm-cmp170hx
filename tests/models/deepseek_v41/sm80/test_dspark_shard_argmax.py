# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-rank greedy argmax of the DSpark Markov sampling
(``vllm_sm80/spec_decode/shard_argmax.py``) on CPU.

- The two Triton kernels run in the Triton interpreter (a subprocess with
  ``TRITON_INTERPRET=1``). Every rank's slots, summed as the all-reduce sums
  them, pick torch.argmax's index of the gathered row on adversarial rows:
  ties across rank and chunk boundaries, NaN of either sign before and after
  +inf, -0.0 against +0.0, subnormals, one-ulp gaps, all -inf, a rank
  without columns, at small chunks and at the served shape (4 ranks of
  32320 columns, 4096-column chunks).
- ``DSparkSpeculator._sample_sequential`` with the switch on and off, in four
  gloo ranks: the same drafts and confidences on every rank, from base logits
  and Markov biases full of ties and NaN, with NaN and +inf in the padding
  columns of the last rank's shard.
- The switch is off by default and refuses the speculators it cannot serve.
- Both kernels compile for sm_80 at the served shape (no GPU needed).

    pytest -q tests/models/deepseek_v41/sm80/test_dspark_shard_argmax.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SWITCH = "VLLM_SM80_SHARD_ARGMAX"


def _sum_slots(per_rank: list[torch.Tensor]) -> torch.Tensor:
    """The all-reduce: FP32 sum rounded once to BF16 (the custom kernels)."""
    return torch.stack([s.float() for s in per_rank]).sum(0).to(torch.bfloat16)


def _sum_slots_bf16(per_rank: list[torch.Tensor]) -> torch.Tensor:
    """The all-reduce accumulating in BF16, rank by rank (a ring in BF16)."""
    acc = per_rank[0].clone()
    for s in per_rank[1:]:
        acc = acc + s
    return acc


def sharded_pick(full: torch.Tensor, tp: int, width: int) -> tuple[torch.Tensor, ...]:
    """The tokens per-rank slots pick for the gathered rows ``full`` [rows, V]
    split into ``tp`` shards of ``width`` columns, under both sums."""
    from vllm_sm80.spec_decode import shard_argmax as sa

    vocab = full.shape[1]
    chunks = -(-width // sa.BLOCK)
    per_rank = []
    for rank in range(tp):
        start = rank * width
        n = min(max(vocab - start, 0), width)
        cols = full[:, start : start + n].contiguous()
        slots = sa.chunk_slots(cols, start, rank, tp, chunks)
        lanes = slots.float()
        assert torch.equal(lanes, lanes.round()) and lanes.min() >= 0 and lanes.max() <= 255
        own = slots.view(full.shape[0], tp, chunks, sa.LANES)
        others = torch.cat([own[:, :rank], own[:, rank + 1 :]], dim=1)
        assert not others.float().any()
        per_rank.append(slots)
    return sa.pick(_sum_slots(per_rank), vocab), sa.pick(_sum_slots_bf16(per_rank), vocab)


def _f32(bits: int) -> float:
    return torch.tensor([bits], dtype=torch.int64).to(torch.int32).view(torch.float32).item()


def named_rows(vocab: int, width: int) -> dict[str, tuple[torch.Tensor, int]]:
    """Adversarial rows with the index ATen's GreaterOrNan argmax gives."""
    nan, inf = float("nan"), float("inf")
    neg_nan = _f32(0xFFC00001 - (1 << 32))
    sub_min, sub_two = _f32(1), _f32(2)
    one_ulp_up = _f32(0x3F800001)
    w, b = width, 8  # shard width and the chunk the subprocess sets
    rows = {}

    def row(fill=-1.0):
        return torch.full((vocab,), fill, dtype=torch.float32)

    r = row(3.0)
    rows["all equal"] = (r, 0)
    r = row(); r[w - 1] = 5.0; r[w] = 5.0
    rows["tie across ranks 0|1"] = (r, w - 1)
    r = row(); r[2 * w + b - 1] = 5.0; r[2 * w + b] = 5.0; r[3 * w + 1] = 5.0
    rows["tie across chunks in rank 2"] = (r, 2 * w + b - 1)
    r = row(); r[3 * w + 2] = 5.0; r[w + 3] = 5.0
    rows["tie, the lower rank wins"] = (r, w + 3)
    r = row(); r[2 * w + 1] = nan; r[w + 5] = nan; r[0] = inf
    rows["first NaN, after +inf"] = (r, w + 5)
    r = row(); r[3] = inf; r[2 * w] = neg_nan
    rows["negative NaN beats +inf"] = (r, 2 * w)
    r = row(); r[w + 2] = -0.0; r[2 * w + 1] = 0.0
    rows["-0.0 ties +0.0 (first wins)"] = (r, w + 2)
    r = row(-inf)
    rows["all -inf"] = (r, 0)
    r = row(-inf); r[3 * w] = -3.0e38
    rows["one finite among -inf"] = (r, 3 * w)
    r = row(); r[w + 1] = inf; r[2 * w + 1] = inf
    rows["+inf ties"] = (r, w + 1)
    r = row(0.0); r[3 * w + 3] = sub_min; r[2] = -sub_min
    rows["smallest subnormal above zero"] = (r, 3 * w + 3)
    r = row(0.0); r[2 * w + 3] = sub_two; r[w] = sub_min
    rows["two subnormals"] = (r, 2 * w + 3)
    r = row(1.0); r[2 * w + 4] = one_ulp_up
    rows["one ulp above"] = (r, 2 * w + 4)
    r = row(-5.0); r[vocab - 1] = -4.0
    rows["last column"] = (r, vocab - 1)
    r = row(-2.0); r[w] = -1.5; r[0] = -1.75
    rows["negatives only"] = (r, w)
    return rows


def kernel_cases() -> dict:
    """Run in the Triton interpreter: picked vs expected tokens per case."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    from vllm_sm80.spec_decode import shard_argmax as sa

    out = {"named": {}, "random": [], "empty_rank": None, "served": None, "unwritten": None}
    served_block = sa.BLOCK
    sa.BLOCK = 8
    tp, width, vocab = 4, 25, 97  # 3 chunks per rank, last rank 22 columns
    for name, (r, want) in named_rows(vocab, width).items():
        got = sharded_pick(r[None], tp, width)
        out["named"][name] = [int(g[0]) for g in got] + [int(torch.argmax(r)), want]
    g = torch.Generator().manual_seed(0)
    for seed in range(40):
        rows = torch.randint(-3, 4, (5, vocab), generator=g).float()
        if seed % 4 == 1:
            rows[torch.rand(rows.shape, generator=g) < 0.05] = float("nan")
        if seed % 4 == 2:
            rows[torch.rand(rows.shape, generator=g) < 0.1] = float("inf")
        if seed % 4 == 3:
            rows = rows * 0.0 * torch.where(torch.rand(rows.shape, generator=g) < 0.5, -1.0, 1.0)
        got = sharded_pick(rows, tp, width)
        out["random"].append([g_.tolist() for g_ in got] + [torch.argmax(rows, -1).tolist()])
    # Rank 3 holds no column of a 70-token vocabulary.
    rows = torch.randint(-2, 3, (3, 70), generator=g).float()
    rows[1] = 2.0
    got = sharded_pick(rows, tp, width)
    out["empty_rank"] = [g_.tolist() for g_ in got] + [torch.argmax(rows, -1).tolist()]
    # The served shape: 4 ranks of 32320 columns, 4096-column chunks.
    sa.BLOCK = served_block
    vocab, width = 129280, 32320
    rows = torch.randint(-2, 3, (3, vocab), generator=g).float()
    rows[:, :] = torch.where(rows > 1, 0.5, rows)
    rows[0, [4095, 4096, 32319, 32320, 129279]] = 7.0
    rows[1, [3 * 32320 + 4096 * 7 + 1, 2 * 32320 + 4096 * 3]] = float("nan")
    rows[2] = -float("inf")
    got = sharded_pick(rows, tp, width)
    out["served"] = [g_.tolist() for g_ in got] + [torch.argmax(rows, -1).tolist()]
    # Unwritten slots (a CUDA-graph warmup's all-reduce output): any bytes,
    # NaN and infinities included, pick a token below the vocabulary.
    chunks = -(-width // sa.BLOCK)
    lanes = torch.randint(0, 256, (6, tp * chunks * sa.LANES), generator=g).float()
    lanes[1] = float("nan")
    lanes[2] = float("inf")
    lanes[3] = -float("inf")
    lanes[4] = 3.0e38
    lanes[5] = 255.0
    out["unwritten"] = sa.pick(lanes.to(torch.bfloat16), vocab).tolist()
    return out


def _rank_main(rank: int, tp: int, init_file: str, result_file: str) -> None:
    """One gloo rank: the speculator's sampling with the switch off and on."""
    import torch.distributed as dist

    from vllm_sm80.spec_decode import shard_argmax as sa
    from vllm_sm80.spec_decode.dspark_width import (
        Sm80DSparkSpeculator as DSparkSpeculator,
    )

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=tp)

    def all_reduce(t: torch.Tensor) -> torch.Tensor:
        f = t.float()
        dist.all_reduce(f)
        return f.to(t.dtype)

    sa.BLOCK = 8
    sa.tensor_model_parallel_all_reduce = all_reduce
    vocab, width, n_spec, num_reqs = 97, 25, 5, 3
    results = []
    for seed in range(12):
        g = torch.Generator().manual_seed(seed)
        rows = num_reqs * n_spec + 4
        base = torch.randint(-2, 3, (rows, vocab), generator=g).float()
        bias = torch.randint(-2, 3, (vocab, vocab), generator=g).float()
        if seed % 3 == 1:
            base[torch.rand(base.shape, generator=g) < 0.03] = float("nan")
            bias[torch.rand(bias.shape, generator=g) < 0.03] = float("-inf")
        if seed % 3 == 2:
            sign = torch.where(torch.rand(base.shape, generator=g) < 0.5, -1.0, 1.0)
            base = base.clamp(max=0) * 0.0 * sign  # signed zeros everywhere
            bias = bias.clamp(max=0) * 0.0
        # Padding columns of the last shard hold NaN and +inf: never read.
        padded = torch.cat(
            [base, torch.tensor([float("nan"), float("inf"), float("nan")]).expand(rows, 3)],
            dim=1,
        )
        assert padded.shape[1] == tp * width

        def lookup(h):
            return h[:, 0].long()

        model = SimpleNamespace(
            lm_head=SimpleNamespace(tp_rank=rank, tp_size=tp),
            logits_processor=SimpleNamespace(org_vocab_size=vocab),
            draft_id_to_target_id=None,
            compute_draft_logits=lambda h: base[lookup(h)],
            compute_logits_local=lambda h: padded[lookup(h), rank * width : (rank + 1) * width],
            markov_embed=lambda t: t[:, None].float(),
            markov_bias=lambda e: bias[e[:, 0].long()],
            map_draft_to_target=lambda ids: ids,
            compute_confidence=lambda h, e: h[:, 0] * 1000.0 + e[:, 0],
        )
        perm = torch.randperm(rows, generator=g)[: num_reqs * n_spec]
        drafts = {}
        for on in (False, True):
            os.environ[SWITCH] = "1" if on else "0"
            s = object.__new__(DSparkSpeculator)
            s.model = model
            s.num_speculative_steps = n_spec
            s.sample_indices = perm.clone()
            s.input_buffers = SimpleNamespace(
                input_ids=torch.randint(0, vocab, (num_reqs * n_spec,), generator=torch.Generator().manual_seed(seed))
            )
            s._anchor_idx = torch.arange(num_reqs) * n_spec
            s.sample_idx_mapping = torch.arange(num_reqs * n_spec)
            s.sample_pos = torch.arange(num_reqs * n_spec) + 1
            s._step_cols = torch.arange(n_spec, dtype=torch.int32)
            s.draft_tokens = torch.zeros(num_reqs, n_spec, dtype=torch.int64)
            s.draft_token_confidence_probs = torch.zeros(num_reqs, n_spec)
            s.draft_logits = None
            s._draft_topk = None
            s.acceptance_estimator = None
            s.use_confidence_head = True
            head_hidden = torch.arange(rows, dtype=torch.float32)[:, None]
            s._sample_sequential(num_reqs, head_hidden)
            assert s._shard_argmax is on
            drafts[on] = [s.draft_tokens.tolist(), s.draft_token_confidence_probs.tolist()]
        results.append(drafts[False] + drafts[True])
    with open(result_file, "w") as f:
        json.dump(results, f)
    dist.destroy_process_group()


def gloo_cases(tmp: str) -> list:
    """Run in the Triton interpreter: four gloo ranks' drafts, off and on."""
    import torch.multiprocessing as mp

    tp = 4
    files = [os.path.join(tmp, f"rank{r}.json") for r in range(tp)]
    init = os.path.join(tmp, "gloo-init")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, tp, init, files[r])) for r in range(tp)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(600)
        assert p.exitcode == 0, p.exitcode
    out = []
    for name in files:
        with open(name) as f:
            out.append(json.load(f))
    return out


def _interpret(code: str) -> object:
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.fixture(scope="module")
def kernels():
    return _interpret(
        "import json; from tests.models.deepseek_v41.sm80.test_dspark_shard_argmax "
        "import kernel_cases as f; print('RESULT ' + json.dumps(f()))"
    )


def test_named_rows_pick_the_gathered_argmax(kernels):
    assert len(kernels["named"]) == 15
    for name, (got, got_bf16, torch_cpu, want) in kernels["named"].items():
        assert got == got_bf16 == want, name
        assert torch_cpu == want, name  # the rule the expectations encode


def test_random_tied_rows_pick_the_gathered_argmax(kernels):
    assert len(kernels["random"]) == 40
    for got, got_bf16, want in kernels["random"]:
        assert got == got_bf16 == want


def test_a_rank_without_columns_is_never_picked(kernels):
    got, got_bf16, want = kernels["empty_rank"]
    assert got == got_bf16 == want
    assert want[1] == 0


def test_served_shape(kernels):
    got, got_bf16, want = kernels["served"]
    assert got == got_bf16 == want
    assert want == [4095, 2 * 32320 + 4096 * 3, 0]


def test_unwritten_slots_pick_a_vocabulary_token(kernels):
    assert len(kernels["unwritten"]) == 6
    assert all(0 <= t < 129280 for t in kernels["unwritten"])
    assert kernels["unwritten"][5] == 129280 - 1  # every index byte 255


def test_sampling_matches_the_gathered_path_on_four_ranks(tmp_path):
    ranks = _interpret(
        "import json; from tests.models.deepseek_v41.sm80.test_dspark_shard_argmax "
        f"import gloo_cases as f; print('RESULT ' + json.dumps(f({str(tmp_path)!r})))"
    )
    assert len(ranks) == 4 and all(len(r) == 12 for r in ranks)
    for rank in ranks:
        assert rank == ranks[0]
        for off_tokens, off_conf, on_tokens, on_conf in rank:
            assert on_tokens == off_tokens
            assert on_conf == off_conf


def test_switch_is_off_by_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(SWITCH, raising=False)
    assert getattr(envs, SWITCH) is False


def _speculator(**overrides):
    model = SimpleNamespace(
        compute_logits_local=lambda h: h,
        draft_id_to_target_id=None,
        logits_processor=SimpleNamespace(org_vocab_size=129280),
    )
    s = SimpleNamespace(model=model, draft_logits=None, _draft_topk=None,
                        acceptance_estimator=None)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(model=SimpleNamespace()), "compute_logits_local"),
        (dict(draft_logits=torch.zeros(1)), "probabilistic"),
        (dict(_draft_topk=8), "top-k"),
        (dict(acceptance_estimator=object()), "acceptance estimator"),
    ],
)
def test_switch_refuses_what_it_cannot_serve(monkeypatch, overrides, reason):
    from vllm_sm80.spec_decode import shard_argmax as sa

    monkeypatch.setenv(SWITCH, "1")
    s = _speculator(**overrides)
    calls = []
    wrapped = sa.shard_argmax_sampling(lambda self, n, h: calls.append(n))
    with pytest.raises(ValueError, match=f"{SWITCH}: .*{reason}"):
        wrapped(s, 1, None)
    assert not calls


def test_switch_off_keeps_the_gathered_path(monkeypatch):
    from vllm_sm80.spec_decode import shard_argmax as sa

    monkeypatch.setenv(SWITCH, "0")
    s = _speculator(draft_logits=torch.zeros(1))  # would be refused if on
    calls = []
    wrapped = sa.shard_argmax_sampling(lambda self, n, h: calls.append(n))
    wrapped(s, 3, None)
    wrapped(s, 2, None)
    assert calls == [3, 2] and s._shard_argmax is False


def test_kernels_compile_for_sm80():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm_sm80.spec_decode import shard_argmax as sa

    target = GPUTarget("cuda", 80, 32)

    def build(kern, types, consts, warps):
        sig = {p: types.get(p, "constexpr") for p in kern.arg_names}
        c = triton.compile(ASTSource(kern, sig, constexprs=consts), target=target,
                           options={"num_warps": warps})
        assert c.asm["cubin"]

    tp, chunks = 4, -(-32320 // sa.BLOCK)
    build(sa._chunk_slots_kernel,
          {"logits_ptr": "*fp32", "logits_stride": "i32", "slots_ptr": "*bf16",
           "row_stride": "i32", "n_cols": "i32", "col_start": "i32", "rank": "i32"},
          dict(TP=tp, TP_P2=4, CHUNKS=chunks, BLOCK=sa.BLOCK), 8)
    build(sa._pick_kernel, {"slots_ptr": "*bf16", "row_stride": "i32", "out_ptr": "*i64",
                            "vocab": "i32"},
          dict(SLOTS=tp * chunks, SLOTS_P2=32), 1)
    build(sa._shard_stats_kernel,
          {"logits_ptr": "*fp32", "logits_stride": "i32", "slots_ptr": "*bf16",
           "row_stride": "i32", "base": "i32", "n_cols": "i32", "rank": "i32"},
          dict(TP=tp, TP_P2=4, CHUNKS=chunks, BLOCK=sa.BLOCK), 8)
    build(sa._combine_stats_kernel,
          {"slots_ptr": "*bf16", "row_stride": "i32", "base": "i32", "out_ptr": "*fp32",
           "out_row_stride": "i32", "out_col_stride": "i32"},
          dict(TP=tp, TP_P2=4), 1)
