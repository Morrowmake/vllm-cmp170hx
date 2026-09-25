# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Split-block Marlin W4A16 MoE prefill at TP4 shards (VLLM_GLM5_TP4_MARLIN_PREFILL).

The split path of VLLM_GLM5_PP_MARLIN_PREFILL with N=512 (all 288 experts
sharded four ways). The helpers (routing model, list-layout model, meta
layer, fp64 reference with the incumbent's rounding points, the incumbent
call) are those of ``test_ampere_pp_marlin_prefill.py``, loaded as a separate
module with N=512 and this flag.

CPU (CUDA_VISIBLE_DEVICES=""): flag defaults; which intermediate sizes each
flag opens; the N=512 gate and the unchanged N=2048 gate; the hook with the
TP4 flag off and on; the TP4 minimum-token threshold; the list layout
contract at the TP4 call sizes (the prefill overlap halves each chunk:
1728/1732 for a 3456-token chunk, 640/642 for a 1282-token tail; 3460
without the overlap).

GPU (sm_80 with ~12 GB free; skipped otherwise): Triton lists equal the model
at the TP4 sizes; flag off is bitwise the incumbent; the split path against
an fp64 reference relative to the incumbent's own error (mean <= 1.10x,
max <= 1.25x) at the TP4 sizes and on small M where lists are empty; bitwise
run to run; CUDA-graph replay equal to eager with no allocation growth; the
gate on the device; captured TP4 prefill records when
VLLM_TP4_MARLIN_CAPTURE_DIR points at a capture (``moe_pre/*.pt`` plus the
``once/`` weights they name).
"""

import glob
import importlib.util
import os
import sys

import pytest
import torch

from vllm.ampere_prefill import moe_split_align as sa
from vllm.ampere_prefill import pp_marlin_prefill as pmp

FLAG = "VLLM_GLM5_TP4_MARLIN_PREFILL"
PP_FLAG = "VLLM_GLM5_PP_MARLIN_PREFILL"


def _helpers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_ampere_pp_marlin_prefill.py")
    src = open(path).read()
    for old, new in (
        ("E, TOPK, K, N, G = 288, 8, 4096, 2048, 128",
         "E, TOPK, K, N, G = 288, 8, 4096, 512, 128"),
        ('FLAG = "VLLM_GLM5_PP_MARLIN_PREFILL"',
         'FLAG = "VLLM_GLM5_TP4_MARLIN_PREFILL"'),
    ):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    name = "_marlin_prefill_helpers_n512"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


hp = _helpers()
E, TOPK, K, N = hp.E, hp.TOPK, hp.K, hp.N
assert N == 512 and hp.FLAG == FLAG
gpu = hp.gpu
weights = hp.weights          # the module fixture, at N=512

# reused as is at N=512 under this flag (the list kernels do not depend on N;
# the w13 row-coverage check runs at the TP4 shard shape)
test_every_routed_row_written_once = hp.test_every_routed_row_written_once
test_empty_list_launch_writes_nothing = hp.test_empty_list_launch_writes_nothing

TP4_M = [640, 642, 1728, 1732, 3456, 3460]


@pytest.fixture(autouse=True)
def _pp_flag_off(monkeypatch):
    monkeypatch.delenv(PP_FLAG, raising=False)
    monkeypatch.delenv(PP_FLAG + "_MIN_TOKENS", raising=False)


# --------------------------------------------------------------------------
# CPU: flags


def test_flag_defaults(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(FLAG + "_MIN_TOKENS", raising=False)
    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL is False
    assert envs.VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS == 384
    src = open(envs.__file__).read()
    assert f"    {FLAG}: bool = False\n" in src
    assert f"    {FLAG}_MIN_TOKENS: int = 384\n" in src


@pytest.mark.parametrize("pp,tp4,want", [
    ("0", "0", {}),
    ("1", "0", {2048: (PP_FLAG, 384)}),
    ("0", "1", {512: (FLAG, 700)}),
    ("1", "1", {2048: (PP_FLAG, 384), 512: (FLAG, 700)}),
])
def test_enabled_shapes(monkeypatch, pp, tp4, want):
    monkeypatch.setenv(PP_FLAG, pp)
    monkeypatch.setenv(FLAG, tp4)
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "700")
    assert pmp.enabled_shapes() == want


# --------------------------------------------------------------------------
# CPU: gate


def test_gate_open_at_n512_only_misses_the_device():
    args = hp._meta_args(M=1728, n=512)
    assert pmp.gate_reason(hp._layer(n=512), **args, allowed_n=(512,)) == "not sm_80"
    assert pmp.gate_reason(hp._layer(n=512), **args, allowed_n=(2048, 512)) == "not sm_80"


def test_gate_n_mismatch_reasons():
    a512 = hp._meta_args(M=1728, n=512)
    a2048 = hp._meta_args(M=2304, n=2048)
    # PP flag only: the PP reason text is unchanged
    why = pmp.gate_reason(hp._layer(n=512), **a512)
    assert why == "intermediate size N=512 != 2048 (TP-sharded experts)"
    why = pmp.gate_reason(hp._layer(n=512), **a512, allowed_n=(2048,))
    assert why == "intermediate size N=512 != 2048 (TP-sharded experts)"
    # TP4 flag only: whole experts stay on the incumbent
    why = pmp.gate_reason(hp._layer(n=2048), **a2048, allowed_n=(512,))
    assert why is not None and "N=2048" in why
    why = pmp.gate_reason(hp._layer(n=1024), **hp._meta_args(M=1728, n=1024),
                          allowed_n=(2048, 512))
    assert why is not None and "N=1024" in why


@pytest.mark.parametrize("case,expect", [
    ("expert_map", "expert map"), ("fp8_input", "activation quantisation"),
    ("fp16", "activation dtype"), ("router_on_input", "router weight"),
    ("zero_points", "w1_zp is set"),
])
def test_gate_closed_reasons_at_n512(case, expect):
    args = hp._meta_args(M=1728, n=512)
    kw = {}
    if case == "expert_map":
        args["expert_map"] = torch.arange(E, device="meta")
    elif case == "fp8_input":
        kw["input_dtype"] = torch.float8_e4m3fn
    elif case == "fp16":
        args["hidden_states"] = torch.empty(1728, K, dtype=torch.float16, device="meta")
    elif case == "router_on_input":
        args["apply_router_weight_on_input"] = True
    elif case == "zero_points":
        kw["w1_zp"] = torch.empty(1, device="meta")
    why = pmp.gate_reason(hp._layer(n=512, **kw), **args, allowed_n=(512,))
    assert why is not None and expect in why, why


def _apply_args_512(M):
    a = hp._meta_args(M=M, n=512)
    return dict(
        output=torch.empty(M, K, dtype=torch.bfloat16, device="meta"),
        hidden_states=a["hidden_states"], w1=a["w1"], w2=a["w2"],
        topk_weights=a["topk_weights"], topk_ids=a["topk_ids"],
        activation=a["activation"], global_num_experts=E, expert_map=None,
        a1q_scale=None, a2_scale=None,
        workspace13=torch.empty(M * TOPK, K, dtype=torch.bfloat16, device="meta"),
        workspace2=torch.empty(M * TOPK * K, dtype=torch.bfloat16, device="meta"),
        expert_tokens_meta=None, apply_router_weight_on_input=False)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_hook_falls_through_to_fused_marlin_moe(monkeypatch, flag):
    """TP4 flag off (and the PP flag off): the split module is never
    consulted. On with the gate closed (no sm_80 here): consulted, declines,
    and the incumbent call is the same."""
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    monkeypatch.setenv(FLAG, flag)
    consulted, calls = [], []
    real = pmp.maybe_apply

    def spy(*a, **k):
        consulted.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pmp, "maybe_apply", spy)
    monkeypatch.setattr(marlin_moe, "fused_marlin_moe", lambda **k: calls.append(k))
    layer = hp._layer(n=512)
    layer._lora_context = None
    layer.apply(**_apply_args_512(1728))
    assert len(calls) == 1 and calls[0]["expert_map"] is None
    assert len(consulted) == (1 if flag == "1" else 0)


def _maybe_args(M):
    a = _apply_args_512(M)
    for k in ("a1q_scale", "a2_scale", "expert_tokens_meta"):
        a.pop(k)
    return a


def test_below_min_tokens_is_not_gated(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: pytest.fail("gate consulted"))
    assert pmp.maybe_apply(hp._layer(n=512), **_maybe_args(383)) is False


def test_tp4_min_tokens_applies_with_both_flags(monkeypatch):
    """PP threshold 384, TP4 threshold 1000: an N=512 call of 500 rows passes
    the open gate but stays on the incumbent (TP4's own threshold)."""
    monkeypatch.setenv(PP_FLAG, "1")
    monkeypatch.setenv(PP_FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "1000")
    seen = []
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: seen.append(1))
    monkeypatch.setattr(pmp, "run", lambda *a, **k: pytest.fail("split path ran"))
    assert pmp.maybe_apply(hp._layer(n=512), **_maybe_args(500)) is False
    assert seen, "gate consulted (M >= the smaller threshold)"


def test_open_gate_runs_the_split_path(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setattr(pmp, "gate_reason", lambda *a, **k: None)
    monkeypatch.setattr(pmp, "_buffers", lambda *a, **k: {"t_max": 1 << 20})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ran = []
    monkeypatch.setattr(pmp, "run", lambda *a, **k: ran.append(a[2].shape[0]))
    assert pmp.maybe_apply(hp._layer(n=512), **_maybe_args(1728)) is True
    assert ran == [1728]


# --------------------------------------------------------------------------
# CPU: list layout at the TP4 call sizes


@pytest.mark.parametrize("M,seed,invalid", [(640, 1, 0), (642, 2, 3), (1728, 3, 0),
                                            (1732, 4, 5), (3456, 5, 0), (3460, 6, 0)])
def test_split_model_covers_every_row_once(M, seed, invalid):
    _, ids = hp._routing(M, seed, invalid=invalid)
    flat = ids.view(-1).tolist()
    tab = sa.cover_table(M * TOPK).tolist()
    lists = hp.split_align_model(flat, E, tab)
    hp._check_lists(lists, flat, E)
    padded = sum(len(v[0]) for v in lists.values())
    one_size = sum(-(-flat.count(e) // 64) * 64 for e in range(E))
    assert padded <= one_size


def test_cover_table_pads_less_than_its_smallest_block_tp4_budget():
    R = 3460 * TOPK
    tab = sa.cover_table(R)
    for r in range(1, R + 1):
        n = tab[r].tolist()
        rows = sum(c * s for c, s in zip(n, sa.SIZES))
        used = [s for c, s in zip(n, sa.SIZES) if c]
        assert 0 <= rows - r < min(used), (r, n)


# --------------------------------------------------------------------------
# GPU


@gpu
@pytest.mark.parametrize("M", [1, 17, 384] + TP4_M)
def test_triton_lists_equal_model_and_are_deterministic(M):
    _, _, ti = hp._inputs(M, 10 + M)
    if M > 1:
        ti.view(-1)[::97] = -1
    buf = sa.buffers(M * TOPK, E, torch.device("cuda"))
    got = []
    for _ in range(2):
        lists = sa.split_align(ti, E, buf)
        torch.cuda.synchronize()
        got.append([(bs, int(n.item()), s[: int(n.item())].tolist(),
                     e[: int(n.item()) // bs].tolist())
                    for bs, s, e, n in lists])
    assert got[0] == got[1]
    model = hp.split_align_model(ti.view(-1).tolist(), E, buf["tab"].tolist())
    for bs, n, sids, eids in got[0]:
        assert (sids, eids) == model[bs], bs


def _gpu_layer(wd):
    layer = hp._layer(device="cuda", n=512, w1_scale=wd["w1_scale"],
                      w2_scale=wd["w2_scale"])
    layer._lora_context = None
    return layer


@gpu
@pytest.mark.parametrize("M", [640, 1728])
def test_flag_off_is_bitwise_the_incumbent(weights, monkeypatch, M):
    layer = _gpu_layer(weights)
    x, tw, ti = hp._inputs(M, M)
    off = hp._apply(layer, weights, x, tw, ti, monkeypatch, "0")
    inc = hp._incumbent(weights, x, tw, ti, monkeypatch)
    assert torch.equal(off, inc)


@gpu
@pytest.mark.parametrize("M,min_tokens", [(m, 384) for m in [384, 1282] + TP4_M]
                         + [(1, 1), (17, 1), (299, 1)])
def test_split_path_accuracy_vs_fp64(weights, monkeypatch, M, min_tokens):
    layer = _gpu_layer(weights)
    x, tw, ti = hp._inputs(M, 100 + M)
    on = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.isfinite(on).all(), "every token row written"
    again = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.equal(on, again), "bitwise run to run"
    inc = hp._incumbent(weights, x, tw, ti, monkeypatch)
    rows = hp._sample_rows(M)
    ref = hp.reference_fp64(x, weights, tw, ti, rows)
    hp._check_accuracy(on[rows], inc[rows], ref, f"M={M}")


@gpu
def test_graph_replay_equals_eager_without_allocation_growth(weights, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    M = 1728
    layer = _gpu_layer(weights)
    x, tw, ti = hp._inputs(M, 3)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    pmp.warmup(x.device, E, 3460, TOPK)
    eager = hp._apply(layer, weights, x, tw, ti, monkeypatch, "1")
    out = torch.empty_like(x)
    ws13 = torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=x.device)

    def step():
        layer.apply(out, x, weights["w1"], weights["w2"], tw, ti,
                    MoEActivation.SILU, E, None, None, None, ws13, ws2, None,
                    False)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        out.zero_()
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert torch.equal(out, eager)


@gpu
def test_gate_on_device(weights):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    layer = _gpu_layer(weights)
    x, tw, ti = hp._inputs(1728, 5)
    args = dict(hidden_states=x, w1=weights["w1"], w2=weights["w2"],
                topk_weights=tw, topk_ids=ti, activation=MoEActivation.SILU,
                global_num_experts=E, expert_map=None,
                apply_router_weight_on_input=False)
    assert pmp.gate_reason(layer, **args, allowed_n=(512,)) is None
    assert "N=512" in pmp.gate_reason(layer, **args)            # PP flag only
    args["expert_map"] = torch.arange(E, device="cuda", dtype=torch.int32)
    assert "expert map" in pmp.gate_reason(layer, **args, allowed_n=(512,))


def _capture_records():
    root = os.environ.get("VLLM_TP4_MARLIN_CAPTURE_DIR")
    if not root:
        return []
    files = sorted(glob.glob(os.path.join(root, "moe_pre", "*.pt")))
    limit = int(os.environ.get("VLLM_TP4_MARLIN_CAPTURE_LIMIT", "10"))
    return files[:limit]


@gpu
@pytest.mark.skipif(not _capture_records(), reason="VLLM_TP4_MARLIN_CAPTURE_DIR not set")
@pytest.mark.parametrize("path", _capture_records())
def test_captured_prefill_records(monkeypatch, path):
    root = os.environ["VLLM_TP4_MARLIN_CAPTURE_DIR"]
    rec = torch.load(path, map_location="cpu", weights_only=False)
    wrec = torch.load(os.path.join(root, rec["weights"]), map_location="cpu",
                      weights_only=False)
    wd = {k: wrec[k].cuda().contiguous() for k in ("w1", "w2", "w1_scale", "w2_scale")}
    del wrec
    assert wd["w2"].size(1) * 16 == 512, "a TP4 (N=512) capture"
    x = rec["hidden_states"].cuda()
    tw = rec["topk_weights"].cuda().float()
    ti = rec["topk_ids"].cuda().to(torch.int32)
    inc = hp._incumbent(wd, x, tw, ti, monkeypatch)
    assert torch.equal(inc.cpu(), rec["out"]), "replay reproduces the live call"
    on = hp._apply(_gpu_layer(wd), wd, x, tw, ti, monkeypatch, "1")
    again = hp._apply(_gpu_layer(wd), wd, x, tw, ti, monkeypatch, "1")
    assert torch.equal(on, again), "bitwise run to run"
    rows = hp._sample_rows(x.size(0), n=64, seed=1)
    ref = hp.reference_fp64(x, wd, tw, ti, rows)
    hp._check_accuracy(on[rows], inc[rows], ref, os.path.basename(path))
    del wd
    torch.cuda.empty_cache()
