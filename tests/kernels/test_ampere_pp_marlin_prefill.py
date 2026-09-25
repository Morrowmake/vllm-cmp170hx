# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-block Marlin W4A16 MoE prefill (VLLM_GLM5_PP_MARLIN_PREFILL).

CPU (CUDA_VISIBLE_DEVICES=""): the cover table, the block-list layout contract
on a pure-Python model of the split alignment, the gate and its fall-throughs,
the hook in MarlinExperts.apply with the flag off and on.

GPU (sm_80 with ~12 GB free; skipped otherwise): the Triton lists equal the
Python model and are deterministic; the flag-off path is bitwise the
incumbent; the split path against an fp64 reference relative to the
incumbent's own error (mean <= 1.10x, max <= 1.25x) on M in {1282, 2304, 2312}
and on small M where lists are empty; every routed row written exactly once;
bitwise run-to-run; CUDA-graph replay equal to eager with no allocation
growth; the gate-closed fall-throughs on the device.  Captured prefill
records are used when VLLM_PP_MARLIN_CAPTURE_DIR points at a capture
(``moe_pre/*.pt`` plus the ``once/`` weights they name).
"""

import glob
import os
import random
from types import SimpleNamespace

import pytest
import torch

from vllm.ampere_prefill import moe_split_align as sa
from vllm.ampere_prefill import pp_marlin_prefill as pmp

E, TOPK, K, N, G = 288, 8, 4096, 2048, 128
FLAG = "VLLM_GLM5_PP_MARLIN_PREFILL"


# --------------------------------------------------------------------------
# pure-Python model of the split alignment (the layout contract)


def split_align_model(ids, n_exp, tab):
    """-> {bs: (sorted_ids list, expert_ids list)} for flattened ``ids``."""
    T = len(ids)
    rows = [[] for _ in range(n_exp)]
    for i, e in enumerate(ids):
        if 0 <= e < n_exp:
            rows[e].append(i)
    out = {bs: ([], []) for bs in sa.SIZES}
    for e in range(n_exp):
        counts = tab[len(rows[e])]
        pos = 0
        for k, bs in enumerate(sa.SIZES):
            cnt = int(counts[k]) * bs
            if cnt == 0:
                continue
            got = rows[e][pos:pos + cnt]
            pos += cnt
            out[bs][0].extend(got + [T] * (cnt - len(got)))
            out[bs][1].extend([e] * (cnt // bs))
    return out


def _brute_cost(r):
    best = None
    lim = {s: r // s + 1 for s in sa.SIZES}
    for a in range(lim[64] + 1):
        for b in range(lim[48] + 1):
            for c in range(lim[32] + 1):
                rest = r - 64 * a - 48 * b - 32 * c
                d = max(0, -(-rest // 16))
                cost = (a * sa.COST[64] + b * sa.COST[48] + c * sa.COST[32]
                        + d * sa.COST[16])
                if best is None or cost < best - 1e-9:
                    best = cost
    return best


def _routing(M, seed, n_exp=E, skew=4.0, invalid=0):
    g = torch.Generator().manual_seed(seed)
    pop = torch._standard_gamma(torch.full((n_exp,), skew), generator=g)
    u = torch.rand(M, n_exp, generator=g).clamp_(1e-12, 1 - 1e-7)
    z = torch.log(pop)[None, :] - torch.log(-torch.log(u))
    ids = torch.topk(z, TOPK, dim=-1).indices.to(torch.int32)
    if invalid:
        flat = ids.view(-1)
        flat[torch.randperm(flat.numel(), generator=g)[:invalid]] = -1
    w = torch.rand(M, TOPK, generator=g) * 0.5 + 0.1
    return (w / w.sum(-1, keepdim=True) * 2.5).float(), ids.contiguous()


# --------------------------------------------------------------------------
# CPU: cover table


def test_cover_table_is_cost_optimal_small():
    tab = sa.cover_table(200)
    for r in range(0, 201):
        n = tab[r].tolist()
        rows = sum(c * s for c, s in zip(n, sa.SIZES))
        assert rows >= r
        cost = sum(c * sa.COST[s] for c, s in zip(n, sa.SIZES))
        assert abs(cost - (_brute_cost(r) if r else 0.0)) < 1e-6, r


def test_cover_table_pads_less_than_its_smallest_block():
    R = 2312 * TOPK
    tab = sa.cover_table(R)
    assert tab.shape == (R + 1, len(sa.SIZES)) and tab.dtype == torch.int32
    assert tab[0].tolist() == [0, 0, 0, 0]
    for r in range(1, R + 1):
        n = tab[r].tolist()
        rows = sum(c * s for c, s in zip(n, sa.SIZES))
        used = [s for c, s in zip(n, sa.SIZES) if c]
        assert 0 <= rows - r < min(used), (r, n)


# --------------------------------------------------------------------------
# CPU: list layout contract on the model


def _check_lists(lists, ids, n_exp):
    T = len(ids)
    seen = [0] * T
    for bs, (sids, eids) in lists.items():
        assert len(sids) % bs == 0 and len(eids) == len(sids) // bs
        last_block_of = {}
        for b, e in enumerate(eids):
            blk = sids[b * bs:(b + 1) * bs]
            valid = [i for i in blk if i < T]
            assert valid, "every block holds a valid row"
            assert all(ids[i] == e for i in valid)
            # padding only at the end of a block
            assert blk[:len(valid)] == valid and all(i == T for i in blk[len(valid):])
            if len(valid) < bs:
                assert e not in last_block_of
                last_block_of[e] = b
            for i in valid:
                seen[i] += 1
    for i, e in enumerate(ids):
        assert seen[i] == (1 if 0 <= e < n_exp else 0), i


@pytest.mark.parametrize("M,seed,invalid", [(1, 0, 0), (17, 1, 3), (299, 2, 0),
                                            (384, 3, 0), (1282, 4, 5),
                                            (2304, 5, 0), (2312, 6, 0)])
def test_split_model_covers_every_row_once(M, seed, invalid):
    _, ids = _routing(M, seed, invalid=invalid)
    flat = ids.view(-1).tolist()
    tab = sa.cover_table(M * TOPK).tolist()
    lists = split_align_model(flat, E, tab)
    _check_lists(lists, flat, E)
    padded = sum(len(v[0]) for v in lists.values())
    # the split layout never pads more than one 64-row block per expert would
    one_size = sum(-(-flat.count(e) // 64) * 64 for e in range(E))
    assert padded <= one_size


# --------------------------------------------------------------------------
# CPU: the gate


def _layer(device="meta", n=N, e=E, **over):
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts,
    )

    qc = SimpleNamespace(
        use_int4_w4a16=True, use_int8_w8a16=False, use_mxfp4_w4a16=False,
        use_nvfp4_w4a16=False, use_fp8_w8a16=False,
        w1_scale=torch.empty(e, K // G, 2 * n, dtype=torch.bfloat16, device=device),
        w2_scale=torch.empty(e, n // G, K, dtype=torch.bfloat16, device=device),
        w1_zp=None, w2_zp=None, w1_bias=None, w2_bias=None, g1_alphas=None,
        g2_alphas=None, a1_gscale=None, a2_gscale=None, a2_scale=None)
    for k, v in over.items():
        setattr(qc, k, v)
    layer = MarlinExperts.__new__(MarlinExperts)
    layer.quant_config = qc
    layer.activation_config = over.get(
        "activation_config", ApplyMoEActivationConfig(clamp_limit=10.0))
    layer.input_dtype = over.get("input_dtype")
    layer.moe_config = SimpleNamespace(num_experts=e)
    return layer


def _meta_args(M=2304, n=N, e=E, device="meta"):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    x = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    w1 = torch.empty(e, K // 16, 2 * n * 2, dtype=torch.int32, device=device)
    w2 = torch.empty(e, n // 16, K * 2, dtype=torch.int32, device=device)
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=device)
    ti = torch.empty(M, TOPK, dtype=torch.int32, device=device)
    return dict(hidden_states=x, w1=w1, w2=w2, topk_weights=tw, topk_ids=ti,
                activation=MoEActivation.SILU, global_num_experts=e,
                expert_map=None, apply_router_weight_on_input=False)


def test_flag_defaults(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(FLAG + "_MIN_TOKENS", raising=False)
    assert envs.VLLM_GLM5_PP_MARLIN_PREFILL is False
    assert envs.VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS == 384


def test_gate_open_config_only_misses_the_device():
    assert pmp.gate_reason(_layer(), **_meta_args()) == "not sm_80"


@pytest.mark.parametrize("case,expect", [
    ("tp_shard_n512", "intermediate size N=512"),
    ("expert_map", "expert map"),
    ("fp8_input", "activation quantisation"),
    ("e256", "E=256"),
    ("topk6", "top-k"),
    ("fp16", "activation dtype"),
    ("zero_points", "w1_zp is set"),
    ("bias", "w2_bias is set"),
    ("router_on_input", "router weight applied on the input"),
    ("no_clamp", "clamp None"),
    ("gelu", "activation MoEActivation.GELU"),
    ("global_e", "E=288 (global 576)"),
])
def test_gate_closed_reasons(case, expect):
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
    )

    kw, args = {}, _meta_args()
    if case == "tp_shard_n512":
        args = _meta_args(n=512)
        layer = _layer(n=512)
    elif case == "e256":
        args = _meta_args(e=256)
        layer = _layer(e=256)
    else:
        layer = None
    if case == "expert_map":
        args["expert_map"] = torch.arange(E, device="meta")
    elif case == "fp8_input":
        kw["input_dtype"] = torch.float8_e4m3fn
    elif case == "topk6":
        args["topk_ids"] = torch.empty(2304, 6, dtype=torch.int32, device="meta")
    elif case == "fp16":
        args["hidden_states"] = torch.empty(2304, K, dtype=torch.float16,
                                            device="meta")
    elif case == "zero_points":
        kw["w1_zp"] = torch.empty(1, device="meta")
    elif case == "bias":
        kw["w2_bias"] = torch.empty(1, device="meta")
    elif case == "router_on_input":
        args["apply_router_weight_on_input"] = True
    elif case == "no_clamp":
        kw["activation_config"] = ApplyMoEActivationConfig(clamp_limit=None)
    elif case == "gelu":
        args["activation"] = MoEActivation.GELU
    elif case == "global_e":
        args["global_num_experts"] = 576
    layer = layer or _layer(**kw)
    why = pmp.gate_reason(layer, **args)
    assert why is not None and expect in why, why


def _apply_args(M=2304, device="meta"):
    a = _meta_args(M=M, device=device)
    return dict(
        output=torch.empty(M, K, dtype=torch.bfloat16, device=device),
        hidden_states=a["hidden_states"], w1=a["w1"], w2=a["w2"],
        topk_weights=a["topk_weights"], topk_ids=a["topk_ids"],
        activation=a["activation"], global_num_experts=E, expert_map=None,
        a1q_scale=None, a2_scale=None,
        workspace13=torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=device),
        workspace2=torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=device),
        expert_tokens_meta=None, apply_router_weight_on_input=False)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_hook_falls_through_to_fused_marlin_moe(monkeypatch, flag):
    """Flag off: the split module is never consulted.  Flag on with the gate
    closed (no sm_80 here): it is consulted, declines, and the incumbent call
    is the same as with the flag off."""
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe

    monkeypatch.setenv(FLAG, flag)
    consulted, calls = [], []
    real = pmp.maybe_apply

    def spy(*a, **k):
        consulted.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pmp, "maybe_apply", spy)
    monkeypatch.setattr(marlin_moe, "fused_marlin_moe",
                        lambda **k: calls.append(k))
    layer = _layer()
    layer._lora_context = None
    layer.apply(**_apply_args())
    assert len(calls) == 1 and calls[0]["expert_map"] is None
    assert len(consulted) == (1 if flag == "1" else 0)


def test_below_min_tokens_is_not_gated(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    monkeypatch.setattr(pmp, "gate_reason",
                        lambda *a, **k: pytest.fail("gate consulted"))
    a = _apply_args(M=383)
    for k in ("a1q_scale", "a2_scale", "expert_tokens_meta"):
        a.pop(k)
    assert pmp.maybe_apply(_layer(), **a) is False


# --------------------------------------------------------------------------
# GPU


def _gpu_ok():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability(0) != (8, 0):
        return False
    free, _ = torch.cuda.mem_get_info(0)
    return free > 12 * 2**30


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU, ~12 GB free")


def _dequant(q, s, size_k, size_n):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_scale_perms,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
    )

    perm = get_weight_perm(4).to(q.device)
    q = q.reshape(-1, perm.numel() // 8)
    q = ((q.unsqueeze(2).expand(-1, -1, 8)
          >> (4 * torch.arange(8, device=q.device)).view(1, 1, -1)) & 15)
    q = q.reshape(q.shape[0], -1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    q = q[:, inv].reshape(size_k // 16, size_n // 16, 16, 16)
    q = q.permute(0, 2, 1, 3).reshape(size_k, size_n)
    sp, _ = get_scale_perms()
    s = s.reshape(-1, len(sp))
    sinv = torch.empty(len(sp), dtype=torch.long, device=s.device)
    sinv[torch.as_tensor(sp, device=s.device)] = torch.arange(len(sp),
                                                              device=s.device)
    s = s[:, sinv].reshape(-1, size_n)
    return (q.double() - 8.0) * s.double().repeat_interleave(G, dim=0)


def _bf(x):
    return x.to(torch.bfloat16).double()


def reference_fp64(x, wd, tw, ti, rows, clamp=10.0):
    """fp64 over the dequantised weights with the incumbent's bf16 rounding
    points (w13 out, silu, h, w2 out, router weight product); returns the fp64
    value before the final bf16 rounding, for token rows ``rows``."""
    x64 = x[rows].double()
    ids = ti[rows].long()
    w = tw[rows].to(torch.bfloat16).double()
    acc = torch.zeros(len(rows), K, dtype=torch.float64, device=x.device)
    for e in torch.unique(ids).tolist():
        r, slot = (ids == e).nonzero(as_tuple=True)
        h1 = _bf(x64[r] @ _dequant(wd["w1"][e], wd["w1_scale"][e], K, 2 * N))
        gate = h1[:, :N].clamp(max=clamp)
        up = h1[:, N:].clamp(min=-clamp, max=clamp)
        h = _bf(_bf(gate / (1.0 + torch.exp(-gate))) * up)
        y = _bf(h @ _dequant(wd["w2"][e], wd["w2_scale"][e], N, K))
        acc.index_add_(0, r, _bf(y * w[r, slot].unsqueeze(1)))
    return acc


@pytest.fixture(scope="module")
def weights():
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    info = torch.iinfo(torch.int32)

    def packed(*shape):
        return torch.randint(info.min, info.max, shape, generator=g, device=dev,
                             dtype=torch.int32)

    def scales(*shape):
        return ((torch.rand(*shape, generator=g, device=dev) * 0.2 + 0.9)
                * 0.0034).to(torch.bfloat16)

    wd = {"w1": packed(E, K // 16, 2 * N * 2), "w2": packed(E, N // 16, K * 2),
          "w1_scale": scales(E, K // G, 2 * N), "w2_scale": scales(E, N // G, K)}
    yield wd
    del wd
    torch.cuda.empty_cache()


def _gpu_layer(wd):
    layer = _layer(device="cuda", w1_scale=wd["w1_scale"],
                   w2_scale=wd["w2_scale"])
    layer._lora_context = None
    return layer


def _inputs(M, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, K, generator=g, device=dev).to(torch.bfloat16)
    tw, ti = _routing(M, seed)
    return x, tw.to(dev), ti.to(dev)


def _apply(layer, wd, x, tw, ti, monkeypatch, flag, min_tokens=384):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", str(min_tokens))
    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    M = x.size(0)
    out = torch.full((M, K), float("nan"), dtype=torch.bfloat16, device=x.device)
    ws13 = torch.empty(M * TOPK, K, dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(M * TOPK * K, dtype=torch.bfloat16, device=x.device)
    layer.apply(out, x, wd["w1"], wd["w2"], tw, ti, MoEActivation.SILU, E, None,
                None, None, ws13, ws2, None, False)
    torch.cuda.synchronize()
    return out


def _incumbent(wd, x, tw, ti, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
        MoEActivation,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.scalar_type import scalar_types
    import vllm._custom_ops as ops

    monkeypatch.setenv("VLLM_GLM5_DETERMINISTIC_MOE_ALIGN", "1")
    out = torch.empty_like(x)
    fused_marlin_moe(
        hidden_states=x, w1=wd["w1"], w2=wd["w2"], bias1=None, bias2=None,
        w1_scale=wd["w1_scale"], w2_scale=wd["w2_scale"], topk_weights=tw,
        topk_ids=ti, quant_type_id=scalar_types.uint4b8.id,
        activation=MoEActivation.SILU,
        activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
        moe_sum=lambda i, o, t, m: ops.moe_sum(i, o), output=out)
    torch.cuda.synchronize()
    return out


def _err(y, ref):
    d = (y.double() - ref).abs()
    return float(d.mean()), float(d.max())


def _check_accuracy(y, inc, ref, tag):
    cm, cx = _err(y, ref)
    im, ix = _err(inc, ref)
    assert torch.isfinite(y).all(), tag
    assert cm <= 1.10 * im + 1e-12, f"{tag}: mean {cm:.3e} vs incumbent {im:.3e}"
    assert cx <= 1.25 * ix + 1e-12, f"{tag}: max {cx:.3e} vs incumbent {ix:.3e}"


def _sample_rows(M, n=48, seed=0):
    rng = random.Random(seed)
    return torch.tensor(sorted(rng.sample(range(M), min(n, M))), device="cuda")


@gpu
@pytest.mark.parametrize("M", [1, 17, 384, 1282, 2304, 2312])
def test_triton_lists_equal_model_and_are_deterministic(M):
    _, _, ti = _inputs(M, 10 + M)
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
    model = split_align_model(ti.view(-1).tolist(), E, buf["tab"].tolist())
    for bs, n, sids, eids in got[0]:
        assert (sids, eids) == model[bs], bs


@gpu
@pytest.mark.parametrize("M", [1282, 2304])
def test_flag_off_is_bitwise_the_incumbent(weights, monkeypatch, M):
    layer = _gpu_layer(weights)
    x, tw, ti = _inputs(M, M)
    off = _apply(layer, weights, x, tw, ti, monkeypatch, "0")
    inc = _incumbent(weights, x, tw, ti, monkeypatch)
    assert torch.equal(off, inc)


@gpu
@pytest.mark.parametrize("M,min_tokens", [(1282, 384), (2304, 384), (2312, 384),
                                          (1, 1), (17, 1), (299, 1)])
def test_split_path_accuracy_vs_fp64(weights, monkeypatch, M, min_tokens):
    layer = _gpu_layer(weights)
    x, tw, ti = _inputs(M, 100 + M)
    on = _apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.isfinite(on).all(), "every token row written"
    again = _apply(layer, weights, x, tw, ti, monkeypatch, "1", min_tokens)
    assert torch.equal(on, again), "bitwise run to run"
    inc = _incumbent(weights, x, tw, ti, monkeypatch)
    rows = _sample_rows(M)
    ref = reference_fp64(x, weights, tw, ti, rows)
    _check_accuracy(on[rows], inc[rows], ref, f"M={M}")


@gpu
def test_every_routed_row_written_once(weights, monkeypatch):
    """Poison w13's output buffer, run the split w13 lists only, and check
    that exactly the routed rows were written."""
    M = 2304
    layer = _gpu_layer(weights)
    x, tw, ti = _inputs(M, 7)
    ti.view(-1)[::131] = -1
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_workspace,
    )
    from vllm.scalar_type import scalar_types

    buf = pmp._buffers(x.device, E, M * TOPK, create=True)
    c1 = torch.full((M * TOPK, 2 * N), float("nan"), dtype=torch.bfloat16,
                    device=x.device)
    for bs, s, e, n in sa.split_align(ti, E, buf):
        tk, tn, bps = pmp._thread_cfg(bs, 2 * N, K, pmp.THREAD_CFG)
        ops.moe_wna16_marlin_gemm(
            x, c1, weights["w1"], None, layer.w1_scale, None, None, None,
            get_marlin_workspace(x.device), s, e, n, tw, moe_block_size=bs,
            top_k=TOPK, mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=2 * N, size_k=K, use_atomic_add=False,
            use_fp32_reduce=True, is_zp_float=False, thread_k=tk, thread_n=tn,
            blocks_per_sm=bps)
    torch.cuda.synchronize()
    written = torch.isfinite(c1).all(dim=1)
    untouched = torch.isnan(c1).all(dim=1)
    valid = ti.view(-1) >= 0
    assert torch.equal(written, valid) and torch.equal(untouched, ~valid)


@gpu
@pytest.mark.parametrize("bs", sa.SIZES)
def test_empty_list_launch_writes_nothing(weights, bs):
    """A list with no block (num_tokens_past_padded = 0) is launched as is:
    the compiled kernel must return without reading ids or writing C."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_workspace,
    )
    from vllm.scalar_type import scalar_types

    M = 384
    layer = _gpu_layer(weights)
    x, tw, _ = _inputs(M, 11)
    dev = x.device
    # valid ids (row 0, expert 0): if the kernel read them it would write C
    # row 0, which the NaN check sees, instead of faulting on a wild index
    sids = torch.zeros(M * TOPK + bs * E, dtype=torch.int32, device=dev)
    eids = torch.zeros(sids.numel() // bs + 1, dtype=torch.int32, device=dev)
    ntpp = torch.zeros(1, dtype=torch.int32, device=dev)
    c1 = torch.full((M * TOPK, 2 * N), float("nan"), dtype=torch.bfloat16,
                    device=dev)
    for tk, tn, bps in {(-1, -1, -1), pmp._thread_cfg(bs, 2 * N, K,
                                                      pmp.THREAD_CFG)}:
        ops.moe_wna16_marlin_gemm(
            x, c1, weights["w1"], None, layer.w1_scale, None, None, None,
            get_marlin_workspace(dev), sids, eids, ntpp, tw, moe_block_size=bs,
            top_k=TOPK, mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=2 * N, size_k=K, use_atomic_add=False,
            use_fp32_reduce=True, is_zp_float=False, thread_k=tk, thread_n=tn,
            blocks_per_sm=bps)
    torch.cuda.synchronize()
    assert torch.isnan(c1).all()


@gpu
def test_graph_replay_equals_eager_without_allocation_growth(weights, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    M = 2304
    layer = _gpu_layer(weights)
    x, tw, ti = _inputs(M, 3)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(FLAG + "_MIN_TOKENS", "384")
    pmp.warmup(x.device, E, 2312, TOPK)
    eager = _apply(layer, weights, x, tw, ti, monkeypatch, "1")
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
    layer = _gpu_layer(weights)
    x, tw, ti = _inputs(2304, 5)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    args = dict(hidden_states=x, w1=weights["w1"], w2=weights["w2"],
                topk_weights=tw, topk_ids=ti, activation=MoEActivation.SILU,
                global_num_experts=E, expert_map=None,
                apply_router_weight_on_input=False)
    assert pmp.gate_reason(layer, **args) is None
    args["expert_map"] = torch.arange(E, device="cuda", dtype=torch.int32)
    assert "expert map" in pmp.gate_reason(layer, **args)
    args["expert_map"] = None
    layer.input_dtype = torch.float8_e4m3fn
    assert "activation quantisation" in pmp.gate_reason(layer, **args)
    layer.input_dtype = None
    w2_512 = weights["w2"][:, : 512 // 16]
    assert "N=512" in pmp.gate_reason(layer, **dict(args, w2=w2_512))


def _capture_records():
    root = os.environ.get("VLLM_PP_MARLIN_CAPTURE_DIR")
    if not root:
        return []
    files = sorted(glob.glob(os.path.join(root, "moe_pre", "*.pt")))
    limit = int(os.environ.get("VLLM_PP_MARLIN_CAPTURE_LIMIT", "4"))
    # one per stage first, largest M first, so a small limit spans the stages
    files.sort(key=lambda f: (-int(f.split("_b")[1].split("_")[0]), f))
    picked, stages = [], set()
    for f in files:
        st = os.path.basename(f).split("_")[0]
        if st not in stages:
            picked.append(f)
            stages.add(st)
    picked += [f for f in files if f not in picked]
    return picked[:limit]


@gpu
@pytest.mark.skipif(not _capture_records(),
                    reason="VLLM_PP_MARLIN_CAPTURE_DIR not set")
@pytest.mark.parametrize("path", _capture_records())
def test_captured_prefill_records(monkeypatch, path):
    root = os.environ["VLLM_PP_MARLIN_CAPTURE_DIR"]
    rec = torch.load(path, map_location="cpu", weights_only=False)
    wrec = torch.load(os.path.join(root, rec["weights"]), map_location="cpu",
                      weights_only=False)
    wd = {k: wrec[k].cuda().contiguous() for k in
          ("w1", "w2", "w1_scale", "w2_scale")}
    del wrec
    x = rec["hidden_states"].cuda()
    tw = rec["topk_weights"].cuda().float()
    ti = rec["topk_ids"].cuda().to(torch.int32)
    inc = _incumbent(wd, x, tw, ti, monkeypatch)
    assert torch.equal(inc.cpu(), rec["out"]), "replay reproduces the live call"
    on = _apply(_gpu_layer(wd), wd, x, tw, ti, monkeypatch, "1")
    rows = _sample_rows(x.size(0), n=64, seed=1)
    ref = reference_fp64(x, wd, tw, ti, rows)
    _check_accuracy(on[rows], inc[rows], ref, os.path.basename(path))
    del wd
    torch.cuda.empty_cache()
