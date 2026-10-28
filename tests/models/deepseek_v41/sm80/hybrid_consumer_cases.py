# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay byte-linked hybrid producer/consumer calls with explicit storage points."""

import hashlib
from pathlib import Path

import torch

from tests.models.deepseek_v41.sm80 import fp64_cases as fp64
from tests.models.deepseek_v41.sm80 import hybrid_linear_cases as linear


def _move(a, device):
    return {k: v.to(device, copy=True) if isinstance(v, torch.Tensor) else v for k, v in a.items()}


def _rope(a):
    from tests.models.deepseek_v41.sm80.test_sm80_kernels_gpu import _inverse_rope_fp64
    x = a["x"]
    shape = x.shape
    value = x.reshape(shape[0], -1, shape[-1])
    freqs = a["freqs"] if a.get("inverse", False) else -a["freqs"]
    return _inverse_rope_fp64(value, a["positions"], freqs).reshape(shape)


def actual(op, a):
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    if op == "gate":
        from pytest import MonkeyPatch
        from tests.models.deepseek_v41.sm80 import construct
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm_sm80.models.deepseek_v41.sm80_moe import _GATE_TIERS
        with MonkeyPatch.context() as mp:
            construct.set_groups(mp, 1, 0)
            with torch.device(a["weight"].device):
                gate = GateLinear(a["weight"].shape[1], a["weight"].shape[0],
                                  params_dtype=torch.float32, out_dtype=torch.float32,
                                  return_bias=a["return_bias"])
        gate.weight = torch.nn.Parameter(a["weight"], requires_grad=False)
        for flag in _GATE_TIERS:
            setattr(gate, flag, False)
        return gate(a["x"])
    if op == "route":
        from vllm_sm80.models.deepseek_v41.sm80_moe import route
        return route(**a)
    if "mhc_" in op:
        from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused
        module = sm80_mhc_fused if op.startswith("fused_") else sm80_mhc
        return getattr(module, op.removeprefix("fused_"))(**a)
    if op == "add":
        return torch.add(a["x"], a["y"], alpha=a["alpha"])
    if op == "greedy":
        from vllm.v1.sample.sampler import Sampler
        return Sampler.greedy_sample(a["logits"])
    if op == "compress_rows":
        result = core.compress_rows(**a)
        return {"0": result[0], "1": result[1], "ring": a["ring"]}
    if op == "swiglu":
        return SiluAndMulWithClamp(a["limit"], alpha=a["alpha"], beta=a["beta"])(a["x"])
    if op == "store_rows":
        core.store_rows(**a)
        return a["cache"]
    return getattr(core, op)(**a)


def ideal(op, a):
    from tests.models.deepseek_v41.sm80.test_act_quant import ref_fp4, ref_fp8
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    if op == "gate":
        result = a["x"].double() @ a["weight"].double().T
        return (result, None) if a["return_bias"] else result
    if op == "route":
        logits = a["logits"].double() / a["gate_temp"]
        scores = torch.nn.functional.softplus(logits).sqrt()
        bias = a["bias"].double()
        if a["image_mask"] is not None and a["bias_vl"] is not None:
            bias = torch.where(a["image_mask"][:, None], a["bias_vl"].double(), bias)
        ids = (scores + bias).topk(a["top_k"], dim=-1).indices
        weights = scores.gather(1, ids)
        if a["renormalize"] and a["top_k"] > 1:
            weights /= weights.sum(-1, keepdim=True) + 1e-20
        return weights * a["routed_scaling_factor"], ids
    if "mhc_" in op:
        if op.removeprefix("fused_") == "mhc_post":
            n, hc, _ = a["residual"].shape
            return a["post_layer_mix"].double().reshape(n, hc, 1) * a["x"].double()[:, None] + torch.einsum(
                "tjk,tjh->tkh", a["comb_res_mix"].double().reshape(n, hc, hc), a["residual"].double())
        values = fp64.mhc_delayed_fp64(a, rounding=True)
        result = (values["post"], values["comb"], values["layer_input"], values["pre"])
        if a.get("x") is not None:
            stored = values["residual"].to(torch.bfloat16).double()
            aux = stored.mean(1) if a.get("capture_aux") else stored.new_empty(0, stored.shape[-1])
            return values["residual"], *result, aux
        return result
    if op == "add":
        return a["x"].double() + a["y"].double() * a["alpha"]
    if op == "greedy":
        return a["logits"].double().argmax(-1).view(-1)
    if op == "rms_norm":
        x = a["x"].double()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + a["eps"]) * a["weight"].double()
    if op == "rope":
        return _rope(a)
    if op == "swiglu":
        gate, up = a["x"].double().chunk(2, -1)
        gate = gate.clamp(max=a["limit"])
        up = up.clamp(min=-a["limit"], max=a["limit"])
        return gate * torch.sigmoid(a["alpha"] * gate) * (up + a["beta"])
    if op == "qdq_fp8":
        return ref_fp8(a["x"], a["block"]).double()
    if op == "qdq_fp4":
        return ref_fp4(a["x"], a["block"], a["scale"]).double()
    if op in ("window_kv", "compressed_kv"):
        key = "kv" if op == "window_kv" else "latent"
        pos = a["positions"] if op == "window_kv" else a["group_pos"]
        rotated = _rope({"x": a[key], "positions": pos, "freqs": a["freqs"]}).to(torch.bfloat16)
        return (ref_fp8(rotated, 32) if op == "window_kv" else ref_fp4(rotated, 16, "e4m3")).double()
    if op == "output_qdq":
        o = a["o"]
        n, h, d = o.shape
        rotated = _rope({"x": o, "positions": a["positions"], "freqs": a["freqs"],
                         "inverse": True}).to(torch.bfloat16)
        return ref_fp8(rotated.reshape(n, a["n_groups"], -1), 32).reshape(n, h, d).double()
    if op == "index_key_rows":
        norm = ideal("rms_norm", {"x": a["k"], "weight": a["k_norm"], "eps": a["eps"]})
        rotated = _rope({"x": norm.to(torch.bfloat16), "positions": a["group_pos"],
                         "freqs": a["freqs"]}).to(torch.bfloat16)
        return ref_fp4(rotated, 32, "e8m0").double()
    if op == "compress_rows":
        d, kv = a["head_dim"], a["kv_score"].double()
        if a["compress_ratio"] == 1:
            closes, pooled = a["comp_slots"] >= 0, kv[:, :d].to(torch.bfloat16)
        else:
            if a["compress_ratio"] != 2:
                raise ValueError("unsupported captured compressor ratio")
            t = torch.arange(kv.shape[0])
            req, qsl, pos = a["token_to_req"].long(), a["query_start_loc"].long(), a["positions"].long()
            slots, ring = a["ring_slots"].long().clamp_min(0), a["ring"].double()
            prior = ring[slots // ring.shape[1], (pos - 1) % ring.shape[1]]
            previous = torch.where((t == qsl[req])[:, None], prior, kv[(t - 1).clamp_min(0)])
            pair = torch.stack((previous[:, :d], kv[:, :d]), 1)
            scores = torch.stack((previous[:, d:], kv[:, d:]), 1)
            pooled = (pair * scores.softmax(1)).sum(1).to(torch.bfloat16)
            closes = (a["comp_slots"] >= 0) & ((pos + 1) % 2 == 0)
        if a["ring"] is not None:
            end = a["query_start_loc"].long()[a["token_to_req"].long() + 1]
            t = torch.arange(kv.shape[0])
            tail = (t >= end - a["ring"].shape[1]) & (t < end)
            core.store_rows(a["ring"], torch.where(tail, a["ring_slots"], -1), a["kv_score"])
        return {"0": closes, "1": ideal("rms_norm", {"x": pooled, "weight": a["norm_weight"], "eps": a["eps"]}),
                "ring": a["ring"]}
    if op in ("index_scores", "sparse_attention"):
        # These independent captured-input references already implement paging,
        # visibility, sink attention and the head/rank arithmetic contracts.
        return fp64.FAMILIES[op](a, torch.device("cpu"), fp64.Source({}, {})).exact
    if op in ("select_topk", "select_candidate_blocks", "mask_to_candidates", "store_rows"):
        # Selection, masks and stores are exact contracts, not approximate math.
        return actual(op, a)
    raise ValueError(f"no independent FP64 consumer definition for {op}")


def _outputs(value, prefix="out"):
    if isinstance(value, torch.Tensor):
        return {prefix: value} if value.numel() else {}
    children = value.items() if isinstance(value, dict) else enumerate(value or ())
    return {key: tensor for name, child in children
            for key, tensor in _outputs(child, f"{prefix}.{name}").items()}


def load_source(path, sha):
    with Path(path).open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != sha:
        raise ValueError(f"consumer source hash changed: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _producer_view(value, stored, view):
    """Apply a runtime link to a replay's logical output in its saved layout."""
    if value.stride() != stored.stride():
        shaped = torch.empty_strided(value.shape, stored.stride(), dtype=value.dtype, device=value.device)
        shaped.copy_(value)
        value = shaped
    return value.as_strided(view["shape"], view["stride"], value.storage_offset() + view["offset"])


def _owned_state(op, args, outputs):
    if op not in ("store_rows", "compress_rows"):
        return
    name = "cache" if op == "store_rows" else "ring"
    before = args[name]
    if before is None:
        return
    slots = args["slots" if op == "store_rows" else "ring_slots"].long()
    if op == "compress_rows":
        t = torch.arange(slots.numel(), device=slots.device)
        ends = args["query_start_loc"].long()[args["token_to_req"].long() + 1]
        slots = torch.where((t >= ends - before.shape[1]) & (t < ends), slots, -1)
    mask = torch.zeros(before.shape[:2], dtype=torch.bool, device=before.device).flatten()
    mask[slots[slots >= 0]] = True
    after = outputs["out" if op == "store_rows" else "out.ring"]
    if not fp64.bits_equal(before.flatten(0, 1)[~mask], after.flatten(0, 1)[~mask]):
        raise ValueError("hybrid consumer writes outside owned slots")


def evaluate(obj, device, checkpoint, stack=()):
    meta, tensors = obj["meta"], obj["tensors"]
    if meta.get("tp_boundary"):
        raise ValueError("TP collective replay requires the distributed consumer check")
    if meta["call_id"] in stack:
        raise ValueError("cyclic hybrid consumer link")
    stack = (*stack, meta["call_id"])
    args = {k: v for k, v in tensors.items() if not k.startswith("out")}
    arms = [_move(args, device), _move(args, device), _move(args, "cpu")]
    invariant = True
    for name, source in (meta.get("parent_sources") or {}).items():
        parent = load_source(source["path"], source["sha256"])
        pvals, pi = evaluate(parent, device, checkpoint, stack)
        invariant &= pi
        view = source["view"]
        stored = parent["tensors"][view["output"]]
        expected = stored.as_strided(view["shape"], view["stride"],
                                     stored.storage_offset() + view["offset"])
        if not fp64.bits_equal(expected, tensors[name]):
            raise ValueError("actual-consumer link is not byte-identical to the recorded producer view")
        for arm, values, target in zip(arms, pvals, (device, device, "cpu")):
            value = values[view["output"]]
            arm[name] = _producer_view(value, stored, view).to(device=target, dtype=tensors[name].dtype)
    if meta.get("linear") is not None:
        vals = []
        for index, arm in enumerate(arms):
            inp = {**arm, "out": tensors["out"]}
            got = linear.linear_outputs(inp, device, meta, checkpoint)
            vals.append({"out": got[index]})
        apply = got[3]
        whole = vals[0]["out"]
        invariant &= all(fp64.bits_equal(whole[i:i + 1], apply(arms[0]["x"][i:i + 1]))
                         for i in fp64.picks(whole.shape[0]))
        return vals, invariant
    if not meta.get("parent_sources"):
        raise ValueError("missing actual-consumer link")
    outputs = [_outputs(actual(meta["op"], _move(arms[0], device))),
               _outputs(actual(meta["op"], _move(arms[1], device))),
               _outputs(ideal(meta["op"], _move(arms[2], "cpu")))]
    for before, after in zip(arms, outputs):
        _owned_state(meta["op"], before, after)
    # Replay the unchanged captured call before substituting upstream outputs.
    replay = _outputs(actual(meta["op"], _move(args, device)))
    if any(not fp64.bits_equal(replay[name], tensors[name]) for name in meta["outputs"]):
        raise ValueError("actual consumer does not replay its captured output")
    op = meta["op"]
    if op == "store_rows":
        # Distinct owned writes commute; replay each write on the same pre-state.
        state = arms[0]["cache"].clone()
        from vllm_sm80.models.deepseek_v41.ops.sm80_attn_core import store_rows
        for row in range(arms[0]["slots"].numel()):
            store_rows(state, arms[0]["slots"][row:row + 1], arms[0]["rows"][row:row + 1])
        invariant &= fp64.bits_equal(state, outputs[0]["out"])
    elif op in ("compress_rows", "index_key_rows", "index_scores", "sparse_attention"):
        family = {"compress_rows": "compress", "index_key_rows": "index_keys"}.get(op, op)
        invariant &= fp64.FAMILIES[family](arms[0], device, fp64.Source({}, {})).row_invariant
    else:
        n = next(v.shape[0] for v in args.values() if isinstance(v, torch.Tensor) and v.ndim)
        constants = {"weight", "freqs", "cache", "norm_weight", "k_norm", "fn", "hc_scale", "hc_base", "bias", "bias_vl"}
        for row in fp64.picks(n):
            sub = {k: v[row:row + 1] if isinstance(v, torch.Tensor) and v.ndim and
                   v.shape[0] == n and k not in constants else v for k, v in arms[0].items()}
            single = _outputs(actual(op, _move(sub, device)))
            invariant &= all(fp64.bits_equal(outputs[0][key][row:row + 1], value)
                             for key, value in single.items())
    return outputs, invariant


def hybrid_linear_consumer(a, device, src):
    linear.reference_settings()
    obj = {"meta": src.meta, "tensors": a}
    outputs, invariant = evaluate(obj, device, src.checkpoint["hybrid"])
    if not invariant:
        raise ValueError("hybrid consumer violates row invariance")
    name = src.output or "out"
    cand, ref, exact = (values[name] for values in outputs)
    if not cand.is_floating_point():
        if not fp64.bits_equal(cand, ref) or not fp64.bits_equal(cand.cpu(), exact.cpu()):
            raise ValueError("discrete hybrid consumer flip; no blanket tie forgiveness")
    return fp64.Call(cand, ref, exact.double(), invariant)
