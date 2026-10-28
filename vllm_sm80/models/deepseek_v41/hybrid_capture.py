# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid projection and successor snapshots, only in diagnostic capture boots.

Weight fingerprints are taken before repacking. Consumer links name the producer
call and its output view; the exporter must check the bytes, not trust an address
which the allocator may have reused. Each replay's step/owner/rank scopes the IDs.
No hook is installed with VLLM_SM80_CAPTURE unset.
"""

import functools
import inspect
import itertools

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from vllm_sm80 import capture

CORE_POINTS = (
    "rms_norm", "rope", "window_kv", "compressed_kv", "qdq_fp8", "qdq_fp4",
    "compress_rows", "store_rows", "index_key_rows", "index_scores", "select_topk",
    "select_candidate_blocks", "mask_to_candidates", "sparse_attention", "output_qdq",
)
_serial = itertools.count()
_outputs = {}
_installed = False


def reset_links():
    _outputs.clear()


def _leaves(value, prefix="out"):
    if isinstance(value, torch.Tensor):
        if value.numel():
            yield prefix, value
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from _leaves(child, f"{prefix}.{i}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{prefix}.{key}")


def _link(value):
    if not isinstance(value, torch.Tensor):
        return None
    storage = value.untyped_storage()
    saved = _outputs.get(storage.data_ptr())
    if saved is None or saved[1].expired() or saved[1] != StorageWeakRef(storage):
        return None
    source = saved[0]
    if source["dtype"] != str(value.dtype):
        return None
    return dict(source, shape=list(value.shape), stride=list(value.stride()),
                offset=value.storage_offset() - source["storage_offset"])


def _compact_blocks(cache, table, used):
    """At the synchronized drain, keep each visible physical block once."""
    table = torch.where(used, table.clamp_min(0), 0)
    ids = torch.unique(torch.cat((table.new_zeros(1), table.flatten())), sorted=True)
    compact = cache[ids.to(device=cache.device, dtype=torch.long)].cpu()
    return compact, torch.searchsorted(ids, table).to(table.dtype)


def _defer_reader(rec, op, args):
    """Snapshot metadata in-stream, not an expansion of KV table capacity.

    These readers do not write their model-owned caches, which remain valid
    until this eager call / graph's synchronized drain. All point-local
    metadata is copied by Record.put, so later metadata-buffer reuse cannot
    change the selection and every graph replay refreshes it. Mutating ops
    keep their separate pre/post snapshots and write guards below.
    """
    cache_names = ("idxk",) if op == "index_scores" else ("swa_cache", "comp_cache")
    caches = {name: args[name] for name in cache_names}
    rec.put(**{name: value for name, value in args.items() if name not in caches})
    # The callback owns the remapped fields. Other controls share the copies
    # already owned by Record.put; retain no Record/back-reference cycle.
    req = rec.tensors["req"]
    if op == "index_scores":
        table = rec.tensors.pop("block_table")
        n_vis = rec.tensors["n_vis"]
        n_cols = args["n_cols"]

        def materialize():
            bt = table.cpu()
            e = caches["idxk"].shape[1]
            visible = n_vis.cpu().clamp(min=0, max=n_cols).long()
            per_req = torch.zeros(bt.shape[0], dtype=torch.long)
            per_req.scatter_reduce_(0, req.cpu().long(), visible,
                                    reduce="amax", include_self=True)
            used = torch.arange(bt.shape[1])[None, :] < ((per_req + e - 1) // e)[:, None]
            cache, mapped = _compact_blocks(caches["idxk"], bt, used)
            return {"idxk": cache, "block_table": mapped}
    else:
        indices = rec.tensors.pop("swa_indices")
        table = rec.tensors.pop("comp_block_table")
        local = rec.tensors["comp_local"]

        def materialize():
            slots = indices.cpu().long()
            e = caches["swa_cache"].shape[1]
            cache, blocks = _compact_blocks(
                caches["swa_cache"], slots.clamp_min(0) // e, slots >= 0)
            out = {"swa_cache": cache,
                   "swa_indices": torch.where(slots >= 0, blocks * e + slots % e, -1).to(indices.dtype),
                   "comp_cache": None, "comp_block_table": None}
            if caches["comp_cache"] is not None:
                bt = table.cpu()
                used = torch.zeros_like(bt, dtype=torch.bool)
                if local is not None:
                    positions = local.cpu().long()
                    valid = positions >= 0
                    requests = req.cpu().long()[:, None].expand_as(positions)
                    columns = positions[valid] // caches["comp_cache"].shape[1]
                    used[requests[valid], columns] = True
                out["comp_cache"], out["comp_block_table"] = _compact_blocks(
                    caches["comp_cache"], bt, used)
            return out

    rec.defer(materialize)


def _snapshot_args(op, args, *, in_graph):
    """Retain pre-write state and neighboring write guards for mutating ops."""
    saved = dict(args)
    if op in ("compress_rows", "store_rows"):
        name = "ring" if op == "compress_rows" else "cache"
        slots_name = "ring_slots" if op == "compress_rows" else "slots"
        cache, slots = args[name], args[slots_name]
        if cache is not None:
            blocks = slots.clamp_min(0).long() // cache.shape[1]
            # Include the null block and real adjacent blocks as write guards.
            ids = torch.cat((blocks.new_zeros(1), blocks,
                             (blocks + 1).remainder(cache.shape[0])))
            if not in_graph:
                ids = torch.unique(ids, sorted=True)
            saved[name] = cache[ids]
            saved["snapshot_blocks"] = ids
            saved["snapshot_name"] = name
    return saved


def _record(op, args, run, *, linear=None, boundary=False):
    parents = {name: link for name, value in args.items() if (link := _link(value))}
    if linear is None and not parents:
        return run()
    prefix = linear["prefix"] if linear else next(iter(parents.values()))["prefix"]
    values = [args[k] for k in ("x", "rows", "kv_score", "q", "scores", "logits") if k in args]
    values.extend(args.values())
    first = next((v for v in values if isinstance(v, torch.Tensor) and v.ndim), None)
    if first is None:
        return run()
    family = "hybrid_linear" if linear is not None else "hybrid_linear_consumer"
    point_phase = ("draft" if linear and linear.get("draft") else
                   next((p.get("point_phase") for p in parents.values() if p.get("point_phase")), None))
    rec = capture.begin("hybrid", family, f"{prefix}.{op}.D{first.shape[-1]}", first.shape[0],
                        point_phase=point_phase, op=op,
                        call_id=next(_serial), parents=parents, linear=linear, tp_boundary=boundary)
    if rec is None:
        return run()
    saved = _snapshot_args(op, args, in_graph=rec.in_graph)
    if op in ("index_scores", "sparse_attention"):
        _defer_reader(rec, op, saved)
    else:
        rec.put(**saved)
    result = run()
    outputs = dict(_leaves(result))
    if op == "compress_rows" and args["ring"] is not None:
        outputs["out.ring"] = args["ring"][saved["snapshot_blocks"]]
    elif op == "store_rows":
        outputs["out"] = args["cache"][saved["snapshot_blocks"]]
    rec.put(preserve_layout=True, **outputs)
    rec.meta["outputs"] = list(outputs)
    for name, value in outputs.items():
        _outputs[value.untyped_storage().data_ptr()] = ({
            "call_id": rec.meta["call_id"], "output": name, "prefix": prefix, "point_phase": rec.point_phase,
            "storage_offset": value.storage_offset(), "dtype": str(value.dtype),
        }, StorageWeakRef(value.untyped_storage()))
    rec.done()
    return result


def register_model(model):
    """Fingerprint the actual loader output before any method packs it."""
    cfg = capture.config()
    for module in model.modules():
        sources = getattr(module, "_hybrid_capture_sources", None)
        if not sources or not hasattr(module, "weight"):
            continue
        prefix = getattr(module, "prefix", sources[0][0])
        if not (cfg.match.search(prefix) or cfg.prefill_match.search(prefix)):
            continue
        weight = module.weight
        # The retained FP32 router is a consumer, not a changed hybrid GEMM.
        if weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            continue
        scale = getattr(module, "weight_scale", None)
        indices = getattr(module, "shard_indices", None)
        module._hybrid_capture = {
            "prefix": prefix, "sources": sources, "draft": any(source.startswith("mtp.") for source, _ in sources),
            "layer_type": type(module).__name__,
            "rank": int(getattr(module, "tp_rank", 0)),
            "tp": int(getattr(module, "tp_size", 1)),
            "shape": list(weight.shape), "dtype": str(weight.dtype),
            "weight_sha256": capture.sha256_of(weight),
            "scale_sha256": capture.sha256_of(scale) if scale is not None else None,
            "groups": int(getattr(module.quant_method, "groups", getattr(module, "bmm_batch_size", 1))),
            "grouped": hasattr(module.quant_method, "apply_grouped"),
            "head_slice": ([indices.org_vocab_start_index, indices.org_vocab_end_index]
                           if indices is not None else None),
        }


def _linear_method(cls, name="apply", grouped=False):
    original = getattr(cls, name)

    @functools.wraps(original)
    def wrapped(self, layer, x, *args, **kwargs):
        facts = getattr(layer, "_hybrid_capture", None)
        if facts is None:
            return original(self, layer, x, *args, **kwargs)
        return _record("grouped" if grouped else "linear", {"x": x},
                       lambda: original(self, layer, x, *args, **kwargs), linear=facts)

    setattr(cls, name, wrapped)


def install():
    global _installed
    if _installed or not capture.active:
        return
    _installed = True
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
    from vllm_sm80.kernels.exl3.quant import OutDtypeLinearMethod
    from vllm_sm80.layers.grouped_linear import GroupedLinearMethod

    for cls in (ModelOptLinearMethod, UnquantizedLinearMethod, OutDtypeLinearMethod):
        _linear_method(cls)
    _linear_method(GroupedLinearMethod, "apply_grouped", grouped=True)
    for name in CORE_POINTS:
        original = getattr(core, name)
        signature = inspect.signature(original)

        def wrap(fn, sig, point):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return _record(point, dict(bound.arguments), lambda: fn(*args, **kwargs))
            return wrapped

        setattr(core, name, wrap(original, signature, name))
    activation = SiluAndMulWithClamp.forward

    @functools.wraps(activation)
    def swiglu(self, x):
        return _record("swiglu", {"x": x, "limit": self.swiglu_limit,
                                  "alpha": self.alpha, "beta": self.beta},
                       lambda: activation(self, x))

    SiluAndMulWithClamp.forward = swiglu
    gather = LogitsProcessor._gather_logits

    @functools.wraps(gather)
    def gather_head(self, logits):
        return _record("tp_logits_gather", {"x": logits, "use_all_gather": self.use_all_gather},
                       lambda: gather(self, logits), boundary=True)

    LogitsProcessor._gather_logits = gather_head
    head = LogitsProcessor._apply_head

    @functools.wraps(head)
    def apply_head(self, lm_head, hidden_states, embedding_bias):
        facts = getattr(lm_head, "_hybrid_capture", None)
        if facts is None:
            return head(self, lm_head, hidden_states, embedding_bias)
        if embedding_bias is not None:
            raise ValueError("hybrid head capture requires the served bias-free head")
        return _record("head", {"x": hidden_states},
                       lambda: head(self, lm_head, hidden_states, embedding_bias), linear=facts)

    LogitsProcessor._apply_head = apply_head

    from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm_sm80.models.deepseek_v41 import sm80_moe
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc, sm80_mhc_fused
    from vllm.v1.sample.sampler import Sampler
    from torch.overrides import TorchFunctionMode

    for module, names in ((sm80_moe, ("route",)),
                          (sm80_mhc, ("mhc_post", "mhc_pre_delayed", "mhc_fused_post_pre_delayed")),
                          (sm80_mhc_fused, ("mhc_pre_delayed", "mhc_fused_post_pre_delayed"))):
        for name in names:
            fn = getattr(module, name)
            point = ("fused_" if module is sm80_mhc_fused else "") + name
            setattr(module, name, wrap(fn, inspect.signature(fn), point))

    gate = GateLinear.forward

    @functools.wraps(gate)
    def gate_forward(self, x):
        return _record("gate", {"x": x, "weight": self.weight, "return_bias": self.return_bias},
                       lambda: gate(self, x))

    GateLinear.forward = gate_forward
    greedy = Sampler.greedy_sample
    Sampler.greedy_sample = staticmethod(lambda logits: _record(
        "greedy", {"logits": logits}, lambda: greedy(logits)))

    def collective(fn, point):
        @functools.wraps(fn)
        def wrapped(self, input_, *args, **kwargs):
            if self is not get_tp_group():
                return fn(self, input_, *args, **kwargs)
            values = {"x": input_, "tp": self.world_size, "rank": self.rank_in_group}
            if point == "tp_all_gather":
                values["dim"] = kwargs.get("dim", args[0] if args else -1)
            return _record(point, values, lambda: fn(self, input_, *args, **kwargs), boundary=True)
        return wrapped

    GroupCoordinator.all_reduce = collective(GroupCoordinator.all_reduce, "tp_all_reduce")
    GroupCoordinator.all_gather = collective(GroupCoordinator.all_gather, "tp_all_gather")

    class SharedCombine(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in (torch.add, torch.Tensor.add, torch.Tensor.__add__) and len(args) == 2:
                return _record("add", {"x": args[0], "y": args[1],
                                       "alpha": kwargs.get("alpha", 1)},
                               lambda: func(*args, **kwargs))
            return func(*args, **kwargs)

    moe_forward = MoERunner.forward

    @functools.wraps(moe_forward)
    def combine_forward(self, *args, **kwargs):
        with SharedCombine():
            return moe_forward(self, *args, **kwargs)

    MoERunner.forward = combine_forward

    from vllm_sm80.models.deepseek_v41.sm80_model import DSparkDeepseekV4ForCausalLM

    def draft_scope(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with capture.phase_scope("draft"):
                return fn(*args, **kwargs)
        return wrapped

    for name in ("forward", "compute_logits"):
        setattr(DSparkDeepseekV4ForCausalLM, name,
                draft_scope(getattr(DSparkDeepseekV4ForCausalLM, name)))


    from vllm_sm80.models.deepseek_v41 import sm80_pipeline_capture
    sm80_pipeline_capture.install()
