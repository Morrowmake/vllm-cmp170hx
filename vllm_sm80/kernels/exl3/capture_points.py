# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 capture points of the kernel call capture (vllm_sm80/capture.py,
VLLM_SM80_CAPTURE; inert when it is unset).

- ``dense`` (point ``exl3_linear``): every call of ops._linear on a
  registered matrix: tensors ``x`` (the input as the operator received it)
  and ``out`` (the output it returned; BF16, or FP32 where the method asks for
  it). The prefix is the matrix's checkpoint prefix; meta carries the shard
  (logical_in, logical_out, input_slice, output_slices, padded_in,
  padded_out, bits_x2, codebook, out_dtype, row_chunk, weight_sha256 of the
  rank's trellis, suh and svh bytes) and the operator that ran.
- ``moe`` (points ``exl3_moe_out``, ``exl3_moe_out_v2``,
  ``exl3_moe_out_prefill``, ``exl3_moe_out_prefill_v2``: the operator family
  the dispatch chose; moe_out_v2_fold and moe_out_v2_try, which return
  moe_out_v2's bits, record as ``exl3_moe_out_v2`` with the operator in
  meta): every call of Exl3Experts.apply on a registered layer, with the
  arguments of the FP64 case driver's capture point (x, topk_ids,
  topk_weights, packed_meta, expert_map, intermediate_size, swiglu_limit, and
  bits_x2 for moe_out_v2) and the output ``out``. The prefix is the layer's
  module prefix.
"""

import torch

from vllm_sm80 import capture

# Operators that return another operator's output bits record under its
# point, so a case driver replays them as that operator.
_MOE_POINTS = {"moe_out_v2_fold": "moe_out_v2", "moe_out_v2_try": "moe_out_v2"}


def register_linear(weight, row_chunk: int, out_dtype: torch.dtype) -> None:
    """Key a loaded dense matrix by its trellis address."""
    if not capture.active:
        return
    cfg = capture.config()
    prefix = weight.matrix.prefix
    if not (cfg.match.search(prefix) or cfg.prefill_match.search(prefix)):
        return
    capture.register_weight(
        weight.trellis.data_ptr(),
        prefix=prefix,
        logical_in=weight.logical_in,
        logical_out=weight.logical_out,
        input_slice=[weight.input_slice.start, weight.input_slice.stop],
        output_slices=[[s.start, s.stop] for s in weight.output_slices],
        padded_in=weight.padded_in,
        padded_out=weight.padded_out,
        bits_x2=weight.matrix.bits_x2,
        codebook=weight.matrix.codebook,
        out_dtype=str(out_dtype).removeprefix("torch."),
        row_chunk=row_chunk,
        weight_sha256=capture.sha256_of(weight.trellis, weight.suh, weight.svh),
    )


def linear(x: torch.Tensor, trellis: torch.Tensor, out: torch.Tensor, operator) -> None:
    facts = capture.weight(trellis.data_ptr())
    if facts is None:
        return
    facts = dict(facts)
    prefix = facts.pop("prefix")
    name = getattr(operator, "__name__", type(operator).__name__)
    rec = capture.begin("dense", "exl3_linear", prefix, x.shape[0], operator=name, **facts)
    if rec is not None:
        rec.put(x=x, out=out).done()


def register_moe(pointers: torch.Tensor, prefix: str) -> None:
    """Key a loaded MoE layer by its expert pointer table's address."""
    if capture.active:
        capture.config()
        capture.register_weight(pointers.data_ptr(), prefix=prefix)


def moe(
    operator,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    pointers: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    intermediate_size: int,
    swiglu_limit: float,
    out: torch.Tensor,
) -> None:
    facts = capture.weight(pointers.data_ptr())
    if facts is None:
        return
    func = getattr(operator, "func", operator)
    keywords = dict(getattr(operator, "keywords", None) or {})
    name = func.__name__
    rec = capture.begin(
        "moe", f"exl3_{_MOE_POINTS.get(name, name)}", facts["prefix"], x.shape[0],
        operator=name,
    )
    if rec is None:
        return
    rec.put(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        packed_meta=packed_meta,
        expert_map=expert_map,
        intermediate_size=int(intermediate_size),
        swiglu_limit=float(swiglu_limit),
        **({"bits_x2": int(keywords["bits_x2"])} if "bits_x2" in keywords else {}),
        out=out,
    ).done()
