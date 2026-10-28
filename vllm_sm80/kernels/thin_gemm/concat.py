# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concat groups for the sm_80 thin GEMM (``VLLM_THIN_GEMM_CONCAT``).

Two projections of an MLA layer with an indexer read the same activation:

- ``qb_wqb``: ``q_b_proj`` and ``indexer.wq_b`` both read ``q_c``;
- ``qkva_kpool``: ``fused_qkv_a_proj`` and ``indexer.index_kpool_compress_gate``
  both read ``hidden_states``.

The pairing lives in the existing dispatch, so no model code changes. Warmup
registers each group under its members' weight ``data_ptr``. When member 0
(``q_b_proj`` / ``fused_qkv_a_proj``) is dispatched, one
``thin_gemm_group_split`` launch computes both members into two contiguous
tensors. Member 0's tensor is returned. Member 1's tensor is stashed in the
group's one-entry slot, keyed by the input tensor: its identity, version
counter (when it has one), shape and strides. When member 1 is dispatched it
takes the stash if the slot holds its own group and its own input, and the
slot is cleared. Anything else runs member 1 alone, with the same bytes, and
logs a one-time line. A layer that skips its indexer leaves the stash until
the next layer's member 0 overwrites it.

The model runner's activations are inference tensors, which carry no version
counter. For them the slot matches on identity, shape and strides alone. That
holds because nothing writes ``q_c`` or ``hidden_states`` in place between
the two members (``mla.py`` and ``Indexer.forward``).

Every member keeps its solo schedule inside the group launch, so each output
is bitwise what the separate ``thin_gemm`` call returns. Only the launch
count and the stream position of member 1's GEMM change.

The gate requires ``VLLM_THIN_GEMM``, an sm_80 device and
``CompilationMode.NONE``. A Python-side stash cannot cross Dynamo-traced
graph pieces, so traced calls (``torch.compiler.is_compiling()``) never pair.
This module is imported only with ``VLLM_THIN_GEMM_CONCAT`` set. In that case
the package exports this module's ``ampere_thin_gemm`` and ``thin_linear``
in place of its own.
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm_sm80.kernels import thin_gemm as _tg
from vllm_sm80.kernels.thin_gemm import (
    _device_supported,
    max_rows,
    thin_gemm_supported,
)

logger = init_logger(__name__)

GROUPS = ("qb_wqb", "qkva_kpool")


class _Slot:
    """Per group name: the stash and its counters."""

    __slots__ = ("name", "stash", "launches", "hits", "misses")

    def __init__(self, name: str):
        self.name = name
        self.stash = None
        self.launches = 0
        self.hits = 0
        self.misses = 0


class _Group:
    """One layer's group: the member weights and the M served in one launch."""

    __slots__ = ("slot", "prefix", "ws", "layouts", "ms")

    def __init__(self, slot, prefix, ws, ms):
        self.slot = slot
        self.prefix = prefix
        self.ws = ws
        self.layouts = tuple((w.shape, w.stride()) for w in ws)
        self.ms = frozenset(ms)


# Member weight data_ptr -> (group, member index). Empty unless registered.
_MEMBERS: dict[int, tuple[_Group, int]] = {}
_SLOTS: dict[str, _Slot] = {}


def _version(x: torch.Tensor) -> int:
    return -1 if x.is_inference() else x._version


def _run(group: _Group, i: int, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    slot = group.slot
    if (w.shape, w.stride()) != group.layouts[i]:
        return _tg.impl().thin_gemm(x, w)
    if x.shape[0] not in group.ms:
        if i == 0:
            slot.stash = None
        return _tg.impl().thin_gemm(x, w)
    if i == 0:
        y0, y1 = _tg.impl().thin_gemm_group_split(x, (w, group.ws[1]))
        slot.stash = (group, x, _version(x), x.shape, x.stride(), y1)
        slot.launches += 1
        return y0
    stash = slot.stash
    slot.stash = None
    if (
        stash is not None
        and stash[0] is group
        and stash[1] is x
        and stash[2] == _version(x)
        and stash[3] == x.shape
        and stash[4] == x.stride()
    ):
        slot.hits += 1
        return stash[5]
    slot.misses += 1
    logger.warning_once(
        "VLLM_THIN_GEMM_CONCAT: group %s member 1 ran alone (input not the "
        "group's)",
        slot.name,
    )
    return _tg.impl().thin_gemm(x, w)


def ampere_thin_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The package's ``ampere_thin_gemm`` with the group pairing."""
    if thin_gemm_supported(x, weight, bias):
        if _MEMBERS and not torch.compiler.is_compiling():
            member = _MEMBERS.get(weight.data_ptr())
            if member is not None:
                return _run(member[0], member[1], x, weight)
        return _tg.impl().thin_gemm(x, weight)
    return torch.nn.functional.linear(x, weight, bias)


def thin_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The package's ``thin_linear`` with the group pairing."""
    if _tg.use_ampere_thin_gemm() and thin_gemm_supported(x, weight, None):
        if _MEMBERS and not torch.compiler.is_compiling():
            member = _MEMBERS.get(weight.data_ptr())
            if member is not None:
                return _run(member[0], member[1], x, weight)
        return _tg.impl().thin_gemm(x, weight)
    return torch.nn.functional.linear(x, weight)


def _bf16_weight(w) -> bool:
    return (
        isinstance(w, torch.Tensor)
        and w.ndim == 2
        and w.dtype is torch.bfloat16
        and w.stride(1) == 1
    )


def _linear_weight(module):
    """The weight of a bias-free linear layer whose GEMM is this dispatch."""
    method = getattr(module, "quant_method", None)
    if getattr(method, "_gemm_impl", None) is not ampere_thin_gemm:
        return None
    if getattr(module, "bias", None) is not None:
        return None
    w = getattr(module, "weight", None)
    return w if _bf16_weight(w) else None


def _find_groups(models) -> list[tuple[str, str, tuple[torch.Tensor, ...]]]:
    """(group name, layer prefix, member weights) for every MLA layer that runs
    its indexer, in module order. Layers reachable twice are listed once."""
    found = []
    seen = set()
    for model in models:
        for prefix, module in model.named_modules():
            indexer = getattr(module, "indexer", None)
            if not isinstance(indexer, torch.nn.Module):
                continue
            # The MLA wrapper calls the indexer only for sparse layers that do
            # not reuse another layer's top-k.
            if not getattr(module, "is_sparse", False) or getattr(
                module, "skip_topk", True
            ):
                continue
            gate = getattr(indexer, "index_kpool_compress_gate", None)
            for name, w0, w1 in (
                (
                    "qb_wqb",
                    _linear_weight(getattr(module, "q_b_proj", None)),
                    _linear_weight(getattr(indexer, "wq_b", None)),
                ),
                (
                    "qkva_kpool",
                    _linear_weight(getattr(module, "fused_qkv_a_proj", None)),
                    gate if _bf16_weight(gate) else None,
                ),
            ):
                if w0 is None or w1 is None or w0.shape[1] != w1.shape[1]:
                    continue
                key = (w0.data_ptr(), w1.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                found.append((name, prefix, (w0, w1)))
    return found


def _set_but_off(reason: str) -> None:
    logger.warning_once("VLLM_THIN_GEMM_CONCAT set but off: %s", reason)


def register_groups(worker, capture_sizes) -> None:
    """Find, warm and register the concat groups (before graph capture)."""
    from vllm.config import CompilationMode
    from vllm_sm80.kernels.thin_gemm.warmup import _candidate_models, _token_sizes

    _MEMBERS.clear()
    _SLOTS.clear()
    if not envs.VLLM_THIN_GEMM:
        return _set_but_off("VLLM_THIN_GEMM is off")
    if not _device_supported():
        return _set_but_off("not an sm_80 device")
    mode = worker.vllm_config.compilation_config.mode
    if mode != CompilationMode.NONE:
        return _set_but_off(f"compilation mode {mode.name}, needs NONE")
    found = _find_groups(_candidate_models(worker))
    if not found:
        return _set_but_off("no MLA layer with an indexer on the thin GEMM")
    sizes = [
        _token_sizes(
            capture_sizes, max(max_rows(w.shape[0], w.shape[1]) for w in ws)
        )
        for _name, _prefix, ws in found
    ]
    if not any(sizes):
        return _set_but_off("no token size within dense_gemm/thin_max_rows")

    groups = []
    for (name, prefix, ws), group_sizes in zip(found, sizes):
        K = ws[0].shape[1]
        group_ms = []
        for M in group_sizes:
            x = torch.empty((M, K), dtype=torch.bfloat16, device="meta")
            if all(thin_gemm_supported(x, w, None) for w in ws):
                group_ms.append(M)
        if group_ms:
            groups.append((name, prefix, ws, group_ms))
    if not groups:
        return _set_but_off("no warmed M reaches the thin GEMM for both members")

    shapes = {}
    for name, _prefix, ws, group_ms in groups:
        key = (tuple(int(w.shape[0]) for w in ws), int(ws[0].shape[1]))
        shapes.setdefault(key, set()).update(group_ms)
    for key, group_ms in sorted(shapes.items()):
        _tg.impl().warmup((), sorted(group_ms), device=worker.device, groups=(key,))

    for name, prefix, ws, group_ms in groups:
        slot = _SLOTS.get(name)
        if slot is None:
            slot = _SLOTS[name] = _Slot(name)
        group = _Group(slot, prefix, ws, group_ms)
        for i, w in enumerate(ws):
            _MEMBERS[w.data_ptr()] = (group, i)

    names = [name for name in GROUPS if name in _SLOTS]
    for name in names:
        mine = [g for g in groups if g[0] == name]
        Ns, K = tuple(int(w.shape[0]) for w in mine[0][2]), int(mine[0][2][0].shape[1])
        logger.info(
            "Thin GEMM concat group %s: %d layers, (%s) x %d, one launch at M in %s.",
            name, len(mine), "+".join(map(str, Ns)), K, sorted(mine[0][3]),
        )
    logger.info_once(
        "sm_80 thin GEMM concat groups active (%s) [VLLM_THIN_GEMM_CONCAT=1]",
        ", ".join(names),
    )
