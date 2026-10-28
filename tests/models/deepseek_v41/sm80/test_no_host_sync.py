# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm_80 DeepSeek-V4.1 attention core reads no device value on the host.

Every aten op one layer issues is observed (prefill chunks, decode, a
spec-verify step, every layer type); none may be a host read: a scalar read,
``nonzero``, ``unique``, ``masked_select`` or indexing with a boolean mask.
The CPU fallbacks of the two Triton kernels (``index_scores``,
``sparse_attention``) are left out: CUDA runs the kernels instead.

    pytest -q tests/models/deepseek_v41/sm80/test_no_host_sync.py
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

from .dsv41_ref import (
    LayerSpec,
    PagedRun,
    RefArgs,
    core_config,
    core_weights,
    layer_inputs,
    make_weights,
)

_SYNC_OPS = {
    "_local_scalar_dense",
    "item",
    "is_nonzero",
    "nonzero",
    "unique",
    "_unique2",
    "unique_dim",
    "masked_select",
}


class _SyncCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.syncs: list[str] = []
        self.paused = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]
        if not self.paused:
            if name in _SYNC_OPS:
                self.syncs.append(name)
            if name in ("index", "index_put", "index_put_", "_index_put_impl_"):
                idx = args[1] if len(args) > 1 else ()
                if any(
                    isinstance(t, torch.Tensor) and t.dtype == torch.bool
                    for t in (idx or ())
                ):
                    self.syncs.append(f"{name}[bool mask]")
        return func(*args, **(kwargs or {}))


def _paused(counter, fn):
    def run(*a, **k):
        counter.paused += 1
        try:
            return fn(*a, **k)
        finally:
            counter.paused -= 1

    return run


def test_no_host_reads_in_any_layer_type(monkeypatch) -> None:
    args = RefArgs(window_size=16, index_topk=8)
    stacks = [
        [LayerSpec(0)],
        [LayerSpec(2, kv_source=True, index_source=True), LayerSpec(2)],
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ]
    # Prefill chunks (one closing a ratio-2 group across chunks), decode on
    # odd and even positions, a 6-token verify, two requests interleaved.
    schedule = [[(0, 23)], [(0, 18), (1, 5)], [(0, 1), (1, 1)], [(0, 1)],
                [(1, 6), (0, 1)]]
    g = torch.Generator().manual_seed(0)
    h = args.n_heads // args.world_size
    for specs in stacks:
        counter = _SyncCounter()
        monkeypatch.setattr(core, "index_scores", _paused(counter, core.index_scores))
        monkeypatch.setattr(
            core, "sparse_attention", _paused(counter, core.sparse_attention)
        )
        weights = [make_weights(args, s, 10 + i) for i, s in enumerate(specs)]
        run = PagedRun(args, specs)
        cfgs = [core_config(args, s) for s in specs]
        cws = [core_weights(args, s, w, "cpu") for s, w in zip(specs, weights)]
        for sched in schedule:
            batch = run.build(sched)
            T = batch.num_tokens
            for li, spec in enumerate(specs):
                x = torch.randn(T, args.dim, generator=g).to(torch.bfloat16)
                qr_kv, kv_score, iw = layer_inputs(args, spec, weights[li], x)
                out = torch.empty((T + 2, h, args.head_dim), dtype=torch.bfloat16)
                qr_kv = torch.cat([qr_kv, qr_kv[:2]])
                kv_score = None if kv_score is None else torch.cat([kv_score, kv_score[:2]])
                iw = None if iw is None else torch.cat([iw, iw[:2]])
                with counter:
                    core.forward_layer(
                        cfgs[li], cws[li], run.caches(li), batch, qr_kv, kv_score,
                        iw, out,
                    )
                assert counter.syncs == [], (specs[li], sched, counter.syncs)
            run.after_step()
        monkeypatch.undo()
