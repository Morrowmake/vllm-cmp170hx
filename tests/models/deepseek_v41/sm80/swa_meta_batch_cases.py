# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cases of ``test_sm80_swa_meta_batch.py`` (CPU, in a subprocess with
``TRITON_INTERPRET=1``) and ``test_sm80_swa_meta_batch_gpu.py`` (CUDA).

The served layout: 43 sliding-window KV cache groups (window 128, 32-token
blocks, one ``Sm80SWAMetadataBuilder`` each, every group with its own block
ids) and one other group after them, built by the runner's
``build_attn_metadata`` with DeepSeek-V4.1's ``ReplayAttnMetadata``, the
block tables and the stacked slot mappings held in persistent buffers as
``BlockTables`` holds them. Every scenario runs per group (switch off), then
twice with the switch on (the first call of the run learns the groups),
each on builder buffers filled with the same poison first. The same runs
with groups whose block tables are kernel-block tables (16-token kernel
blocks: ``AttentionGroup`` fills them in the group's own build), and a run
that discards a learned layout and checks that nothing keeps it alive.
:func:`run_all` returns what :func:`check` asserts on.

    TRITON_INTERPRET=1 python -m tests.models.deepseek_v41.sm80.swa_meta_batch_cases
"""

import dataclasses
import gc
import json
import os
import weakref
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

NUM_SWA = 43
WINDOW = 128
BLOCK = 32
KERNEL_BLOCK = 16
SPEC_K = 5
MAX_TOKENS = 256
MAX_SEQS = 8
MAX_BLOCKS = 64
LENS = [400, 40, 127, 128, 129, 300, 70, 1000]
SWITCH = "VLLM_DEEPSEEK_V41_SWA_META_BATCH"

# name: (requests (rows of LENS), query lengths, padding rows, replay starts,
# variant)
SCENARIOS = {
    "c1_verify_6": ([0], [6], 0, None, None),
    "c8_32": (list(range(8)), [4] * 8, 0, None, None),
    "c1_verify_6_padded": ([4], [6], 2, None, None),
    # Two decode requests and a prefill whose window starts at its replay
    # start.
    "mixed_prefill_decode": ([0, 3, 5], [6, 6, 60], 0, [0, 0, 200], None),
    "window_crossing": ([2, 3, 4], [6, 6, 6], 0, None, None),
    # Group 5 reads another block-table tensor than the one learned: it
    # builds per group, the others take the batch.
    "changed_block_table": ([0], [6], 0, None, "changed_block_table"),
    # DSpark's draft metadata (non-causal): every group per group.
    "noncausal_draft": ([0], [6], 0, None, "noncausal"),
}
CHANGED_GROUP = 5
# Layouts with kernel-block groups: every window group, or one.
KERNEL_LAYOUTS = {"all_kernel_blocks": tuple(range(NUM_SWA)), "one_kernel_block": (7,)}
KERNEL_SCENARIOS = ("c1_verify_6", "c8_32")
SWA = "_compute_swa_indices_and_lens_kernel"
PREFILL = "kernel"  # ComputePrefillMetadataKernel.kernel
BATCH = "_swa_meta_batch_kernel"
T2R = "_token_request_mapping_kernel"
GE, FILL = "aten.ge", "aten.fill_"
BUFFERS = (
    "decode_swa_indices",
    "decode_swa_lens",
    "prefill_swa_indices",
    "prefill_swa_lens",
    "is_valid_token",
)
# Metadata fields that must be views of the same builder buffers either way
# (the captured graphs read those buffers).
ALIASED = (*BUFFERS, "token_to_req_indices")


def vllm_config():
    """What the window builder reads of the served config."""
    hf = SimpleNamespace(sliding_window=WINDOW, compress_ratios=[0, 1, 2])
    speculative = SimpleNamespace(
        num_speculative_tokens=SPEC_K,
        parallel_drafting=False,
        use_dspark=lambda: True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096, hf_config=hf),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_TOKENS, max_num_seqs=MAX_SEQS
        ),
        speculative_config=speculative,
        num_speculative_tokens=SPEC_K,
        use_v2_model_runner=True,
    )


class _OtherBuilder:
    """The group after the window groups (a compressed cache's builder)."""

    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        return "other"


def layout(device: torch.device, kernel_block_groups=()):
    """The groups, as ``AttentionGroup.create_metadata_builders`` makes them
    (the groups in ``kernel_block_groups`` with 16-token kernel blocks), and
    the runner's persistent inputs."""
    from vllm_sm80.models.deepseek_v41.sm80_attention import (
        Sm80SWABackend,
        Sm80SWAMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    from vllm.v1.worker.utils import AttentionGroup

    config = vllm_config()
    groups, builders = [], []
    for i in range(NUM_SWA):
        spec = SlidingWindowMLASpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=WINDOW,
            cache_dtype_str="auto",
            model_version="deepseek_v4",
        )
        name = f"layers.{i}.swa_cache"
        group = AttentionGroup(Sm80SWABackend, [name], spec, i)
        builder_spec = spec
        if i in kernel_block_groups:
            per_block = BLOCK // KERNEL_BLOCK
            builder_spec = spec.copy_with_new_block_size(KERNEL_BLOCK)
            group.kernel_block_size = KERNEL_BLOCK
            group.kernel_block_table = torch.zeros(
                1, MAX_SEQS, MAX_BLOCKS * per_block, dtype=torch.int32, device=device
            )
            group.kernel_block_offsets = torch.arange(
                per_block, dtype=torch.int32, device=device
            )
        builder = Sm80SWAMetadataBuilder(builder_spec, [name], config, device)
        group.metadata_builders = [builder]
        builders.append(builder)
        groups.append([group])
    other = AttentionGroup(
        Sm80SWABackend, ["layers.0.attn"], None, NUM_SWA,
        metadata_builders=[_OtherBuilder()],
    )
    groups.append([other])
    kv_cache_config = SimpleNamespace(kv_cache_groups=[None] * len(groups))
    # Every window group draws its own blocks: distinct ids, in no order.
    gen = torch.Generator().manual_seed(5)
    per_group = MAX_SEQS * MAX_BLOCKS
    ids = torch.randperm(NUM_SWA * per_group, generator=gen).to(torch.int32) + 1
    sources = [
        ids[i * per_group : (i + 1) * per_group].view(MAX_SEQS, MAX_BLOCKS).clone()
        for i in range(NUM_SWA)
    ] + [torch.zeros(MAX_SEQS, MAX_BLOCKS, dtype=torch.int32)]
    # The runner's persistent input block tables and stacked slot mappings.
    block_tables = [torch.zeros_like(s, device=device) for s in sources]
    slot_mappings = torch.full(
        (len(groups), MAX_TOKENS), -1, dtype=torch.int64, device=device
    )
    return groups, builders, kv_cache_config, sources, block_tables, slot_mappings


def step_inputs(scenario, sources, block_tables, slot_mappings) -> dict:
    """Write the scenario into the persistent block tables (the batch's rows
    first, as ``BlockTables.gather_block_tables`` does) and slot mappings;
    return the step's batch-level inputs."""
    rows, query_lens, pad, replay, variant = SCENARIOS[scenario]
    device = slot_mappings.device
    seq_lens = [LENS[r] for r in rows]
    num_reqs = len(rows)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)])
    req = torch.repeat_interleave(torch.arange(num_reqs), torch.tensor(query_lens))
    slots = torch.full_like(slot_mappings, -1, device="cpu")
    for g, (table, source) in enumerate(zip(block_tables, sources)):
        table[:num_reqs] = source[rows].to(device)
        if g < NUM_SWA:
            rows_g = source[rows]
            slots[g, : len(pos)] = rows_g[req, pos // BLOCK].long() * BLOCK + pos % BLOCK
    slot_mappings.copy_(slots)
    return dict(
        num_reqs=num_reqs,
        num_tokens=sum(query_lens) + pad,
        qsl=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_query_len=max(query_lens),
        replay=None if replay is None else torch.tensor(replay, dtype=torch.int32),
        variant=variant,
    )


def call_inputs(inp: dict, block_tables, slot_mappings) -> dict:
    """One call's arguments, the batch-level tensors fresh as the runner's
    ``prepare_attn`` slices them."""
    device = slot_mappings.device
    n, t = inp["num_reqs"], inp["num_tokens"]
    changed = inp["variant"] == "changed_block_table"
    replay = inp["replay"]
    return dict(
        num_reqs=n,
        num_tokens=t,
        query_start_loc_gpu=inp["qsl"].to(device)[: n + 1],
        query_start_loc_cpu=inp["qsl"].clone()[: n + 1],
        max_query_len=inp["max_query_len"],
        seq_lens=inp["seq_lens"].to(device),
        max_seq_len=int(inp["seq_lens"].max()),
        block_tables=tuple(
            bt[:n].clone() if changed and g == CHANGED_GROUP else bt[:n]
            for g, bt in enumerate(block_tables)
        ),
        slot_mappings=slot_mappings[:, :t],
        seq_lens_cpu_upper_bound=inp["seq_lens"].clone(),
        is_prefilling=torch.zeros(n, dtype=torch.bool),
        replay=None if replay is None else replay.to(device),
        causal=inp["variant"] != "noncausal",
    )


def build(groups, kv_cache_config, call: dict):
    from vllm.models.deepseek_v41.nvidia.model_state import ReplayAttnMetadata
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    call = dict(call)
    replay = call.pop("replay")
    return build_attn_metadata(
        attn_groups=groups,
        kv_cache_config=kv_cache_config,
        model_specific_attn_metadata=(
            None if replay is None else ReplayAttnMetadata(replay)
        ),
        **call,
    )


def poison(builders, groups):
    """The same stale bytes in every buffer a build writes, before each run
    (kernel-block tables included: a launch that read one before its group's
    build refreshed it reads these)."""
    for b in builders:
        for name in BUFFERS:
            buf = getattr(b, name)
            buf.fill_(True if buf.dtype == torch.bool else -7)
        b.token_to_req_indices.fill_(-7)
        if b.decode_swa_indices_noncausal is not None:
            b.decode_swa_indices_noncausal.fill_(-7)
    for (group,) in groups[:NUM_SWA]:
        if group.kernel_block_table is not None:
            group.kernel_block_table.fill_(-7)


def _bytes(t: torch.Tensor) -> str:
    return t.detach().cpu().numpy().tobytes().hex()


def snapshot(metadata, builders):
    """Per window group: every metadata field (tensors as dtype, shape and
    bytes; the builder-buffer views also by address) and every buffer."""
    out = []
    for i, b in enumerate(builders):
        md = metadata[f"layers.{i}.swa_cache"]
        fields = {}
        for f in dataclasses.fields(md):
            v = getattr(md, f.name)
            if isinstance(v, torch.Tensor):
                fields[f.name] = (
                    str(v.dtype),
                    tuple(v.shape),
                    _bytes(v),
                    v.data_ptr() if f.name in ALIASED else None,
                )
            elif f.name.startswith("tile_sched"):
                fields[f.name] = v is None
            else:
                fields[f.name] = repr(v)
        buffers = {name: _bytes(getattr(b, name)) for name in BUFFERS}
        out.append((fields, buffers))
    return out


class _Ops(TorchDispatchMode):
    """The torch ops that launch a CUDA compare or fill kernel, counted as
    the CUDA profiler counts those kernels: a zero-filled allocation is a
    fill launch on CUDA."""

    FILLS = frozenset(
        ("fill_", "fill", "zero_", "zeros", "zeros_like", "full", "full_like")
    )

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._overloadpacket.__name__
        if name == "ge":
            self.counts[GE] += 1
        elif name in self.FILLS:
            self.counts[FILL] += 1
        return func(*args, **(kwargs or {}))


@contextmanager
def _interpreted_launches():
    from triton.runtime.interpreter import InterpretedFunction

    run = InterpretedFunction.run
    counts = Counter()

    def counting(self, *args, grid, warmup, **kwargs):
        if not warmup:
            counts[self.fn.__name__] += 1
        return run(self, *args, grid=grid, warmup=warmup, **kwargs)

    InterpretedFunction.run = counting
    ops = _Ops()
    try:
        with ops:
            yield counts
    finally:
        InterpretedFunction.run = run
        counts.update(ops.counts)


def _cuda_name(name: str) -> str:
    if "compare_scalar_kernel" in name:
        return GE
    if "FillFunctor" in name:
        return FILL
    return name


@contextmanager
def _cuda_launches():
    from torch.profiler import ProfilerActivity, profile

    counts = Counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield counts
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[_cuda_name(event.name)] += 1


def launches(device: torch.device):
    """Counts the launches of the block by kernel name; the torch ge and fill
    kernels as ``aten.ge`` and ``aten.fill_``."""
    return _cuda_launches() if device.type == "cuda" else _interpreted_launches()


def _differ(run, ref) -> tuple[list[int], list[str]]:
    groups = [i for i in range(NUM_SWA) if run[i] != ref[i]]
    names = sorted(
        {k for i in groups for k in ref[i][0] if run[i][0][k] != ref[i][0][k]}
        | {k for i in groups for k in ref[i][1] if run[i][1][k] != ref[i][1][k]}
    )
    return groups, names


@contextmanager
def _switch():
    from vllm_sm80.models.deepseek_v41 import sm80_swa_meta_batch

    saved = os.environ.get(SWITCH)
    sm80_swa_meta_batch.reset()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = saved
        sm80_swa_meta_batch.reset()


def _scenarios(device: torch.device, names, kernel_block_groups=()) -> dict:
    """Each scenario per group, then twice batched, on one layout."""
    groups, builders, kv_cache_config, sources, block_tables, slot_mappings = layout(
        device, kernel_block_groups
    )
    result = {}
    with _switch():
        for scenario in names:
            inp = step_inputs(scenario, sources, block_tables, slot_mappings)
            runs = {}
            for arm in ("off", "learn", "on"):
                os.environ[SWITCH] = "0" if arm == "off" else "1"
                poison(builders, groups)
                call = call_inputs(inp, block_tables, slot_mappings)
                with launches(device) as counts:
                    md = build(groups, kv_cache_config, call)
                runs[arm] = (snapshot(md, builders), dict(counts), md["layers.0.attn"])
            ref = runs["off"][0]
            result[scenario] = {}
            for arm, (snap, counts, other) in runs.items():
                differ, fields = _differ(snap, ref)
                result[scenario][arm] = {
                    "differ": differ,
                    "differ_fields": fields,
                    "launches": counts,
                    "other": other,
                }
            # Each window group's metadata is its own: its indices differ.
            decode = {ref[i][0]["decode_swa_indices"][2] for i in range(NUM_SWA)}
            result[scenario]["distinct_decode_indices"] = len(decode)
    return result


def _released(device: torch.device) -> dict:
    """Learn a layout and launch for it, then discard it as the runner
    discards builders and buffers (profiling, decoder-replay copies, a KV
    cache reinitialisation): nothing may keep them alive."""
    groups, builders, kv_cache_config, sources, block_tables, slot_mappings = layout(
        device
    )
    refs = [weakref.ref(b) for b in builders]
    refs += [weakref.ref(t) for t in block_tables] + [weakref.ref(slot_mappings)]
    with _switch():
        os.environ[SWITCH] = "1"
        inp = step_inputs("c1_verify_6", sources, block_tables, slot_mappings)
        total = Counter()
        for _ in range(2):
            with launches(device) as counts:
                md = build(
                    groups, kv_cache_config, call_inputs(inp, block_tables, slot_mappings)
                )
            total.update(counts)
        del md, groups, builders, kv_cache_config, sources, block_tables, slot_mappings
        gc.collect()
        alive = sum(r() is not None for r in refs)
    return {"launched": total.get(BATCH, 0), "alive": alive, "checked": len(refs)}


def run_all(device: torch.device) -> dict:
    return {
        "scenarios": _scenarios(device, list(SCENARIOS)),
        "kernel_blocks": {
            name: _scenarios(device, KERNEL_SCENARIOS, groups)
            for name, groups in KERNEL_LAYOUTS.items()
        },
        "released": _released(device),
    }


def _check_identical(scenarios: dict) -> None:
    for scenario, case in scenarios.items():
        # Why one result cannot serve every layer: each group's indices are
        # slots of its own blocks.
        assert case["distinct_decode_indices"] == NUM_SWA, scenario
        for arm in ("learn", "on"):
            assert case[arm]["differ"] == [], (scenario, arm, case[arm]["differ_fields"])
            assert case[arm]["other"] == "other"
        off, on = case["off"]["launches"], case["on"]["launches"]
        # The token-to-request map launches as often as per group (once a
        # step; kernel-block groups build it again on their own copy of the
        # common metadata).
        assert on.get(T2R) == off.get(T2R), scenario
        assert off.get(BATCH, 0) == 0, scenario


def check(result: dict) -> None:
    """The batched metadata is the per-group build's, in one launch."""
    scenarios = result["scenarios"]
    assert list(scenarios) == list(SCENARIOS)
    _check_identical(scenarios)
    for scenario, case in scenarios.items():
        off, on = case["off"]["launches"], case["on"]["launches"]
        assert on.get(T2R) == 1, scenario
        if scenario == "noncausal_draft":
            assert on.get(BATCH, 0) == 0 and on.get(GE) == NUM_SWA
            continue
        prefill_rows = scenario == "mixed_prefill_decode"
        # Per group: ge, the lens fill (decode rows), the decode and prefill
        # index launches and the prefill gather lengths.
        assert off.get(GE) == NUM_SWA, scenario
        assert off.get(FILL) == NUM_SWA, scenario
        assert off.get(SWA) == NUM_SWA * (2 if prefill_rows else 1), scenario
        assert off.get(PREFILL, 0) == (NUM_SWA if prefill_rows else 0), scenario
        fallback = 1 if scenario == "changed_block_table" else 0
        assert on.get(BATCH) == 1, scenario
        assert on.get(GE, 0) == fallback, scenario
        assert on.get(FILL, 0) == fallback, scenario
        assert on.get(SWA, 0) == fallback, scenario
        assert on.get(PREFILL, 0) == 0, scenario
    # The first call with the switch on learns the groups: per group.
    learn = scenarios[next(iter(SCENARIOS))]["learn"]["launches"]
    assert learn.get(BATCH, 0) == 0 and learn.get(SWA) == NUM_SWA

    # Kernel-block tables are filled in their group's own build: such a group
    # builds per group; the others still take the batch.
    kernel = result["kernel_blocks"]
    for name, mapped in KERNEL_LAYOUTS.items():
        _check_identical(kernel[name])
        for scenario, case in kernel[name].items():
            on = case["on"]["launches"]
            batched = NUM_SWA - len(mapped)
            assert on.get(BATCH, 0) == (1 if batched > 1 else 0), (name, scenario)
            assert on.get(GE, 0) == len(mapped), (name, scenario)
            assert on.get(SWA, 0) == len(mapped), (name, scenario)

    released = result["released"]
    assert released["launched"] == 1
    assert released["alive"] == 0, released


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1"
    print("RESULT " + json.dumps(run_all(torch.device("cpu"))))
