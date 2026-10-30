# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is the compressed flags all-reduce (VLLM_SM80_ALLREDUCE_COMPRESS) bitwise
the incumbent at TP4?

Under the served all-reduce settings with the switch on, the TP group's
custom all-reduce (``Sm80CustomAllreduce``) carries both paths, so every
message runs, in one process:

* ``inc``: ``_custom_all_reduce_base``, the dispatch with the switch off (the
  flags-in-data kernel up to its limit, else upstream's two-stage kernel);
* ``cand``: ``custom_all_reduce``, the dispatch with the switch on (one-shot,
  flags-in-data, compressed);
* ``z``: the compressed kernel itself (``_flags_z.all_reduce``) at every size
  of the custom all-reduce's range, also below the raw cut-off where the
  dispatch keeps the uncompressed kernels;
* ``one``: the one-shot kernel itself (``_flags.all_reduce(one_shot=True)``)
  at every size up to 64 KiB, also above its cut-off.

Messages: every captured served call of ``--real`` (all four ranks' inputs
and the live output, which ``inc``, ``cand``, ``z`` and ``one`` must all
equal), messages composed of captured rows at every row count 1..64 and
prefill sizes up to the largest the custom all-reduce takes, and generated
kinds (randn, wide exponents, the order-sensitive ``cancel``,
NaN/Inf/-0/subnormals and sentinel words, mostly zeros); then CUDA-graph
replays of the decode sizes (one graph per size with ``cand``, replayed over
many inputs, against eager ``inc``).

Per message: bits equal (NaN equals NaN), the input unchanged, every rank's
result the same; at the end no wait timed out (error records). Rank 0
prints one line per group and ``Z_BITS PASS`` or ``Z_BITS FAIL`` and writes
the JSON at ``--out``; the exit code is 0 when every case ran. Not a pytest
module: test_custom_all_reduce_flags_z_gpu.py runs it::

    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 VLLM_SM80_ALLREDUCE=1 \\
    VLLM_SM80_ALLREDUCE_COMPRESS=1 VLLM_CUSTOM_ALLREDUCE_ALGO=2stage \\
    NCCL_P2P_LEVEL=SYS VLLM_SM80_TUNING_FILE=served.toml \\
    torchrun --standalone --nproc-per-node 4 \\
        tests/distributed/flags_z_bits.py --out result.json \\
        --real ~/kernels/ampere-kernels/dsv41_allreduce_tp4/data/real_index.json
"""

import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist

from tests.models.deepseek_v41.sm80.wo_b_allreduce_bits import _engine_like_setup

HIDDEN = 5120
DECODE_ROWS = [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 24, 32, 40, 48]
PREFILL_ROWS = [51, 52, 64, 65, 96, 128, 256, 512, 819]
KINDS = ["randn", "wide", "cancel", "specials", "zeros"]
GRAPH_REPLAYS = 16
ONE_SHOT_FORCED_MAX = 65536


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bits differ, NaN equal to NaN (payloads are not part of
    the contract)."""
    differ = a.view(torch.int16) != b.view(torch.int16)
    return int((differ & ~(torch.isnan(a) & torch.isnan(b))).sum())


def _partial(kind: str, rows: int, rank: int, seed: int, dev) -> torch.Tensor:
    """Rank ``rank``'s input of a generated message."""
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + rank)
    shape = (rows, HIDDEN)
    x = torch.randn(shape, generator=g, device=dev)
    if kind == "randn":
        x = x * 0.05
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-30, 31, shape, generator=g, device=dev).float())
    elif kind == "cancel":
        # +-B (~2^20) on ranks 0 and 1: owner and rank order give other sums
        gb = torch.Generator(device=dev).manual_seed(seed * 7919 + 1000003)
        b = torch.randn(shape, generator=gb, device=dev) * 2.0**20
        x = (b if rank == 0 else -b) if rank < 2 else x * 2.0**-4
    elif kind == "specials":
        u = torch.rand(shape, generator=g, device=dev)
        x = x * 0.05
        x = torch.where(u < 0.01, torch.full_like(x, float("nan")), x)
        x = torch.where((u >= 0.01) & (u < 0.02), torch.full_like(x, float("inf")), x)
        x = torch.where((u >= 0.02) & (u < 0.03), torch.full_like(x, float("-inf")), x)
        x = torch.where((u >= 0.03) & (u < 0.05), torch.full_like(x, -0.0), x)
        x = torch.where((u >= 0.05) & (u < 0.07), x * 2.0**-124, x)  # subnormals
        out = x.to(torch.bfloat16).contiguous()
        words = out.view(torch.int32)
        words[torch.rand(words.shape, generator=g, device=dev) < 0.02] = -1  # sentinel
        out.view(torch.int16)[(u >= 0.07) & (u < 0.08)] = -1
        return out
    elif kind == "zeros":
        keep = torch.rand(shape, generator=g, device=dev) < 0.03
        x = torch.where(keep, x, torch.zeros_like(x))
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16).contiguous()


def _composed(pool: torch.Tensor, rows: int, rank: int, seed: int) -> torch.Tensor:
    """``rows`` captured rows (pool: [4, R, 5120]; every rank's partial of a
    row comes from the same captured call), drawn by a seed every rank
    shares."""
    g = torch.Generator().manual_seed(seed)
    n = pool.shape[1]
    idx = torch.cat([torch.randperm(n, generator=g) for _ in range(-(-rows // n))])[:rows]
    return pool[rank, idx.to(pool.device)].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--real", help="real_index.json of the captured served calls")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import get_tp_group, graph_capture

    from vllm_sm80.distributed.communicator import Sm80CustomAllreduce

    _engine_like_setup()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 4, "run with --nproc-per-node 4"
    torch.cuda.set_device(local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world, rank=rank, local_rank=local_rank,
            distributed_init_method="env://", backend="nccl",
        )
        ensure_model_parallel_initialized(world, 1)
    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    if not isinstance(ca, Sm80CustomAllreduce) or ca._flags is None or ca._flags_z is None:
        if rank == 0:
            print("REFUSED: the TP group's custom all-reduce has no compressed flags "
                  "kernel (VLLM_SM80_ALLREDUCE=1, VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1, "
                  "VLLM_SM80_ALLREDUCE_COMPRESS=1; see the log lines above)", flush=True)
        dist.destroy_process_group()
        raise SystemExit(2)
    dev = torch.device("cuda", local_rank)
    z = ca._flags_z
    config = {
        "flags_max_bytes": ca._flags.max_bytes,
        "one_shot_max_bytes": ca._flags_one_shot_max_bytes,
        "z_min_bytes": ca._flags_z_min_bytes,
        "z_upstream_from": ca._flags_z_upstream_from,
        "z_range_max_bytes": ca._flags_z_max_bytes,
        "z_max_bytes": z.max_bytes,
        "z_buffer_bytes": z.buffer_bytes,
        "custom_max_size": ca.max_size,
        "algo": os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO", ""),
    }

    cases = []

    def agree(t: torch.Tensor) -> bool:
        got = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(got, t)
        return all(_mismatch(g, got[0]) == 0 for g in got[1:])

    def route_of(x: torch.Tensor) -> str:
        if ca._flags_z_takes(x):
            return "z"
        if ca._one_shot_takes(x):
            return "one_shot"
        return "flags" if ca._flags.eligible(x) else "custom"

    def run(group: str, name: str, x: torch.Tensor, live: torch.Tensor | None = None):
        before = x.clone()
        route = route_of(x)
        inc = ca._custom_all_reduce_base(x)
        cand = ca.custom_all_reduce(x)
        forced = z.all_reduce(x) if x.nbytes <= ca._flags_z_max_bytes else None
        one = (ca._flags.all_reduce(x, one_shot=True)
               if x.nbytes <= min(ONE_SHOT_FORCED_MAX, ca._flags.max_bytes // 2) else None)
        torch.cuda.synchronize()
        case = {"group": group, "name": name, "bytes": x.nbytes, "route": route,
                "input_unchanged": torch.equal(before.view(torch.int16), x.view(torch.int16))}
        if inc is None:  # above the custom all-reduce's buffer: not ours
            case["skipped"] = True
            cases.append(case)
            return
        ref = live if live is not None else inc
        case["inc_ne_live"] = _mismatch(inc, live) if live is not None else 0
        case["cand_ne"] = _mismatch(cand, ref)
        case["z_ne"] = _mismatch(forced, ref) if forced is not None else 0
        case["one_ne"] = _mismatch(one, ref) if one is not None else 0
        case["one_forced"] = one is not None
        case["ranks_agree"] = agree(cand)
        cases.append(case)

    # 1. Captured served calls.
    pool = None
    if args.real:
        idx = json.loads(open(args.real).read())
        if _sha256(idx["pack"]) != idx["pack_sha256"]:
            raise SystemExit(f"captured pack {idx['pack']} changed (sha256)")
        blob = torch.load(idx["pack"], map_location="cpu", weights_only=True)
        rows_of_pool = []
        for c in idx["cases"]:
            xs = [blob[c["id"]]["x"][r] for r in range(world)]
            live = blob[c["id"]]["out"].to(dev)
            run("real", c["id"], xs[rank].to(dev).contiguous(), live)
            rows_of_pool.append(torch.stack([t.reshape(-1, HIDDEN) for t in xs]))
        pool = torch.cat(rows_of_pool, dim=1).to(dev)

    # 2. Composed captured rows and generated kinds at every size.
    sizes = list(range(1, 65)) + PREFILL_ROWS
    for seed in range(args.seeds):
        for rows in sizes:
            name, key = f"rows{rows}-s{seed}", 1000 * seed + rows
            if pool is not None:
                run("composed", name, _composed(pool, rows, rank, key))
            for kind in KINDS:
                run(kind, name, _partial(kind, rows, rank, key, dev))
    # Partitions that do not fill a block, and the largest messages.
    for packs in (1, 3, 4, 5, 31, 124, 125, 4 * 31 * 37 + 7, 524287):
        rows = -(-8 * packs // HIDDEN)
        x = _partial("wide", rows, rank, packs, dev).view(-1)[: 8 * packs].contiguous()
        run("packs", f"packs{packs}", x)

    # 3. CUDA-graph replays of the decode sizes and the slot sums (the switch-on dispatch).
    graph_msgs = [(rows, HIDDEN) for rows in DECODE_ROWS] + [(1, 320), (2, 320), (8, 256), (24, 256)]
    for shape in graph_msgs:
        static = torch.zeros(shape, dtype=torch.bfloat16, device=dev)
        graph = torch.cuda.CUDAGraph()
        with graph_capture(dev) as gc:
            ca.custom_all_reduce(static)  # warm-up, as vLLM's capture runs it
            with torch.cuda.graph(graph, stream=gc.stream):
                outs = [ca.custom_all_reduce(static) for _ in range(3)]
        for i in range(GRAPH_REPLAYS):
            kind = KINDS[i % len(KINDS)]
            if pool is not None and i % 2 and shape[1] == HIDDEN:
                x = _composed(pool, shape[0], rank, 77 * i + shape[0])
                kind = "composed"
            else:
                rows = -(-shape[0] * shape[1] // HIDDEN)
                x = _partial(kind, rows, rank, 77 * i + shape[0], dev).view(-1)[: static.numel()]
                x = x.view(shape).contiguous()
            static.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            ref = ca._custom_all_reduce_base(x)
            torch.cuda.synchronize()
            ne = sum(_mismatch(o, ref) for o in outs)
            cases.append({"group": "graph", "name": f"{shape}-{kind}-r{i}",
                          "bytes": x.nbytes, "route": route_of(x),
                          "cand_ne": ne, "z_ne": 0, "one_ne": 0, "inc_ne_live": 0,
                          "input_unchanged": True, "ranks_agree": True})
        del graph

    errors = [list(z.errors.read()), list(ca._flags.errors.read())]
    by_rank = [None] * world
    dist.all_gather_object(by_rank, {"cases": cases, "errors": errors})
    if rank == 0:
        all_cases = [c for r in by_rank for c in r["cases"]]
        ran = [c for c in all_cases if not c.get("skipped")]
        bad = [c for c in ran if c["cand_ne"] or c["z_ne"] or c["one_ne"] or c["inc_ne_live"]
               or not c["input_unchanged"] or not c["ranks_agree"]]
        # Word 0 of an error record is set when a wait timed out
        # (custom_all_reduce_flags.describe).
        no_errors = all(not rec[0] for r in by_rank for rec in r["errors"])
        routes_all = ("one_shot", "flags", "z", "custom")
        groups = sorted({c["group"] for c in ran})
        for g in groups:
            gc_ = [c for c in ran if c["group"] == g]
            routes = {r: sum(c["route"] == r for c in gc_) for r in routes_all}
            routes = {r: n for r, n in routes.items() if n}
            print(f"{g:15s} cases {len(gc_):5d} routes {routes} "
                  f"bad {sum(c in bad for c in gc_)}", flush=True)
        real = [c for c in ran if c["group"] == "real"]
        counts = {
            "z_for_upstream": sum(c["route"] == "z" and c["bytes"] > config["flags_max_bytes"]
                                  for c in ran),
            "one_shot": sum(c["route"] == "one_shot" for c in ran),
            "one_forced": sum(bool(c.get("one_forced")) for c in ran),
        }
        ok = (not bad and no_errors and counts["z_for_upstream"] > 0 and counts["one_shot"] > 0
              and (not args.real or len(real) == world * len(idx["cases"])))
        print(f"Z_BITS {'PASS' if ok else 'FAIL'}: {len(ran)} case-ranks, {len(bad)} bad, "
              f"real {len(real) // world}, wait errors {'none' if no_errors else 'SET'}; "
              f"counts {counts}; config {config}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"pass": ok, "config": config, "bad": bad[:50], "no_errors": no_errors,
                       "real_calls": len(real) // world, "cases": len(ran), **counts},
                      f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
