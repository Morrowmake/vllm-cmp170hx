# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do the TP4 all-reduces of BF16 wo_b partials give the same bits as the
FP32 path the sm_80 DeepSeek-V4.1 attention uses today?

Today ``DeepseekV41Sm80Attention._o_proj`` all-reduces ``partial.float()``
and rounds the sum to BF16 once. The served kernels (flags two-shot, custom
1/2-stage, host-staged) accumulate BF16 inputs in FP32 in a fixed order and
round once, so all-reducing the BF16 partial directly may give the same bits
at half the bytes. For every row count, on rank-distinct random partials of
the wo_b output shape ``[rows, 5120]``, this compares

* ``bf16``: ``all_reduce(partial)`` (BF16 message);
* ``fp32``: ``all_reduce(partial.float()).bfloat16()`` (today's path);
* ``exact``: the FP64 sum of the four gathered partials rounded to BF16;

* ``switched``: ``wo_b_all_reduce(partial, switch)``, the layer's own call
  with ``switch`` = VLLM_SM80_ALLREDUCE_ROUND_ONCE as the process environment
  sets it (on: BF16 where the host-staged or custom all-reduce takes the
  message whole, else through the custom all-reduce in pieces where the
  message is too large for it whole); the JSON records the switch and the
  path each message took (``bf16``, ``bf16_pieces`` or ``fp32``);

reports which backend each message takes, bitwise equality, mismatch counts
and the largest BF16-ulp distance (also of ``switched`` and ``fp32`` against
``exact``), and the median time of each all-reduce. ``switched`` must equal
``fp32`` (verdict line ``SWITCHED``), or where it differs be no less accurate
against ``exact`` in mismatch count and largest ulp distance.
Every rank's switched bits are gathered and compared byte for byte. Rank 0
prints a table and writes every rank's diagnostics to JSON at ``--out``. The exit code is 0 when
every case ran; the verdict line says whether ``bf16`` equals ``fp32``.

Not a pytest module: it needs four peer-connected sm_80 GPUs, one process
each, under the served all-reduce switches (test_wo_b_allreduce_bits_gpu.py
runs it and judges its JSON). It loads the plugins and puts ``ninja`` on PATH
as the engine has them, and exits 2 without
measuring when the communicator is not the served one (with
VLLM_SM80_ALLREDUCE=1: the flags-in-data kernel must be up)::

    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 VLLM_SM80_ALLREDUCE=1 \\
    VLLM_CUSTOM_ALLREDUCE_ALGO=2stage NCCL_P2P_LEVEL=SYS \\
    torchrun --standalone --nproc-per-node 4 \\
        tests/models/deepseek_v41/sm80/wo_b_allreduce_bits.py --out result.json
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from vllm_sm80.distributed.step_errors import copy_error_flags

from tests.models.deepseek_v41.sm80.exact_round import exact_sum, reference_round

# 819 rows of 5120 BF16 are the largest message below the 8 MiB custom
# all-reduce buffer, 820 the smallest above it.
ROWS = [1, 2, 4, 8, 16, 24, 25, 26, 32, 51, 52, 64, 128, 256, 512, 819, 820, 1024, 2048]
HIDDEN = 5120


def _ordered(t: torch.Tensor) -> torch.Tensor:
    bits = t.view(torch.int16).int() & 0xFFFF
    return torch.where(bits >= 0x8000, 0x8000 - bits, bits)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_ordered(a) - _ordered(b)).abs().max()) if a.numel() else 0


def _backend(comm, t: torch.Tensor) -> str:
    """Which backend the fork communicator dispatches ``t`` to (its own
    predicates, evaluated in its dispatch order)."""
    host = getattr(comm, "hostshm_comm", None)
    if host is not None and host.should_host_ar(t):
        return "hostshm"
    ca = getattr(comm, "ca_comm", None)
    if ca is not None and not ca.disabled and ca.should_custom_ar(t):
        flags = getattr(ca, "_flags", None)
        if flags is not None and flags.eligible(t):
            return "flags"
        return "custom"
    return "nccl"


def _ninja_dir() -> str | None:
    """The directory of the ``ninja`` torch's JIT loader runs: one already on
    PATH, else the interpreter's ``bin``, else the binary the ``ninja``
    package installed (its scripts directory, which is not the interpreter's
    ``bin`` when the package comes from another environment's site-packages).
    None when there is none."""
    import shutil
    import sys
    from importlib import metadata

    found = shutil.which("ninja")
    if found:
        return os.path.dirname(found)
    candidates = [os.path.join(os.path.dirname(sys.executable), "ninja")]
    try:
        dist = metadata.distribution("ninja")
    except metadata.PackageNotFoundError:
        dist = None
    for f in (dist.files or ()) if dist is not None else ():
        if f.name == "ninja" and f.parent.name == "bin":
            candidates.append(os.path.realpath(dist.locate_file(f)))
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.dirname(c)
    return None


def _engine_like_setup() -> None:
    """What the served engine has before it builds the TP group: vLLM's
    general plugins loaded (the fork's included, checked by its guard), and
    ``ninja`` on PATH (the served venv's ``bin`` holds it). The flags-in-data
    kernel is a JIT extension and torch's loader needs ``ninja`` even for a
    cached build; without it the communicator logs "Ninja is required" and
    leaves the flags path off. A test interpreter whose packages come from
    another environment has no ``ninja`` in its own ``bin``: see _ninja_dir."""
    from vllm.plugins import load_general_plugins
    from vllm_sm80.guard import check_plugin_loaded

    ninja_dir = _ninja_dir()
    path = os.environ.get("PATH", "")
    if ninja_dir is not None and ninja_dir not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join(p for p in (ninja_dir, path) if p)
    load_general_plugins()
    check_plugin_loaded()


def _refuse_unless_served(comm) -> str | None:
    """Why this communicator is not the served one, or None."""
    import vllm.envs as envs
    from vllm_sm80.distributed.communicator import (
        Sm80CudaCommunicator,
        needs_fork_communicator,
    )

    if needs_fork_communicator() and not isinstance(comm, Sm80CudaCommunicator):
        return f"the TP communicator is {type(comm).__name__}, not the fork's"
    if envs.VLLM_SM80_ALLREDUCE:
        ca = getattr(comm, "ca_comm", None)
        if ca is None or ca.disabled:
            return "VLLM_SM80_ALLREDUCE=1 but the custom all-reduce is off"
        if getattr(ca, "_flags", None) is None:
            return (
                "VLLM_SM80_ALLREDUCE=1 but the flags-in-data kernel is "
                "not up (see the 'flags-in-data' log line above for why)"
            )
    return None


def _time(fn, iters: int) -> float:
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        errors = copy_error_flags()
        end.record()
        torch.cuda.synchronize()
        if errors is not None:
            errors.raise_if_failed()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_sm80.models.deepseek_v41.sm80_attention import (
        custom_all_reduce_pieces,
        custom_all_reduce_takes,
        wo_b_all_reduce,
    )

    _engine_like_setup()
    import vllm.envs as envs

    switch = bool(envs.VLLM_SM80_ALLREDUCE_ROUND_ONCE)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 4, "run with --nproc-per-node 4"
    torch.cuda.set_device(local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        ensure_model_parallel_initialized(world, 1)
    comm = get_tp_group().device_communicator
    reason = _refuse_unless_served(comm)
    if reason is not None:
        if rank == 0:
            print(f"REFUSED: {reason}", flush=True)
        dist.destroy_process_group()
        raise SystemExit(2)
    dev = torch.device("cuda", local_rank)

    results = []
    for rows in ROWS:
        for seed in range(args.seeds):
            g = torch.Generator(device=dev).manual_seed(1000 * seed + rows + 7 * rank)
            # wo_b partials: sums of ~2048 products, magnitudes around 1 with
            # a heavy tail; a scale per row so exponents vary across rows.
            scale = torch.exp2(
                torch.randint(-6, 4, (rows, 1), generator=g, device=dev).float()
            )
            partial = (
                torch.randn(rows, HIDDEN, generator=g, device=dev) * scale
            ).to(torch.bfloat16)

            # Just above a BF16 midpoint; conversion through FP32 loses the
            # final summand. Include both signs on every tested message size.
            midpoint = (1.0, 2.0**-8, 2.0**-30, 0.0)[rank]
            partial[0, 0] = midpoint
            partial[0, 1] = -midpoint
            bf16 = tensor_model_parallel_all_reduce(partial.clone())
            fp32 = tensor_model_parallel_all_reduce(partial.float()).to(torch.bfloat16)
            switched = wo_b_all_reduce(partial.clone(), switch)
            errors = copy_error_flags(dev)
            gathered = [torch.empty_like(partial) for _ in range(world)]
            dist.all_gather(gathered, partial)
            exact = reference_round(exact_sum(torch.stack(gathered)), torch.bfloat16)
            switched_bits = switched.view(torch.uint8)
            all_bits = [torch.empty_like(switched_bits) for _ in range(world)]
            dist.all_gather(all_bits, switched_bits)
            ranks_agree = all(torch.equal(b, all_bits[0]) for b in all_bits[1:])
            host = getattr(get_tp_group().device_communicator, "hostshm_comm", None)
            host_takes = host is not None and host.should_host_ar(partial)
            if switch and (host_takes or custom_all_reduce_takes(partial)):
                path = "bf16"
            elif switch and custom_all_reduce_pieces(partial) is not None:
                path = "bf16_pieces"
            else:
                path = "fp32"

            case = {
                "rank": rank,
                "switched_ranks_agree": ranks_agree,
                "rows": rows,
                "seed": seed,
                "backend_bf16": _backend(comm, partial),
                "backend_fp32": _backend(comm, partial.float()),
                "switched_bf16": path != "fp32",
                "switched_path": path,
                "switched_eq_fp32": bool(torch.equal(switched, fp32)),
                "switched_ne_exact": int(
                    (switched.view(torch.int16) != exact.view(torch.int16)).sum()
                ),
                "ulps_switched_exact": _ulps(switched, exact),
                "ulps_fp32_exact": _ulps(fp32, exact),
                "bf16_eq_fp32": bool(torch.equal(bf16, fp32)),
                "bf16_ne_fp32": int((bf16.view(torch.int16) != fp32.view(torch.int16)).sum()),
                "ulps_bf16_fp32": _ulps(bf16, fp32),
                "bf16_ne_exact": int((bf16.view(torch.int16) != exact.view(torch.int16)).sum()),
                "fp32_ne_exact": int((fp32.view(torch.int16) != exact.view(torch.int16)).sum()),
            }
            if errors is not None:
                errors.raise_if_failed()
            if seed == 0:
                case["us_bf16"] = _time(
                    lambda: tensor_model_parallel_all_reduce(partial), args.iters
                )
                pf = partial.float()
                case["us_fp32"] = _time(
                    lambda: tensor_model_parallel_all_reduce(pf).to(torch.bfloat16),
                    args.iters,
                )
                case["us_switched"] = _time(
                    lambda: wo_b_all_reduce(partial, True), args.iters
                )
            results.append(case)

    # Preserve every rank's diagnostics, not just rank zero's verdict.
    results_by_rank = [None] * world
    dist.all_gather_object(results_by_rank, results)
    all_cases = [case for rank_cases in results_by_rank for case in rank_cases]
    ranks_agree = all(c["switched_ranks_agree"] for c in all_cases)
    if rank == 0:
        equal = all(c["bf16_eq_fp32"] for c in all_cases)
        switched_ok = ranks_agree and all(c["switched_eq_fp32"] for c in all_cases)
        no_less_accurate = ranks_agree and all(
            c["switched_eq_fp32"]
            or (
                c["switched_ne_exact"] <= c["fp32_ne_exact"]
                and c["ulps_switched_exact"] <= c["ulps_fp32_exact"]
            )
            for c in all_cases
        )
        for c in results:
            t = (
                f" bf16 {c['us_bf16']:.1f} us fp32 {c['us_fp32']:.1f} us"
                f" switched {c['us_switched']:.1f} us"
                if "us_bf16" in c
                else ""
            )
            print(
                f"rows {c['rows']:5d} seed {c['seed']} "
                f"{c['backend_bf16']:>16s}/{c['backend_fp32']:<16s} "
                f"switched {c['switched_path']:<11s} "
                f"eq {c['switched_eq_fp32']!s:5s} "
                f"bf16==fp32 {c['bf16_eq_fp32']!s:5s} ne {c['bf16_ne_fp32']:7d} "
                f"ulps {c['ulps_bf16_fp32']:3d} | vs exact bf16 "
                f"{c['bf16_ne_exact']:7d} fp32 {c['fp32_ne_exact']:7d}{t}"
            )
        verdict = (
            "EQUAL: the BF16 all-reduce gives the FP32 path's bits"
            if equal
            else "DIFFERS: the BF16 all-reduce changes bits"
        )
        print(f"VERDICT {verdict}; max ne over ranks {max(c['bf16_ne_fp32'] for c in all_cases)}")
        n_bf16 = sum(c["switched_bf16"] for c in results)
        state = "EQUAL to" if switched_ok else "DIFFERS from"
        accuracy = "" if switched_ok else (
            "; no less accurate against FP64"
            if no_less_accurate
            else "; LESS ACCURATE against FP64"
        )
        print(
            f"SWITCHED {state} the FP32 path "
            f"(BF16 messages: {n_bf16}/{len(results)}){accuracy}"
        )
        with open(args.out, "w") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "switch": switch,
                    "equal": equal,
                    "switched_equal": switched_ok,
                    "switched_no_less_accurate": no_less_accurate,
                    "switched_ranks_agree": ranks_agree,
                    "world": world,
                    "cases": all_cases,
                },
                f,
                indent=1,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
