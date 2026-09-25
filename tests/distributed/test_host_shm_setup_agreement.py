# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests: host-shm all-reduce setup is collective.

    CUDA_VISIBLE_DEVICES="" python tests/distributed/test_host_shm_setup_agreement.py

Four real processes on a gloo group run `HostShmAllreduce.__init__` for real
(segment in /dev/shm, the agreement collectives, the unlink); only the CUDA
pieces are faked: the JIT module (its `host_register` can be told to fail on
one rank, or to fail once and then succeed) and the device the sequence tensor
lives on. What must hold:

  * a failure on any one rank, at any step, disables the path on **every**
    rank, and no rank waits forever (the old behaviour: the failed rank went on
    alone and its peers hung in the next collective);
  * a transient registration failure is retried and the path still comes up;
  * ranks register one at a time, rank 0 first;
  * nothing is left in /dev/shm afterwards, on success or on failure;
  * the stand-aside decision is agreed too, when the peer-access probe runs.

Runs standalone, like test_host_shm_all_reduce.py (no pytest on this box).
"""

from __future__ import annotations

import os
import sys
import time
import traceback

WORLD = 4
TIMEOUT_S = 120


class _FakeMod:
    """The JIT module, minus CUDA. Records the registration order."""

    def __init__(self, rank, fail_mode, log_path):
        self.rank = rank
        self.fail_mode = fail_mode
        self.log_path = log_path
        self.calls = 0

    def host_register(self, ptr, nbytes):
        self.calls += 1
        with open(self.log_path, "a") as f:
            f.write(f"start {self.rank} {time.monotonic():.6f}\n")
        time.sleep(0.05)  # long enough that an overlap would show
        with open(self.log_path, "a") as f:
            f.write(f"end {self.rank} {time.monotonic():.6f}\n")
        mode = self.fail_mode
        if mode == "register" or (mode == "register_once" and self.calls == 1):
            raise RuntimeError("cudaHostRegister: invalid argument (1)")
        return ptr

    def host_unregister(self, ptr):
        pass


def _worker(rank, port, scenario, log_path, q):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE", None)
        if scenario.get("env_fail_rank") is not None:
            os.environ["VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK"] = str(
                scenario["env_fail_rank"])
        if scenario.get("handover"):
            os.environ["VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE"] = "1"
        import torch.distributed as dist

        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=rank, world_size=WORLD)
        group = dist.new_group(backend="gloo")

        import vllm.distributed.parallel_state as ps
        from vllm.distributed.device_communicators import host_shm_all_reduce as hsm

        hsm.current_platform.is_cuda = lambda: True
        ps.in_the_same_node_as = lambda group, source_rank=0: [True] * WORLD
        # The peer-access probe: rank `peer_rank` alone sees peer access.
        peer_rank = scenario.get("peer_rank")
        hsm._p2p_unavailable = lambda world_size: rank != peer_rank

        fail_rank = scenario.get("fail_rank")
        fail_step = scenario.get("fail_step")
        mode = fail_step if rank == fail_rank else None
        if mode in ("register", "register_once"):
            fake = _FakeMod(rank, mode, log_path)
        else:
            fake = _FakeMod(rank, None, log_path)

        def _load():
            if mode == "load":
                raise RuntimeError("Ninja is required to load C++ extensions")
            return fake

        hsm._load_module = _load
        if mode == "map":
            def _bad_map(self):
                raise OSError(12, "Cannot allocate memory")

            hsm.HostShmAllreduce._map_segment = _bad_map
        # Keep the retry pauses short in tests.
        hsm.REGISTER_RETRY_DELAYS_S = (0.01, 0.01, 0.01)

        ar = hsm.HostShmAllreduce(group=group, device="cpu")
        name = getattr(ar, "name", None)
        enabled = not ar.disabled
        attempts = getattr(ar, "_register_attempts", None)
        # A collective after setup: every rank must reach it. With a split this
        # is where the old code deadlocked.
        box = [None] * WORLD
        dist.all_gather_object(box, (rank, enabled), group=group)
        ar.close()
        dist.barrier(group=group)
        q.put(("ok", rank, enabled, [b[1] for b in box], name, attempts))
    except Exception:
        q.put(("err", rank, traceback.format_exc()))


def _run(scenario, tag):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    log_path = f"/tmp/hostshm_setup_test_{os.getpid()}_{tag}.log"
    if os.path.exists(log_path):
        os.unlink(log_path)
    port = 20000 + (os.getpid() * 7 + hash(tag)) % 20000
    before = set(n for n in os.listdir("/dev/shm") if n.startswith("vllm_hostshm_ar_"))
    procs = [ctx.Process(target=_worker, args=(r, port, scenario, log_path, q))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    results = []
    deadline = time.time() + TIMEOUT_S
    while len(results) < WORLD and time.time() < deadline:
        try:
            results.append(q.get(timeout=1))
        except Exception:
            pass
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    hung = len(results) < WORLD
    after = set(n for n in os.listdir("/dev/shm") if n.startswith("vllm_hostshm_ar_"))
    order = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            order = [line.split() for line in f]
        os.unlink(log_path)
    errs = [r for r in results if r[0] == "err"]
    assert not errs, "worker raised:\n" + "\n".join(e[2] for e in errs)
    assert not hung, (
        f"HANG: only {len(results)}/{WORLD} ranks finished within {TIMEOUT_S}s"
    )
    leaked = after - before
    assert not leaked, f"segment(s) left in /dev/shm: {sorted(leaked)}"
    return sorted(results, key=lambda r: r[1]), order


def _enabled(results):
    return [r[2] for r in results]


# --------------------------------------------------------------------------
def test_all_ranks_ok_enables_everywhere():
    res, order = _run({}, "ok")
    assert _enabled(res) == [True] * WORLD
    assert len({r[4] for r in res}) == 1, "ranks disagree on the segment name"


def test_registration_is_serialised_rank_0_first():
    res, order = _run({}, "order")
    starts = [(float(t), int(r)) for k, r, t in order if k == "start"]
    ends = {int(r): float(t) for k, r, t in order if k == "end"}
    assert [r for _, r in sorted(starts)] == list(range(WORLD)), starts
    # No overlap: each rank starts after the previous one finished.
    for (t, r) in sorted(starts)[1:]:
        assert t >= ends[r - 1], f"rank {r} started before rank {r - 1} finished"


def _assert_all_fell_back(scenario, tag):
    res, _ = _run(scenario, tag)
    assert _enabled(res) == [False] * WORLD, _enabled(res)
    for r in res:
        assert r[3] == [False] * WORLD  # every rank saw every rank disabled


def test_register_failure_on_rank_2_disables_every_rank():
    _assert_all_fell_back({"fail_rank": 2, "fail_step": "register"}, "reg2")


def test_register_failure_on_rank_0_disables_every_rank():
    _assert_all_fell_back({"fail_rank": 0, "fail_step": "register"}, "reg0")


def test_register_failure_on_last_rank_disables_every_rank():
    _assert_all_fell_back({"fail_rank": WORLD - 1, "fail_step": "register"}, "reg3")


def test_module_load_failure_on_rank_0_disables_every_rank():
    _assert_all_fell_back({"fail_rank": 0, "fail_step": "load"}, "load0")


def test_module_load_failure_on_rank_1_disables_every_rank():
    _assert_all_fell_back({"fail_rank": 1, "fail_step": "load"}, "load1")


def test_map_failure_on_rank_3_disables_every_rank():
    _assert_all_fell_back({"fail_rank": 3, "fail_step": "map"}, "map3")


def test_env_forced_failure_disables_every_rank():
    """VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK, the hook the GPU check uses."""
    _assert_all_fell_back({"env_fail_rank": 1}, "env1")


def test_transient_register_failure_is_retried():
    res, order = _run({"fail_rank": 1, "fail_step": "register_once"}, "once")
    assert _enabled(res) == [True] * WORLD
    assert res[1][5] == 2, f"rank 1 should have needed 2 attempts, got {res[1][5]}"
    assert all(r[5] == 1 for r in res if r[1] != 1)


def test_stand_aside_is_agreed():
    """Hand-over env set, and only rank 2's probe sees peer access: without
    agreement rank 2 would return early while the others entered setup."""
    res, _ = _run({"handover": True, "peer_rank": 2}, "aside")
    assert _enabled(res) == [False] * WORLD


def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"[PASS] {name} ({time.time() - t:.1f}s)")
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
