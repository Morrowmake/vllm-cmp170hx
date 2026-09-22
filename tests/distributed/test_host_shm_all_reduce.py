# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the host-staged no-P2P all-reduce.

    CUDA_VISIBLE_DEVICES="" python tests/distributed/test_host_shm_all_reduce.py

Runs standalone: there is no pytest in any of this box's virtualenvs, so the
file carries a ~30-line harness instead of depending on one. Under pytest it
would also collect, since every check is a plain `test_*` function.

Nothing here touches a GPU. `torch.cuda.can_device_access_peer` is always
patched out, and one of the tests asserts that importing the module does not
compile the CUDA extension.

What is worth testing without hardware:

  * the **gate** (`should_host_ar`) accepts and rejects the right messages and
    depends only on dtype and byte count -- every rank must take the same
    branch on every call, or the in-kernel barrier deadlocks until its spin
    bound fires;
  * the **algorithm thresholds** are pure functions of size, for the same
    reason;
  * the **OFF path** constructs nothing, so a run with the flag off is
    byte-identical to upstream;
  * the communicator **stands aside only when told to** -- the operator sets
    VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE to hand the fast path to the
    device-memory custom all-reduce; a driver that merely starts reporting
    peer access must not silently turn this path into NCCL;
  * every early return leaves a usable, `disabled` object rather than a
    half-built one.
"""

from __future__ import annotations

import inspect
import sys
import traceback

import torch

from vllm.distributed.device_communicators import host_shm_all_reduce as hsm

MiB = 1 << 20


# --------------------------------------------------------------------------
# minimal harness
# --------------------------------------------------------------------------
class MonkeyPatch:
    """The slice of pytest's monkeypatch fixture these tests use."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name), True))
        setattr(obj, name, value)

    def setitem(self, dct, key, value):
        had = key in dct
        self._undo.append((dct, key, dct.get(key), had))
        dct[key] = value

    def setenv(self, name, value):
        import os

        self.setitem(os.environ, name, value)

    def delenv(self, name, raising=True):
        import os

        if name in os.environ:
            self._undo.append((os.environ, name, os.environ[name], True))
            del os.environ[name]
        elif raising:
            raise KeyError(name)

    def undo(self):
        for obj, name, old, had in reversed(self._undo):
            if isinstance(obj, dict):
                if had:
                    obj[name] = old
                else:
                    obj.pop(name, None)
            else:
                setattr(obj, name, old)
        self._undo.clear()


def run_all() -> int:
    tests = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failed = []
    for name, fn in tests:
        mp = MonkeyPatch()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            print(f"[PASS] {name}")
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _stub(**kw):
    """A gate-only instance: no group, no segment, no GPU.

    `should_host_ar` reads exactly these attributes, which is the point -- the
    decision may not depend on anything else.
    """
    o = object.__new__(hsm.HostShmAllreduce)
    o.disabled = False
    o.host_cap = 512 * 1024
    for k, v in kw.items():
        setattr(o, k, v)
    return o


class _FakeTensor:
    """Enough of a tensor for the gate: dtype, element count, device kind."""

    def __init__(self, nbytes, dtype=torch.bfloat16, is_cuda=True):
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._n = nbytes // dtype.itemsize

    def numel(self):
        return self._n

    def element_size(self):
        return self.dtype.itemsize


def _no_gpu_init(mp, *, is_cuda=True, world=4, same_node=True):
    mp.delenv("VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE", raising=False)
    mp.setattr(torch.cuda, "can_device_access_peer", lambda a, b: False)
    mp.setattr(hsm.current_platform, "is_cuda", lambda: is_cuda)
    mp.setattr(hsm.dist, "get_backend", lambda group: "gloo")
    mp.setattr(hsm.dist, "get_rank", lambda group=None: 0)
    mp.setattr(hsm.dist, "get_world_size", lambda group=None: world)
    # `in_the_same_node_as` is imported inside __init__, so patch the attribute
    # on the real module. Replacing the whole module in sys.modules also works
    # for this call but breaks every later import from it (the logger's
    # warning_once needs is_local_first_rank).
    import vllm.distributed.parallel_state as ps

    mp.setattr(ps, "in_the_same_node_as", lambda group, source_rank=0: [same_node] * world)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def test_gate_accepts_the_measured_band():
    ar = _stub()
    for nbytes in (16, 8192, 16384, 32768, 65536, 131072, 262144, 512 * 1024):
        assert ar.should_host_ar(_FakeTensor(nbytes)) is True, nbytes


def test_gate_rejects_above_the_cap():
    """Prefill-sized messages stay on NCCL's ring, which is already near wire
    speed for them. This is also what keeps the prefill-overlap feature's
    >= 2 MiB sliced messages out of the host path."""
    ar = _stub()
    for nbytes in (512 * 1024 + 16, 2 * MiB, 16 * MiB):
        assert ar.should_host_ar(_FakeTensor(nbytes)) is False, nbytes


def test_gate_rejects_zero_bytes():
    assert _stub().should_host_ar(_FakeTensor(0)) is False


def test_gate_rejects_non_multiple_of_16():
    """The kernel has no scalar tail path; NCCL can have these."""
    t = _FakeTensor(8192)
    t._n = 4093  # 8186 bytes in bf16
    assert _stub().should_host_ar(t) is False


def test_gate_rejects_unsupported_dtypes():
    ar = _stub()
    for dtype in (torch.float16, torch.float64, torch.int8):
        assert ar.should_host_ar(_FakeTensor(32768, dtype=dtype)) is False, dtype


def test_gate_accepts_supported_dtypes():
    ar = _stub()
    for dtype in (torch.bfloat16, torch.float32):
        assert ar.should_host_ar(_FakeTensor(32768, dtype=dtype)) is True, dtype


def test_gate_rejects_cpu_tensors():
    assert _stub().should_host_ar(_FakeTensor(32768, is_cuda=False)) is False


def test_gate_closed_when_disabled():
    """`close()` and a spin timeout both set `disabled`; the gate must then
    decline everything so the chain falls through to NCCL."""
    ar = _stub()
    ar.disabled = True
    assert ar.should_host_ar(_FakeTensor(32768)) is False


def test_gate_honours_a_lowered_cap():
    ar = _stub(host_cap=64 * 1024)
    assert ar.should_host_ar(_FakeTensor(65536)) is True
    assert ar.should_host_ar(_FakeTensor(65536 + 16)) is False


def test_gate_depends_only_on_dtype_and_size():
    """Ranks with different local state must agree, so the gate must not read
    anything rank-local."""
    a, b = _stub(), _stub()
    b.rank = 3
    b.calls = 10_000
    for nbytes in (8192, 32768, 262144, 4 * MiB):
        assert a.should_host_ar(_FakeTensor(nbytes)) == b.should_host_ar(
            _FakeTensor(nbytes)
        ), nbytes


# --------------------------------------------------------------------------
# algorithm dispatch
# --------------------------------------------------------------------------
def _algo(ar, nbytes):
    """Mirror of the dispatch in `host_all_reduce`."""
    blocks = ar._blocks(nbytes)
    if nbytes >= ar.dma_min_bytes and blocks == 1:
        return 2
    if nbytes >= ar.two_shot_min_bytes:
        return 1
    return 0


def test_algorithm_thresholds_are_size_only():
    """Measured thresholds, and pure functions of nbytes
    for the same rank-agreement reason as the gate."""
    ar = _stub(
        threads=hsm.DEFAULT_THREADS,
        block_cap=hsm.DEFAULT_BLOCK_CAP,
        two_shot_min_bytes=hsm.TWO_SHOT_MIN_BYTES,
        dma_min_bytes=hsm.DMA_MIN_BYTES,
    )
    cases = [
        (8192, 0),  # one-shot
        (32768, 0),
        (65536 - 16, 0),
        (65536, 1),  # two-shot from TWO_SHOT_MIN_BYTES
        (131072, 1),
        (262144 - 16, 1),
        (262144, 2),  # DMA publish from DMA_MIN_BYTES
        (512 * 1024, 2),
    ]
    for nbytes, expect in cases:
        assert _algo(ar, nbytes) == expect, (nbytes, expect)


def test_measured_thresholds_match_the_benchmarks():
    assert hsm.TWO_SHOT_MIN_BYTES == 64 * 1024
    assert hsm.DMA_MIN_BYTES == 256 * 1024
    assert hsm.DEFAULT_THREADS == 1024
    assert hsm.DEFAULT_BLOCK_CAP == 1


def test_block_count_is_capped_and_size_only():
    ar = _stub(threads=hsm.DEFAULT_THREADS, block_cap=hsm.DEFAULT_BLOCK_CAP)
    for nbytes in (16, 8192, 262144, 512 * 1024):
        assert ar._blocks(nbytes) == 1, nbytes
    wide = _stub(threads=256, block_cap=32)
    assert wide._blocks(32768) == 8  # 2048 packed units / 256 threads
    assert wide._blocks(16) == 1


# --------------------------------------------------------------------------
# P2P probe and the stand-aside policy built on it
# --------------------------------------------------------------------------
def test_probe_sees_p2p_when_it_works(monkeypatch):
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    assert hsm._p2p_unavailable(4) is False


def test_probe_reports_no_p2p_when_refused(monkeypatch):
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: False)
    assert hsm._p2p_unavailable(4) is True


def test_probe_sees_p2p_when_any_pair_can_peer(monkeypatch):
    """Partial P2P is not our case; the probe must not call it absent."""
    monkeypatch.setattr(
        torch.cuda, "can_device_access_peer", lambda a, b: {a, b} == {0, 1}
    )
    assert hsm._p2p_unavailable(4) is False


def test_p2p_probe_failure_does_not_claim_no_p2p(monkeypatch):
    """If the probe raises, do not assume no-P2P."""

    def boom(a, b):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", boom)
    assert hsm._p2p_unavailable(4) is False


def test_stand_aside_needs_the_env_and_working_p2p(monkeypatch):
    """The two fast paths are exclusive by the operator's choice.

    Only VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE -- the same env that lets
    CustomAllreduce count PCIe peer-to-peer as fully connected above two GPUs
    -- hands the fast path over, and only where peer access is really there.
    """
    for env_on, peer, expected in (
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ):
        mp = MonkeyPatch()
        try:
            mp.setenv("VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE", "1" if env_on else "0")
            mp.setattr(torch.cuda, "can_device_access_peer", lambda a, b: peer)
            got = hsm._stand_aside_for_custom_allreduce(4)
            assert got is expected, (env_on, peer, got)
        finally:
            mp.undo()


# --------------------------------------------------------------------------
# early returns
# --------------------------------------------------------------------------
def test_disabled_on_non_cuda_platform(monkeypatch):
    _no_gpu_init(monkeypatch, is_cuda=False)
    ar = hsm.HostShmAllreduce(group=object(), device="cuda:0")
    assert ar.disabled is True
    assert ar.should_host_ar(_FakeTensor(32768)) is False


def test_disabled_for_world_size_one(monkeypatch):
    _no_gpu_init(monkeypatch, world=1)
    assert hsm.HostShmAllreduce(group=object(), device="cuda:0").disabled is True


def test_disabled_for_unsupported_world_size(monkeypatch):
    for world in (3, 5, 6, 7):
        mp = MonkeyPatch()
        try:
            _no_gpu_init(mp, world=world)
            ar = hsm.HostShmAllreduce(group=object(), device="cuda:0")
            assert ar.disabled is True, world
        finally:
            mp.undo()


def test_disabled_across_nodes(monkeypatch):
    """A /dev/shm segment cannot span hosts."""
    _no_gpu_init(monkeypatch, same_node=False)
    assert hsm.HostShmAllreduce(group=object(), device="cuda:0").disabled is True


def _counting_loader(mp):
    """Replace the JIT build with a counter, so a test can assert it never ran
    without needing nvcc."""
    calls = {"n": 0}

    def _load():
        calls["n"] += 1
        raise RuntimeError("the CPU tests never build the extension")

    mp.setattr(hsm, "_load_module", _load)
    return calls


def test_disabled_when_p2p_available_and_handed_over(monkeypatch):
    """With the env set and peer access present, stand aside -- and do not
    compile the extension on the way out."""
    _no_gpu_init(monkeypatch)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setenv("VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE", "1")
    calls = _counting_loader(monkeypatch)
    assert hsm.HostShmAllreduce(group=object(), device="cuda:0").disabled is True
    assert calls["n"] == 0


def test_p2p_availability_alone_does_not_stand_aside(monkeypatch):
    """Installing a P2P-capable driver must not silently turn
    VLLM_GLM5_HOST_ALLREDUCE=1 into NCCL: above two PCIe-only GPUs upstream's
    CustomAllreduce refuses anyway unless the operator opts in."""
    _no_gpu_init(monkeypatch)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    calls = _counting_loader(monkeypatch)
    hsm.HostShmAllreduce(group=object(), device="cuda:0")
    assert calls["n"] == 1, "the host path should have gone on to set itself up"


def test_disabled_when_max_size_is_useless(monkeypatch):
    _no_gpu_init(monkeypatch)
    ar = hsm.HostShmAllreduce(group=object(), device="cuda:0", max_size=0)
    assert ar.disabled is True


def test_every_early_return_leaves_a_usable_object(monkeypatch):
    """A disabled instance still answers the gate and survives close()."""
    _no_gpu_init(monkeypatch, world=1)
    ar = hsm.HostShmAllreduce(group=object(), device="cuda:0")
    assert ar.should_host_ar(_FakeTensor(32768)) is False
    assert ar.error_flags() == []
    ar.close()
    ar.close()  # idempotent


def test_nccl_group_is_rejected(monkeypatch):
    """The communicator takes the CPU group; attaching it to the NCCL group
    would deadlock the setup barrier against the collectives it guards."""
    _no_gpu_init(monkeypatch)
    monkeypatch.setattr(hsm.dist, "get_backend", lambda group: hsm.dist.Backend.NCCL)
    try:
        hsm.HostShmAllreduce(group=object(), device="cuda:0")
    except AssertionError:
        return
    raise AssertionError("a NCCL group should have been refused")


# --------------------------------------------------------------------------
# the OFF path
# --------------------------------------------------------------------------
def test_env_default_is_off():
    import vllm.envs as envs

    assert envs.VLLM_GLM5_HOST_ALLREDUCE is False
    assert envs.VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE == 512 * 1024


def test_env_vars_are_declared():
    """An undeclared VLLM_* name makes validate_environ() warn at every start,
    and `envs.X` would not report what the feature will do."""
    import vllm.envs as envs

    assert "VLLM_GLM5_HOST_ALLREDUCE" in envs.environment_variables
    assert "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE" in envs.environment_variables
    # The type hints live in a TYPE_CHECKING block, so assert on the source
    # rather than on runtime __annotations__ (which is empty).
    src = inspect.getsource(envs)
    assert "VLLM_GLM5_HOST_ALLREDUCE: bool = False" in src
    assert "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE: int = 512 * 1024" in src


def test_env_reads_are_live(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_GLM5_HOST_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE", "65536")
    assert envs.environment_variables["VLLM_GLM5_HOST_ALLREDUCE"]() is True
    assert envs.environment_variables["VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE"]() == 65536


def test_cuda_communicator_off_path_constructs_nothing():
    """With the flag off `hostshm_comm` stays None, so `all_reduce` cannot
    reach the host path and the dispatch chain is upstream's."""
    from vllm.distributed.device_communicators import cuda_communicator as cc

    src = inspect.getsource(cc.CudaCommunicator.__init__)
    assert "self.hostshm_comm: HostShmAllreduce | None = None" in src
    assert "envs.VLLM_GLM5_HOST_ALLREDUCE" in src
    assert 'unique_name.split(":")[0] == "tp"' in src  # TP-only

    dispatch = inspect.getsource(cc.CudaCommunicator.all_reduce)
    # `is not None` first, so the OFF path short-circuits before touching us.
    assert "hostshm_comm is not None and hostshm_comm.should_host_ar" in dispatch


def test_host_path_is_tried_before_custom_allreduce():
    """Ordering matters: on real-P2P hardware we disable ourselves, and there
    ca_comm must still be reached."""
    from vllm.distributed.device_communicators import cuda_communicator as cc

    src = inspect.getsource(cc.CudaCommunicator.all_reduce)
    assert src.index("hostshm_comm.should_host_ar") < src.index(
        "ca_comm.should_custom_ar"
    )


def test_cuda_communicator_cleans_up():
    from vllm.distributed.device_communicators import cuda_communicator as cc

    assert "self.hostshm_comm.close()" in inspect.getsource(
        cc.CudaCommunicator.destroy
    )


def test_backend_selection_log_mentions_us():
    """That log is how a silent fallback gets noticed."""
    from vllm.distributed.device_communicators import cuda_communicator as cc

    src = inspect.getsource(cc.CudaCommunicator._log_all_reduce_backend_selection)
    assert '"HOSTSHM"' in src
    assert 'enabled_ar_backends.append("HOSTSHM")' in src


def test_pynccl_is_left_intact():
    """The prefill-overlap feature reaches past us for `pynccl_comm` and runs
    its own collectives on a side stream; we must not disturb that."""
    from vllm.distributed.device_communicators import cuda_communicator as cc

    src = inspect.getsource(cc.CudaCommunicator.__init__)
    assert "self.pynccl_comm = PyNcclCommunicator(" in src
    destroy = inspect.getsource(cc.CudaCommunicator.destroy)
    assert "self.pynccl_comm.destroy()" in destroy


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------
def test_import_does_not_compile_anything():
    """The JIT build needs nvcc and takes tens of seconds; importing the
    module happens on every vLLM start, flag or no flag."""
    assert hsm._MODULE is None


def test_build_dir_is_outside_the_source_tree(monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR", raising=False)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    d = hsm._build_dir()
    pkg = hsm.__file__.rsplit("/", 1)[0]
    assert not d.startswith(pkg), f"build dir {d} is inside the source tree"


def test_supported_world_sizes_match_the_kernel():
    assert hsm.SUPPORTED_WORLD_SIZES == (2, 4, 8)
    for w in hsm.SUPPORTED_WORLD_SIZES:
        assert f"case {w}:" in hsm.CUDA_SRC, w


def test_no_fast_math_in_build_flags():
    """Rounding is frozen: the reduce accumulates in FP32 and must stay
    bitwise reproducible across runs and across ranks."""
    flags = [
        line
        for line in inspect.getsource(hsm._load_module).splitlines()
        if "cflags" in line or ('"-' in line and "#" not in line)
    ]
    assert not any("use_fast_math" in line for line in flags), flags


def test_kernel_targets_sm80_only():
    assert "compute_80" in inspect.getsource(hsm._load_module)


def test_spin_is_bounded():
    """An unbounded in-kernel spin would wedge the GPU until reboot."""
    assert "clock64" in hsm.CUDA_SRC
    assert "spin_cycles" in hsm.CUDA_SRC
    assert hsm.DEFAULT_SPIN_TIMEOUT_S > 0


def test_flags_use_release_acquire_not_atomics():
    """Global atomics on this card collapse 25x under 32-way contention; the
    flag protocol must stay plain release/acquire stores."""
    assert "st.release.sys" in hsm.CUDA_SRC
    assert "ld.acquire.sys" in hsm.CUDA_SRC
    assert "atomicCAS" not in hsm.CUDA_SRC
    assert "atomicAdd" not in hsm.CUDA_SRC


def test_error_flag_check_raises_rather_than_hangs(monkeypatch):
    ar = _stub(
        world_size=4,
        spin_cycles=10**10,
        _mm=bytearray(1 << 20),
    )
    monkeypatch.setattr(
        hsm.HostShmAllreduce, "error_flags", lambda self: [0, 0, 0x80000005, 0]
    )
    try:
        ar._check_error_flags()
    except RuntimeError as exc:
        assert "spin" in str(exc)
        assert ar.disabled is True  # and it takes itself out of the chain
        return
    raise AssertionError("a set error flag must raise")


def test_error_flag_check_is_quiet_when_clean(monkeypatch):
    ar = _stub(world_size=4, spin_cycles=10**10, _mm=bytearray(1 << 20))
    monkeypatch.setattr(hsm.HostShmAllreduce, "error_flags", lambda self: [0, 0, 0, 0])
    ar._check_error_flags()
    assert ar.disabled is False


if __name__ == "__main__":
    sys.exit(run_all())
