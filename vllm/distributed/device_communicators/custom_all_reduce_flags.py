# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write-based two-shot all-reduce with the ready signal carried in the data.

Why this exists
---------------
On four PCIe-only GPUs with peer access (the NVIDIA CMP 170HX on PCIe Gen2 x16
is the case this was built for), `cross_device_reduce_2stage` spends most of a
small decode all-reduce in its two flag barriers: each barrier is a fenced
flag write to every peer and a poll, and a fenced round trip costs ~8 us on this
link. A 4096-wide bf16 row (8 KiB) took ~20 us, of which the barriers alone
were ~10 us.

Here no separate flag exists. Each rank pushes its data into the peers'
receive buffers with plain 16-byte stores, and a receiver knows a 16-byte pack
has arrived once none of its four 32-bit words is the sentinel 0xFFFFFFFF (the
buffers are pre-filled with it). Two shots, as in the incumbent:

  1. reduce-scatter by push: partition q of my input goes to rank q;
  2. rank q reduces its partition and pushes the result to every peer;
  3. every rank gathers the other partitions from its own buffer.

Three buffer stages rotate. The stage the previous call used is re-filled with
the sentinel during the current call: a peer can be at most one call ahead
(it cannot finish a call without my data), so it can only be writing into the
next stage. A per-communicator device word holds the stage; CTA 0 advances it
after every CTA has read it, so a captured CUDA graph replays correctly.

Measured on 4x CMP 170HX against `cross_device_reduce_2stage` at 512 threads,
block limit 36, CUDA-graph replay of back-to-back calls, this launch rule
(256 threads, one per pack of the largest partition, at most 36 blocks):
8 KiB 20.5 -> 10.1 us, 32 KiB 23.9 -> 13.2, 64 KiB 30.9 -> 20.3,
128 KiB 48.4 -> 36.0, 256 KiB 86.4 -> 66.3.

Exactness
---------
Bitwise identical to `cross_device_reduce_2stage`: the element in partition p
is accumulated in fp32 in the order p, p+1, p+2, p+3 and rounded once, as
there, and the partitions are the same. The sentinel is a pair of negative
NaNs with full payload. An input word equal to it is sent as a pair of
canonical NaNs; any NaN input makes the fp32 sum the canonical NaN, so the
result bits do not change. A reduced word can never be the sentinel (bf16
rounding of a NaN gives the canonical NaN), and is sanitised anyway.

Only bf16, four ranks, compute capability 8.0 and the opt-in PCIe peer-to-peer
custom all-reduce: the configuration this was validated on. Messages above
VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES (default 256 KiB, the largest measured)
and other dtypes keep the incumbent kernel.

Safety
------
A wait never gives up quietly. Every wait is bounded by
VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S (default 60 s, measured on the GPU's
global timer, far above any normal skew between ranks). On overrun the
first timed-out thread writes rank, call, stage, size, phase, the awaited
peer and the wait time into a host-mapped error record and the kernel
executes a trap, so the CUDA context fails and no partial result is ever
returned. A host thread polls the record (and it is checked again every 256
eager calls and at exit): it logs the record and ends the process. The other
ranks then time out the same way. The incumbent kernel waits without limit.

The extension is compiled on first use with torch's JIT loader and cached
outside the source tree (VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR, else
TORCH_EXTENSIONS_DIR, else ~/.cache/vllm/custom_all_reduce_flags).
"""

import atexit
import ctypes
import hashlib
import math
import os
import sys
import threading

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

WORLD_SIZE = 4
THREADS = 256  # launch rule chosen from a threads x blocks sweep (1..32 rows)
MAX_BLOCKS = 36
EPOCH_WORDS = 16
CHECK_EVERY = 256  # eager calls between host checks of the error record
EXIT_CODE = 70

KERNEL_SRC = r"""
#include <cuda_bf16.h>
#include <cstdint>

namespace arflags {

constexpr int kRanks = 4;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kCanonicalPair = 0x7FFF7FFFu;

struct __align__(16) Ptrs {
  void* p[kRanks];
};

// w[0] stage (0..2), w[1] CTAs that read w[0] this call, w[2..4] packs last
// written into each stage, w[8] calls completed, w[15] timed-out waits.
struct __align__(16) Epoch {
  uint32_t w[16];
};

// Owner of pack idx in cross_device_reduce_2stage: part = size / 4, ranks 0..2
// own [r*part, (r+1)*part), rank 3 the rest (all of it when part == 0).
__host__ __device__ __forceinline__ int part_len(int q, int size) {
  int part = size / kRanks;
  return q == kRanks - 1 ? size - (kRanks - 1) * part : part;
}

__host__ __device__ __forceinline__ uint32_t sanitize_word(uint32_t w) {
  return w == kSentinel ? kCanonicalPair : w;
}

// One element, the incumbent's arithmetic: fp32 sum from the owner onwards.
__host__ __device__ __forceinline__ __nv_bfloat16 sum4_owner_order(
    const __nv_bfloat16 v[kRanks], int owner) {
  float acc = __bfloat162float(v[owner]);
#pragma unroll
  for (int j = 1; j < kRanks; j++)
    acc += __bfloat162float(v[(owner + j) % kRanks]);
  return __float2bfloat16(acc);
}

#define AFD __device__ __forceinline__

AFD uint4 ld_volatile16(const void* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

AFD void st_volatile16(void* ptr, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

AFD bool dirty(const uint4& v) {
  return v.x == kSentinel || v.y == kSentinel || v.z == kSentinel ||
         v.w == kSentinel;
}

AFD uint4 sanitize(uint4 v) {
  v.x = sanitize_word(v.x);
  v.y = sanitize_word(v.y);
  v.z = sanitize_word(v.z);
  v.w = sanitize_word(v.w);
  return v;
}

// Error record in host-mapped memory (int64 words): [0] set, [1] rank,
// [2] stage, [3] call, [4] size in packs, [5] phase (1 reduce-scatter wait,
// 2 all-gather wait), [6] awaited peer, [7] waited ns, [8] block, [9] thread,
// [10] limit ns.
struct WaitCtx {
  Epoch* ep;
  long long* err;
  unsigned long long limit_ns;
  int rank;
  int stage;
  int size;
};

AFD unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Overrun: record it once (first thread), make it visible to the host, trap.
// The trap fails the CUDA context, so the caller never sees a partial result.
__device__ __noinline__ void wait_failed(WaitCtx c, int phase, int peer,
                                        unsigned long long waited) {
  if (atomicAdd(&c.ep->w[15], 1u) == 0u) {
    volatile long long* e = c.err;
    e[1] = c.rank;
    e[2] = c.stage;
    e[3] = c.ep->w[8];
    e[4] = c.size;
    e[5] = phase;
    e[6] = peer;
    e[7] = (long long)waited;
    e[8] = blockIdx.x;
    e[9] = threadIdx.x;
    e[10] = (long long)c.limit_ns;
    __threadfence_system();
    e[0] = 1;
    __threadfence_system();
  }
  __trap();
}

// Buffer pointers in registers, selected by static index only (a dynamically
// indexed kernel parameter would be copied to local memory).
struct Regs4 {
  void* p0;
  void* p1;
  void* p2;
  void* p3;
  AFD explicit Regs4(const Ptrs& b)
      : p0(b.p[0]), p1(b.p[1]), p2(b.p[2]), p3(b.p[3]) {}
  AFD void* operator[](int q) const {
    return q == 0 ? p0 : (q == 1 ? p1 : (q == 2 ? p2 : p3));
  }
};

AFD uint4 pick4(const uint4 (&v)[kRanks], int q) {
  return q == 0 ? v[0] : (q == 1 ? v[1] : (q == 2 ? v[2] : v[3]));
}

union Pack {
  uint4 u;
  __nv_bfloat16 h[8];
};

// The incumbent's packed_reduce over pointers rotated to the owner.
AFD uint4 reduce_owner_order(const uint4 (&v)[kRanks], int owner) {
  float acc[8];
  Pack x;
  x.u = pick4(v, owner);
#pragma unroll
  for (int e = 0; e < 8; e++) acc[e] = __bfloat162float(x.h[e]);
#pragma unroll
  for (int j = 1; j < kRanks; j++) {
    x.u = pick4(v, (owner + j) & (kRanks - 1));
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] += __bfloat162float(x.h[e]);
  }
#pragma unroll
  for (int e = 0; e < 8; e++) x.h[e] = __float2bfloat16(acc[e]);
  return x.u;
}

AFD void wait3(const uint4* base, int rank_stride, int rank,
               uint4 (&v)[kRanks], const WaitCtx& c) {
  bool ready[kRanks];
#pragma unroll
  for (int s = 0; s < kRanks; s++) ready[s] = (s == rank);
  int remaining = kRanks - 1;
  const unsigned long long t0 = now_ns();
  while (remaining) {
#pragma unroll
    for (int s = 0; s < kRanks; s++) {
      if (!ready[s]) {
        uint4 x = ld_volatile16(base + s * rank_stride);
        if (!dirty(x)) {
          v[s] = x;
          ready[s] = true;
          --remaining;
        }
      }
    }
    if (remaining) {
      unsigned long long waited = now_ns() - t0;
      if (waited > c.limit_ns) {
        int peer = 0;
#pragma unroll
        for (int s = kRanks - 1; s >= 0; s--)
          if (!ready[s]) peer = s;
        wait_failed(c, 1, peer, waited);
      }
    }
  }
}

AFD uint4 wait1(const uint4* ptr, int peer, const WaitCtx& c) {
  uint4 x = ld_volatile16(ptr);
  const unsigned long long t0 = now_ns();
  while (dirty(x)) {
    unsigned long long waited = now_ns() - t0;
    if (waited > c.limit_ns) wait_failed(c, 2, peer, waited);
    x = ld_volatile16(ptr);
  }
  return x;
}

// size: message length in 16-byte packs. Layout per stage (L = largest
// partition): reduce-scatter slot s at s*L, all-gather slot q at (4+q)*L.
// stage_packs >= 8*L.
__global__ void __launch_bounds__(512, 1)
    two_shot(Ptrs bufs, const uint4* __restrict__ in, uint4* __restrict__ out,
             Epoch* ep, int rank, int size, int stage_packs,
             unsigned long long limit_ns, long long* err) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int stage = ep->w[0];
  const int dirty_stage = (stage + 2) % 3;
  const int dirty_size = ep->w[2 + dirty_stage];
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&ep->w[1], 1u);
  const Regs4 b(bufs);
  const WaitCtx wc{ep, err, limit_ns, rank, stage, size};

  const int part = size / kRanks;
  const int L = part_len(kRanks - 1, size);

  for (int idx = tid; idx < L; idx += stride) {
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int q = (rank + i) & (kRanks - 1);
      if (idx < part_len(q, size)) {
        uint4* dst = reinterpret_cast<uint4*>(b[q]) + stage * stage_packs +
                     rank * L + idx;
        st_volatile16(dst, sanitize(in[q * part + idx]));
      }
    }
  }

  uint4* self = reinterpret_cast<uint4*>(b[rank]);
  uint4* dirty_base = self + dirty_stage * stage_packs;
  const uint4 sent = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
  for (int idx = tid; idx < dirty_size; idx += stride) dirty_base[idx] = sent;

  const uint4* cur = self + stage * stage_packs;
  const int mine = part_len(rank, size);
  for (int idx = tid; idx < mine; idx += stride) {
    uint4 own = sanitize(in[rank * part + idx]);
    uint4 v[kRanks] = {own, own, own, own};
    wait3(cur + idx, L, rank, v, wc);
    uint4 red = sanitize(reduce_owner_order(v, rank));
    out[rank * part + idx] = red;
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int q = (rank + i) & (kRanks - 1);
      uint4* dst = reinterpret_cast<uint4*>(b[q]) + stage * stage_packs +
                   (kRanks + rank) * L + idx;
      st_volatile16(dst, red);
    }
  }

  for (int idx = tid; idx < L; idx += stride) {
#pragma unroll
    for (int i = 1; i < kRanks; i++) {
      int q = (rank + i) & (kRanks - 1);
      if (idx < part_len(q, size))
        out[q * part + idx] = wait1(cur + (kRanks + q) * L + idx, q, wc);
    }
  }

  if (tid == 0) {
    volatile uint32_t* arrived = &ep->w[1];
    while (*arrived < gridDim.x) {
    }
    ep->w[2 + stage] = 2 * kRanks * L;
    ep->w[1] = 0;
    ep->w[8] += 1;
    ep->w[0] = (stage + 1) % 3;
  }
}

#undef AFD

}  // namespace arflags
"""

BIND_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void all_reduce(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> bufs,
                int64_t epoch_ptr, int64_t rank, int64_t stage_packs,
                int64_t threads, int64_t max_blocks, int64_t limit_ns,
                int64_t err_dev_ptr) {
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(inp.numel() == out.numel());
  TORCH_CHECK(inp.nbytes() % 16 == 0 && inp.nbytes() > 0);
  TORCH_CHECK(bufs.size() == 4);
  TORCH_CHECK(limit_ns > 0 && err_dev_ptr != 0);
  int size = static_cast<int>(inp.nbytes() / 16);
  int L = arflags::part_len(3, size);
  TORCH_CHECK(8 * L <= stage_packs, "message larger than the flags buffer");
  int blocks = static_cast<int>(
      std::min<int64_t>(max_blocks, (L + threads - 1) / threads));
  blocks = std::max(blocks, 1);
  arflags::Ptrs p;
  for (int i = 0; i < 4; i++) p.p[i] = reinterpret_cast<void*>(bufs[i]);
  arflags::two_shot<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      p, reinterpret_cast<const uint4*>(inp.data_ptr()),
      reinterpret_cast<uint4*>(out.data_ptr()),
      reinterpret_cast<arflags::Epoch*>(epoch_ptr), static_cast<int>(rank), size,
      static_cast<int>(stage_packs), static_cast<unsigned long long>(limit_ns),
      reinterpret_cast<long long*>(err_dev_ptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_sentinel(int64_t ptr, int64_t bytes) {
  // 0xFF bytes = the 0xFFFFFFFF sentinel word everywhere.
  C10_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(ptr), 0xFF, bytes));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

// Host-mapped, portable error record (16 int64 words, zeroed). Returns the host
// address and the device address; never freed (lives as long as the process).
std::vector<int64_t> alloc_error_record() {
  void* h = nullptr;
  C10_CUDA_CHECK(cudaHostAlloc(&h, 16 * sizeof(int64_t),
                               cudaHostAllocMapped | cudaHostAllocPortable));
  memset(h, 0, 16 * sizeof(int64_t));
  void* d = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, h, 0));
  return {reinterpret_cast<int64_t>(h), reinterpret_cast<int64_t>(d)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("all_reduce", &all_reduce, "flags-in-data two-shot all-reduce");
  m.def("fill_sentinel", &fill_sentinel, "fill a buffer with the sentinel");
  m.def("alloc_error_record", &alloc_error_record,
        "host-mapped error record for timed-out waits");
}
"""

CUDA_SRC = KERNEL_SRC + BIND_SRC

_MODULE = None


def _build_dir() -> str:
    """Where the JIT extension is cached; never inside the source tree."""
    d = (
        envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR
        or os.environ.get("TORCH_EXTENSIONS_DIR")
        or os.path.join(
            os.path.expanduser("~"), ".cache", "vllm", "custom_all_reduce_flags"
        )
    )
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def load_module():
    """Compile (or reuse) the extension. Frozen across calls."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    build = _build_dir()
    tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:10]
    name = f"custom_ar_flags_{tag}"
    src = os.path.join(build, f"{name}.cu")
    if not os.path.exists(src):
        tmp = src + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(CUDA_SRC)
        os.replace(tmp, src)
    _MODULE = load(
        name=name,
        sources=[src],
        build_directory=build,
        extra_cflags=["-O3", "-std=c++17"],
        # No --use_fast_math: the reduction must round exactly as the incumbent.
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode",
            "arch=compute_80,code=sm_80",
        ],
        verbose=False,
    )
    return _MODULE


def wait_limit_ns() -> int:
    s = float(envs.VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S)
    if not s > 0:
        raise ValueError(f"VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S must be > 0, got {s}")
    return int(s * 1e9)


_PHASES = {1: "reduce-scatter", 2: "all-gather"}


def describe(rec) -> str | None:
    """The error message for a record (sequence of 16 ints), or None if unset."""
    if not rec[0]:
        return None
    return (
        "Custom all-reduce flags-in-data wait timed out on rank %d after %.1f s "
        "(limit %.1f s, VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S): call %d, stage %d, "
        "%d packs, %s wait for rank %d (block %d, thread %d). The kernel trapped "
        "instead of returning a partial result; rank %d never delivered its data."
        % (rec[1], rec[7] / 1e9, rec[10] / 1e9, rec[3], rec[2], rec[4],
           _PHASES.get(rec[5], "?"), rec[6], rec[8], rec[9], rec[6])
    )


class ErrorRecord:
    """Host-mapped record the kernel fills before it traps, and its watchers.

    A daemon thread polls it (host memory: no CUDA call, works after the
    context has failed), it is checked again at interpreter exit, and callers
    may check it. When set, the message is logged and printed and the process
    ends with EXIT_CODE.
    """

    def __init__(self, module, poll_s: float = 0.25, on_error=None):
        self.host_ptr, self.dev_ptr = module.alloc_error_record()
        self._words = (ctypes.c_int64 * 16).from_address(self.host_ptr)
        self._on_error = on_error or _die
        self._stop = threading.Event()
        self._reported = False
        self._thread = threading.Thread(
            target=self._poll, args=(poll_s,), name="flags-ar-watch", daemon=True
        )
        self._thread.start()
        atexit.register(self.check)

    def read(self) -> list[int]:
        return list(self._words)

    def check(self) -> None:
        msg = describe(self.read())
        if msg is not None and not self._reported:
            self._reported = True
            self._on_error(msg)

    def _poll(self, poll_s: float) -> None:
        while not self._stop.wait(poll_s):
            self.check()

    def stop(self) -> None:
        self._stop.set()


def _die(msg: str) -> None:
    logger.error(msg)
    print(msg, file=sys.stderr, flush=True)
    os._exit(EXIT_CODE)


def largest_part_packs(nbytes: int) -> int:
    size = nbytes // 16
    return size - 3 * (size // WORLD_SIZE)


def launch_blocks(nbytes: int) -> int:
    return max(1, min(MAX_BLOCKS, math.ceil(largest_part_packs(nbytes) / THREADS)))


def gate_reason(
    world_size: int,
    same_node: bool,
    capability: tuple[int, int] | None,
    is_cuda: bool,
    pcie_p2p_allowed: bool,
) -> str | None:
    """Why the flags path stays off for this communicator (None: it may run)."""
    if not is_cuda:
        return "not a CUDA platform"
    if world_size != WORLD_SIZE:
        return f"world size {world_size} (validated at {WORLD_SIZE} only)"
    if not same_node:
        return "the group spans nodes"
    if capability != (8, 0):
        return f"compute capability {capability} (validated on 8.0 only)"
    if not pcie_p2p_allowed:
        return (
            "the PCIe peer-to-peer custom all-reduce is not enabled "
            "(VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE with expandable_segments off)"
        )
    return None


class FlagsAllreduce:
    """Per-communicator state: shared stage buffers and the epoch block.

    Construction is collective over `group` (IPC handle exchange, barrier).
    """

    def __init__(self, group, device: torch.device, rank: int, max_bytes: int,
                 create_shared_buffer, free_shared_buffer):
        self.module = load_module()
        self.group = group
        self.rank = rank
        self.max_bytes = int(max_bytes)
        self.stage_packs = 8 * largest_part_packs(self.max_bytes)
        self.buffer_bytes = 3 * self.stage_packs * 16
        self._free = free_shared_buffer
        self.ptrs = create_shared_buffer(self.buffer_bytes, group=group)
        self.module.fill_sentinel(self.ptrs[rank], self.buffer_bytes)
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=device)
        self.limit_ns = wait_limit_ns()
        self.errors = ErrorRecord(self.module)
        self._calls = 0
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def eligible(self, inp: torch.Tensor) -> bool:
        return inp.dtype == torch.bfloat16 and 0 < inp.nbytes <= self.max_bytes

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None):
        if out is None:
            out = torch.empty_like(inp)
        self.module.all_reduce(
            inp,
            out,
            self.ptrs,
            self.epoch.data_ptr(),
            self.rank,
            self.stage_packs,
            THREADS,
            launch_blocks(inp.nbytes),
            self.limit_ns,
            self.errors.dev_ptr,
        )
        self._calls += 1
        if self._calls % CHECK_EVERY == 0 and not torch.cuda.is_current_stream_capturing():
            self.errors.check()
        return out

    def close(self) -> None:
        self.errors.check()
        if self.ptrs is not None:
            self._free(self.ptrs, rank=self.rank)
            self.ptrs = None
