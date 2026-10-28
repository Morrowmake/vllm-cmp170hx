// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/library.h>

// Registers the class EngramHostFetch(fd, base, row_bytes, owner_lo,
// owner_hi, scales, table, ids, values, scales_out, cache_bytes, ways,
// retries, max_threads, slots): one rank's rows [owner_lo, owner_hi) of one
// DeepSeek V4.1 Engram table. scales is the resident [owner rows,
// scale_bytes] uint8 table; table is the resident [owner rows, row_bytes]
// value table, or empty to read values from fd (row r at base + r *
// row_bytes) through a set-associative LRU host row cache of at most
// cache_bytes. With fd opened O_DIRECT the rows are read directly, each
// through its aligned 4 KiB span (no page-cache probe, no pages left in the
// page cache), and the constructor refuses when a direct read of the first
// owned row fails; otherwise reads are buffered after a non-blocking
// page-cache probe. A fetch of ids[0:count) (global row ids) fills
// values[0:count) and scales_out[0:count): owned rows with their bytes,
// other ids with zero values and the scale byte 127; a row that still
// fails after `retries` read retries gets an E4M3 NaN first value and
// zeros, and is never cached. enqueue(count) queues the fetch as a host
// function on the current CUDA stream (never under capture); run(count)
// runs it on the calling thread. take() reports each finished fetch once,
// in queue order: [sequence, count, status (0 done, 1 stopped by an
// unexpected error with every row poisoned), failed rows, first failed row
// id, its errno (0: end of file), byte offset reached]. counters() holds
// the totals of every finished fetch (see vllm/models/deepseek_v41/nvidia/
// sm80_engram.py NATIVE_COUNTERS). prefault() touches every cache page and
// returns the time it took in ns. At most `slots` fetches are queued or not
// yet taken. For CUDA graphs, add_fixed(ids, values, scales_out, published)
// adds the one fixed staging of up to ids.numel() rows and a CPU int64 the
// fixed fetches add their failed rows to; enqueue_fixed(count, dev_ids,
// dev_values, dev_scales) queues, also under capture, a fetch of `count`
// rows on it: the copy of the ids, the fetch as a host function and the
// copies of its rows to the device, every argument fixed; run_fixed(count)
// runs that fetch on the calling thread; take_fixed() returns [failed rows,
// first failed row id, errno, offset, stopped
// fetches] of the fixed fetches since the last call. The fetch keeps its own
// duplicate of fd and the tensors; one dropped while a fetch is still queued,
// or after a fixed fetch was ever queued, is left allocated.
void register_engram_host_fetch(torch::Library& m);
