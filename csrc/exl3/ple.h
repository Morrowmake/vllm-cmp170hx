// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

// All tensors are contiguous CUDA tensors on one device. packed_rows contains
// selected rows, in global_ids order, rather than the full embedding table.
// A mapped pinned-host buffer is supported through its CUDA/UVA tensor view;
// its CPU owner and transfer/readiness lifetime remain the caller's responsibility.
// head_offsets must be sorted and describe nonoverlapping heads; the loader
// validates metadata values before capture. owner_hi is exclusive.
// No tensor allocation, host synchronization, or codebook table is needed.
void ple_decode_out(const torch::Tensor& packed_rows,
                    const torch::Tensor& global_ids,
                    const torch::Tensor& head_bias,
                    const torch::Tensor& head_offsets,
                    const torch::Tensor& head_vocab_sizes, int64_t owner_lo,
                    int64_t owner_hi, int64_t bits, torch::Tensor& out);

// Host only. Reads rows ids[0:count) of the compressed table in file fd
// (trellis at byte base, row_bytes per row) into dest, row i at byte
// i * row_bytes: the same bytes as Exl3PLERowReader.read_rows. Rows outside
// [owner_lo, owner_hi) or after their head's vocabulary are zero-filled.
// Rows in the page cache are read on the calling thread; the others on up
// to max_threads threads, after a non-blocking probe that also starts their
// reads (one attempt more than read_rows makes for a row not yet cached).
// A row that still fails after retries (1 ms
// backoff, doubling) is left unspecified and returned as one [index, errno
// (0: end of file), byte offset reached] row of the int64 result, sorted by
// index.
torch::Tensor ple_read_rows(int64_t fd, const torch::Tensor& ids,
                            torch::Tensor& dest, int64_t count, int64_t base,
                            int64_t row_bytes, int64_t owner_lo,
                            int64_t owner_hi, const torch::Tensor& head_offsets,
                            const torch::Tensor& head_vocab_sizes,
                            int64_t retries, int64_t max_threads);

// Registers the class PleHostReader(fd, ids, rows, base, row_bytes, owner_lo,
// owner_hi, head_offsets, head_vocab_sizes, retries, max_threads, slots):
// ple_read_rows of ids[0:count) into rows as a host function on a CUDA
// stream. enqueue(count) queues it on the current stream (never under
// capture); run(count) runs the same function on the calling thread.
// Rows that fail are poisoned (an FP16 NaN scale) instead of returned, and
// take() reports each finished read once, in queue order: [sequence, count,
// status (0 read, 1 stopped by an unexpected error with every row poisoned),
// failed rows, first failed index, its row id, its errno (0: end of file),
// byte offset reached, read ns]. At most `slots` reads are queued or not yet
// taken. The reader keeps its own duplicate of fd and the ids and rows
// tensors; a reader dropped while a read is still queued is left allocated.
void register_ple_host_reader(torch::Library& m);
