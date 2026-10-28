// SPDX-License-Identifier: Apache-2.0
// Positional reads of selected fixed-size rows of a file region: a
// non-blocking page-cache probe, then threaded preads with bounded retries;
// or, for a descriptor opened with O_DIRECT, threaded aligned direct reads
// that leave the page cache alone. The core of the PLE row reader
// (ple_read.cpp) and the Engram row fetch (engram_fetch.cpp).
#pragma once

#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace exl3_rows {

// Direct reads: file offset, length and buffer aligned to this many bytes.
constexpr int64_t kDirectAlign = 4096;

// The most bytes the aligned direct read of one row covers.
inline int64_t direct_span_bytes(int64_t row_bytes) {
  return (row_bytes + kDirectAlign - 1) / kDirectAlign * kDirectAlign +
         kDirectAlign;
}

struct Failure {
  int64_t index;
  int64_t error;  // errno, or 0 for end of file
  int64_t offset;
};

struct Table {
  int fd;
  int64_t base;
  int64_t row_bytes;
  int64_t owner_lo;
  int64_t owner_hi;
  const int64_t* head_offsets;
  const int64_t* head_ends;  // offsets[h] + vocab_sizes[h]
  int64_t heads;
  int64_t retries;
  // fd was opened with O_DIRECT: rows are read through aligned spans.
  bool direct = false;
};

// Exl3PLERowReader.owns: inside the owner range and inside its head's
// vocabulary (padding rows after a head belong to nobody).
inline bool owns(const Table& t, int64_t row) {
  if (row < t.owner_lo || row >= t.owner_hi) return false;
  const int64_t* end = t.head_offsets + t.heads;
  const int64_t* after = std::upper_bound(t.head_offsets, end, row);
  if (after == t.head_offsets) return false;
  return row < t.head_ends[after - t.head_offsets - 1];
}

// pread_exact_into: partial reads continue; EINTR is retried as os.preadv
// retries it; any other error or end of file is returned with the offset
// reached. Returns -1 on success.
inline int64_t read_exact(int fd, uint8_t* dest, int64_t length, int64_t offset,
                   int64_t* reached) {
  int64_t done = 0;
  while (done < length) {
    ssize_t count = pread(fd, dest + done, length - done, offset + done);
    if (count < 0) {
      if (errno == EINTR) continue;
      *reached = offset + done;
      return errno;
    }
    if (count == 0) {
      *reached = offset + done;
      return 0;
    }
    done += count;
  }
  return -1;
}

// The direct read of the aligned [start, start + length) into the aligned
// `span` until at least its first `need` bytes are in. A direct read ends
// short only at the end of the file, so a short read that leaves `need`
// unmet is end of file. Returns as read_exact.
inline int64_t read_direct(int fd, uint8_t* span, int64_t length,
                           int64_t start, int64_t need, int64_t* reached) {
  int64_t done = 0;
  while (done < need) {
    ssize_t count = pread(fd, span + done, length - done, start + done);
    if (count < 0) {
      if (errno == EINTR) continue;
      *reached = start + done;
      return errno;
    }
    if (count == 0) break;
    done += count;
    if (done % kDirectAlign) break;
  }
  if (done >= need) return -1;
  *reached = start + done;
  return 0;
}

// Exl3PLERowReader.read_exact: the same bounded retries and backoff (1 ms,
// doubling). A direct table reads the row's aligned span into `span`
// (direct_span_bytes, aligned) and copies the row out of it.
inline bool read_row(const Table& t, uint8_t* dest, int64_t row, uint8_t* span,
                     Failure* failure) {
  const int64_t offset = t.base + row * t.row_bytes;
  const int64_t start = offset - offset % kDirectAlign;
  const int64_t need = offset - start + t.row_bytes;
  const int64_t length =
      (need + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
  int64_t reached = offset;
  int64_t error = -1;
  for (int64_t attempt = 0; attempt <= t.retries; ++attempt) {
    error = t.direct ? read_direct(t.fd, span, length, start, need, &reached)
                     : read_exact(t.fd, dest, t.row_bytes, offset, &reached);
    if (error < 0) {
      if (t.direct) std::memcpy(dest, span + (offset - start), t.row_bytes);
      return true;
    }
    if (attempt == t.retries) break;
    const int64_t ms = int64_t{1} << std::min<int64_t>(attempt, 40);
    timespec wait{static_cast<time_t>(ms / 1000),
                  static_cast<long>(ms % 1000) * 1000000L};
    while (nanosleep(&wait, &wait) != 0 && errno == EINTR) {
    }
  }
  failure->error = error;
  failure->offset = reached;
  return false;
}

// Reads the rows at indices[begin, end); stores their failures in order from
// failures[0] and returns how many there are. `span` is this caller's
// aligned buffer for a direct table, else unused.
inline int64_t read_rows(const Table& t, uint8_t* dest, const int64_t* ids,
                  const int64_t* indices, int64_t begin, int64_t end,
                  uint8_t* span, Failure* failures) {
  int64_t failed = 0;
  for (int64_t i = begin; i < end; ++i) {
    const int64_t index = indices[i];
    Failure failure{index, 0, 0};
    if (!read_row(t, dest + index * t.row_bytes, ids[index], span, &failure)) {
      failures[failed++] = failure;
    }
  }
  return failed;
}

struct AlignedFree {
  void operator()(uint8_t* p) const { std::free(p); }
};

// `bytes` aligned to kDirectAlign, or none for zero bytes.
inline std::unique_ptr<uint8_t, AlignedFree> aligned_bytes(int64_t bytes) {
  if (bytes <= 0) return nullptr;
  void* p = nullptr;
  if (posix_memalign(&p, kDirectAlign, static_cast<size_t>(bytes)) != 0) {
    throw std::bad_alloc();
  }
  return std::unique_ptr<uint8_t, AlignedFree>(static_cast<uint8_t*>(p));
}

// What a batch read needs besides its rows, allocated for up to `rows` rows
// and max_threads threads before the read starts; span_bytes > 0 adds one
// aligned direct-read span per thread.
struct Scratch {
  Scratch(int64_t rows, int64_t max_threads, int64_t span_bytes = 0)
      : failures(new Failure[std::max<int64_t>(rows, 1)]),
        failed(new int64_t[max_threads]),
        span_bytes(span_bytes),
        spans(aligned_bytes(max_threads * span_bytes)) {
    pending.reserve(rows);
    workers.reserve(max_threads);
    inline_chunks.reserve(max_threads);
  }
  // The direct-read span of chunk `chunk`, or null for buffered reads.
  uint8_t* span(int64_t chunk) const {
    return spans ? spans.get() + chunk * span_bytes : nullptr;
  }
  std::vector<int64_t> pending;
  std::unique_ptr<Failure[]> failures;
  std::unique_ptr<int64_t[]> failed;  // per chunk
  std::vector<std::thread> workers;
  std::vector<int64_t> inline_chunks;
  int64_t span_bytes;
  std::unique_ptr<uint8_t, AlignedFree> spans;  // per chunk
};

// Reads rows ids[0:count) into out as ple_read_rows (ple.h) describes and
// returns how many failed; their failures are scratch.failures[0, n), sorted
// by index. Only starting a thread allocates, and a thread that cannot start
// leaves its chunk to the calling thread, so no started thread is left
// joinable by an exception. A direct table needs a scratch with spans; it
// skips the page-cache probe, so every owned row is read from storage.
inline int64_t read_batch(const Table& t, uint8_t* out, const int64_t* id,
                   int64_t count, int64_t max_threads, Scratch& s) {
  // Rows already in the page cache are read here without waiting for
  // storage; the rest are read below, on up to max_threads threads, so their
  // storage latencies overlap. For a row that is not cached the probe also
  // starts the read of its page, so such a row gets one more read attempt
  // than Exl3PLERowReader.read_rows makes.
  s.pending.clear();
  bool nowait = !t.direct;
  for (int64_t index = 0; index < count; ++index) {
    uint8_t* row = out + index * t.row_bytes;
    if (!owns(t, id[index])) {
      std::memset(row, 0, t.row_bytes);
      continue;
    }
    if (nowait) {
      iovec vector{row, static_cast<size_t>(t.row_bytes)};
      ssize_t done = preadv2(t.fd, &vector, 1, t.base + id[index] * t.row_bytes,
                             RWF_NOWAIT);
      if (done == t.row_bytes) continue;
      if (done < 0 &&
          (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
        nowait = false;
      }
    }
    s.pending.push_back(index);
  }

  const int64_t rows = static_cast<int64_t>(s.pending.size());
  if (rows == 0) return 0;
  const int64_t threads = std::min(max_threads, rows);
  const int64_t step = (rows + threads - 1) / threads;
  const int64_t* pending = s.pending.data();
  Failure* failures = s.failures.get();
  int64_t* failed = s.failed.get();
  s.workers.clear();
  s.inline_chunks.clear();
  int64_t chunks = 1;
  for (; chunks < threads && chunks * step < rows; ++chunks) {
    const int64_t chunk = chunks;
    const int64_t begin = chunk * step;
    const int64_t end = std::min(rows, begin + step);
    uint8_t* span = s.span(chunk);
    try {
      s.workers.emplace_back(
          [&t, out, id, pending, begin, end, span, failures, failed, chunk] {
            failed[chunk] = read_rows(t, out, id, pending, begin, end, span,
                                      failures + begin);
          });
    } catch (...) {
      s.inline_chunks.push_back(chunk);
    }
  }
  failed[0] = read_rows(t, out, id, pending, 0, std::min(rows, step),
                        s.span(0), failures);
  for (int64_t chunk : s.inline_chunks) {
    const int64_t begin = chunk * step;
    failed[chunk] = read_rows(t, out, id, pending, begin,
                              std::min(rows, begin + step), s.span(chunk),
                              failures + begin);
  }
  for (std::thread& worker : s.workers) worker.join();

  // Chunk c stored its failures from failures[c * step]; chunks hold
  // increasing indices, so packing them in chunk order keeps index order.
  int64_t total = failed[0];
  for (int64_t chunk = 1; chunk < chunks; ++chunk) {
    std::memmove(failures + total, failures + chunk * step,
                 failed[chunk] * sizeof(Failure));
    total += failed[chunk];
  }
  return total;
}

}  // namespace exl3_rows
