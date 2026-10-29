# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed PLE rows staged on the GPU without materializing the table."""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger

from ..common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)


class Qwen4ExpPLEMmapEmbedding(Qwen4ExpPLEEmbedding):
    """PLE rows read from safetensors CPU mappings and staged per step.

    Checkpoint shards arrive as ``num_shards`` row-contiguous CPU tensors
    that stay memory-mapped; only the rows the current step selects are
    gathered into a persistent GPU buffer, so the full table never becomes a
    RAM copy or a device allocation. One step's staging waits for the
    previous step's host-to-device copy through a CUDA event on the gather
    worker; the main thread performs no GPU synchronization.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEUnquantizedEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
        num_shards: int,
    ) -> None:
        if not isinstance(embedding_method, Qwen4ExpPLEUnquantizedEmbeddingMethod):
            raise ValueError(
                "mmap PLE stores raw unquantized rows; quantized PLE formats "
                f"require their own backend, got {type(embedding_method).__name__}"
            )
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("mmap PLE supports BF16 and FP16 tables only")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        if num_shards <= 0:
            raise ValueError("mmap PLE requires a positive shard count")
        if self.tp_size != 1:
            raise ValueError(
                f"mmap PLE requires ETP=TP=1, but the ETP world size is {self.tp_size}"
            )
        if not self.reduce_results:
            raise NotImplementedError(
                "mmap PLE does not support sequence parallelism (reduce_results=False)"
            )
        if max_total_tokens <= 0:
            raise ValueError("mmap PLE needs a positive staging token capacity")
        self._heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim
        self._storage_dtype = params_dtype
        self._row_bytes = self.embedding_dim * params_dtype.itemsize
        self._capacity = max_total_tokens
        self._num_shards = int(num_shards)
        self._shard_rows = (
            self.org_vocab_size + self._num_shards - 1
        ) // self._num_shards
        self._workers = min(8, os.cpu_count() or 1)
        self._shards: dict[int, torch.Tensor] = {}
        self._shard_bytes: dict[int, np.ndarray] = {}
        self._finalized = False
        self._closed = False
        self._rows_gpu: torch.Tensor | None = None
        self._ids_host: torch.Tensor | None = None
        self._rows_host: torch.Tensor | None = None
        self._rows_typed: torch.Tensor | None = None
        self._ids_np: np.ndarray | None = None
        self._rows_np: np.ndarray | None = None
        self._ids_ready: torch.cuda.Event | None = None
        self._staging_stream: torch.cuda.Stream | None = None
        self._gatherer: ThreadPoolExecutor | None = None
        self._pool: ThreadPoolExecutor | None = None
        # Engine shutdown does not close modules; join workers at exit.
        atexit.register(self.close)

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reserve a placeholder; checkpoint rows stay in their mappings."""
        return torch.empty(1, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, *_):
        raise RuntimeError(
            "mmap PLE reads checkpoint rows from their CPU mappings via "
            "load_shard(); nothing may be copied into the placeholder weight"
        )

    def _expected_rows(self, index: int) -> int:
        start = index * self._shard_rows
        return max(0, min(self._shard_rows, self.org_vocab_size - start))

    def load_shard(self, index: int, weight: torch.Tensor) -> None:
        """Retain a row-contiguous CPU shard without reading its contents."""
        if self._closed:
            raise RuntimeError("mmap PLE mappings were released by close()")
        if not isinstance(index, (int, np.integer)) or not (
            0 <= index < self._num_shards
        ):
            raise ValueError(
                f"PLE shard index {index} is outside [0, {self._num_shards})"
            )
        index = int(index)
        expected = self._expected_rows(index)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"PLE shard {index} must be a tensor")
        if weight.device.type != "cpu":
            raise ValueError(f"PLE shard {index} must stay on CPU mappings")
        if weight.dtype != self._storage_dtype:
            raise ValueError(
                f"PLE shard {index} must have dtype {self._storage_dtype}, "
                f"got {weight.dtype}"
            )
        if tuple(weight.shape) != (expected, self.embedding_dim):
            raise ValueError(
                f"PLE shard {index} must have shape "
                f"({expected}, {self.embedding_dim}), got {tuple(weight.shape)}"
            )
        if not weight.is_contiguous():
            raise ValueError(f"PLE shard {index} must be row-contiguous")
        rows = weight.view(torch.uint8).numpy().reshape(expected, self._row_bytes)
        stored = self._shards.get(index)
        if stored is not None:
            if stored.data_ptr() != weight.data_ptr():
                raise ValueError(f"PLE shard {index} was already loaded")
            return
        self._shards[index] = weight
        self._shard_bytes[index] = rows

    def finalize_load(self) -> None:
        """Validate shard completeness and freeze the row-to-mapping index.

        Raises:
            RuntimeError: Some checkpoint shard was never loaded, or the
                loaded rows do not cover the vocabulary exactly once.

        """
        if self._closed:
            raise RuntimeError("mmap PLE mappings were released by close()")
        if self._finalized:
            return
        missing = sorted(set(range(self._num_shards)) - self._shards.keys())
        if missing:
            raise RuntimeError(
                f"mmap PLE is missing {len(missing)} of {self._num_shards} "
                f"checkpoint shards (first missing: {missing[:5]})"
            )
        covered = sum(tensor.shape[0] for tensor in self._shards.values())
        if covered != self.org_vocab_size:
            raise RuntimeError(
                f"mmap PLE shards cover {covered} rows but the vocabulary "
                f"has {self.org_vocab_size}"
            )
        self._finalized = True
        logger.info(
            "mmap PLE: %d shards, %d rows, dim=%d storage=%s row_bytes=%d "
            "capacity=%d tokens",
            self._num_shards,
            self.org_vocab_size,
            self.embedding_dim,
            self._storage_dtype,
            self._row_bytes,
            self._capacity,
        )

    def gather_into(self, ids, destination: np.ndarray) -> None:
        """Copy the raw storage bytes of ``ids`` into ``(rows, row_bytes)``.

        Output rows follow the id order, repetitions included; duplicated
        ids are read from the mappings once. Negative or out-of-vocabulary
        ids become zero rows, matching the masked lookups of the resident
        backends.

        Args:
            ids: 1-D integer row indices (numpy array or CPU tensor).
            destination: Contiguous uint8 ``(len(ids), row_bytes)`` target.

        """
        if not self._finalized:
            raise RuntimeError("mmap PLE gather requires finalize_load() first")
        if self._closed:
            raise RuntimeError("mmap PLE mappings were released by close()")
        if isinstance(ids, torch.Tensor):
            if ids.device.type != "cpu":
                raise ValueError("mmap PLE gather reads ids on the CPU")
            ids = ids.detach().numpy()
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("mmap PLE gather ids must be one-dimensional")
        if (
            not isinstance(destination, np.ndarray)
            or destination.dtype != np.uint8
            or destination.ndim != 2
            or destination.shape[1] != self._row_bytes
            or destination.shape[0] != ids.shape[0]
            or not destination.flags["C_CONTIGUOUS"]
        ):
            raise ValueError(
                f"mmap PLE gather destination must be contiguous uint8 "
                f"({ids.shape[0]}, {self._row_bytes})"
            )
        if ids.shape[0] == 0:
            return
        self._gather_rows(ids, destination)

    def _gather_rows(self, ids: np.ndarray, destination: np.ndarray) -> None:
        valid = (ids >= 0) & (ids < self.org_vocab_size)
        if not bool(valid.all()):
            destination[~valid] = 0
        picked = ids[valid]
        if picked.size == 0:
            return
        unique, inverse = np.unique(picked, return_inverse=True)
        groups = unique // self._shard_rows
        bounds = np.searchsorted(groups, np.arange(self._num_shards + 1))
        values = np.empty((unique.shape[0], self._row_bytes), dtype=np.uint8)
        for shard in np.flatnonzero(bounds[1:] > bounds[:-1]):
            lo, hi = int(bounds[shard]), int(bounds[shard + 1])
            local = unique[lo:hi] - shard * self._shard_rows
            values[lo:hi] = self._shard_bytes[int(shard)][local]
        destination[valid] = values[inverse]

    def _ensure_buffers(self) -> torch.Tensor:
        """Allocate the persistent staging buffers on first GPU use."""
        if self._closed:
            raise RuntimeError("mmap PLE staging buffers were closed")
        rows = self._rows_gpu
        if rows is not None:
            return rows
        if not torch.cuda.is_available():
            raise RuntimeError("mmap PLE staging requires a CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "mmap PLE staging buffers must be allocated before graph "
                "capture (a dummy run precedes it)"
            )
        device = torch.device("cuda")
        capacity_rows = self._capacity * self._heads
        self._rows_gpu = torch.zeros(
            (self._capacity, self._output_dim),
            dtype=self._storage_dtype,
            device=device,
        )
        self._ids_host = torch.empty(
            capacity_rows, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._rows_host = torch.empty(
            capacity_rows * self._row_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._rows_typed = self._rows_host.view(self._storage_dtype).reshape(
            self._capacity, self._output_dim
        )
        self._ids_np = self._ids_host.numpy()
        self._rows_np = self._rows_host.numpy().reshape(-1, self._row_bytes)
        self._ids_ready = torch.cuda.Event()
        if self._gatherer is None:
            self._gatherer = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ple-gpu-ids"
            )
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="ple-read"
            )
        return self._rows_gpu

    def prepare(self, ngram_ids: torch.Tensor, num_tokens_after_padding: int) -> None:
        """Stage the current step's rows before the model forward runs.

        The ids go to pinned memory without a host wait and a CUDA event
        covers both that copy and the previous step's host-to-device copy,
        so the gather worker reuses the pinned row buffer only after its
        last reader finished. The gather runs in the worker and its chunks
        in the read pool; a failed gather raises here, failing the step.

        Args:
            ngram_ids: Current step's n-gram row indices as ``[tokens,
                heads]`` (or flat), CPU or CUDA. CUDA ids keep the GPU-side
                values authoritative.
            num_tokens_after_padding: Rows to make valid, including the
                padded rows of the full replay batch.

        Raises:
            ValueError: The ids shape or the batch sizes exceed capacity.
            RuntimeError: A gather failed or staging runs during capture.

        """
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mmap PLE prepare must run outside CUDA graph capture")
        if ngram_ids.ndim == 2:
            if ngram_ids.shape[1] != self._heads:
                raise ValueError(
                    f"PLE ids must carry {self._heads} heads per token, got "
                    f"{tuple(ngram_ids.shape)}"
                )
            tokens = ngram_ids.shape[0]
        elif ngram_ids.ndim == 1:
            if ngram_ids.numel() % self._heads:
                raise ValueError(
                    "flat PLE ids must contain a whole number of "
                    f"{self._heads}-head tokens"
                )
            tokens = ngram_ids.numel() // self._heads
        else:
            raise ValueError("PLE ids must be [tokens, heads] or flat")
        if ngram_ids.is_floating_point() or ngram_ids.dtype == torch.bool:
            raise ValueError("PLE ids must be integer row indices")
        if ngram_ids.dtype != torch.int64:
            ngram_ids = ngram_ids.to(torch.int64)
        padded = int(num_tokens_after_padding)
        if not 0 <= tokens <= padded <= self._capacity:
            raise ValueError(
                f"PLE staging needs {tokens} rows padded to {padded}, "
                f"capacity is {self._capacity} tokens"
            )
        rows = self._ensure_buffers()
        if not self._finalized:
            raise RuntimeError("mmap PLE staging requires finalize_load() first")
        self._staging_stream = torch.cuda.current_stream()
        if tokens:
            assert self._ids_host is not None
            assert self._ids_ready is not None
            assert self._gatherer is not None
            assert self._rows_typed is not None
            ids_flat = ngram_ids.reshape(-1)
            self._ids_host[: ids_flat.numel()].copy_(ids_flat, non_blocking=True)
            self._ids_ready.record()
            # Joins the gather: its exception fails this step's serving.
            self._gatherer.submit(self._gather, tokens).result()
            rows[:tokens].copy_(self._rows_typed[:tokens], non_blocking=True)
        if tokens < padded:
            rows[tokens:padded].zero_()

    def _gather(self, tokens: int) -> None:
        """Wait for this step's ids, then gather its rows in the read pool."""
        assert self._ids_ready is not None
        assert self._ids_np is not None
        assert self._rows_np is not None
        assert self._pool is not None
        self._ids_ready.synchronize()
        count = tokens * self._heads
        ids = self._ids_np[:count]
        destination = self._rows_np[:count]
        step = (count + self._workers - 1) // self._workers
        futures = [
            self._pool.submit(
                self._gather_rows,
                ids[begin : begin + step],
                destination[begin : begin + step],
            )
            for begin in range(0, count, step)
        ]
        # Join every chunk before propagating the first failure.
        errors: list[BaseException] = []
        for future in futures:
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            raise errors[0]

    def view(self, total_tokens: int) -> torch.Tensor:
        """Return persistent flattened GPU rows valid for the current step."""
        if not 0 <= total_tokens <= self._capacity:
            raise ValueError(
                f"PLE rows capacity exceeded: {total_tokens}/{self._capacity}"
            )
        return self._ensure_buffers()[:total_tokens]

    def zero_rows(self, total_tokens: int) -> None:
        """Zero the current row window for dummy runs and graph capture."""
        self.view(total_tokens).zero_()

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        raise RuntimeError("mmap PLE staging uses prepare() and view()")

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise RuntimeError("mmap PLE staging uses prepare() and view()")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "mmap PLE lookup requires prepare() before the forward and "
            "view() for its rows; direct embedding lookup is not supported"
        )

    def close(self) -> None:
        """Join the gather workers, then release staging and mappings."""
        if self._closed:
            return
        self._closed = True
        if self._gatherer is not None:
            self._gatherer.shutdown(wait=True)
            self._gatherer = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._rows_gpu is not None and torch.cuda.is_initialized():
            # The last host-to-device copy read the pinned rows; let it
            # finish on the stream that queued it before freeing them.
            try:
                done = torch.cuda.Event()
                done.record(self._staging_stream)
                done.synchronize()
            except RuntimeError:
                logger.debug("mmap PLE: CUDA context is gone at close")
        self._rows_gpu = None
        self._rows_typed = None
        self._ids_host = None
        self._rows_host = None
        self._ids_np = None
        self._rows_np = None
        self._ids_ready = None
        self._shards.clear()
        self._shard_bytes.clear()
