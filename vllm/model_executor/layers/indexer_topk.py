# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k kernels for the DSA sparse attention indexer."""

import functools

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# cooperative_topk's hard row limit: one cluster wave covers at most this many
# rows. "auto" uses cooperative_topk for every batch within the limit and falls
# back to persistent_topk past it; explicit ``cooperative`` requests are
# validated against the same bound.
AUTO_COOPERATIVE_MAX_ROWS = 64

# ---------------------------------------------------------------------------
# DeepSelect (vllm._deepselect_C)
# ---------------------------------------------------------------------------

# Matches the -1 fill convention used for topk_indices_buffer elsewhere.
IDX_OOB_FILL_VALUE = -1

try:
    import vllm._deepselect_C  # noqa: F401  (registers torch.ops.deep_select)
except ImportError as e:
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "Failed to import the DeepSelect extension (vllm._deepselect_C): %s", e
    )


@functools.lru_cache(maxsize=1)
def get_deep_select_stride_requirement() -> tuple[int, int]:
    """Stride alignment requirement (input, output) in bytes."""
    return torch.ops.deep_select.get_alignment_requirement()


def is_deep_select_supported(input: torch.Tensor, topk: int) -> bool:
    """Whether the kernel accepts this input (dtype/stride/topk constraints)."""
    return (
        topk <= 4096
        and input.shape[1] < 2**23
        and input.dtype in (torch.float32, torch.bfloat16)
        and input.stride(1) == 1
        and input.stride(0)
        * input.element_size()
        % get_deep_select_stride_requirement()[0]
        == 0
    )


def _get_empty_and_aligned_tensor(
    dim0: int, dim1: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Tensor with shape (dim0, dim1) whose stride(0) is 32B-aligned."""
    output_stride_requirement = get_deep_select_stride_requirement()[1] // (
        dtype.itemsize
    )
    assert output_stride_requirement > 0
    dim1_rounded = (
        (dim1 + output_stride_requirement - 1) // output_stride_requirement
    ) * output_stride_requirement
    return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]


def deep_select_topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Select the top-k indices per row of `input` with DeepSelect.

    Args:
        input: (num_rows, vocab_size), bf16 or fp32. stride(1) must be 1 and
            stride(0) must be 1024B-aligned.
        topk: Number of elements to select per row; must be <= 4096.
        end: Optional (num_rows,) int32 tensor with the exclusive right
            boundary of each row. Rows with `end[i] < topk` get their
            remaining indices filled with -1.
        output_idx: Optional preallocated (num_rows, topk) output tensor whose
            stride(0) is 32B-aligned (e.g. a slice of a wider buffer).
        indices_dtype: Output dtype when `output_idx` is not provided.

    Returns:
        The (num_rows, topk) indices tensor.

    """
    assert input.dim() == 2 and input.stride(1) == 1
    assert (
        input.stride(0) * input.element_size() % get_deep_select_stride_requirement()[0]
        == 0
    )

    num_rows = input.shape[0]
    if output_idx is None:
        output_idx = _get_empty_and_aligned_tensor(
            num_rows, topk, input.device, indices_dtype
        )
    else:
        assert output_idx.dtype == indices_dtype
        assert output_idx.shape[0] >= num_rows and output_idx.shape[1] >= topk

    torch.ops.deep_select.topk(
        input,
        topk,
        None,  # begin is not supported
        end,
        False,  # sorted_value
        False,  # sorted_index
        None,  # output_value
        output_idx,
        None,  # output_idx_offset
        IDX_OOB_FILL_VALUE,
        float("-inf"),  # value_oob_fill_value
        False,  # return_value
        False,  # abort_when_nan_found: dummy/capture passes can feed NaN
        # logits from uninitialized KV cache; the reference topk kernels do
        # not trap on NaN, and trapping would abort CUDA graph capture.
    )
    return output_idx


# ---------------------------------------------------------------------------
# Canonical (tie-stable) top-k -- VLLM_GLM5_TOPK_CANONICAL
# ---------------------------------------------------------------------------
#
# Every stock top-k here selects the correct top-k *scores*, but none of them
# defines which of several equal scores wins a contested slot: persistent_topk
# and topKPerRowJob both place threshold-bin entries with an atomicAdd, so the
# selected index SET is a per-call lottery whenever the row contains exact
# ties. The DSA indexer logits are a dot product against an fp8-e4m3 K cache,
# which quantises many keys onto the same grid point, so ties are common and
# a different set means a different KV page set and a different attention
# output for an identical prompt.
#
# The canonical order is (score descending, column index ascending). It is
# realised without a full sort by packing both fields into one strictly
# ordered int64 key and running an ordinary top-k on that: with no two keys
# equal, every top-k implementation returns the same answer.


# Widest row the composite key can address. 32 score bits + this must stay
# below 63 so the key is always a positive int64.
_MAX_INDEX_BITS = 30

# Row budget for the int64 key buffer. A 1152-row prefill chunk over a 32768
# wide logits tensor would otherwise allocate a single 302 MB temporary on top
# of the 151 MB of logits.
_CANONICAL_KEY_BUDGET_BYTES = 64 << 20


@functools.cache
def use_canonical_topk() -> bool:
    """Whether to take the canonical tie-break. Cached: the flag is read once
    per process because it is a boot-time choice (tests call cache_clear)."""
    import vllm.envs as envs

    return envs.VLLM_GLM5_TOPK_CANONICAL


@functools.cache
def use_sorted_topk() -> bool:
    """Whether the indexer sorts its selected indices (VLLM_GLM5_TOPK_SORTED).
    Read once per process; tests call cache_clear."""
    import vllm.envs as envs

    on = bool(envs.VLLM_GLM5_TOPK_SORTED)
    if on:
        logger.info_once(
            "GLM-5 sorted top-k active: the sparse indexer's selected indices "
            "are put in ascending order before attention "
            "(VLLM_GLM5_TOPK_SORTED=1; set 0 to disable)"
        )
    return on


_SORT_FILL = 2**31 - 1


def sort_selected_topk_(ids: torch.Tensor) -> torch.Tensor:
    """Sort each row of a selected-index tensor in place, ascending, with the
    -1 fill kept at the end of the row.

    ``ids`` is (rows, k) int32 or int64 and may be a strided view (a column
    slice of the persistent top-k buffer). The selected set of every row is
    unchanged; only its order becomes a function of the set, so consumers
    that accumulate in index order give the same result for the same set.
    Static shapes and no host reads: safe inside CUDA graph capture.
    """
    if ids.numel() == 0:
        return ids
    key = ids.masked_fill(ids < 0, _SORT_FILL)
    key = torch.sort(key, dim=-1).values
    ids.copy_(key.masked_fill(key == _SORT_FILL, -1))
    return ids


def _monotonic_int_key(logits: torch.Tensor) -> torch.Tensor:
    """Map fp32 to int64 preserving order: a > b iff key(a) > key(b).

    The standard IEEE-754 trick. Non-negative floats keep their bit pattern
    with the high bit set; negative floats are inverted, which reverses their
    magnitude ordering. -0.0 is folded onto +0.0 first, because the two are
    equal under `==` and must therefore be one tie group, not two. NaN sorts
    above +inf, matching torch.topk's own behaviour (the dummy/capture passes
    that feed NaN logits are documented in deep_select_topk above).
    """
    src = logits if logits.is_contiguous() else logits.contiguous()
    bits = src.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where(bits == 0x80000000, torch.zeros_like(bits), bits)
    return torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)


def canonical_topk(
    logits: torch.Tensor,
    k: int,
    *,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    relative: bool = False,
    identity_when_short: bool = False,
) -> torch.Tensor:
    """Top-k column indices per row, broken canonically on exact ties.

    Args:
        logits: (num_rows, num_cols) fp32.
        k: number of columns to select per row.
        row_ends: (num_rows,) int, exclusive right boundary of each row.
        row_starts: (num_rows,) int, inclusive left boundary; default 0.
        out: optional (num_rows, >= k) int32 destination.
        relative: emit indices relative to ``row_starts`` -- the convention
            ``top_k_per_row_prefill`` uses (csrc/libtorch_stable/sampler.cu,
            ``topKPerRowJob``'s store loop subtracts ``rowStart``).
        identity_when_short: reproduce the same kernel's shortcut, which for a
            window no longer than k emits 0..len-1 in *identity* order rather
            than score order, then -1 fill. The selected set is the whole
            window either way, so this only preserves byte-compatibility with
            the incumbent on rows that were never ambiguous.

    Returns:
        ``out`` (int32), with -1 past each row's length.

    """
    assert logits.dim() == 2, "canonical_topk expects 2-D logits"
    assert logits.dtype == torch.float32, f"expected fp32 logits, got {logits.dtype}"
    num_rows, num_cols = logits.shape
    assert k <= num_cols, f"k={k} exceeds row width {num_cols}"
    assert num_cols < (1 << _MAX_INDEX_BITS), (
        f"row width {num_cols} exceeds the composite key's index field"
    )

    device = logits.device
    if out is None:
        out = torch.empty((num_rows, k), dtype=torch.int32, device=device)

    shift = max(1, (num_cols - 1).bit_length())
    cols = torch.arange(num_cols, device=device, dtype=torch.int64)
    inv_cols = (num_cols - 1) - cols
    slot = torch.arange(k, device=device, dtype=torch.int64)

    ends = row_ends.reshape(-1).to(torch.int64)
    starts = (
        torch.zeros_like(ends)
        if row_starts is None
        else row_starts.reshape(-1).to(torch.int64)
    )
    lengths = (ends - starts).clamp_(min=0)

    rows_per_chunk = max(1, _CANONICAL_KEY_BUDGET_BYTES // (num_cols * 8))
    for r0 in range(0, num_rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, num_rows)
        blk_starts = starts[r0:r1].unsqueeze(1)
        blk_ends = ends[r0:r1].unsqueeze(1)
        blk_lengths = lengths[r0:r1].unsqueeze(1)

        key = (_monotonic_int_key(logits[r0:r1]) << shift) | inv_cols
        # Keys are non-negative, so -1 is strictly below every in-window key
        # and needs no -inf sentinel in the score domain.
        in_window = (cols.unsqueeze(0) >= blk_starts) & (cols.unsqueeze(0) < blk_ends)
        key = torch.where(in_window, key, torch.full_like(key, -1))

        idx = key.topk(k, dim=-1).indices
        if relative:
            idx = idx - blk_starts

        filled = slot.unsqueeze(0) < blk_lengths
        sel = torch.where(filled, idx, torch.full_like(idx, -1))
        if identity_when_short:
            ident = torch.where(
                filled, slot.unsqueeze(0).expand_as(idx), torch.full_like(idx, -1)
            )
            sel = torch.where(blk_lengths <= k, ident, sel)
        out[r0:r1, :k].copy_(sel.to(torch.int32))

    return out


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@functools.cache
def get_indexer_topk(backend: str) -> "SparseIndexerTopk":
    return SparseIndexerTopk(backend)


class SparseIndexerTopk(torch.nn.Module):
    """The sparse indexer's decode top-k stage.

    Selects among the available top-k kernels (see
    kernel_config.sparse_indexer_topk_backend) and runs the chosen one.
    """

    def __init__(self, backend: str | None = None) -> None:
        super().__init__()
        if backend is None:
            backend = (
                get_current_vllm_config().kernel_config.sparse_indexer_topk_backend
            )
        self._backend = backend
        self._is_cuda = current_platform.is_cuda()
        self._has_deep_select = self._is_cuda and (
            current_platform.is_device_capability_family(100)
        )
        self._has_flashinfer_topk = has_flashinfer()
        self._cooperative_capable = self._is_cuda and (
            current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )

    def resolve_backend(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """Resolve the decode top-k implementation from the configured
        backend ("auto" = the pre-existing chain, or a validated explicit
        value)."""
        if self._backend == "auto":
            return self._resolve_auto(logits, topk_tokens, num_rows)

        failures: list[str] = []
        if self._backend == "cooperative":
            failures = self._cooperative_constraints(logits, topk_tokens, num_rows)
        elif self._backend == "persistent":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if topk_tokens not in (512, 1024, 2048):
                failures.append(
                    f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
                )
        elif self._backend == "deep_select":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            elif not self._has_deep_select:
                failures.append("requires SM100a/SM103a (10.x device family)")
            elif not is_deep_select_supported(logits, topk_tokens):
                failures.append(
                    f"inputs violate DeepSelect's constraints: dtype={logits.dtype},"
                    f" stride={logits.stride()}, topk={topk_tokens}"
                )
        elif self._backend == "flashinfer":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if not self._has_flashinfer_topk:
                failures.append(
                    "flashinfer.topk.top_k_ragged_transform is not importable"
                )
            if logits.dtype != torch.float32 or logits.stride(1) != 1:
                failures.append(
                    f"requires fp32 logits with stride(1) == 1, got "
                    f"dtype={logits.dtype}, stride={logits.stride()}"
                )
        if failures:
            raise RuntimeError(
                f"sparse_indexer_topk_backend='{self._backend}' was requested, but: "
                + "; ".join(failures)
            )
        return self._backend

    def _resolve_auto(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """The priority chain: cooperative -> persistent -> per_row.
        deep_select/flashinfer/torch are opt-in only.

        cooperative_topk is preferred whenever it is applicable, i.e. within
        its AUTO_COOPERATIVE_MAX_ROWS row limit; larger batches go to
        persistent_topk.
        """
        if not self._cooperative_constraints(logits, topk_tokens, num_rows):
            return "cooperative"
        if self._is_cuda and topk_tokens in (512, 1024, 2048):
            return "persistent"
        return "per_row"

    def _cooperative_constraints(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> list[str]:
        """Unmet constraints of cooperative_topk (empty when applicable)."""
        failures = []
        if not self._is_cuda:
            failures.append("requires a CUDA platform")
        if topk_tokens not in (512, 1024, 2048):
            failures.append(
                f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
            )
        if num_rows > AUTO_COOPERATIVE_MAX_ROWS:
            failures.append(
                f"num_rows must be <= {AUTO_COOPERATIVE_MAX_ROWS}, got {num_rows}"
            )
        if logits.stride(0) % 4 != 0:
            failures.append(
                f"logits.stride(0) must be divisible by 4, got {logits.stride(0)}"
            )
        if self._is_cuda and not self._cooperative_capable:
            failures.append("requires SM90+ and is not supported on the SM12x family")
        return failures

    @staticmethod
    def _row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
        """Per-row exclusive end offsets (int32, (num_rows,)) for top-k
        kernels that take ragged lengths (DeepSelect, FlashInfer, torch
        reference).

        seq_lens is (B, next_n) per-row effective lens for native spec decode
        and (B, 1) otherwise, in which case per-row lens are derived the same
        way as the other decode top-k kernels.
        """
        if seq_lens.numel() == num_rows:
            row_ends = seq_lens.reshape(-1)
        else:
            next_n_offsets = torch.arange(
                next_n, dtype=torch.int32, device=seq_lens.device
            )
            row_ends = (
                (seq_lens.reshape(-1, 1) - next_n + 1 + next_n_offsets)
                .clamp_(min=0)
                .reshape(-1)
            )
        assert row_ends.dtype == torch.int32
        return row_ends

    def forward(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk_indices: torch.Tensor,
        topk_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Run the resolved decode top-k implementation, writing into
        topk_indices (int32, -1 fill for rows shorter than topk_tokens)."""
        if use_canonical_topk():
            # Selection is done canonically rather than by any of the
            # kernels below, all of which place tied entries by atomic
            # arrival order. Same scores, defined set.
            canonical_topk(
                logits,
                topk_tokens,
                row_ends=self._row_ends(seq_lens, next_n, logits.shape[0]),
                out=topk_indices,
            )
            return
        backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])
        if backend == "deep_select":
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            deep_select_topk(logits, topk_tokens, end=row_ends, output_idx=topk_indices)
        elif backend == "cooperative":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                max_seq_len,
            )
        elif backend == "persistent":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            # persistent_topk's last argument is the host-side "max seq_len
            # across rows" bound (it gates the sampled_topk path and clamps
            # per-row lengths), not the padded row width: logits is sized to
            # max_model_len, so passing its width would incorrectly enable the
            # long-row kernels on short-context batches.
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                max_seq_len,
            )
        elif backend == "flashinfer":
            # Deferred: importing flashinfer initializes CUDA at import time.
            from flashinfer.topk import top_k_ragged_transform

            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            offsets = torch.zeros(
                logits.shape[0], dtype=torch.int32, device=logits.device
            )
            # top_k_ragged_transform selects within [0, row_ends[i]) per
            # row and -1-fills past the row length.
            indices = top_k_ragged_transform(logits, offsets, row_ends, topk_tokens)
            topk_indices.copy_(indices)
        elif backend == "torch":
            # Debug reference: mask everything past each row's end, then topk.
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            cols = torch.arange(logits.shape[1], device=logits.device)
            masked = logits.masked_fill(
                cols.unsqueeze(0) >= row_ends.unsqueeze(1), float("-inf")
            )
            indices = masked.topk(topk_tokens, dim=-1).indices
            in_range = torch.arange(topk_tokens, device=logits.device).unsqueeze(
                0
            ) < row_ends.unsqueeze(1)
            indices = torch.where(in_range, indices, -1)
            topk_indices.copy_(indices)
        else:
            assert backend == "per_row", f"unknown topk backend: {backend}"
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
