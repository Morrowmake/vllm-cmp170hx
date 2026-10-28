# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen QSA canonical top-k on the real kernel, under exact ties.

Shapes follow prefill chunk 4 of the doc32768 repro (positions 12800-15999:
3200 rows, 3200-4000 visible compressed blocks, 4032 logits columns, block
top-k 512) plus a 4-row decode batch over the same width. Scores are
ReLU-floored on a coarse grid, so each row's k-th score is an exact tie that
straddles the selection boundary (at the 0.0 floor or at a positive level).

These two shapes reach two of the ``persistent_topk`` branches Qwen uses:
FilteredTopK (more than 32 rows) and histogram_2048 (at most 8,192 visible
blocks). Long-context decode batches of at most 32 rows reach the other two,
by each row's visible count: histogram_256 up to 32,768 and the multi-CTA
radix path above it. They run as 4-row decode batches at 16,384 and 40,000
columns.

``persistent_topk`` alone may return a different set for identical scores;
the count of distinct native outputs is printed, not asserted (it depends on
timing). With VLLM_SM80_TOPK_CANONICAL_TIES the set must be identical on
every call and equal to the (score descending, block index ascending) choice.
Without ties the tie fix must leave the native output byte for byte.
Needs a CUDA device and the compiled _C ops; skipped otherwise.
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm_sm80.kernels.indexer import topk_tiefix_gate as gate

COMPRESS_RATIO = 4
BLOCK_TOPK = 512
WIDTH = 4032
REPEATS = 20

# persistent_topk branch -> (logits columns, visible blocks of the 4 rows).
LONG_DECODE = {
    "histogram_256": (16384, (16384, 15000, 12289, 8193)),
    "radix": (40000, (40000, 38000, 35000, 32769)),
}


def _has_persistent_topk() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import vllm._custom_ops  # noqa: F401

        torch.ops._C.persistent_topk  # noqa: B018
    except (AttributeError, ImportError, RuntimeError):
        return False
    return True


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not _has_persistent_topk(),
        reason="needs a CUDA device and torch.ops._C.persistent_topk",
    ),
]


def _clear_gate() -> None:
    for f in (gate.use_tiefix_topk, gate.use_sorted_topk, gate.tiefix_split_rows):
        f.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_gate():
    """The gate caches its reads; drop them around each test (after the
    environment is restored)."""
    _clear_gate()
    yield
    _clear_gate()


def _switches(monkeypatch, *, canonical: bool, sort: bool) -> None:
    monkeypatch.setenv("VLLM_SM80_TOPK_CANONICAL_TIES", "1" if canonical else "0")
    monkeypatch.setenv("VLLM_SM80_TOPK_SORTED", "1" if sort else "0")
    _clear_gate()


def _tied_logits(visible: torch.Tensor, width: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = visible.shape[0]
    logits = torch.zeros(rows, width, device="cuda")
    # 256..767 positive blocks per row on 8 levels; the rest is the 0.0 floor.
    positives = torch.randint(256, 768, (rows, 1), device="cuda", generator=g)
    rank = torch.rand(rows, width, device="cuda", generator=g).argsort(dim=1)
    level = torch.randint(1, 9, (rows, width), device="cuda", generator=g)
    logits = torch.where(rank < positives, level.float() / 8, logits)
    return _mask_unwritten(logits, visible)


def _mask_unwritten(logits: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    # Never written by the score kernels: must never be selected.
    columns = torch.arange(logits.shape[1], device="cuda")
    logits.masked_fill_(columns >= visible[:, None].long(), 1e30)
    return logits.contiguous()


def _chunk4_case(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rows == 3200:
        positions = torch.arange(12800, 16000, device="cuda")
        visible = ((positions + 1) // COMPRESS_RATIO).to(torch.int32)
    else:
        visible = torch.full((rows,), 4000, dtype=torch.int32, device="cuda")
    return _tied_logits(visible, WIDTH, seed), visible


def _long_decode_visible(branch: str) -> tuple[int, torch.Tensor]:
    width, counts = LONG_DECODE[branch]
    return width, torch.tensor(counts, dtype=torch.int32, device="cuda")


def _reference(logits: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Canonical set, ascending: a stable descending sort keeps equal scores
    in index order, so its first k are (score desc, index asc)."""
    columns = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(columns >= visible[:, None].long(), float("-inf"))
    order = torch.sort(masked, dim=1, descending=True, stable=True).indices
    return order[:, :BLOCK_TOPK].sort(dim=1).values.to(torch.int32)


def _workspace() -> torch.Tensor:
    return torch.empty(
        qsa_indexer_ops._TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"
    )


def _native(logits, visible, workspace) -> torch.Tensor:
    out = torch.empty(logits.shape[0], BLOCK_TOPK, dtype=torch.int32, device="cuda")
    torch.ops._C.persistent_topk(
        logits, visible, out, workspace, BLOCK_TOPK, logits.shape[1]
    )
    return out


def _qsa_topk(logits, visible, workspace) -> torch.Tensor:
    out = torch.empty(logits.shape[0], BLOCK_TOPK, dtype=torch.int32, device="cuda")
    qsa_indexer_ops._topk(
        logits,
        visible,
        BLOCK_TOPK * COMPRESS_RATIO,
        COMPRESS_RATIO,
        out,
        workspace,
    )
    return out


def _row_sets_sorted(x: torch.Tensor) -> torch.Tensor:
    # Every row here has > k visible blocks, so no -1 fill.
    return x.sort(dim=1).values


def _time_ms(fn, *args) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    fn(*args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 10


def _check_canonical_every_call(monkeypatch, logits, visible, sort, label) -> None:
    workspace = _workspace()
    want = _reference(logits, visible)

    native_variants = set()
    for _ in range(REPEATS):
        native = _row_sets_sorted(_native(logits, visible, workspace))
        assert (native >= 0).all() and (native < visible[:, None]).all()
        native_variants.add(native.cpu().numpy().tobytes())

    _switches(monkeypatch, canonical=True, sort=sort)
    first = None
    for _ in range(REPEATS):
        got = _qsa_topk(logits, visible, workspace)
        if sort:
            assert torch.equal(got, want)
        else:
            assert torch.equal(_row_sets_sorted(got), want)
        if first is None:
            first = got.clone()
        elif sort:
            assert torch.equal(got, first)
    canonical_ms = _time_ms(_qsa_topk, logits, visible, workspace)
    _switches(monkeypatch, canonical=False, sort=sort)
    stock_ms = _time_ms(_qsa_topk, logits, visible, workspace)
    print(
        f"QSA_TIE {label} sort={int(sort)} k={BLOCK_TOPK} width={logits.shape[1]} "
        f"repeats={REPEATS} native_set_variants={len(native_variants)} "
        f"canonical_set_variants=1 stock_ms={stock_ms:.3f} "
        f"canonical_ms={canonical_ms:.3f}"
    )


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("rows", [3200, 4])
def test_canonical_set_is_identical_every_call(monkeypatch, rows, sort):
    logits, visible = _chunk4_case(rows, seed=rows)
    _check_canonical_every_call(monkeypatch, logits, visible, sort, f"rows={rows}")


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("branch", sorted(LONG_DECODE))
def test_long_decode_canonical_set_is_identical_every_call(monkeypatch, branch, sort):
    width, visible = _long_decode_visible(branch)
    logits = _tied_logits(visible, width, seed=width)
    _check_canonical_every_call(
        monkeypatch, logits, visible, sort, f"rows=4 branch={branch}"
    )


@pytest.mark.parametrize("branch", sorted(LONG_DECODE))
def test_long_decode_without_ties_is_a_no_op(monkeypatch, branch):
    """Distinct scores: the native output is already the canonical set, and
    the tie fix keeps every byte of it (sort off) or only sorts it (sort on)."""
    width, visible = _long_decode_visible(branch)
    g = torch.Generator(device="cuda").manual_seed(width + 1)
    # A permutation of 0..width-1 per row: exact in fp32, no two scores equal.
    distinct = torch.rand(4, width, device="cuda", generator=g).argsort(dim=1)
    logits = _mask_unwritten(distinct.float(), visible)
    native = _native(logits, visible, _workspace())
    assert torch.equal(_row_sets_sorted(native), _reference(logits, visible))

    for sort in (False, True):
        _switches(monkeypatch, canonical=True, sort=sort)
        got = native.clone()
        gate.canonical_topk_(logits, got, indexer="qsa", row_ends=visible)
        assert torch.equal(got, _row_sets_sorted(native) if sort else native)
