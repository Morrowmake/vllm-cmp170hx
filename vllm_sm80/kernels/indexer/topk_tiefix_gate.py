# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the indexer top-k determinism switches, shared by every
sparse indexer that selects with a fast top-k kernel.

``VLLM_SM80_TOPK_CANONICAL_TIES``: entries tied with the k-th score that
straddle the selection boundary go to the lowest index, so the selected set
is a function of the scores alone. ``VLLM_SM80_TOPK_SORTED``: each selected
row is put in ascending order, -1 fill last, so attention reads the set in a
fixed order. With both on, the sort runs inside the tie-fix launch, split
over (row, chunk) programs up to the topk_select/split_rows row of the sm80
tuning table, keyed by the selected width and the calling indexer
(``indexer``: "kpool" for GLM, "qsa" for Qwen), so a row tuned for one
indexer never reaches another of the same width.

The GLM kpool sparse indexer (``vllm/models/glm5next/nvidia/sparse_indexer.py``)
calls :func:`canonical_topk_` after its prefill top-k and
:func:`canonical_decode_topk_` after its decode top-k (behind upstream's
``SparseIndexerTopk``). The Qwen QSA indexer
(``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py``) calls
:func:`canonical_topk_` after its prefill and decode top-k. The kernels
(``topk_tiefix.py``) are imported only once a switch is on, so with both
unset nothing new is loaded and the indexers run upstream's path.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm_sm80 import hot_swap, tuning

logger = init_logger(__name__)


@functools.cache
def _warm() -> None:
    """Compile the kernels on the first call, which is the first indexer
    forward (the profiling run, eager), before any graph capture."""
    from vllm_sm80.kernels.indexer.topk_tiefix import warm_tiefix

    warm_tiefix()


@functools.cache
def use_sorted_topk() -> bool:
    """Whether the indexers sort their selected indices
    (VLLM_SM80_TOPK_SORTED). Read once per process; tests call cache_clear."""
    import vllm.envs as envs

    on = bool(envs.VLLM_SM80_TOPK_SORTED)
    if on:
        logger.info_once(
            "sm80 sorted top-k active: the sparse indexer's selected indices "
            "are put in ascending order, -1 fill last, before attention "
            "(VLLM_SM80_TOPK_SORTED=1; set 0 to disable)"
        )
        _warm()
    return on


@functools.cache
def use_tiefix_topk() -> bool:
    """Whether the indexers repair boundary ties of the fast top-k kernels
    (VLLM_SM80_TOPK_CANONICAL_TIES). Read once per process; tests call
    cache_clear."""
    import vllm.envs as envs

    on = bool(envs.VLLM_SM80_TOPK_CANONICAL_TIES)
    if on:
        logger.info_once(
            "sm80 canonical top-k ties active: the sparse indexer's selected "
            "set breaks ties at the k-th score by lowest index "
            "(VLLM_SM80_TOPK_CANONICAL_TIES=1; set 0 to disable)"
        )
        _warm()
    return on


@functools.cache
def tiefix_split_rows(top_k: int, indexer: str) -> int:
    """Row bound of the split tie fix + sort at ``top_k`` selected slots per
    row of ``indexer`` (the topk_select/split_rows row of the sm80 tuning
    table; 0 = off, the default). Only meaningful with the tie fix and the
    sort both on. Read once per process, width and indexer; tests call
    cache_clear."""
    rows = _split_rows_row(top_k, indexer)
    if rows and not (use_tiefix_topk() and use_sorted_topk()):
        logger.info_once(
            "sm80 split tie-fix scan NOT active: the topk_select/split_rows "
            "row needs VLLM_SM80_TOPK_CANONICAL_TIES=1 and VLLM_SM80_TOPK_SORTED=1"
        )
        return 0
    if rows:
        logger.info_once(
            "sm80 split tie-fix scan active: batches of at most %d rows over "
            "wide logits run the tie fix + sort over (row, chunk) programs "
            "(topk_select/split_rows = %d at top_k %d, indexer %s in the sm80 "
            "tuning table)",
            rows,
            rows,
            top_k,
            indexer,
        )
    return rows


def _split_rows_row(top_k: int, indexer: str) -> int:
    """The topk_select/split_rows table value at (``top_k``, ``indexer``)."""
    return max(0, tuning.lookup(
        "topk_select", "split_rows", {"top_k": top_k, "indexer": indexer}, 0
    ))


def tiefix_topk_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    indexer: str,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    sort: bool = False,
    split_rows: int | None = None,
) -> torch.Tensor:
    """Resolve boundary ties of an exact top-k to the lowest indices, in
    place; with ``sort`` also sort each row (VLLM_SM80_TOPK_SORTED) in the
    same launch (two launches in all on the split path, ``use_split``, up to
    ``split_rows`` rows; None: :func:`tiefix_split_rows` of ``indexer``).
    See ``topk_tiefix.py``; imported lazily so that with the flag unset
    nothing new is loaded."""
    from vllm_sm80.kernels.indexer.topk_tiefix import (
        topk_tiefix_,
        use_split,
    )

    if split_rows is None:
        split_rows = tiefix_split_rows(int(topk_indices.shape[1]), indexer)
    split = sort and use_split(
        topk_indices.shape[0],
        logits.shape[1],
        split_rows,
    )
    return topk_tiefix_(
        logits,
        topk_indices,
        row_ends=row_ends,
        row_starts=row_starts,
        relative=relative,
        sort=sort,
        split=split,
    )


# The hot pairs of this site (vllm_sm80.hot_swap): a server lists at most one.
_HOT_PAIRS = ("topk_canonical_ties", "topk_sorted")


def _hot_pair() -> str | None:
    for name in _HOT_PAIRS:
        if hot_swap.listed(name):
            return name
    return None


def _canonical(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    indexer: str,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None,
    relative: bool,
    ties: bool,
    sort: bool,
    split_rows: int | None = None,
) -> torch.Tensor:
    if ties:
        tiefix_topk_(
            logits,
            topk_indices,
            indexer=indexer,
            row_ends=row_ends,
            row_starts=row_starts,
            relative=relative,
            sort=sort,
            split_rows=split_rows,
        )
    elif sort:
        from vllm_sm80.kernels.indexer.topk_tiefix import sort_topk_rows_

        sort_topk_rows_(topk_indices)
    return topk_indices


def canonical_topk_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    indexer: str,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
) -> None:
    """After an exact top-k wrote ``topk_indices`` from ``logits``: with
    VLLM_SM80_TOPK_CANONICAL_TIES make its set canonical on boundary ties,
    sorting each row in the same launch when VLLM_SM80_TOPK_SORTED is on too;
    with the sort alone, sort each row and keep the set. Both unset: nothing
    runs. ``row_ends``/``row_starts``/``relative`` describe each row's window
    as the top-k kernel saw it (see ``topk_tiefix_``); ``indexer`` names the
    calling indexer ("kpool", "qsa") for the split row bound's table key.

    With a hot pair of ``_HOT_PAIRS`` listed, this is its pair site: the
    incumbent runs the switches as booted, the candidate with the pair's
    switch on, both in place on ``topk_indices``. With both switches on, the
    candidate's split row bound is the topk_select/split_rows row, as served."""
    ties, sort = use_tiefix_topk(), use_sorted_topk()
    hot = _hot_pair()
    if hot is None:
        _canonical(
            logits, topk_indices, indexer, row_ends, row_starts, relative, ties, sort
        )
        return
    cand = (True, sort) if hot == "topk_canonical_ties" else (ties, True)
    split_rows = (
        _split_rows_row(int(topk_indices.shape[1]), indexer)
        if all(cand) else 0
    )
    hot_swap.pair(
        hot,
        lambda: _canonical(
            logits, topk_indices, indexer, row_ends, row_starts, relative, ties, sort
        ),
        lambda out: _canonical(
            logits, topk_indices, indexer, row_ends, row_starts, relative, *cand,
            split_rows,
        ),
    )


def canonical_decode_topk_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    *,
    indexer: str,
) -> None:
    """:func:`canonical_topk_` after the decode top-k that
    ``SparseIndexerTopk`` (``get_indexer_topk``) just wrote into
    ``topk_indices``, over the same per-row ends as its kernels
    (``SparseIndexerTopk._row_ends`` of the same arguments)."""
    if use_tiefix_topk() or use_sorted_topk() or _hot_pair() is not None:
        canonical_topk_(
            logits,
            topk_indices,
            indexer=indexer,
            row_ends=SparseIndexerTopk._row_ends(seq_lens, next_n, logits.shape[0]),
        )
