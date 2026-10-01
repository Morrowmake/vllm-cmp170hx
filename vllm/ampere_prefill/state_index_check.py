# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime range checks for KDA / Mamba state indices (debug only).

``VLLM_GLM5_STATE_INDEX_CHECK=1`` checks, on the host and before the kernels
that use them run:

* every step's GDN/KDA attention metadata (also for steps replayed from CUDA
  graphs, whose kernels run without Python): the state-pool rows the conv / recurrent kernels
  will read or write (non-spec rows, spec rows and the accepted-token column
  they start from) lie in ``[0, pool)``, the initial-state rows cover the
  prefill sequences, and the ``cu_seqlens`` describe at most the tokens that
  are there;
* every block-table gather: each batch row names a request slot;
* the align-mode state pre-copy (before the forward) and post-copy (after
  sampling): for every request that will copy, the block-table columns lie in
  the row and the block ids they hold lie in ``[0, num_blocks)``; a copy from
  or into the null block is reported as a warning.

A violation raises ``StateIndexError`` with the step's numbers instead of
letting a kernel address memory outside the pool. Every check synchronizes
with the device, so this is for diagnosis, never for serving.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG = "VLLM_GLM5_STATE_INDEX_CHECK"
BANNER = (
    "GLM5 state index check: every KDA/Mamba state index is range-checked on "
    "the host before use (debug; synchronizes every step); "
    "VLLM_GLM5_STATE_INDEX_CHECK=1"
)
NULL_BLOCK_ID = 0


class StateIndexError(RuntimeError):
    pass


def enabled() -> bool:
    from vllm import envs

    on = bool(envs.VLLM_GLM5_STATE_INDEX_CHECK)
    if on:
        logger.warning_once(BANNER)
    return on


def _host(t: torch.Tensor | None) -> torch.Tensor | None:
    return None if t is None else t.detach().to("cpu", torch.int64)


def _range(problems: list[str], name: str, v: torch.Tensor | None, hi: int) -> None:
    if v is None or v.numel() == 0:
        return
    lo_v, hi_v = int(v.min()), int(v.max())
    if lo_v < 0 or hi_v >= hi:
        bad = ((v < 0) | (v >= hi)).nonzero().tolist()[:8]
        problems.append(
            f"{name}: values {lo_v}..{hi_v} outside [0, {hi}) at {bad}"
        )


def _cu(problems: list[str], name: str, cu: torch.Tensor | None, num_tokens: int):
    if cu is None or cu.numel() == 0:
        return
    if int(cu[0]) != 0:
        problems.append(f"{name}[0] = {int(cu[0])}")
    if cu.numel() > 1 and bool((cu[1:] < cu[:-1]).any()):
        problems.append(f"{name} decreases: {cu.tolist()[:40]}")
    if int(cu[-1]) > num_tokens:
        problems.append(f"{name}[-1] = {int(cu[-1])} > {num_tokens} tokens")


def _state_pools(vllm_config) -> dict[str, int]:
    """Rows of each mamba-family layer's state pool (conv and recurrent)."""
    pools: dict[str, int] = {}
    ctx = vllm_config.compilation_config.static_forward_context
    for name, layer in ctx.items():
        kv = getattr(layer, "kv_cache", None)
        if isinstance(kv, (list, tuple)) and kv and all(
            isinstance(t, torch.Tensor) for t in kv
        ) and len(kv) == 2 and kv[0].dim() >= 2:
            rows = {int(t.shape[0]) for t in kv}
            pools[name] = min(rows)
    return pools


_POOLS: dict[int, dict[str, int]] = {}


def check_attn_metadata(attn_metadata: dict, vllm_config) -> None:
    """Range-check every GDN/KDA metadata object of one step (once per object,
    against the smallest pool of the layers that share it)."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    key = id(vllm_config)
    if key not in _POOLS:
        _POOLS[key] = _state_pools(vllm_config)
    pools = _POOLS[key]
    by_md: dict[int, tuple] = {}
    for name, md in attn_metadata.items():
        if isinstance(md, GDNAttentionMetadata) and name not in pools:
            # Never pass vacuously: a KDA layer whose pool is unknown is a failure.
            raise StateIndexError(f"no state pool found for KDA layer {name}")
        if isinstance(md, GDNAttentionMetadata):
            prev = by_md.get(id(md))
            pool = pools[name] if prev is None else min(prev[2], pools[name])
            by_md[id(md)] = (name, md, pool)
    for name, md, pool in by_md.values():
        check_kda_metadata(name, md, pool, int(md.num_actual_tokens))


def check_kda_metadata(layer: str, md, pool: int, num_tokens: int) -> None:
    """Range-check the state indices the KDA kernels will use (module doc)."""
    problems: list[str] = []
    ns_idx = _host(md.non_spec_state_indices_tensor)
    ns_cu = _host(md.non_spec_query_start_loc)
    has_init = (None if md.has_initial_state is None
                else md.has_initial_state.detach().to("cpu"))
    if md.num_prefills > 0:
        _range(problems, "non-spec state index", ns_idx, pool)
        _cu(problems, "non_spec_query_start_loc", ns_cu, num_tokens)
        if ns_idx is not None and ns_cu is not None and ns_idx.numel() < ns_cu.numel() - 1:
            problems.append(
                f"{ns_idx.numel()} non-spec state rows for {ns_cu.numel() - 1} sequences")
        if has_init is not None and ns_idx is not None and has_init.numel() != ns_idx.numel():
            problems.append(
                f"has_initial_state {has_init.numel()} rows, state indices {ns_idx.numel()}")
    elif md.num_decodes > 0:
        _range(problems, "decode state index",
               None if ns_idx is None else ns_idx[: md.num_decodes], pool)
    n_spec = int(md.num_spec_decodes or 0)
    # VLLM_GLM5_KDA_RECOVER: one state per request. Only column 0 of the spec
    # state table is filled and read (the fused verify path leaves the other
    # columns of its persistent buffer unwritten), and the verify reads
    # num_accepted = 1; the query still spans up to num_spec + 1 tokens.
    commit = getattr(md, "recover_commit", None)
    if n_spec > 0 and md.spec_state_indices_tensor is not None:
        sp_dev = md.spec_state_indices_tensor[:n_spec]
        width = int(sp_dev.shape[1]) if sp_dev.dim() == 2 else 1
        if commit is not None and sp_dev.dim() == 2:
            sp_dev = sp_dev[:, :1]
        sp = _host(sp_dev)
        _range(problems, "spec state index", sp, pool)
        cols = int(sp.shape[1]) if sp.dim() == 2 else 1
        acc = _host(md.num_accepted_tokens)
        if acc is not None:
            acc = acc[:n_spec]
            if acc.numel() and (int(acc.min()) < 1 or int(acc.max()) > cols):
                problems.append(
                    f"num_accepted_tokens {acc.tolist()} outside [1, {cols}]")
        sq = _host(md.spec_query_start_loc)
        if sq is not None:
            sq = sq[: n_spec + 1]
            _cu(problems, "spec_query_start_loc", sq, num_tokens)
            if sq.numel() > 1 and int((sq[1:] - sq[:-1]).max()) > width:
                problems.append(
                    f"spec query lengths {(sq[1:] - sq[:-1]).tolist()} > {width} "
                    f"token columns")
        if commit is not None:
            _check_recover_commit(problems, commit, n_spec, pool,
                                  int(md.num_prefills) + int(md.num_decodes) + n_spec)
    if problems:
        detail = (f"{layer}: pool {pool}, prefills {md.num_prefills}, decodes "
                  f"{md.num_decodes}, spec {n_spec}, tokens {num_tokens}; "
                  f"non-spec idx {None if ns_idx is None else ns_idx.tolist()[:40]}, "
                  f"cu {None if ns_cu is None else ns_cu.tolist()[:40]}")
        logger.error("state index check failed: %s | %s", "; ".join(problems), detail)
        raise StateIndexError("; ".join(problems) + " | " + detail)


def _check_recover_commit(problems: list[str], commit, n_spec: int, pool: int,
                          num_seqs: int) -> None:
    """The commit metadata of VLLM_GLM5_KDA_RECOVER (kda_recover.py): the
    state rows the commit replays from and the batch rows it reads."""
    si = _host(commit.state_indices)
    if si is not None and si.numel() != n_spec:
        problems.append(f"recover commit: {si.numel()} state rows for {n_spec} spec decodes")
    _range(problems, "recover commit state index", si, pool)
    ri = _host(commit.request_indices)
    if ri is not None:
        if ri.numel() != n_spec:
            problems.append(
                f"recover commit: {ri.numel()} request rows for {n_spec} spec decodes")
        _range(problems, "recover commit request index", ri, num_seqs)
    cq = _host(commit.query_start_loc)
    if cq is not None and cq.numel() != n_spec + 1:
        problems.append(
            f"recover commit: query_start_loc has {cq.numel()} entries "
            f"for {n_spec} spec decodes")


def check_gather_mapping(idx_mapping: torch.Tensor, max_num_reqs: int) -> None:
    """Rows gathered into the batch-order block tables must name a request
    slot: a negative slot has no row (reading row -1 reads memory before the
    table, whose contents would become state block ids)."""
    v = _host(idx_mapping)
    if v.numel() and (int(v.min()) < 0 or int(v.max()) >= max_num_reqs):
        logger.error("state index check failed (gather): request slots %s", v.tolist())
        raise StateIndexError(
            f"block-table gather with request slots outside [0, {max_num_reqs}): "
            f"{v.tolist()}")


def _full_table(bt: torch.Tensor, rows: int) -> torch.Tensor:
    """The persistent batch-order table behind a leading-rows slice (as many
    of ``rows`` as its storage holds)."""
    elems = bt.untyped_storage().nbytes() // bt.element_size() - bt.storage_offset()
    fit = (elems - (bt.shape[1] - 1) * bt.stride(1)) // bt.stride(0) + 1
    return torch.as_strided(bt, (max(0, min(rows, fit)), bt.shape[1]), bt.stride())


def check_align_copies(kind: str, block_tables: list[torch.Tensor], max_rows: int,
                       num_blocks: int, idx_mapping: torch.Tensor, src_col, dst_col,
                       bias, needs) -> None:
    """Shared range check of one align copy pass. ``src_col`` / ``dst_col`` /
    ``bias`` / ``needs`` are per batch row (host int64 / bool); rows whose
    ``idx_mapping`` is negative or whose ``needs`` is False do not copy."""
    req = _host(idx_mapping)
    problems: list[str] = []
    warnings: list[str] = []
    for g, bt in enumerate(block_tables):
        table = _host(_full_table(bt, max_rows))
        width = int(table.shape[1])
        for row in range(req.numel()):
            if int(req[row]) < 0 or not bool(needs[row]):
                continue
            s, d, b = int(src_col[row]), int(dst_col[row]), int(bias[row])
            cols = {"src": s, "src+bias": s + b, "dst": d}
            for name, c in cols.items():
                if not 0 <= c < width:
                    problems.append(f"{kind} group {g} row {row} (req slot {int(req[row])}): "
                                    f"{name} column {c} outside [0, {width})")
                    continue
                if row >= table.shape[0]:
                    problems.append(f"{kind} group {g}: row {row} beyond {table.shape[0]} rows")
                    continue
                bid = int(table[row, c])
                if not 0 <= bid < num_blocks:
                    problems.append(f"{kind} group {g} row {row} (req slot {int(req[row])}): "
                                    f"{name} column {c} holds block {bid} outside [0, {num_blocks})")
                elif bid == NULL_BLOCK_ID and name != "src":
                    warnings.append(f"{kind} group {g} row {row}: {name} column {c} is the null block")
                elif bid == NULL_BLOCK_ID:
                    warnings.append(f"{kind} group {g} row {row}: copies from the null block (column {c})")
    if warnings:
        logger.warning("state index check (%s): %s", kind, "; ".join(warnings[:16]))
    if problems:
        logger.error("state index check failed (%s): %s", kind, "; ".join(problems[:16]))
        raise StateIndexError("; ".join(problems[:16]))


def precopy_plan(src_col_slots, state_idx_slots, src_off_slots, idx_mapping):
    """Per batch row (src, dst, bias, needs) of precopy_mamba_align_fused_kernel."""
    req = _host(idx_mapping)
    src_s, dst_s, off_s = _host(src_col_slots), _host(state_idx_slots), _host(src_off_slots)
    n = req.numel()
    src = torch.full((n,), -1, dtype=torch.int64)
    dst = torch.full((n,), -1, dtype=torch.int64)
    bias = torch.zeros(n, dtype=torch.int64)
    ok = req >= 0
    src[ok], dst[ok], bias[ok] = src_s[req[ok]], dst_s[req[ok]], off_s[req[ok]]
    needs = ok & (src >= 0) & (src != dst)
    return src, dst, bias, needs


def postcopy_plan(num_accepted_slots, state_idx_slots, num_computed_slots, idx_mapping,
                  block_size: int):
    """Per batch row (src, dst, bias, needs) of postprocess_mamba_fused_kernel
    with PRECOMPUTED_NEW_COMPUTED (the V2 call)."""
    req = _host(idx_mapping)
    acc_s, st_s, nc_s = (_host(num_accepted_slots), _host(state_idx_slots),
                         _host(num_computed_slots))
    n = req.numel()
    src = torch.full((n,), -1, dtype=torch.int64)
    dst = torch.full((n,), -1, dtype=torch.int64)
    bias = torch.zeros(n, dtype=torch.int64)
    needs = torch.zeros(n, dtype=torch.bool)
    for row in range(n):
        r = int(req[row])
        if r < 0:
            continue
        acc, s, new = int(acc_s[r]), int(st_s[r]), int(nc_s[r])
        running = new - acc + 1
        aligned = (new // block_size) * block_size
        if aligned < running:
            continue
        b = aligned - running
        d = aligned // block_size - 1
        if s == d and b == 0:
            continue
        src[row], dst[row], bias[row], needs[row] = s, d, b, True
    return src, dst, bias, needs
