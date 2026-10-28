# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning rules of the CMP 170HX (sm_80) fork.

Upstream files call in from two places:

- ``kv_cache_planning._get_packed_kv_cache_groups`` passes its groups through
  :func:`fit_packed_kv_cache_groups`, which refits the groups of an MLA +
  Mamba hybrid (GLM-5.3-Flash): a DFlash drafter's sliding-window layers get
  an exact-fit block size and groups no wider than the block already is, and
  under pipeline parallelism the state groups are split so that no stage's
  share of a group widens the block.
- ``kv_cache_utils.get_max_concurrency_for_kv_cache_config`` returns
  :func:`max_concurrency_for_kv_cache_config`, which charges a spec's
  speculative scratch (``KVCacheSpec.speculative_scratch_bytes``) once per
  running request instead of once per resident request.
"""

import math
from collections import defaultdict
from dataclasses import replace

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# Packed (block-outermost) groups of an MLA + Mamba hybrid
# --------------------------------------------------------------------------


def fit_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    """Refit packed KV cache groups of an MLA + Mamba hybrid model.

    In a block-outermost layout every group draws block ids from one pool and
    a block is as wide as the widest group's pages on any worker. The packed
    grouping sizes the state groups from whole-model layer counts and keeps a
    drafter's sliding-window layers at the target block size, which on the
    CMP 170HX layouts widens the block:

    - a DFlash drafter at pipeline parallelism (its page is several MLA pages
      wide at the target block size) and
    - a pipeline stage holding more Mamba layers of one group than its block
      has room for.

    The drafter's layers get the block size at which their page equals the
    MLA page and are chunked into groups no wider than the block (see
    :func:`_fit_drafter_groups`); under pipeline parallelism each state
    group family is re-split until every stage's share fits the block (see
    :func:`_fit_state_groups`). Groups of any other model are returned
    unchanged.
    """
    if not _is_mla_mamba_hybrid(kv_cache_spec):
        return kv_cache_groups
    pp_size = _pipeline_parallel_size(vllm_config)
    if pp_size == 1 and not any(_is_drafter_group(g) for g in kv_cache_groups):
        return kv_cache_groups
    stage_of = _layer_stages(vllm_config, kv_cache_spec, pp_size)
    if stage_of is None:
        return kv_cache_groups
    anchor_bytes = _anchor_bytes(kv_cache_groups, stage_of)
    if not anchor_bytes:
        return kv_cache_groups
    groups = _fit_drafter_groups(
        vllm_config, kv_cache_spec, kv_cache_groups, anchor_bytes
    )
    if pp_size > 1:
        groups = _fit_state_groups(kv_cache_spec, groups, stage_of, anchor_bytes)
    return groups


def _pipeline_parallel_size(vllm_config: VllmConfig) -> int:
    """Pipeline stages of the config; one when it has no parallel settings.

    get_kv_cache_groups reads only the config fields its grouping needs and
    is called with partial configs, so this hook must not require more.
    """
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return 1
    return parallel_config.pipeline_parallel_size


def _is_mla_mamba_hybrid(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """MLA attention + Mamba state layers, optionally with circular-buffer
    tails and a drafter's plain sliding-window layers, and nothing else."""
    has_mamba = has_mla = False
    for spec in kv_cache_spec.values():
        if isinstance(spec, MambaSpec):
            has_mamba = True
        elif type(spec) is MLAAttentionSpec:
            has_mla = has_mla or spec.tokens_per_state == 1
        elif not (
            isinstance(spec, CircularBufferSpec) or type(spec) is SlidingWindowSpec
        ):
            return False
    return has_mamba and has_mla


def _layer_spec(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def _group_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    return {name: _layer_spec(group, name) for name in group.layer_names}


def _is_state_group(group: KVCacheGroupSpec) -> bool:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.first_spec
    return isinstance(spec, (MambaSpec, CircularBufferSpec))


def _is_drafter_group(group: KVCacheGroupSpec) -> bool:
    """Plain sliding-window layers: the drafter's, in an MLA + Mamba hybrid
    (CircularBufferSpec and SlidingWindowMLASpec are not exact matches)."""
    return bool(group.layer_names) and all(
        type(_layer_spec(group, name)) is SlidingWindowSpec
        for name in group.layer_names
    )


def _layer_stages(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec], pp_size: int
) -> dict[str, int] | None:
    """The pipeline stage of every layer; a layer past the target's layers
    (a drafter's) runs on the last stage. None if a name has no layer index."""
    if pp_size == 1:
        return dict.fromkeys(kv_cache_spec, 0)

    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index

    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    bounds = [get_pp_indices(total_layers, rank, pp_size) for rank in range(pp_size)]
    stage_of: dict[str, int] = {}
    for name in kv_cache_spec:
        try:
            index = extract_layer_index(name)
        except AssertionError:
            return None
        stage_of[name] = next(
            (rank for rank, (start, end) in enumerate(bounds) if start <= index < end),
            pp_size - 1,
        )
    return stage_of


def _anchor_bytes(
    kv_cache_groups: list[KVCacheGroupSpec], stage_of: dict[str, int]
) -> int:
    """Bytes per block the target's attention groups already need: the widest
    attention group's pages on any one stage."""
    widths: defaultdict[tuple[int, int], int] = defaultdict(int)
    for index, group in enumerate(kv_cache_groups):
        if _is_state_group(group) or _is_drafter_group(group):
            continue
        for name in group.layer_names:
            widths[index, stage_of[name]] += _layer_spec(group, name).page_size_bytes
    return max(widths.values(), default=0)


def _drafter_group(
    layer_names: list[str], specs: dict[str, KVCacheSpec], is_eagle: bool
) -> KVCacheGroupSpec:
    """A drafter group with one spec when its layers share it.

    The drafter runs on the last pipeline stage only. The per-worker
    projection of the groups keeps a UniformTypeKVCacheSpecs that has none of
    the worker's layers whole, so the other stages would get cache tensors
    for the drafter's layers that no group of theirs owns. A plain spec
    projects to an empty group with no tensors.
    """
    spec = specs[layer_names[0]]
    if any(specs[name] != spec for name in layer_names):
        spec = UniformTypeKVCacheSpecs.from_specs({n: specs[n] for n in layer_names})
        assert spec is not None
    return KVCacheGroupSpec(layer_names, spec, is_eagle_group=is_eagle)


def _fit_drafter_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    anchor_bytes: int,
) -> list[KVCacheGroupSpec]:
    """Exact-fit groups for a DFlash-style drafter's sliding-window layers.

    The drafter's block size is chosen so its page is byte-for-byte the MLA
    page, and its layers are split into near-equal groups (in layer order) of
    at most ``anchor_bytes // mla_page`` layers, so a drafter group is never
    wider than the block the target already needs and drafting costs no
    bytes per block; each extra group costs its sliding window's worth of
    block ids per request. The drafter groups are flagged as eagle groups
    when the speculative config drops volatile trailing blocks.

    An unsupported geometry keeps the packed groups (correct, but the block
    may widen several-fold) and is logged.
    """
    draft_indices = [
        index
        for index, group in enumerate(kv_cache_groups)
        if _is_drafter_group(group)
    ]
    spec_config = vllm_config.speculative_config
    if not draft_indices or spec_config is None or not spec_config.use_eagle():
        return kv_cache_groups
    is_eagle = spec_config.use_eagle_block_drop()
    draft_layer_set = {
        name for index in draft_indices for name in kv_cache_groups[index].layer_names
    }
    draft_specs = {
        name: spec for name, spec in kv_cache_spec.items() if name in draft_layer_set
    }
    mla_specs = [
        spec
        for spec in kv_cache_spec.values()
        if type(spec) is MLAAttentionSpec and spec.tokens_per_state == 1
    ]
    fitted = _exact_fit_drafter_specs(draft_specs, mla_specs)
    if fitted is None:
        for index in draft_indices:
            group = kv_cache_groups[index]
            kv_cache_groups[index] = _drafter_group(
                group.layer_names,
                {name: _layer_spec(group, name) for name in group.layer_names},
                is_eagle,
            )
        return kv_cache_groups

    mla_page = mla_specs[0].page_size_bytes
    names = list(fitted)
    max_per_group = max(anchor_bytes // mla_page, 1)
    num_groups = cdiv(len(names), max_per_group)
    per_group = cdiv(len(names), num_groups)
    chunks = [names[i : i + per_group] for i in range(0, len(names), per_group)]
    draft_groups = [_drafter_group(chunk, fitted, is_eagle) for chunk in chunks]
    any_draft = next(iter(draft_specs.values()))
    logger.info(
        "DFlash drafter KV: %d sliding-window layers in %d group(s) of %s "
        "(block_size %d -> %d, page %d B = the MLA page; at most %d per group, "
        "no extra bytes per block)",
        len(names),
        len(chunks),
        [len(chunk) for chunk in chunks],
        any_draft.block_size,
        next(iter(fitted.values())).block_size,
        mla_page,
        max_per_group,
    )
    # The drafter groups take the place of the first packed drafter group.
    first = draft_indices[0]
    kept = [
        group
        for index, group in enumerate(kv_cache_groups)
        if index not in draft_indices
    ]
    return kept[:first] + draft_groups + kept[first:]


def _exact_fit_drafter_specs(
    draft_specs: dict[str, KVCacheSpec], mla_specs: list[MLAAttentionSpec]
) -> dict[str, KVCacheSpec] | None:
    """The drafter's specs at the block size whose page equals the MLA page,
    or None (logged) if the geometry does not allow it."""
    mla_pages = {spec.page_size_bytes for spec in mla_specs}
    mla_blocks = {spec.block_size for spec in mla_specs}
    if len(mla_pages) != 1 or len(mla_blocks) != 1:
        logger.warning(
            "DFlash drafter KV: MLA layers have different pages or block sizes; "
            "the drafter keeps its own block size (KV capacity may drop)"
        )
        return None
    mla_page = mla_pages.pop()
    mla_block = mla_blocks.pop()
    any_draft = next(iter(draft_specs.values()))
    if not all(spec == any_draft for spec in draft_specs.values()):
        logger.warning(
            "DFlash drafter KV: drafter layers have different specs; the "
            "drafter keeps its own block size (KV capacity may drop)"
        )
        return None
    if getattr(any_draft, "page_size_padded", None) is not None:
        logger.warning(
            "DFlash drafter KV: the drafter page is padded; the drafter keeps "
            "its own block size (KV capacity may drop)"
        )
        return None
    bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
    if not bytes_per_token or mla_page % bytes_per_token:
        logger.warning(
            "DFlash drafter KV: drafter %d B/token does not divide the MLA page "
            "(%d B); the drafter keeps its own block size (KV capacity may drop)",
            bytes_per_token,
            mla_page,
        )
        return None
    fit_block = mla_page // bytes_per_token
    mla_bytes_per_token = mla_page // mla_block
    if fit_block % 64 or (fit_block % mla_block and mla_block % fit_block):
        # A 64-divisible block that divides or is divided by the MLA block
        # keeps the scheduler's block-size LCM at max(mla_block, fit_block).
        logger.warning(
            "DFlash drafter KV: the exact-fit drafter block (%d tokens for the "
            "%d-token MLA block) is not a multiple of 64 or does not divide the "
            "MLA block; the drafter keeps its own block size and KV capacity "
            "may drop several-fold. Choose --block-size so that "
            "block_size * %d / %d is a multiple of 64 (e.g. round the attention "
            "block size up to a multiple of %d).",
            fit_block,
            mla_block,
            mla_bytes_per_token,
            bytes_per_token,
            64 * bytes_per_token // math.gcd(64 * bytes_per_token, mla_bytes_per_token),
        )
        return None
    fitted = {
        name: replace(spec, block_size=fit_block) for name, spec in draft_specs.items()
    }
    assert all(spec.page_size_bytes == mla_page for spec in fitted.values())
    return fitted


def _fit_state_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    stage_of: dict[str, int],
    anchor_bytes: int,
) -> list[KVCacheGroupSpec]:
    """Split each state group family until no stage's share widens the block.

    The packed grouping caps a state group at the states one block holds,
    counted over the whole model. With pipeline parallelism each worker holds
    only its stage's layers of a group, but a stage holding more of a group's
    layers than ``anchor_bytes`` has room for widens every block of that
    worker, and the smallest worker pool sets num_blocks for all. A family
    whose groups overflow on some stage is re-split round-robin in layer
    order (as the packed grouping splits it) into the fewest groups that fit
    on every stage.
    """
    families: list[list[int]] = []
    for index, group in enumerate(kv_cache_groups):
        if not _is_state_group(group):
            continue
        for family in families:
            merged: dict[str, KVCacheSpec] = {}
            for member in family:
                merged.update(_group_specs(kv_cache_groups[member]))
            merged.update(_group_specs(group))
            if UniformTypeKVCacheSpecs.is_uniform_type(merged):
                family.append(index)
                break
        else:
            families.append([index])

    def fits(chunks: list[list[str]], per_block: int) -> bool:
        for chunk in chunks:
            per_stage: defaultdict[int, int] = defaultdict(int)
            for name in chunk:
                per_stage[stage_of[name]] += 1
            if max(per_stage.values(), default=0) > per_block:
                return False
        return True

    replacements: dict[int, list[KVCacheGroupSpec]] = {}
    for family in families:
        members = [kv_cache_groups[index] for index in family]
        specs: dict[str, KVCacheSpec] = {}
        for group in members:
            specs.update(_group_specs(group))
        page = max(spec.page_size_bytes for spec in specs.values())
        per_block = max(anchor_bytes // page, 1)
        if fits([group.layer_names for group in members], per_block):
            continue
        names = [name for name in kv_cache_spec if name in specs]
        for num_groups in range(len(family) + 1, len(names) + 1):
            chunks = [names[i::num_groups] for i in range(num_groups)]
            if fits(chunks, per_block):
                break
        new_groups: list[KVCacheGroupSpec] = []
        for chunk in chunks:
            uniform = UniformTypeKVCacheSpecs.from_specs({n: specs[n] for n in chunk})
            assert uniform is not None
            new_groups.append(
                KVCacheGroupSpec(
                    chunk,
                    uniform,
                    is_eagle_group=any(group.is_eagle_group for group in members),
                )
            )
        logger.info(
            "Pipeline-parallel KV: %d %s layers split into %d groups (was %d) "
            "so that no stage holds more than %d of a group's states per block",
            len(names),
            type(next(iter(specs.values()))).__name__,
            len(new_groups),
            len(family),
            per_block,
        )
        replacements[family[0]] = new_groups
        for index in family[1:]:
            replacements[index] = []

    if not replacements:
        return kv_cache_groups
    result: list[KVCacheGroupSpec] = []
    for index, group in enumerate(kv_cache_groups):
        result.extend(replacements.get(index, [group]))
    return result


# --------------------------------------------------------------------------
# Capacity and scheduler config
# --------------------------------------------------------------------------


def max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """How many max_model_len requests the KV cache pools hold.

    Each group charges cdiv(per-request bytes, page bytes) blocks per request
    out of its pool (device groups share one pool, host groups another); the
    smaller pool limit applies.

    Speculative scratch (`KVCacheSpec.speculative_scratch_bytes`: the extra
    Mamba state pages a linear-attention model needs while drafting, and Mamba
    in-flight states) is part of that per-request footprint, but only a
    *running* request holds it, so the engine never needs more than
    `max_num_seqs` copies of it no matter how many requests the pool can hold;
    see `pool_concurrency_limit`.
    """
    num_blocks_per_request = 0
    host_blocks_per_request = 0
    scratch_blocks_per_request = 0
    host_scratch_blocks_per_request = 0
    plan: list[str] = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        required = cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
        scratch = cdiv(
            spec.speculative_scratch_bytes(vllm_config), spec.page_size_bytes
        )
        # The scratch is a subset of the per-request footprint; never let a
        # spec claim more scratch than it claims memory.
        scratch = min(scratch, required)
        if envs.VLLM_KV_PLAN_LOG:
            layer_spec = (
                _layer_spec(group, group.layer_names[0]) if group.layer_names else spec
            )
            plan.append(
                f"[{len(plan)}] {type(layer_spec).__name__} "
                f"layers={len(group.layer_names)} "
                f"block={spec.block_size} page={spec.page_size_bytes} "
                f"charged={required} scratch={scratch}"
                + (" eagle" if group.is_eagle_group else "")
                + (" host" if group.host_resident else "")
            )
        if group.host_resident:
            host_blocks_per_request += required
            host_scratch_blocks_per_request += scratch
        else:
            num_blocks_per_request += required
            scratch_blocks_per_request += scratch
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    if envs.VLLM_KV_MAMBA_INFLIGHT_STATES:
        _log_mamba_inflight_states(vllm_config, kv_cache_config)
    limits = [
        pool_concurrency_limit(
            kv_cache_config.num_blocks,
            num_blocks_per_request,
            scratch_blocks_per_request,
            max_num_seqs,
        )
    ]
    if host_blocks_per_request:
        assert kv_cache_config.hisparse_host_num_blocks is not None
        limits.append(
            pool_concurrency_limit(
                kv_cache_config.hisparse_host_num_blocks,
                host_blocks_per_request,
                host_scratch_blocks_per_request,
                max_num_seqs,
            )
        )
    concurrency = min(limits)
    if envs.VLLM_KV_PLAN_LOG:
        logger.info_once(
            "KV cache plan (VLLM_KV_PLAN_LOG): num_blocks=%d, %d blocks per "
            "request of which %d scratch, max_num_seqs=%d, concurrency %.4f; %s",
            kv_cache_config.num_blocks,
            num_blocks_per_request,
            scratch_blocks_per_request,
            max_num_seqs,
            concurrency,
            "; ".join(plan),
        )
    return concurrency


def _log_mamba_inflight_states(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """One line saying whether VLLM_KV_MAMBA_INFLIGHT_STATES is live."""
    per_group: list[int] = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, MambaSpec):
            per_group.append(spec.inflight_state_blocks(vllm_config))
    if not any(per_group):
        logger.info_once(
            "VLLM_KV_MAMBA_INFLIGHT_STATES is set but no align-mode Mamba "
            "group holds in-flight states (max_concurrent_batches=%d, "
            "mamba_cache_mode=%s); the KV reservation is unchanged.",
            vllm_config.max_concurrent_batches,
            vllm_config.cache_config.mamba_cache_mode,
        )
        return
    logger.info_once(
        "Mamba in-flight states: %d extra state blocks per request (%d per "
        "group x %d align-mode groups) reserved for prefill chunks in flight "
        "(max_concurrent_batches=%d, max_in_flight_tokens=%d), charged %s "
        "(VLLM_KV_MAMBA_INFLIGHT_STATES=1).",
        sum(per_group),
        max(per_group),
        len(per_group),
        vllm_config.max_concurrent_batches,
        vllm_config.max_in_flight_tokens,
        "per concurrency slot (KV connector)"
        if vllm_config.kv_transfer_config is not None
        else "once per running request",
    )


def pool_concurrency_limit(
    num_blocks: int,
    blocks_per_request: int,
    scratch_blocks_per_request: int,
    max_num_seqs: int,
) -> float:
    """
    How many max_model_len requests one block pool holds.

    `blocks_per_request` includes `scratch_blocks_per_request`, the part of
    the footprint only a running request holds. At most `max_num_seqs`
    requests run at a time, so the pool sets aside
    `max_num_seqs * scratch_blocks_per_request` blocks once and divides what
    is left by the resident (scratch-free) per-request cost.

    Below `max_num_seqs` every resident request may also be running, so the
    whole footprint is charged per request -- which is the plain
    `num_blocks / blocks_per_request` formula, and also what happens whenever
    there is no scratch at all.
    """
    resident_blocks_per_request = blocks_per_request - scratch_blocks_per_request
    if scratch_blocks_per_request <= 0 or resident_blocks_per_request <= 0:
        return num_blocks / blocks_per_request
    all_running = num_blocks / blocks_per_request
    if all_running <= max_num_seqs:
        # Fewer resident requests than running slots: nothing to save, and the
        # two branches agree exactly at all_running == max_num_seqs.
        return all_running
    usable_blocks = num_blocks - max_num_seqs * scratch_blocks_per_request
    return usable_blocks / resident_blocks_per_request
