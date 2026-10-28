# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host gate of the ``VLLM_AMPERE_DECODE_IDX_GLUE`` decode folds.

The folds live in ``vllm_sm80/models/glm5next/ops/idx_glue.py`` (weights,
glue, fwht), ``vllm_sm80/kernels/attention/triton_mla_remap_cache.py`` (cache) and
``vllm_sm80/kernels/moe/sum.py`` (moesum). The shared MLA attention layer and
MoE runner ask this gate, so it is kept here and imports none of them: with the
switch unset no fold module is imported and every call site keeps its code
path.
"""

from vllm.logger import init_logger
from vllm_sm80.kernels._platform import _is_sm80

__all__ = ["use_idx_glue"]

logger = init_logger(__name__)

PARTS = ("weights", "glue", "fwht", "cache", "moesum")

_PARTS_CACHE: tuple[str, frozenset] | None = None
_BANNER_DONE = False


def _parts() -> frozenset:
    global _PARTS_CACHE
    from vllm import envs

    raw = envs.VLLM_GLM5_DECODE_IDX_GLUE_PARTS or ""
    if _PARTS_CACHE is None or _PARTS_CACHE[0] != raw:
        parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
        unknown = parts - set(PARTS)
        if unknown:
            logger.warning(
                "VLLM_GLM5_DECODE_IDX_GLUE_PARTS: ignoring unknown parts %s",
                sorted(unknown),
            )
        _PARTS_CACHE = (raw, parts & frozenset(PARTS))
    return _PARTS_CACHE[1]


def idx_glue_part(part: str) -> bool:
    """True when the switch is on, ``part`` is selected and the part is sm_80."""
    global _BANNER_DONE
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_IDX_GLUE:
        return False
    parts = _parts()
    if part not in parts:
        return False
    if not _is_sm80():
        return False
    if not _BANNER_DONE:
        _BANNER_DONE = True
        logger.info(
            "VLLM_AMPERE_DECODE_IDX_GLUE=1: sm_80 MLA indexer / MoE tail folds "
            "on, parts=%s",
            ",".join(p for p in PARTS if p in parts),
        )
    return True


def use_idx_glue(part: str) -> bool:
    """Gate for one fold of ``VLLM_AMPERE_DECODE_IDX_GLUE``.

    Reads ``VLLM_AMPERE_DECODE_IDX_GLUE`` first, so with the switch unset it
    costs one bool read and every call site keeps its code path.
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_IDX_GLUE:
        return False
    return idx_glue_part(part)
