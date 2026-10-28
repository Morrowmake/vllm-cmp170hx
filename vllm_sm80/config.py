# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's engine settings: ``VllmConfig.additional_config["sm80"]``.

vLLM keeps platform settings in ``additional_config`` (hashed into the
compile cache key, carried to every engine process with the config). The
fork's served flags land there (``vllm_sm80.engine_args``):

- ``prefill_chunk_with_decodes``, ``max_num_partial_prefills``: fair chunked
  prefill (``vllm_sm80.sched``).

This module imports nothing from vLLM: config construction calls it.
"""

from typing import Any

SECTION = "sm80"

# Fair chunked prefill knobs, both off (0) by default.
FAIR_PREFILL_KEYS = ("prefill_chunk_with_decodes", "max_num_partial_prefills")


def section(vllm_config: Any) -> dict[str, Any]:
    """The fork's settings of ``vllm_config`` (empty when none are set)."""
    additional = vllm_config.additional_config
    if not isinstance(additional, dict):
        return {}
    return additional.get(SECTION) or {}


def with_settings(additional_config: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``additional_config`` whose fork section also holds
    ``settings``; a key the section already sets to another value is an
    error (the same setting given twice)."""
    if not isinstance(additional_config, dict):
        raise ValueError(
            "the sm80 settings need additional_config to be a dict, got "
            f"{type(additional_config).__name__}"
        )
    current = dict(additional_config.get(SECTION) or {})
    for key, value in settings.items():
        if key in current and current[key] != value:
            raise ValueError(
                f"{key} is set twice: {value!r} and additional_config"
                f"[{SECTION!r}][{key!r}] = {current[key]!r}"
            )
        current[key] = value
    return {**additional_config, SECTION: current}


def fair_prefill(vllm_config: Any) -> tuple[int, int]:
    """``(prefill_chunk_with_decodes, max_num_partial_prefills)``."""
    settings = section(vllm_config)
    return tuple(int(settings.get(key, 0)) for key in FAIR_PREFILL_KEYS)  # type: ignore[return-value]
