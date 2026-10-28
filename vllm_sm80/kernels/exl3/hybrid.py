# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 checkpoints that mix EXL3 with another quantization per module class.

A hybrid checkpoint keeps some modules as EXL3 matrices and takes the others
from a source checkpoint in that checkpoint's own format. Its
``quantization_config`` is an EXL3 one with two more keys::

    "quant_method": "exl3",
    "module_quantization": {
        "rules": [{"match": REGEX, "method": METHOD}, ...],
        "default": METHOD                      # optional
    },
    "non_routed_quantization": {"quant_method": NAME, ...}

``non_routed_quantization`` is the second method's own config, built through
vLLM's registry (any registered method). Each module's checkpoint name (the
EXL3 layout's name, before ``.weight``) is matched against the rules in order
with ``re.search``; the first match gives its method: ``"exl3"`` (the module's
EXL3 matrices), ``"unquantized"`` (a plain float weight) or the second
method's NAME. A module no rule matches takes ``default``, or is refused
when there is none. A fused projection's members must all resolve to the
same method.

The checkpoint's ``model.safetensors.index.json`` selects its tensors: only
the tensors it names, each from the file it names, are checkpoint tensors
(``Exl3Manifest(select_by_index=True)``), so a directory of whole shards from
both checkpoints presents exactly one copy of each tensor.

Honoured only with ``VLLM_EXL3_HYBRID=1``; without it such a config is
refused, and EXL3 checkpoints without these keys load as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re

from vllm import envs

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    from vllm_sm80.kernels.exl3.utils import Exl3Manifest

EXL3 = "exl3"
UNQUANTIZED = "unquantized"
HYBRID_KEYS = ("module_quantization", "non_routed_quantization")
# Weight dtypes of a plain (unquantized) linear.
FLOAT_DTYPES = ("BF16", "F16", "F32")


@dataclass(frozen=True)
class HybridSpec:
    """The per-module method rules of a hybrid checkpoint."""

    rules: tuple[tuple[re.Pattern, str], ...]
    default: str | None
    secondary: QuantizationConfig
    secondary_name: str

    def method(self, source: str) -> str:
        """The method of the checkpoint module ``source``."""
        for pattern, method in self.rules:
            if pattern.search(source):
                return method
        if self.default is None:
            raise ValueError(
                f"No module_quantization rule matches {source} and the config "
                "has no default"
            )
        return self.default

    def method_for(self, sources: list[str]) -> str:
        """The one method of a module's checkpoint members."""
        methods = {source: self.method(source) for source in sources}
        if len(set(methods.values())) != 1:
            raise ValueError(f"Members of one module take different methods: {methods}")
        return next(iter(methods.values()))


def parse_hybrid(config: dict[str, Any]) -> HybridSpec | None:
    """The hybrid rules of an EXL3 ``quantization_config``, or None."""
    present = [key for key in HYBRID_KEYS if key in config]
    if not present:
        return None
    if not envs.VLLM_EXL3_HYBRID:
        raise ValueError(
            f"This EXL3 quantization_config mixes methods per module ({present}); "
            "set VLLM_EXL3_HYBRID=1 to load it"
        )
    if len(present) != len(HYBRID_KEYS):
        raise ValueError(f"A hybrid EXL3 config needs both {HYBRID_KEYS}")
    from vllm.model_executor.layers.quantization import get_quantization_config

    sub = dict(config["non_routed_quantization"])
    name = sub.get("quant_method")
    if not isinstance(name, str) or name in (EXL3, UNQUANTIZED):
        raise ValueError(f"Invalid non_routed_quantization method: {name!r}")
    secondary = get_quantization_config(name).from_config(sub)
    methods = (EXL3, UNQUANTIZED, name)
    spec = config["module_quantization"]
    rules = []
    for rule in spec.get("rules", []):
        if set(rule) != {"match", "method"} or rule["method"] not in methods:
            raise ValueError(f"Invalid module_quantization rule: {rule}")
        rules.append((re.compile(rule["match"]), rule["method"]))
    default = spec.get("default")
    if default is not None and default not in methods:
        raise ValueError(f"Invalid module_quantization default: {default!r}")
    if not rules and default is None:
        raise ValueError("module_quantization has neither rules nor a default")
    return HybridSpec(tuple(rules), default, secondary, name)


def check_plain_sources(
    manifest: Exl3Manifest, sources: list[str], method: str
) -> None:
    """Refuse a module routed away from EXL3 whose checkpoint tensors do not
    fit the method: EXL3 matrices, no ``.weight``, a float weight for a
    quantized method or a quantized one for ``"unquantized"``."""
    for source in sources:
        grouped = f"{source}.slice.0" in manifest.matrices
        if source in manifest.matrices or grouped:
            raise ValueError(f"{source} holds EXL3 matrices but takes {method}")
        weight = manifest.tensors.get(f"{source}.weight")
        if weight is None:
            raise ValueError(f"{source} takes {method}; the checkpoint has no weight")
        if (weight.dtype in FLOAT_DTYPES) != (method == UNQUANTIZED):
            raise ValueError(f"{source} takes {method}; its weight is {weight.dtype}")
