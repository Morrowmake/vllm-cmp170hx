# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model registry and model-config entries of the fork's model classes.

vLLM caches the model information of a lazily registered class on disk
(``$VLLM_CACHE_ROOT/modelinfos``), keyed by a hash of the class's module
file, or of the whole package when the module is a package ``__init__.py``.
The fork model classes subclass the upstream ones in another module, so a key
over their own module alone would keep serving stale information after a
change to the upstream model code. :class:`PackageHashedLazyModel` keys it
on the fork model package (``vllm_sm80/models/<model>/``) and the whole
upstream model package (``vllm/models/<model>/``), as upstream's own package
entry does; :class:`UpstreamHashedLazyModel` (fork classes outside a model
package, e.g. the drafters in ``vllm_sm80/spec_decode/``) on the fork module
and the upstream module its class extends.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from vllm.model_executor.models.registry import _LazyRegisteredModel
from vllm.utils.hashing import safe_hash


def upstream_model_package_init(model: str) -> Path:
    """``vllm/models/<model>/__init__.py``, located without importing it."""
    spec = importlib.util.find_spec(f"vllm.models.{model}")
    assert spec is not None and spec.origin, model
    return Path(spec.origin)


@dataclass(frozen=True)
class PackageHashedLazyModel(_LazyRegisteredModel):
    """A lazy entry for a class in ``vllm_sm80/models/<model>/``."""

    @staticmethod
    def _get_modelinfo_module_hash(model_path: Path) -> str:
        # vllm_sm80/models/<model>/sm80_model.py -> vllm_sm80/models/<model>/
        # and vllm/models/<model>/
        fork_init = model_path.parent / "__init__.py"
        assert fork_init.parent.parent.name == "models", model_path
        hasher = safe_hash(b"", usedforsecurity=False)
        for package_init in (fork_init, upstream_model_package_init(fork_init.parent.name)):
            hasher.update(_LazyRegisteredModel._get_modelinfo_module_hash(package_init).encode())
            hasher.update(b"\0")
        return hasher.hexdigest()


@dataclass(frozen=True)
class UpstreamHashedLazyModel(_LazyRegisteredModel):
    """A lazy entry for a fork class that extends the upstream model class in
    ``upstream_module`` (a module name, located without importing it)."""

    upstream_module: str

    def _get_modelinfo_module_hash(self, model_path: Path) -> str:
        spec = importlib.util.find_spec(self.upstream_module)
        assert spec is not None and spec.origin, self.upstream_module
        hasher = safe_hash(b"", usedforsecurity=False)
        for name, path in (
            (self.module_name, model_path),
            (self.upstream_module, Path(spec.origin)),
        ):
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
        return hasher.hexdigest()


def with_hf_config_defaults(hook: type | None, defaults: dict[str, object]) -> type:
    """A ``MODELS_CONFIG_MAP`` entry that runs ``hook`` (the entry it replaces,
    if any) and then sets each attribute of ``defaults`` that the model's HF
    config leaves unset, so a checkpoint or ``--hf-overrides`` value wins."""
    from vllm.model_executor.models.config import VerifyAndUpdateConfig

    base = hook or VerifyAndUpdateConfig

    class HfConfigDefaults(base):  # type: ignore[valid-type, misc]
        hf_config_defaults = dict(defaults)
        replaced_hook = hook

        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            base.verify_and_update_model_config(model_config)
            for name, value in defaults.items():
                if getattr(model_config.hf_config, name, None) is None:
                    setattr(model_config.hf_config, name, value)

    return HfConfigDefaults
