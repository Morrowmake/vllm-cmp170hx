# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop-in ``triton.autotune`` that can pin one config per autotune key.

``pinned_autotune(...)`` takes exactly the arguments of ``triton.autotune``.

* ``VLLM_GLM5_FLA_PIN_AUTOTUNE`` unset (the default): it returns the object
  ``triton.autotune(...)(fn)`` returns, unchanged; the kernel is only recorded
  by name in ``REGISTRY``.
* ``VLLM_GLM5_FLA_PIN_AUTOTUNE=1``: the ``Autotuner`` becomes a
  ``PinnedAutotuner``. On its first launch it checks the device; on sm_80 every
  launch uses the config from ``autotune_pins.PINS[(name, key)]`` or, for a key
  not in the table, the kernel's ``autotune_pins.DEFAULTS[name]``. Nothing is
  benchmarked, so the config (and with it the reduction order: BK tiles over a
  contracted dimension, chunk lengths, warp and stage counts) depends only on
  the shape. On any other device it autotunes as usual.

The key is built exactly as ``triton.runtime.autotuner.Autotuner.run`` builds
it: the values of ``key`` in argument order, then ``str(dtype)`` of every
tensor argument in call order. The pin table uses the same tuples.

Kernel names are ``fla.<module>.<function>`` for the vendored
flash-linear-attention ops and ``glm5_kda.<function>`` for the GLM-5 KDA
chunked-prefill kernels.
"""

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, triton

logger = init_logger(__name__)

_MODULE_PREFIXES = (
    ("vllm.third_party.flash_linear_attention.ops.", "fla."),
    ("vllm.models.glm5next.nvidia.ops.third_party.kda.kernels", "glm5_kda"),
)

# kernel name -> the Autotuner object (always filled; never changes behaviour).
REGISTRY: dict[str, Any] = {}

_gate: bool | None = None


def kernel_name(fn: Any) -> str:
    # The wrapped Python function: JITFunction and the interpreter's
    # InterpretedFunction both keep it as ``.fn``.
    fn = getattr(fn, "fn", fn)
    module = getattr(fn, "__module__", "") or ""
    for prefix, short in _MODULE_PREFIXES:
        if module.startswith(prefix):
            module = short + module[len(prefix) :]
            break
    return f"{module}.{fn.__name__}"


def tuning_key(tuner: Any, args: tuple, kwargs: dict) -> tuple:
    """The key ``Autotuner.run`` caches the winner under (Triton 3.7)."""
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    _args = {k: v for (k, v) in all_args.items() if k in tuner.arg_names}
    key = [_args[k] for k in tuner.keys if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


def config_spec(config: Any) -> dict:
    """A table entry for ``config``."""
    return {
        "kwargs": dict(config.kwargs),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def find_config(configs: list, spec: dict) -> Any | None:
    """The member of ``configs`` a table entry names, or None."""
    for config in configs:
        if config_spec(config) == {
            "kwargs": dict(spec.get("kwargs", {})),
            "num_warps": spec["num_warps"],
            "num_stages": spec["num_stages"],
        }:
            return config
    return None


def resolve(name: str, key: tuple, configs: list) -> tuple[Any, str]:
    """The pinned config for (name, key) and where it came from."""
    from vllm.third_party.flash_linear_attention.ops import autotune_pins

    spec = autotune_pins.PINS.get((name, key))
    if spec is not None:
        config = find_config(configs, spec)
        if config is not None:
            return config, "table"
        logger.warning_once(
            "fla autotune pin for %s key=%s is not one of its configs (%s); "
            "using the kernel default",
            name,
            key,
            spec,
        )
    spec = autotune_pins.DEFAULTS.get(name)
    if spec is None:
        raise KeyError(f"no default autotune pin for kernel {name}")
    config = find_config(configs, spec)
    if config is None:
        raise ValueError(f"default autotune pin for {name} is not a config: {spec}")
    return config, "default"


def _gate_open() -> bool:
    global _gate
    if _gate is None:
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        _gate = (
            current_platform.is_cuda()
            and cap is not None
            and (cap.major, cap.minor) == (8, 0)
        )
        if _gate:
            from vllm.third_party.flash_linear_attention.ops import autotune_pins

            logger.info_once(
                "fla autotune pinned: %d kernels, %d table entries%s; unlisted "
                "keys use the per-kernel default",
                len(REGISTRY),
                len(autotune_pins.PINS),
                " (provisional table)" if autotune_pins.PROVISIONAL else "",
            )
        else:
            logger.info_once(
                "VLLM_GLM5_FLA_PIN_AUTOTUNE is set but the device is %s, not "
                "sm_80: fla kernels autotune as usual",
                None if cap is None else f"sm_{cap.major}{cap.minor}",
            )
    return _gate


if HAS_TRITON:
    from triton.runtime.autotuner import Autotuner as _Autotuner

    class PinnedAutotuner(_Autotuner):
        """An Autotuner whose winner per key comes from the pin table."""

        _pin_name: str = ""

        def run(self, *args, **kwargs):
            if len(self.configs) <= 1 or not _gate_open():
                return super().run(*args, **kwargs)
            key = tuning_key(self, args, kwargs)
            config = self.cache.get(key)
            if config is None:
                config, source = resolve(self._pin_name, key, self.configs)
                self.cache[key] = config
                if source == "default":
                    logger.info_once(
                        "fla autotune pin: %s key=%s not in the table, using "
                        "the default %s",
                        self._pin_name,
                        key,
                        config,
                    )
            self.nargs = dict(zip(self.arg_names, args))
            self.best_config = config
            if config.pre_hook is not None:
                config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
            ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret


def pinned_autotune(configs, key, **kwargs):
    """``triton.autotune`` with an optional per-key pin (see module doc)."""
    decorator = triton.autotune(configs=configs, key=key, **kwargs)
    if not HAS_TRITON:
        return decorator

    def wrap(fn):
        tuner = decorator(fn)
        name = kernel_name(fn)
        REGISTRY[name] = tuner
        if envs.VLLM_GLM5_FLA_PIN_AUTOTUNE:
            tuner.__class__ = PinnedAutotuner
            tuner._pin_name = name
        return tuner

    return wrap
