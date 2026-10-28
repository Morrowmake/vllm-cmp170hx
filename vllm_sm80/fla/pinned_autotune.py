# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop-in ``triton.autotune`` that can pin one config per autotune key.

``pinned_autotune(...)`` takes exactly the arguments of ``triton.autotune``.

* ``VLLM_FLA_PIN_AUTOTUNE`` and ``VLLM_FLA_AUTOTUNE_SKIP_SPILLING`` unset
  (the default): it returns the object ``triton.autotune(...)(fn)`` returns,
  unchanged; the kernel is only recorded by name in ``REGISTRY``.
* ``VLLM_FLA_PIN_AUTOTUNE=1``: the ``Autotuner`` becomes a
  ``PinnedAutotuner``. On its first launch it checks the device; on sm_80 every
  launch uses the config the pin table gives for (kernel, autotune key, H,
  T bucket), or the kernel's ``autotune_pins.DEFAULTS[name]`` when the table
  has no entry. Nothing is benchmarked, so the config (and with it the
  reduction order: BK tiles over a contracted dimension, warp and stage
  counts) is a pure function of the launch shape. On any other device it
  autotunes as usual.
* ``VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1`` (pinning off): every ``Autotuner``
  becomes a ``SkipSpillingAutotuner``. When it benchmarks on sm_80, it builds
  each config first and skips (never launches) one whose build needs more
  than ``LOCAL_BYTES_LIMIT`` bytes of local memory per thread; the skipped
  config times as infinite. A benchmark that skips a config and leaves no
  finite time raises instead of launching anything. On a hit in Triton's
  autotune cache only the cached winner is built; if it is over the limit
  (timings written without the flag) the key is benchmarked again. On any
  other device it autotunes as usual.

The autotune key is built exactly as ``triton.runtime.autotuner.Autotuner.run``
builds it: the values of ``key`` in argument order, then ``str(dtype)`` of
every tensor argument in call order. None of these keys holds the token
count, and several do not hold the head count, yet the fastest config depends
on both. The pin lookup therefore adds ``H`` (the kernel's ``H`` argument, or
None) and the T bucket: the first upper bound in ``autotune_pins.T_BUCKETS``
that is >= the kernel's ``T`` argument (the total token count of the launch),
or None past the last bound. The one kernel without a ``T`` argument
(``fla.l2norm.l2norm_fwd_kernel1``) always uses the None bucket.

Kernel names are ``fla.<module>.<function>`` for the vendored
flash-linear-attention ops and ``glm5_kda.<function>`` for the GLM-5 KDA
chunked-prefill kernels.
"""

import math
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, triton

logger = init_logger(__name__)

# The CUDA context's default per-thread stack (cudaLimitStackSize). The
# driver grows the context's local memory to the largest per-thread local
# size launched, for every thread the device can hold (2,048 per SM), and
# never shrinks it. Benchmarking a config above this size during the memory
# profile therefore keeps local size x 2,048 x SMs bytes reserved for the
# life of the process, and the KV cache loses that much (chunk_fwd_kernel_o
# at two warps and the 128 x 128 tile: 11,664 B, about 1.5 GiB on a 74-SM
# card).
LOCAL_BYTES_LIMIT = 1024

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


def _key_of(tuner: Any, all_args: dict) -> tuple:
    _args = {k: v for (k, v) in all_args.items() if k in tuner.arg_names}
    key = [_args[k] for k in tuner.keys if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


def t_bucket(T: int | None, bounds: tuple) -> int | None:
    """The T bucket: the first bound >= T, None past the last (or no T)."""
    if T is None:
        return None
    for b in bounds:
        if T <= b:
            return b
    return None


def pin_lookup_key(tuner: Any, args: tuple, kwargs: dict, bounds: tuple) -> tuple:
    """(autotune key, H, T bucket) for one launch."""
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    if "T" in tuner.arg_names:
        T = all_args.get("T")
        assert isinstance(T, int), f"{tuner.base_fn.__name__}: T must be an int"
    else:
        T = None
    H = all_args.get("H")
    return _key_of(tuner, all_args), H, t_bucket(T, bounds)


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


def resolve(
    name: str, key: tuple, H: int | None, bucket: int | None, configs: list
) -> tuple[Any, str]:
    """The pinned config for (name, key, H, bucket) and where it came from."""
    from vllm_sm80.fla import autotune_pins

    spec = autotune_pins.PINS.get((name, key, H), {}).get(bucket)
    if spec is not None:
        config = find_config(configs, spec)
        if config is not None:
            return config, "table"
        logger.warning_once(
            "fla autotune pin for %s key=%s H=%s T<=%s is not one of its "
            "configs (%s); using the kernel default",
            name,
            key,
            H,
            bucket,
            str(spec),
        )
    spec = autotune_pins.DEFAULTS.get(name)
    if spec is None:
        raise KeyError(f"no default autotune pin for kernel {name}")
    config = find_config(configs, spec)
    if config is None:
        raise ValueError(f"default autotune pin for {name} is not a config: {spec}")
    return config, "default"


def _is_sm80() -> bool:
    from vllm.platforms import current_platform

    cap = current_platform.get_device_capability()
    return bool(
        current_platform.is_cuda()
        and cap is not None
        and (cap.major, cap.minor) == (8, 0)
    )


def local_bytes(kernel: Any) -> int:
    """Local memory per thread of a compiled Triton kernel. Loads its module
    into the context (``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES``, which Triton
    reports in 4-byte words as ``n_spills``); nothing is launched."""
    kernel._init_handles()
    return int(kernel.n_spills) * 4


_skip_gate: bool | None = None


def _skip_gate_open() -> bool:
    global _skip_gate
    if _skip_gate is None:
        _skip_gate = _is_sm80()
        if _skip_gate:
            logger.info_once(
                "fla autotune on sm_80 skips configs whose build needs more "
                "than %d B of local memory per thread "
                "(VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1)",
                LOCAL_BYTES_LIMIT,
            )
        else:
            logger.info_once(
                "VLLM_FLA_AUTOTUNE_SKIP_SPILLING is set but the device is not "
                "sm_80: fla kernels autotune over every config"
            )
    return _skip_gate


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
            from vllm_sm80.fla import autotune_pins

            logger.info_once(
                "fla autotune pinned: %d kernels, %d table entries, T buckets "
                "%s%s; unlisted shapes use the per-kernel default",
                len(REGISTRY),
                sum(len(v) for v in autotune_pins.PINS.values()),
                str(list(autotune_pins.T_BUCKETS)),
                " (provisional table)" if autotune_pins.PROVISIONAL else "",
            )
        else:
            logger.info_once(
                "VLLM_FLA_PIN_AUTOTUNE is set but the device is %s, not "
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
            from vllm_sm80.fla import autotune_pins

            key, H, bucket = pin_lookup_key(
                self, args, kwargs, autotune_pins.T_BUCKETS
            )
            # Own cache key: the Triton key plus H and the T bucket.
            cache_key = (*key, "H", H, "T<=", bucket)
            config = self.cache.get(cache_key)
            if config is None:
                config, source = resolve(self._pin_name, key, H, bucket, self.configs)
                self.cache[cache_key] = config
                if source == "default":
                    logger.info_once(
                        "fla autotune pin: %s key=%s H=%s T<=%s not in the "
                        "table, using the default %s",
                        self._pin_name,
                        key,
                        H,
                        bucket,
                        str(config),
                    )
            self.nargs = dict(zip(self.arg_names, args))
            self.best_config = config
            if config.pre_hook is not None:
                config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
            ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret

    class SkipSpillingAutotuner(_Autotuner):
        """An Autotuner that never launches a config whose sm_80 build needs
        more than ``LOCAL_BYTES_LIMIT`` bytes of local memory per thread."""

        _pin_name: str = ""
        _bench_left: int = 0
        _bench_finite: int = 0
        _bench_skipped: int = 0
        _run_args: tuple = ((), {})

        def run(self, *args, **kwargs):
            # check_disk_cache is not given the launch arguments; keep them.
            self._run_args = (args, kwargs)
            try:
                return super().run(*args, **kwargs)
            finally:
                self._run_args = ((), {})

        def prune_configs(self, kwargs):
            configs = super().prune_configs(kwargs)
            # Autotuner.run prunes once per new key, then benchmarks (or reads
            # the autotune cache), timing each config in turn.
            self._bench_left = len(configs)
            self._bench_finite = self._bench_skipped = 0
            return configs

        def check_disk_cache(self, tuning_key, configs, bench_fn):
            hit = super().check_disk_cache(tuning_key, configs, bench_fn)
            if not (hit and _skip_gate_open()):
                return hit
            args, kwargs = self._run_args
            best = self.cache[tuning_key]
            size = self._local_bytes(args, best, kwargs)
            if size is None or size <= LOCAL_BYTES_LIMIT:
                return hit
            logger.info(
                "fla autotune: the cached winner of %s %s needs %d B of local "
                "memory per thread on sm_80 (limit %d B); benchmarking again",
                self._pin_name,
                best,
                size,
                LOCAL_BYTES_LIMIT,
            )
            del self.cache[tuning_key]
            bench_fn()
            return False

        def _local_bytes(self, args, config, meta) -> int | None:
            current = {**meta, **config.all_kwargs(), "warmup": True}
            try:
                return local_bytes(self.fn.run(*args, **current))
            except Exception:  # noqa: BLE001
                # A config that fails to build is left to the benchmark,
                # which times it as infinite.
                return None

        def _bench(self, *args, config, **meta):
            self._bench_left -= 1
            timing = None
            if _skip_gate_open():
                size = self._local_bytes(args, config, meta)
                if size is not None and size > LOCAL_BYTES_LIMIT:
                    logger.info(
                        "fla autotune skips %s %s: %d B of local memory per "
                        "thread on sm_80 (limit %d B)",
                        self._pin_name,
                        config,
                        size,
                        LOCAL_BYTES_LIMIT,
                    )
                    self._bench_skipped += 1
                    timing = [float("inf")] * 3
            if timing is None:
                timing = super()._bench(*args, config=config, **meta)
                if math.isfinite(timing[0]):
                    self._bench_finite += 1
            # With every time infinite, Autotuner.run would pick the first
            # config, which may be one this benchmark skipped.
            if self._bench_left == 0 and self._bench_skipped and not self._bench_finite:
                raise RuntimeError(
                    f"no autotune config of {self._pin_name} has a finite time "
                    f"on sm_80: {self._bench_skipped} need more than "
                    f"{LOCAL_BYTES_LIMIT} B of local memory per thread and the "
                    "rest failed"
                )
            return timing


def pinned_autotune(configs, key, **kwargs):
    """``triton.autotune`` with an optional per-key pin (see module doc)."""
    decorator = triton.autotune(configs=configs, key=key, **kwargs)
    if not HAS_TRITON:
        return decorator

    def wrap(fn):
        tuner = decorator(fn)
        name = kernel_name(fn)
        REGISTRY[name] = tuner
        if envs.VLLM_FLA_PIN_AUTOTUNE:
            tuner.__class__ = PinnedAutotuner
            tuner._pin_name = name
        elif envs.VLLM_FLA_AUTOTUNE_SKIP_SPILLING:
            tuner.__class__ = SkipSpillingAutotuner
            tuner._pin_name = name
        return tuner

    return wrap
