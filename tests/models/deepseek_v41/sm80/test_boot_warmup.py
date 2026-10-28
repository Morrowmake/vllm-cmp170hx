# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot warmup of the Triton kernels served DeepSeek-V4.1 runs used to compile
mid-request: the sm_80 index scores, the speculative-decoding kernels, the
MoE gate and route of VLLM_DEEPSEEK_V41_ROUTER_MMA and the grouped wo_a GEMM.

* The index-score launch's specialisation key (Triton's: pointer dtypes and
  16-byte alignment, the class of every specialised integer -- 1, divisible by
  16, other -- and the constexprs) is the same for every score width, row
  count and block-table width on one cache, so the one warmup compile per
  cache covers every served call.
* The prepare kernel's BLOCK_SIZE set the warmup walks is exactly the set
  ``prepare_dflash_inputs`` can pick.
* The spec-decode warmup drives the runner's own rejection sampler and the
  speculator's own buffers.
* The router warmup's and the wo_a warmup's launches have, between them, the
  specialisation key (constexprs and launch options exactly) of every launch
  a served call of 1..MAX_ROWS (router: target and drafter shapes, with and
  without image routing, several SM counts) or 1..2048 rows (wo_a) makes.

    pytest -q tests/models/deepseek_v41/sm80/test_boot_warmup.py
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    triton_scalar_specialization_rep,
)
from vllm_sm80.models.deepseek_v41.ops import sm80_index_scores as isc
from vllm_sm80.models.deepseek_v41.ops import sm80_router_mma as rm
from vllm_sm80.models.deepseek_v41.ops import sm80_wo_a_grouped as wg
from vllm_sm80.spec_decode import warmup as spec_warmup
from vllm.triton_utils import triton

_KERNEL = isc._index_scores_kernel


class _Recorder:
    """Stands in for a Triton kernel: records each launch's arguments."""

    def __init__(self, kernel):
        self.arg_names = list(inspect.signature(kernel.fn).parameters)
        self.do_not_specialize = set(kernel.do_not_specialize)
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            bound = dict(zip(self.arg_names, args))
            bound.update(kwargs)
            self.calls.append(bound)

        return run

    def key(self, call: dict) -> tuple:
        """Triton's specialisation key of one launch, without the device."""
        parts = []
        for name in self.arg_names:
            value = call[name]
            if isinstance(value, torch.Tensor):
                parts.append((name, value.dtype, value.data_ptr() % 16 == 0))
            elif name in self.do_not_specialize:
                parts.append((name, "dns"))
            elif isinstance(value, bool) or not isinstance(value, int):
                parts.append((name, value))
            else:
                parts.append((name, triton_scalar_specialization_rep(value)))
        return tuple(parts)


def _launch_keys(monkeypatch, cache: torch.Tensor) -> set[tuple]:
    rec = _Recorder(_KERNEL)
    monkeypatch.setattr(isc, "_index_scores_kernel", rec)
    g = torch.Generator().manual_seed(0)
    q_all = torch.zeros(40, 32, 128, dtype=torch.bfloat16)
    w_all = torch.zeros(40, 32, dtype=torch.bfloat16)
    for n_cols in list(range(1, 70)) + [511, 512, 513, 4096]:
        for rows in (1, 3, 8, 17):
            start = int(torch.randint(0, 40 - rows + 1, (1,), generator=g))
            for table_width in (1, 3, 16, 33):
                n_vis = torch.full((rows,), n_cols, dtype=torch.int64)
                req = torch.zeros(rows, dtype=torch.int64)
                table = torch.zeros((2, table_width), dtype=torch.int32)
                out = isc.index_scores_triton(
                    q_all[start : start + rows],
                    w_all[start : start + rows],
                    n_vis,
                    req,
                    cache,
                    table,
                    n_cols,
                    8,
                )
                assert out.shape == (rows, n_cols)
                assert out.stride(0) % 16 == 0
    return {rec.key(c) for c in rec.calls}


def test_index_scores_one_specialisation_per_cache(monkeypatch) -> None:
    for entries in (64, 128):
        cache = torch.zeros(5, entries, 128, dtype=torch.bfloat16)
        keys = _launch_keys(monkeypatch, cache)
        assert len(keys) == 1, keys


def test_index_scores_warmup_keys_and_launch(monkeypatch) -> None:
    def fake_cache(entries, block_stride):
        return SimpleNamespace(
            kv_cache=SimpleNamespace(
                numel=lambda: 1,
                is_cuda=True,
                shape=(10, entries, 128),
                stride=lambda d: block_stride,
                data_ptr=lambda: 4096,
            )
        )

    warm = isc.IndexScoresWarmup()
    a = warm.get_warmup_keys(fake_cache(64, 64 * 128 * 7), 32, 128, 8)
    b = warm.get_warmup_keys(fake_cache(64, 64 * 128 * 9), 32, 128, 8)
    c = warm.get_warmup_keys(fake_cache(128, 128 * 128), 32, 128, 8)
    assert a == b and a != c  # same geometry class compiles once
    empty = SimpleNamespace(kv_cache=torch.tensor([]))
    assert warm.get_warmup_keys(empty, 32, 128, 8) == []

    # compile() launches on the cache it was given, with served dtypes.
    cache = torch.zeros(3, 64, 128, dtype=torch.bfloat16)
    rec = _Recorder(_KERNEL)
    monkeypatch.setattr(isc, "_index_scores_kernel", rec)
    key = isc.IndexScoresWarmup.Key(64, 32, 128, 8, 16, True, cache)
    warm.compile(key)
    assert len(rec.calls) == 1 and rec.calls[0]["k_ptr"] is cache
    served = _launch_keys(monkeypatch, cache)
    assert {rec.key(rec.calls[0])} == served


@pytest.mark.parametrize("num_query_per_req", [2, 3, 6, 9])
@pytest.mark.parametrize("max_num_tokens", [8, 100, 2048])
def test_prepare_block_sizes_cover_every_reachable_value(
    num_query_per_req, max_num_tokens
) -> None:
    reachable = {
        min(256, triton.next_power_of_2(q + num_query_per_req))
        for q in range(1, max_num_tokens + 1)
    }
    picked = spec_warmup.prepare_block_sizes(num_query_per_req, max_num_tokens)
    assert {b for b, _ in picked} == reachable
    for block, q in picked:
        assert 1 <= q <= max_num_tokens
        assert min(256, triton.next_power_of_2(q + num_query_per_req)) == block


def _no_collective(hidden):
    raise AssertionError("compute_logits gathers across TP ranks")


def _on_sm80(monkeypatch, sm80: bool) -> None:
    platform = spec_warmup.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        platform, "is_device_capability_family", lambda cap, device_id=0: sm80 == (cap == 80)
    )


def _assert_skipped(monkeypatch, runner) -> None:
    calls = []
    for name in ("_warm_rejection_sampler", "_warm_dflash_prepare"):
        monkeypatch.setattr(spec_warmup, name, lambda *a, n=name: calls.append(n))
    spec_warmup.warmup_spec_decode(runner)
    assert calls == []


def _fake_worker(monkeypatch, verify_calls, prepare_calls):
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = object.__new__(DSparkSpeculator)
    for name, value in dict(
        draft_logits=torch.zeros(8, 5, 16),
        num_query_per_req=6,
        num_speculative_steps=5,
        draft_kv_cache_group_ids=[0, 2],
        _context_slot_mappings=["ctx0", "ctx1"],
        block_tables=SimpleNamespace(
            slot_mappings={0: "s0", 2: "s2"},
            input_block_tables={0: "t0", 2: "t2"},
            block_sizes={0: 64, 2: 128},
            cp_rank=0,
            cp_interleave=1,
        ),
        input_buffers="spec_buffers",
        context_positions="cp",
        sample_indices="si",
        sample_pos="sp",
        sample_idx_mapping="sim",
        temperature="t",
        seeds="s",
        dcp_size=1,
        parallel_drafting_token_id=7,
        max_num_reqs=8,
        max_num_tokens=2048,
        # Speculator.max_model_len reads the model config.
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
        sample_from_anchor=True,
    ).items():
        object.__setattr__(spec, name, value)
    rejection = SimpleNamespace(
        num_speculative_steps=5,
        synthetic_conditional_rates=None,
        use_block_verification=False,
        _watermarking_kwargs=lambda *a: {},
        sampler=SimpleNamespace(
            use_fp64_gumbel=False,
            sampling_states=SimpleNamespace(
                temperature=SimpleNamespace(gpu="tg"), seeds=SimpleNamespace(gpu="sg")
            ),
        ),
    )
    runner = SimpleNamespace(
        speculator=spec,
        rejection_sampler=rejection,
        is_last_pp_rank=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_num_tokens=2048,
        model_config=SimpleNamespace(get_vocab_size=lambda: 16, head_dtype=torch.bfloat16),
        model=SimpleNamespace(compute_logits=_no_collective),
        input_buffers=SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int32),
            positions=torch.zeros(4, dtype=torch.int64),
        ),
        req_states=SimpleNamespace(last_sampled_tokens="ls", next_prefill_tokens="np"),
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(
                temperature=SimpleNamespace(gpu="tg"), seeds=SimpleNamespace(gpu="sg")
            )
        ),
    )
    monkeypatch.setattr(
        InputBatch,
        "make_dummy",
        classmethod(
            lambda cls, n, q, buffers: SimpleNamespace(
                num_reqs=n, num_scheduled_tokens=np.array([q]), buffers=buffers
            )
        ),
    )
    monkeypatch.setattr(
        dflash, "prepare_dflash_inputs", lambda *a: prepare_calls.append(a)
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    _on_sm80(monkeypatch, True)
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils

    monkeypatch.setattr(
        rejection_sampler_utils,
        "rejection_sample",
        lambda *a, **k: verify_calls.append((a, k)),
    )
    return runner, spec


def test_spec_decode_warmup_drives_the_served_paths(monkeypatch) -> None:
    verify_calls, prepare_calls = [], []
    runner, spec = _fake_worker(monkeypatch, verify_calls, prepare_calls)
    spec_warmup.warmup_spec_decode(runner)

    # One request with 5 drafts, verified on the model's logits and on their
    # FP32 copy, with the speculator's draft logits and the runner's dtypes.
    assert len(verify_calls) == 2
    (raw, raw_kw), (proc, _) = verify_calls
    assert raw[0].dtype == torch.bfloat16 and proc[0].dtype == torch.float32
    assert raw[0].shape == (6, 16) and raw[1] is spec.draft_logits
    assert raw[2].dtype == torch.int32 and raw[4].dtype == torch.int64
    assert raw[3].tolist() == [0, 6]
    assert raw[10] == 5 and raw_kw["use_block_verification"] is False

    # Every reachable BLOCK_SIZE, for every draft KV-cache group.
    blocks = {
        min(256, triton.next_power_of_2(int(c[9].num_scheduled_tokens.max()) + 6))
        for c in prepare_calls
    }
    assert blocks == {8, 16, 32, 64, 128, 256}
    assert len(prepare_calls) == 2 * len(blocks)
    first = prepare_calls[0]
    assert first[0] == "spec_buffers" and first[1] == "s0" and first[3] == "ctx0"
    assert first[16:19] == ("t0", 64, 0) and first[-1] is True


def test_spec_decode_warmup_failure_does_not_stop_boot(monkeypatch) -> None:
    runner, _ = _fake_worker(monkeypatch, [], [])

    def boom(*a):
        raise RuntimeError("layout changed")

    runner.rejection_sampler._watermarking_kwargs = boom
    spec_warmup.warmup_spec_decode(runner)  # logs, does not raise


def test_spec_decode_warmup_skips_without_speculator() -> None:
    spec_warmup.warmup_spec_decode(
        SimpleNamespace(speculator=None, is_last_pp_rank=True)
    )


def test_spec_decode_warmup_skips_off_sm80(monkeypatch) -> None:
    verify_calls, prepare_calls = [], []
    runner, _ = _fake_worker(monkeypatch, verify_calls, prepare_calls)
    _on_sm80(monkeypatch, False)
    _assert_skipped(monkeypatch, runner)


@pytest.mark.parametrize("kind", ["dflash", "dflash2", "mtp"])
def test_spec_decode_warmup_skips_other_speculators(monkeypatch, kind) -> None:
    """Other models' drafters (GLM's DFlash2, Qwen's MTP) keep upstream's
    boot: only DeepSeek-V4.1's DSpark speculator is warmed."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    cls = {"dflash": DFlashSpeculator, "dflash2": DFlash2Speculator}.get(
        kind, MTPSpeculator
    )
    verify_calls, prepare_calls = [], []
    runner, spec = _fake_worker(monkeypatch, verify_calls, prepare_calls)
    other = object.__new__(cls)
    other.__dict__.update(spec.__dict__)
    runner.speculator = other
    _assert_skipped(monkeypatch, runner)


def test_spec_decode_warmup_is_called_by_the_runner_warmup() -> None:
    import inspect as _inspect

    from vllm.v1.worker.gpu import warmup as gpu_warmup

    assert gpu_warmup.warmup_spec_decode is spec_warmup.warmup_spec_decode
    assert "warmup_spec_decode(model_runner)" in _inspect.getsource(
        gpu_warmup.warmup_kernels
    )


class _SpecRecorder(_Recorder):
    """A :class:`_Recorder` whose key also takes the kernel, its constexprs
    exactly and the launch options (``num_warps``, ``num_stages``)."""

    def __init__(self, kernel):
        super().__init__(kernel)
        self.name = f"{kernel.fn.__module__}.{kernel.fn.__name__}"
        self.constexprs = {p.name for p in kernel.params if p.is_constexpr}

    def key(self, call: dict) -> tuple:
        args = {n: call[n] for n in self.arg_names}
        consts = tuple((n, args[n]) for n in self.arg_names if n in self.constexprs)
        options = tuple(sorted((k, v) for k, v in call.items() if k not in args))
        return (self.name, super().key(args) + consts, options)

    def drain(self) -> set[tuple]:
        keys = {self.key(c) for c in self.calls}
        self.calls.clear()
        return keys


def _record_router(monkeypatch, sms: int) -> list[_SpecRecorder]:
    recs = []
    for module, name in ((rm.G, "_gate_kernel"), (rm.RG, "_route_kernel"), (rm.R, "_route_kernel")):
        rec = _SpecRecorder(getattr(module, name))
        monkeypatch.setattr(module, name, rec)
        recs.append(rec)
    locks = torch.zeros(rm.G.LOCK_SLOTS, dtype=torch.int32)
    monkeypatch.setattr(rm, "_sms", lambda device: sms)
    monkeypatch.setattr(rm, "_locks", lambda device: locks)
    monkeypatch.setattr(rm, "_sm80", lambda device: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    return recs


class _MmaGate:
    """An adopted gate: its weight held once, in BF16 (sm80_moe)."""

    def __init__(self, experts: int):
        self.weight = torch.zeros(experts, rm.HIDDEN, dtype=torch.bfloat16)


def _router(experts: int, vision: bool) -> SimpleNamespace:
    """The attributes of an adopted router the layers' route call reads."""
    return SimpleNamespace(
        mma_gate=_MmaGate(experts),
        e_score_correction_bias=torch.zeros(experts),
        bias_vl=torch.zeros(experts) if vision else None,
        top_k=rm.EXPERTS_TOP_K[experts],
        image_sentinel_lo=rm.SENTINEL_LO if vision else 0,
    )


@pytest.mark.parametrize("sms", [70, 82, 108])
@pytest.mark.parametrize("experts", sorted(rm.EXPERTS_TOP_K))
def test_router_warmup_covers_every_served_launch(monkeypatch, experts, sms) -> None:
    recs = _record_router(monkeypatch, sms)

    def drain():
        return frozenset().union(*(r.drain() for r in recs))

    x_all = torch.zeros(rm.MAX_ROWS, rm.HIDDEN, dtype=torch.bfloat16)
    ids_all = torch.zeros(rm.MAX_ROWS, dtype=torch.int32)  # the runner's input_ids
    for vision in (False, True):
        router = _router(experts, vision)
        w16 = router.mma_gate.weight
        served, by_variant = set(), {}
        for rows in range(1, rm.MAX_ROWS + 1):
            # What Sm80DeepseekV41Gate.forward and the router's
            # _compute_routing call (sm80_moe.py).
            _, pending = rm.gate(x_all[:rows], w16)
            rm.route(pending, w16, router.e_score_correction_bias, router.bias_vl,
                     ids_all[:rows], router.top_k, router.image_sentinel_lo)
            keys = drain()
            served |= keys
            by_variant.setdefault(rm.launch_variant(rows, experts, sms), set()).add(keys)
        # A call's row count reaches its launches' keys only through
        # launch_variant ...
        assert all(len(v) == 1 for v in by_variant.values()), by_variant
        # ... and the warmup runs one call per variant: the same keys.
        monkeypatch.setattr(rm, "_ADOPTED", [router])
        warm = rm.RouterWarmup()
        warm_keys = warm.get_warmup_keys()
        assert [k.rows for k in warm_keys] == list(rm.warmup_rows(experts, sms))
        assert len(warm_keys) == len(by_variant)
        # 1 row; 17 (gate tile of 32 rows); 65 (prefill gate tiling and the
        # Triton route); 257 (single-warp route programs).
        assert {1, 17, 65, 257} <= {k.rows for k in warm_keys}
        warmed = set()
        for key in warm_keys:
            warm.compile(key)
            warmed |= drain()
        assert warmed == served
        # Both routes compile as the served route does (sm80_route.py): no
        # FTZ, no FP contraction; the warmup's launches carry the same options.
        plain = {("enable_fp_fusion", False), ("enable_reflect_ftz", False)}
        routes = [k for k in served if k[0].endswith("._route_kernel")]
        assert routes and all(plain <= set(options) for _, _, options in routes), routes


def test_router_warmup_dedups_layers_and_skips_off_sm80(monkeypatch) -> None:
    _record_router(monkeypatch, 108)
    target = [_router(384, True) for _ in range(3)]
    drafter = [_router(128, True) for _ in range(2)]
    monkeypatch.setattr(rm, "_ADOPTED", target + drafter)
    keys = rm.RouterWarmup().get_warmup_keys()
    assert len(keys) == len(rm.warmup_rows(384, 108)) + len(rm.warmup_rows(128, 108))
    assert {(k.experts, k.top_k) for k in keys} == {(384, 6), (128, 3)}
    monkeypatch.setattr(rm, "_sm80", lambda device: False)
    assert rm.RouterWarmup().get_warmup_keys() == []


def test_wo_a_warmup_covers_every_served_launch(monkeypatch) -> None:
    rec = _SpecRecorder(wg._wo_a_grouped_kernel)
    monkeypatch.setattr(wg, "_wo_a_grouped_kernel", rec)
    monkeypatch.setattr(wg, "_cuda", lambda device: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    # The served wo_a at TP 4: 2 local groups of 1024 outputs over 4096 inputs.
    groups, per_group, inputs, max_rows = 2, 1024, 4096, 2048
    codes = torch.zeros(groups * per_group, inputs, dtype=torch.uint8)
    scales = torch.zeros(groups * per_group, inputs // 32, dtype=torch.uint8)
    x_all = torch.zeros(max_rows, groups, inputs, dtype=torch.bfloat16)
    for rows in range(1, max_rows + 1):
        wg.wo_a_grouped(x_all[:rows], codes, scales)
    served = rec.drain()
    assert len(served) == len(wg.CONFIGS)

    layer = SimpleNamespace(
        wo_a_codes=codes, wo_a_scales=scales, quant_method=SimpleNamespace(groups=groups)
    )
    monkeypatch.setattr(wg, "_ADOPTED", [layer, layer])
    warm = wg.WoAGroupedWarmup()
    keys = warm.get_warmup_keys()
    assert [k.rows for k in keys] == list(wg.WARMUP_ROWS) == [1, 17, 33, 49, 65, 257]
    for key in keys:
        warm.compile(key)
    assert rec.drain() == served
    monkeypatch.setattr(wg, "_cuda", lambda device: False)
    assert wg.WoAGroupedWarmup().get_warmup_keys() == []


def test_router_and_wo_a_warmups_register_at_adoption(monkeypatch) -> None:
    import weakref

    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm_sm80.models.deepseek_v41 import sm80_moe

    monkeypatch.setattr(rm, "_ADOPTED", weakref.WeakSet())
    registry = JitWarmupRegistry(SimpleNamespace())
    routers = [torch.nn.Module(), torch.nn.Module()]
    with registry.activate():
        for router in routers:
            rm.register_warmup(router)
    # One registration (no arguments: get_warmup_keys reads the adopted set,
    # which the drafter's layers, loaded outside the registry, also join).
    assert len(registry) == 1 and set(rm._ADOPTED) == set(routers)
    drafter = torch.nn.Module()
    rm.register_warmup(drafter)  # no active registry: joins the set only
    assert len(registry) == 1 and set(rm._ADOPTED) == {*routers, drafter}
    assert "sm80_router_mma.register_warmup(router)" in inspect.getsource(
        sm80_moe._adopt_router_mma
    )
    adopt = inspect.getsource(wg.WoAGroupedGemmMethod.adopt)
    assert "_ADOPTED.add(layer)" in adopt and "WO_A_GROUPED_WARMUP.register_warmup()" in adopt
