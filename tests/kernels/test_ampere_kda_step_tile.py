# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the step tile of the fused KDA decode kernel (token tile from the
step's max tokens per request; always on, no switch of its own).

CPU part: the tile helper, the metadata field default.
GPU part (skipped without CUDA): for the v2 kernel and the v2 recover verify
(normed and SKIP_NORM), max_query_len = the step's tokens per
request (tile 1/2/4/8) gives bitwise the result of max_query_len = 8 (the
full window, what is launched today): outputs, recurrent state, conv state,
records.

    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_step_tile.py
"""

import importlib.util
import os

try:
    import pytest
except ImportError:
    pytest = None
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _rec_tests():
    spec = importlib.util.spec_from_file_location(
        "_trec", os.path.join(HERE, "test_ampere_kda_recover.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ CPU tests

def test_tile_helper():
    from vllm_sm80.models.glm5next.ops.kda_decode import _kda_tile

    assert _kda_tile(None, 8) == 8
    assert _kda_tile(4, 8) == 4
    assert _kda_tile(12, 8) == 8          # mixed step: prefill rows raise the bound
    assert _kda_tile(0, 8) == 1
    # recover verify: the window unless the token tile shrinks
    assert _kda_tile(4, 8, keep_same_rows=True) == 4
    assert _kda_tile(6, 8, keep_same_rows=True) == 8
    assert _kda_tile(5, 8, keep_same_rows=True) == 8
    assert _kda_tile(2, 4, keep_same_rows=True) == 2
    assert _kda_tile(3, 4, keep_same_rows=True) == 4


def test_metadata_field_default():
    from vllm_sm80.kernels.kda.recover import Glm5NextGDNAttentionMetadata

    f = {x.name: x for x in
         __import__("dataclasses").fields(Glm5NextGDNAttentionMetadata)}
    assert "spec_max_query_len" in f and f["spec_max_query_len"].default is None


# ------------------------------------------------------------------ GPU tests

def _eq(a, b):
    return torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                       b.view(torch.int32) if b.dtype == torch.float32 else b)


def gpu_test_tile_is_bitwise_the_full_window():
    from vllm_sm80.models.glm5next.ops import kda_decode as k2
    from vllm_sm80.kernels.kda import recover as kr

    tr = _rec_tests()
    H, D = tr.H, tr.D
    bad = []
    for T in range(1, 9):
        for nseq in (1, 4, 8):
            nslot = 2 + nseq * 8
            p = tr._inputs(nseq, T, 300 + 10 * T + nseq, nslot)
            p.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
            ssm8 = torch.as_tensor([[1 + s * 8 + t for t in range(8)] for s in range(nseq)],
                                   device="cuda", dtype=torch.int32)
            acc = torch.as_tensor([1 + (s * 3) % T for s in range(nseq)], device="cuda",
                                  dtype=torch.int32)
            M = nseq * T
            for q in sorted({T, 8}):
                k2.warmup(plans=((nseq, q),), width=8)
                k2.warmup(plans=((nseq, q),), recover=True)

            def run(kind, q):
                rec, conv = p.rec.clone(), p.conv.clone()
                out = torch.zeros(1, M, H, D, device="cuda", dtype=torch.bfloat16)
                pool = torch.zeros(1, 3, 8, kr.WS_T, H, D, device="cuda")
                if kind == "v2":
                    k2.kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg,
                                     conv.transpose(-1, -2), p.cw, None, p.nw, rec,
                                     ssm8[:, 0], ssm8, acc, p.qsl, q, p.al, p.gb, out=out)
                else:
                    ssm1 = ssm8[:, :1].contiguous()
                    ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
                    k2.kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg,
                                     conv.transpose(-1, -2), p.cw, None, p.nw, rec,
                                     ssm1[:, 0], ssm1, ones, p.qsl, q, p.al, p.gb, out=out,
                                     records=kr.layer_records(pool, 0),
                                     skip_norm=(kind == "recover_skip"))
                torch.cuda.synchronize()
                return out, rec, conv, pool

            for kind in ("v2", "recover", "recover_skip"):
                a, b = run(kind, T), run(kind, 8)
                if not all(_eq(x, y) for x, y in zip(a, b)):
                    bad.append((kind, T, nseq))
    assert not bad, bad
    print("  tile = tokens per request is bitwise the 8-row window for v2, recover, "
          "recover SKIP_NORM at T 1..8 x nseq 1/4/8")


def gpu_test_warmup_covers_every_tile():
    """The v2 warmup compiles (nseq, t) for t = 1..num_spec+1 where the v2 gate
    admits it."""
    import types

    from vllm_sm80.kernels import _platform
    from vllm_sm80.models.glm5next.ops import kda_decode

    w = torch.zeros(16 * 128, 128, device="cuda", dtype=torch.bfloat16)
    layer = types.SimpleNamespace(_conv_state_dim_first=False, local_num_heads=16,
                                  head_dim=128, num_spec=3, _kda_step_tile=True,
                                  _kda_v2=True, _kda_wide=False, _kda_recover=False,
                                  f_b_proj=types.SimpleNamespace(weight=w),
                                  g_b_proj=types.SimpleNamespace(weight=w))
    model = torch.nn.Module()
    worker = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=2)))
    saved = kda_decode.find_module
    kda_decode.find_module = lambda model, attr: layer
    old = {k: os.environ.get(k) for k in ("VLLM_AMPERE_DECODE_KERNELS",)}
    os.environ.update(VLLM_AMPERE_DECODE_KERNELS="1")
    sm80 = _platform._SM80_CACHE
    _platform._SM80_CACHE = True
    try:
        kda_decode._WARMED.clear()
        kda_decode._warmup_kda_v2(worker, model, "cuda", [1, 2, 4, 8])
        got = sorted({(k[0], k[1]) for k in kda_decode._WARMED})
        assert got == [(n, t) for n in (1, 2) for t in (1, 2, 3, 4)], got
    finally:
        kda_decode.find_module = saved
        _platform._SM80_CACHE = sm80
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


CPU_TESTS = (test_tile_helper, test_metadata_field_default)
GPU_TESTS = (gpu_test_tile_is_bitwise_the_full_window, gpu_test_warmup_covers_every_tile)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an sm_80 GPU")
    test_gpu_tile_is_bitwise_the_full_window = _needs_gpu(gpu_test_tile_is_bitwise_the_full_window)
    test_gpu_warmup_covers_every_tile = _needs_gpu(gpu_test_warmup_covers_every_tile)


def _main():
    import sys
    import traceback

    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
