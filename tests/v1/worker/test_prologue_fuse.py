# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU equivalence tests for the fused decode prologue (VLLM_GLM5_PROLOGUE_FUSE).

Every fused op replaces a run of small upstream kernels with one.  The
replacement must be *bit-identical*, which is checkable here because both families
produce integer index tensors only.  These tests run the upstream code and the
fused code on the same CPU inputs and assert equality; on CPU the fused
wrapper dispatches to its pure-torch reference, which is the transcription the
Triton kernel is written from (the Triton path itself needs a GPU and is
covered by the phase-2 validation).
"""

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu import prologue_fuse
from vllm.v1.worker.gpu.prologue_fuse import PrologueFuseSettings


@pytest.fixture
def fuse_on(monkeypatch):
    """Force the fused path on without touching the process environment."""
    monkeypatch.setattr(
        prologue_fuse, "settings", lambda: PrologueFuseSettings(enabled=True)
    )
    return PrologueFuseSettings(enabled=True)


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def test_settings_default_off():
    assert prologue_fuse.read_settings({}).enabled is False


@pytest.mark.parametrize("raw,expected", [("1", True), ("0", False), ("", False)])
def test_settings_parse(raw, expected):
    s = prologue_fuse.read_settings({"VLLM_GLM5_PROLOGUE_FUSE": raw})
    assert s.enabled is expected


def test_settings_sub_flags():
    s = prologue_fuse.read_settings(
        {"VLLM_GLM5_PROLOGUE_FUSE": "1", "VLLM_GLM5_PROLOGUE_FUSE_GDN": "0"}
    )
    assert s.enabled and s.mamba_block_table and not s.gdn


def test_settings_rejects_garbage():
    with pytest.raises(ValueError):
        prologue_fuse.read_settings({"VLLM_GLM5_PROLOGUE_FUSE": "maybe"})


@pytest.mark.parametrize(
    "name,default",
    [
        ("VLLM_GLM5_PROLOGUE_FUSE", False),
        ("VLLM_GLM5_PROLOGUE_FUSE_GDN", True),
        ("VLLM_GLM5_PROLOGUE_FUSE_MAMBA_BT", True),
    ],
)
@pytest.mark.parametrize("raw", [None, "", "1", "0", "true", "off", " On ", "YES"])
def test_envs_declaration_agrees_with_the_parser(monkeypatch, name, default, raw):
    """envs.py declares these three and mirrors this parser by hand.

    They are declared so validate_environ() does not call them unknown -- the
    launcher exports VLLM_GLM5_PROLOGUE_FUSE -- and the declaration is worth
    nothing if ``envs.X`` and the value the fusion actually uses can drift
    apart. The accept-set here is not overlap.py's: empty reads as false rather
    than as unset, and anything unrecognised raises.
    """
    import vllm.envs as envs

    if raw is None:
        monkeypatch.delenv(name, raising=False)
        env: dict[str, str] = {}
    else:
        monkeypatch.setenv(name, raw)
        env = {name: raw}

    assert envs.environment_variables[name]() is prologue_fuse._env_flag(
        env, name, default
    )


def test_envs_declaration_rejects_garbage(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_GLM5_PROLOGUE_FUSE", "maybe")
    with pytest.raises(ValueError):
        envs.environment_variables["VLLM_GLM5_PROLOGUE_FUSE"]()


# --------------------------------------------------------------------------- #
# 1. mamba_get_block_table_tensor, "align" mode
# --------------------------------------------------------------------------- #


def _mamba_spec(block_size: int, num_speculative_blocks: int) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=num_speculative_blocks,
    )


@pytest.mark.parametrize("block_size", [1, 4, 1152])
@pytest.mark.parametrize("num_spec_blocks", [0, 1, 3])
@pytest.mark.parametrize("num_reqs", [1, 8])
def test_mamba_tail_block_table_matches_upstream(
    monkeypatch, block_size, num_spec_blocks, num_reqs
):
    torch.manual_seed(block_size * 131 + num_spec_blocks * 17 + num_reqs)
    ncols = 64
    block_table = torch.randint(0, 100000, (num_reqs, ncols), dtype=torch.int32)
    spec = _mamba_spec(block_size, num_spec_blocks)
    # Include 0 (the CUDA-graph padded row) and exact block boundaries.
    seq_lens = torch.randint(0, 40, (num_reqs,), dtype=torch.int32) * block_size
    seq_lens[0] = 0
    if num_reqs > 1:
        seq_lens[1] = 1

    expected = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    monkeypatch.setattr(
        prologue_fuse, "settings", lambda: PrologueFuseSettings(enabled=True)
    )
    got = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)


@pytest.mark.parametrize("mode", ["all", "none"])
def test_mamba_tail_block_table_passthrough_modes(fuse_on, mode):
    block_table = torch.arange(24, dtype=torch.int32).view(4, 6)
    spec = _mamba_spec(4, 2)
    seq_lens = torch.tensor([1, 5, 9, 0], dtype=torch.int32)
    out = mamba_get_block_table_tensor(block_table, seq_lens, spec, mode)
    assert out is block_table


def test_mamba_tail_block_table_ref_is_the_upstream_expression():
    """The reference the Triton kernel transcribes must equal upstream."""
    block_table = torch.randint(0, 999, (5, 32), dtype=torch.int32)
    seq_lens = torch.tensor([0, 1, 7, 8, 231], dtype=torch.int32)
    spec = _mamba_spec(8, 3)
    assert torch.equal(
        prologue_fuse.mamba_tail_block_table_ref(block_table, seq_lens, 8, 3),
        mamba_get_block_table_tensor(block_table, seq_lens, spec, "align"),
    )


# --------------------------------------------------------------------------- #
# 2. GDN spec-decode metadata
# --------------------------------------------------------------------------- #


class _FakeGDNBuilder:
    """Just the attributes the fused GDN path reads."""

    def __init__(self, num_spec=3, max_bs=128, use_spec=True, full_graph=True):
        self.num_spec = num_spec
        self.use_spec_decode = use_spec
        self.use_full_cuda_graph = full_graph
        self.decode_cudagraph_max_bs = max_bs
        self.spec_state_indices_tensor = torch.zeros(
            (max_bs, num_spec + 1), dtype=torch.int32
        )
        self.non_spec_state_indices_tensor = torch.zeros(max_bs, dtype=torch.int32)
        self.spec_sequence_masks = torch.zeros(max_bs, dtype=torch.bool)
        self.spec_token_indx = torch.zeros(max_bs * (num_spec + 1), dtype=torch.int32)
        self.non_spec_token_indx = torch.zeros(
            max_bs * (num_spec + 1), dtype=torch.int32
        )
        self.spec_query_start_loc = torch.zeros(max_bs + 1, dtype=torch.int32)
        self.num_accepted_tokens = torch.zeros(max_bs, dtype=torch.int32)


def _upstream_gdn_buffers(builder, block_table, qsl, acc, mask_cpu, batch_size):
    """Verbatim transcription of the upstream buffer-writing block."""
    s = int(mask_cpu.sum().item())
    num_spec = builder.num_spec
    spec_state_indices_tensor = block_table[mask_cpu, : num_spec + 1]
    spec_sequence_masks = mask_cpu.clone()
    spec_token_size = min(s * (num_spec + 1), int(qsl[-1].item()))
    spec_token_indx = torch.arange(spec_token_size, dtype=torch.int32)
    spec_query_start_loc = qsl[: s + 1]
    num_accepted_tokens = acc[mask_cpu]

    builder.spec_state_indices_tensor[:s].copy_(spec_state_indices_tensor)
    out_state = builder.spec_state_indices_tensor[:batch_size]
    out_state[s:].fill_(NULL_BLOCK_ID)

    builder.spec_sequence_masks[:s].copy_(spec_sequence_masks[:s])
    out_masks = builder.spec_sequence_masks[:batch_size]
    out_masks[s:] = False

    builder.spec_token_indx[: spec_token_indx.size(0)].copy_(spec_token_indx)
    out_tokens = builder.spec_token_indx[: spec_token_indx.size(0)]

    builder.spec_query_start_loc[: s + 1].copy_(spec_query_start_loc)
    spec_num_query_tokens = spec_query_start_loc[-1]
    out_qsl = builder.spec_query_start_loc[: batch_size + 1]
    out_qsl[s + 1 :].fill_(spec_num_query_tokens)

    builder.num_accepted_tokens[:s].copy_(num_accepted_tokens)
    out_acc = builder.num_accepted_tokens[:batch_size]
    out_acc[s:].fill_(1)
    return out_state, out_masks, out_tokens, out_qsl, out_acc


class _FakeCommon:
    def __init__(self, num_reqs, qsl, qsl_cpu, num_actual_tokens):
        self.num_reqs = num_reqs
        self.query_start_loc = qsl
        self.query_start_loc_cpu = qsl_cpu
        self.num_actual_tokens = num_actual_tokens


@pytest.mark.parametrize("num_spec", [1, 3, 5])
@pytest.mark.parametrize("nreal,npad", [(1, 0), (1, 3), (4, 4), (8, 0)])
def test_gdn_fused_metadata_matches_upstream(fuse_on, num_spec, nreal, npad):
    torch.manual_seed(num_spec * 31 + nreal * 7 + npad)
    k = num_spec + 1
    batch_size = nreal + npad
    query_lens = [k] * nreal + [0] * npad
    qsl_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0, out=qsl_cpu[1:])
    qsl = qsl_cpu.clone()
    block_table = torch.randint(1, 5000, (batch_size, k + 2), dtype=torch.int32)
    acc = torch.randint(1, k + 1, (batch_size,), dtype=torch.int32)
    draft_counts = torch.tensor(
        [num_spec] * nreal + [-1] * npad, dtype=torch.int32
    )
    mask_cpu = draft_counts >= 0

    ref_builder = _FakeGDNBuilder(num_spec=num_spec)
    want = _upstream_gdn_buffers(
        ref_builder, block_table, qsl, acc, mask_cpu, batch_size
    )

    builder = _FakeGDNBuilder(num_spec=num_spec)
    m = _FakeCommon(batch_size, qsl, qsl_cpu, int(qsl_cpu[-1].item()))
    md = prologue_fuse.try_build_gdn_spec_decode(
        builder, m, block_table, acc, draft_counts
    )
    assert md is not None
    got = (
        md.spec_state_indices_tensor,
        md.spec_sequence_masks,
        md.spec_token_indx,
        md.spec_query_start_loc,
        md.num_accepted_tokens,
    )
    names = ["state_indices", "sequence_masks", "token_indx", "qsl", "num_accepted"]
    for name, g, w in zip(names, got, want):
        assert g.shape == w.shape, name
        assert torch.equal(g, w), (name, g, w)

    # scalar fields
    assert md.num_prefills == 0 and md.num_decodes == 0
    assert md.num_spec_decodes == nreal
    assert md.num_spec_decode_tokens == nreal * k
    assert md.num_actual_tokens == m.num_actual_tokens
    assert md.non_spec_state_indices_tensor is None
    assert md.non_spec_query_start_loc is None
    assert md.non_spec_token_indx.numel() == 0
    assert md.has_initial_state is None
    assert md.chunk_indices is None and md.chunk_offsets is None


def _plan(builder, batch_size, query_lens, draft_counts):
    qsl_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0, out=qsl_cpu[1:])
    return prologue_fuse.gdn_spec_decode_plan(
        builder, batch_size, qsl_cpu, torch.tensor(draft_counts, dtype=torch.int32)
    )


def test_gdn_plan_accepts_the_steady_decode_shape():
    b = _FakeGDNBuilder()
    assert _plan(b, 4, [4, 4, 0, 0], [3, 3, -1, -1]) == {
        "num_spec_decodes": 2,
        "num_spec_decode_tokens": 8,
        "spec_token_size": 8,
    }


@pytest.mark.parametrize(
    "batch_size,query_lens,draft_counts,why",
    [
        (4, [4, 4, 1, 0], [3, 3, -1, -1], "a plain decode row is present"),
        (4, [4, 4, 9, 0], [3, 3, -1, -1], "a prefill row is present"),
        (4, [4, 0, 4, 0], [3, -1, 3, -1], "spec rows are not a leading run"),
        (2, [0, 0], [-1, -1], "no spec rows at all"),
        (2, [1, 1], [0, 0], "no draft tokens"),
    ],
)
def test_gdn_plan_rejects_other_shapes(batch_size, query_lens, draft_counts, why):
    b = _FakeGDNBuilder()
    assert _plan(b, batch_size, query_lens, draft_counts) is None, why


def test_gdn_plan_rejects_without_full_cudagraphs():
    b = _FakeGDNBuilder(full_graph=False)
    assert _plan(b, 2, [4, 4], [3, 3]) is None


def test_gdn_plan_rejects_without_spec_decode():
    b = _FakeGDNBuilder(use_spec=False)
    assert _plan(b, 2, [4, 4], [3, 3]) is None
    b = _FakeGDNBuilder()
    qsl_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
    assert prologue_fuse.gdn_spec_decode_plan(b, 2, qsl_cpu, None) is None


def test_gdn_plan_rejects_oversized_batch():
    b = _FakeGDNBuilder(max_bs=2)
    assert _plan(b, 4, [4, 4, 4, 4], [3, 3, 3, 3]) is None


def test_try_build_returns_none_when_flag_is_off(monkeypatch):
    monkeypatch.setattr(
        prologue_fuse, "settings", lambda: PrologueFuseSettings(enabled=False)
    )
    b = _FakeGDNBuilder()
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    m = _FakeCommon(2, qsl, qsl, 8)
    assert (
        prologue_fuse.try_build_gdn_spec_decode(
            b,
            m,
            torch.ones(2, 6, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            torch.tensor([3, 3], dtype=torch.int32),
        )
        is None
    )


# --------------------------------------------------------------------------- #
# GPU: the Triton kernels against their references.
#
# These are the phase-2 tests -- they are the only thing that exercises the
# Triton code, since on CPU the wrappers dispatch to the reference. Run with
#   CUDA_VISIBLE_DEVICES=<one free gpu> pytest ... -k triton
# Everything above runs on CPU and must be run with CUDA_VISIBLE_DEVICES=""
# so this block is skipped.
# --------------------------------------------------------------------------- #

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton path requires CUDA"
)


@cuda_only
@pytest.mark.parametrize("col_step", [1, 2])
@pytest.mark.parametrize("block_size", [1, 16, 1152])
@pytest.mark.parametrize("num_spec_blocks", [0, 2, 3])
def test_triton_mamba_tail_block_table(col_step, block_size, num_spec_blocks):
    """Covers the strided block-table view that [:, ::2] produces."""
    dev = torch.device("cuda")
    torch.manual_seed(block_size + num_spec_blocks + col_step)
    nreqs, ncols = 65, 128
    storage = torch.arange(nreqs * ncols, dtype=torch.int32, device=dev).reshape(
        nreqs, ncols
    )
    block_table = storage[:, ::col_step]
    seq_lens = (
        torch.tensor(
            [0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 513],
            dtype=torch.int32,
            device=dev,
        )
        .repeat(6)[:nreqs]
        .clamp(max=block_size * (block_table.shape[1] - num_spec_blocks - 1))
    )
    want = prologue_fuse.mamba_tail_block_table_ref(
        block_table, seq_lens, block_size, num_spec_blocks
    )
    got = prologue_fuse.mamba_tail_block_table(
        block_table, seq_lens, block_size, num_spec_blocks
    )
    assert torch.equal(got, want)


@cuda_only
@pytest.mark.parametrize("num_spec", [1, 3, 5])
@pytest.mark.parametrize("nreal,npad", [(1, 0), (1, 3), (4, 4), (8, 0)])
def test_triton_gdn_spec_decode_meta(num_spec, nreal, npad):
    dev = torch.device("cuda")
    torch.manual_seed(num_spec * 31 + nreal * 7 + npad)
    k = num_spec + 1
    batch_size = nreal + npad
    qsl = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
    torch.cumsum(
        torch.tensor([k] * nreal + [0] * npad, dtype=torch.int32, device=dev),
        0,
        out=qsl[1:],
    )
    block_table = torch.randint(
        1, 5000, (batch_size, k + 2), dtype=torch.int32, device=dev
    )
    acc = torch.randint(1, k + 1, (batch_size,), dtype=torch.int32, device=dev)
    spec_token_size = min(nreal * k, int(qsl[-1].item()))

    def _buffers():
        return dict(
            state_idx=torch.zeros((batch_size, k), dtype=torch.int32, device=dev),
            masks=torch.zeros(batch_size, dtype=torch.bool, device=dev),
            token_indx=torch.zeros(batch_size * k, dtype=torch.int32, device=dev),
            qsl_out=torch.zeros(batch_size + 1, dtype=torch.int32, device=dev),
            acc_out=torch.zeros(batch_size, dtype=torch.int32, device=dev),
        )

    want, got = _buffers(), _buffers()
    args = (nreal, batch_size, spec_token_size, k, -1)
    prologue_fuse.gdn_spec_decode_meta_ref(
        block_table, qsl, acc, *want.values(), *args
    )
    prologue_fuse._gdn_spec_decode_meta(block_table, qsl, acc, *got.values(), *args)
    for name in want:
        assert torch.equal(got[name], want[name]), name


def test_debug_flag_is_retired():
    """VLLM_GLM5_PROLOGUE_FUSE_DEBUG was parsed into a field nothing read.

    It is gone from both readers: envs.py no longer declares it (so setting it
    draws the usual unknown-variable warning instead of silently doing
    nothing) and the settings object has no ``debug`` field.
    """
    import dataclasses

    import vllm.envs as envs

    assert "VLLM_GLM5_PROLOGUE_FUSE_DEBUG" not in envs.environment_variables
    fields = {f.name for f in dataclasses.fields(prologue_fuse.PrologueFuseSettings)}
    assert fields == {"enabled", "gdn", "mamba_block_table"}
    # A stale export of the retired name must not change the parsed settings.
    base = prologue_fuse.read_settings({"VLLM_GLM5_PROLOGUE_FUSE": "1"})
    stale = prologue_fuse.read_settings(
        {"VLLM_GLM5_PROLOGUE_FUSE": "1", "VLLM_GLM5_PROLOGUE_FUSE_DEBUG": "1"}
    )
    assert base == stale
