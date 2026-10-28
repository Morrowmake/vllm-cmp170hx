# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of how the sm_80 DeepSeek-V4.1 attention plugs into vLLM:
platform selection, the DSpark context-KV hook, RoPE parameters from the
config, metadata gathering, and the output projection.

    pytest -q tests/models/deepseek_v41/sm80/test_sm80_attention_wiring.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v41.nvidia import model as dsv41_model
from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4Model
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .dsv41_ref import apply_rotary_emb, precompute_freqs_cis

BF16 = torch.bfloat16


class _Platform:
    def __init__(self, major: int):
        self.major = major

    def is_cuda(self) -> bool:
        return True

    def get_device_capability(self) -> DeviceCapability:
        return DeviceCapability(self.major, 0)


def test_selector_takes_sm80_on_compute_capability_8(monkeypatch) -> None:
    cfg = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV41
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    monkeypatch.setattr(sm80, "current_platform", _Platform(8))
    assert (
        dsv41_model._select_dsv4_attn_cls(cfg) is sm80.DeepseekV41Sm80Attention
    )
    monkeypatch.setattr(sm80, "current_platform", _Platform(9))
    assert dsv41_model._select_dsv4_attn_cls(cfg) is DeepseekV4FlashMLAAttention


def test_backends() -> None:
    cap8, cap9 = DeviceCapability(8, 0), DeviceCapability(9, 0)
    for backend in (sm80.Sm80SparseDsv41Backend, sm80.Sm80IndexKBackend):
        assert backend.supports_compute_capability(cap8)
        assert not backend.supports_compute_capability(cap9)
        assert set(backend.supported_kv_cache_dtypes) == {"auto", "bfloat16"}
    assert sm80.Sm80IndexKBackend.get_supported_head_sizes() == [128]
    assert sm80.Sm80SparseDsv41Backend.get_supported_head_sizes() == [512]
    assert sm80.Sm80SWABackend.get_builder_cls() is sm80.Sm80SWAMetadataBuilder
    assert not sm80.DeepseekV41Sm80Attention.use_fp8_ds_mla_layout
    assert sm80.DeepseekV41Sm80Attention.get_padded_num_q_heads(16) == 16


def _config(nested: bool):
    yarn = {
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }
    rp = {"main": {"rope_type": "default"}, "compress": yarn} if nested else yarn
    return SimpleNamespace(rope_theta=10000, compress_rope_theta=160000, rope_parameters=rp)


@pytest.mark.parametrize("nested", [False, True])
def test_rope_params_and_rotation_match_reference(nested: bool) -> None:
    cfg = _config(nested)
    assert sm80._rope_params(cfg, 0, 64)[:3] == (64, 0, 10000.0)
    assert sm80._rope_params(cfg, 2, 64) == (64, 65536, 160000.0, 16.0, 32, 1)
    assert sm80._rope_params(cfg, 1, 64) == sm80._rope_params(cfg, 2, 64)
    g = torch.Generator().manual_seed(0)
    positions = torch.cat([torch.arange(70), torch.randint(0, 300000, (60,))])
    for ratio in (0, 2):
        params = sm80._rope_params(cfg, ratio, 64)
        freqs = core.rope_freqs(*params)
        cis = precompute_freqs_cis(64, 300001, *params[1:])[positions]
        x = torch.randn(positions.numel(), 4, 512, generator=g).to(BF16)
        want = x.clone().unsqueeze(0)
        # The reference rotates [b, s, h, d] with one row of cis per s.
        apply_rotary_emb(want[..., -64:], cis)
        got = core.rope(x, positions, freqs)
        assert torch.equal(got.view(torch.int16), want[0].view(torch.int16))
        back = core.rope(got, positions, freqs, inverse=True)
        apply_rotary_emb(want[..., -64:], cis, inverse=True)
        assert torch.equal(back.view(torch.int16), want[0].view(torch.int16))


def _bare_layer(**attrs) -> sm80.DeepseekV41Sm80Attention:
    layer = object.__new__(sm80.DeepseekV41Sm80Attention)
    nn.Module.__init__(layer)
    for k, v in attrs.items():
        object.__setattr__(layer, k, v)
    return layer


def test_batch_gathers_decode_and_prefill_rows() -> None:
    nd, npf = 2, 3
    dec = torch.arange(10 * 4, dtype=torch.int32).view(10, 1, 4)
    pre = torch.arange(100, 100 + 10 * 6, dtype=torch.int32).view(10, 1, 6)
    swa = SimpleNamespace(
        num_decode_tokens=nd,
        num_prefill_tokens=npf,
        decode_swa_indices=dec[:nd],
        prefill_swa_indices=pre[:npf],
        token_to_req_indices=torch.tensor([0, 1, 2, 2, 2, 9, 9]),
        query_start_loc=torch.tensor([0, 1, 2, 5]),
        slot_mapping=torch.tensor([5, 6, 7, 8, 9, -1, -1]),
    )
    comp = SimpleNamespace(
        block_table=torch.ones(3, 2), slot_mapping=torch.arange(7), max_seq_len=45
    )
    idxk = SimpleNamespace(block_table=torch.zeros(3, 2), slot_mapping=torch.arange(7) + 1)
    ring = SimpleNamespace(slot_mapping=torch.arange(7) + 2)
    layer = _bare_layer(
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        compress_ratio=2,
        compressed_cache_prefix="comp",
        is_kv_source=True,
        indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="idxk")),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="ring")),
    )
    md = {"swa": swa, "comp": comp, "idxk": idxk, "ring": ring}
    positions = torch.arange(7) + 40
    batch = layer._sm80_batch(md, positions)
    assert batch.num_tokens == 5
    assert batch.swa_indices.shape == (5, 6)
    assert batch.swa_indices[:2, :4].tolist() == dec[:2, 0].tolist()
    assert (batch.swa_indices[:2, 4:] == -1).all()
    assert batch.swa_indices[2:].tolist() == pre[:3, 0].tolist()
    assert batch.swa_slots.tolist() == [5, 6, 7, 8, 9]
    assert batch.comp_slots.tolist() == [0, 1, 2, 3, 4]
    assert batch.idxk_slots.tolist() == [1, 2, 3, 4, 5]
    assert batch.ring_slots.tolist() == [2, 3, 4, 5, 6]
    assert batch.positions.tolist() == [40, 41, 42, 43, 44]
    assert batch.max_seq_len == 45


class _Linear:
    def __init__(self, weight, reduce_results=True, tp_size=1, grouped=None):
        self.weight = weight
        self.reduce_results = reduce_results
        self.tp_size = tp_size
        self.quant_method = SimpleNamespace(apply=self.apply)
        if grouped is not None:
            self.quant_method.apply_grouped = grouped

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, self.weight)

    def __call__(self, x):
        return self.apply(self, x)


def test_o_proj_grouped_wo_a_and_fp32_reduce(monkeypatch) -> None:
    g = torch.Generator().manual_seed(1)
    T, G, R, D, H = 3, 2, 8, 64, 5
    wo_a = (torch.randn(G * R, D, generator=g) / 8).to(BF16)
    wo_b = (torch.randn(H, G * R, generator=g) / 4).to(BF16)
    o = torch.randn(T, G, D, generator=g).to(BF16)
    z = torch.einsum("tgd,grd->tgr", o, wo_a.view(G, R, D)).flatten(1)
    layer = _bare_layer(
        n_local_groups=G, o_lora_rank=R, gemm_rs=None,
        wo_a=_Linear(wo_a), wo_b=_Linear(wo_b),
    )
    attn_out = o.view(T, G, D)
    got = layer._o_proj(attn_out, torch.arange(T))
    assert torch.equal(got, torch.nn.functional.linear(z, wo_b))

    # wo_b partials of several ranks are reduced in fp32, rounded once.
    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", lambda t: t * 3)
    layer = _bare_layer(
        n_local_groups=G, o_lora_rank=R, gemm_rs=None,
        wo_a=_Linear(wo_a), wo_b=_Linear(wo_b, tp_size=4),
        _wo_b_round_once=False,
    )
    got = layer._o_proj(attn_out, torch.arange(T))
    want = (torch.nn.functional.linear(z, wo_b).float() * 3).to(BF16)
    assert torch.equal(got, want)

    # A quantized wo_a with a grouped apply gets [T, groups, group input].
    seen = {}

    def grouped(layer_, x):
        seen["shape"] = tuple(x.shape)
        return z

    layer = _bare_layer(
        n_local_groups=G, o_lora_rank=R, gemm_rs=None,
        wo_a=_Linear(wo_a, grouped=grouped), wo_b=_Linear(wo_b),
    )
    layer._o_proj(attn_out, torch.arange(T))
    assert seen["shape"] == (T, G, D)


def test_dspark_context_kv_hook_dispatch() -> None:
    calls = []
    sm80_attn = _bare_layer()
    object.__setattr__(
        sm80_attn,
        "sm80_store_context_kv",
        lambda x, p, s: calls.append(("sm80", s)),
    )
    model = SimpleNamespace(layers=[SimpleNamespace(attn=sm80_attn)] * 2)
    store = DSparkDeepseekV4Model.precompute_and_store_context_kv
    x, pos = torch.zeros(2, 4), torch.arange(2)
    store(model, x, pos, [torch.tensor([1]), None])
    assert [c[0] for c in calls] == ["sm80", "sm80"] and calls[1][1] is None

    other = SimpleNamespace(layers=[])
    upstream = store.__wrapped__
    hits = []
    wrapped = sm80.sm80_dspark_context_kv(lambda *a: hits.append(a))
    wrapped(other, x, pos, None)
    assert len(hits) == 1 and upstream is not None


class _FakeComm:
    """A TP communicator whose custom all-reduce accepts what ``takes`` says
    (whole messages and pieces alike)."""

    def __init__(self, takes: bool, host_takes: bool = False, preempt=None):
        self.ca_comm = SimpleNamespace(
            disabled=False, should_custom_ar=lambda t: takes, max_size=1024
        )
        self.hostshm_comm = SimpleNamespace(should_host_ar=lambda t: host_takes)
        self.pynccl_comm = None
        self.fi_ar_comm = preempt


@pytest.mark.parametrize(
    "switch,comm,bf16_message",
    [
        (False, _FakeComm(True), False),
        (True, _FakeComm(True), True),
        (True, _FakeComm(False), False),
        (True, _FakeComm(True, host_takes=True), False),
        (True, _FakeComm(True, preempt=SimpleNamespace(disabled=False)), False),
        (True, _FakeComm(True, preempt=SimpleNamespace(disabled=True)), True),
    ],
)
def test_wo_b_round_once_whole_message(monkeypatch, switch, comm, bf16_message) -> None:
    """VLLM_SM80_ALLREDUCE_ROUND_ONCE sends the BF16 partial to the all-reduce
    when the custom all-reduce's own predicate takes it whole (and the
    host-staged path does not); otherwise with no round-once transport left
    (no ring, no pieces), and with the switch off, the FP32 sum rounded once."""
    seen = []
    monkeypatch.delenv("VLLM_SM80_ALLREDUCE", raising=False)
    monkeypatch.setattr(sm80.ce_ring_all_reduce, "_RING", None)

    def all_reduce(t):
        seen.append(t.dtype)
        return t * 3

    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    g = torch.Generator().manual_seed(2)
    partial = torch.randn(5, 64, generator=g).to(BF16)
    got = sm80.wo_b_all_reduce(partial, switch)
    assert seen == [BF16 if bf16_message else torch.float32]
    assert got.dtype == BF16
    if not bf16_message:
        assert torch.equal(got, (partial.float() * 3).to(BF16))


def test_allreduce_round_once_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_SM80_ALLREDUCE_ROUND_ONCE", raising=False)
    assert envs.VLLM_SM80_ALLREDUCE_ROUND_ONCE is False
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE_ROUND_ONCE", "1")
    assert envs.VLLM_SM80_ALLREDUCE_ROUND_ONCE is True


@pytest.mark.parametrize("on", [False, True])
def test_allreduce_round_once_banner(monkeypatch, on) -> None:
    """The layer reads the switch through one call that says so at start."""
    said = []
    monkeypatch.setattr(sm80.logger, "info_once", lambda msg, *a, **_: said.append(msg))
    monkeypatch.setenv("VLLM_SM80_ALLREDUCE_ROUND_ONCE", "1" if on else "0")
    assert sm80.wo_b_round_once_switch() is on
    assert len(said) == int(on)
    if on:
        assert "VLLM_SM80_ALLREDUCE_ROUND_ONCE=1" in said[0]


class _PieceCustomAllreduce:
    """A custom all-reduce with a ``max_size`` buffer: like upstream's
    ``should_custom_ar`` it takes contiguous float messages of whole 16-byte
    units below the buffer size when the ranks are fully connected. Its
    ``all_reduce`` stands for four ranks holding the same partial: it writes
    the FP32 sum rounded once into ``out`` and records each piece."""

    def __init__(self, max_size: int, fully_connected: bool = True):
        self.max_size = max_size
        self.fully_connected = fully_connected
        self.disabled = False
        self.pieces: list[torch.Tensor] = []

    def should_custom_ar(self, t: torch.Tensor) -> bool:
        size = t.numel() * t.element_size()
        return (
            self.fully_connected
            and t.dtype in (torch.float32, BF16)
            and size % 16 == 0
            and t.is_contiguous()
            and size < self.max_size
        )

    def all_reduce(self, inp, *, out=None, registered=False):
        assert not registered and out is not None
        assert inp.numel() * inp.element_size() < self.max_size
        self.pieces.append(inp)
        out.copy_((inp.float() * 4).to(inp.dtype))
        return out


def _piece_comm(ca, host_takes=False, preempt=None):
    return SimpleNamespace(
        ca_comm=ca,
        hostshm_comm=SimpleNamespace(should_host_ar=lambda t: host_takes),
        pynccl_comm=None,
        fi_ar_comm=preempt,
    )


# 1 KiB buffer: the largest piece is 63 units of 16 bytes (504 BF16 values).
_MAX = 1024


@pytest.mark.parametrize(
    "case,rows,cols,path",
    [
        ("switch_off", 40, 64, "fp32"),
        ("too_large", 40, 64, "pieces"),
        ("one_row_over", 9, 64, "pieces"),
        ("whole_fits", 7, 64, "bf16"),
        ("host_takes", 40, 64, "fp32"),
        ("preempted", 40, 64, "fp32"),
        ("not_fully_connected", 40, 64, "fp32"),
        ("capture_warmup", 40, 64, "fp32"),
        ("not_contiguous", 40, 64, "fp32"),
        ("odd_bytes", 41, 63, "fp32"),
    ],
)
def test_wo_b_round_once_pieces(
    monkeypatch, case, rows, cols, path
) -> None:
    """VLLM_SM80_ALLREDUCE_ROUND_ONCE sends a BF16 partial too large for the
    custom all-reduce whole through it in storage-order pieces below its
    buffer, each taken by its own predicate, into one output; one the custom
    all-reduce takes whole goes to the communicator as BF16; any other
    message keeps the FP32 sum rounded once."""
    ca = _PieceCustomAllreduce(_MAX, fully_connected=case != "not_fully_connected")
    monkeypatch.delenv("VLLM_SM80_ALLREDUCE", raising=False)
    monkeypatch.setattr(sm80.ce_ring_all_reduce, "_RING", None)
    if case == "capture_warmup":
        ca._IS_CAPTURING = True
    comm = _piece_comm(
        ca,
        host_takes=case == "host_takes",
        preempt=SimpleNamespace(disabled=False) if case == "preempted" else None,
    )
    seen = []

    def all_reduce(t):
        seen.append(t.dtype)
        return t * 4

    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    g = torch.Generator().manual_seed(3)
    partial = torch.randn(rows, cols, generator=g).to(BF16)
    if case == "not_contiguous":
        partial = torch.randn(cols, rows, generator=g).to(BF16).t()
    got = sm80.wo_b_all_reduce(partial, case != "switch_off")
    want = (partial.float() * 4).to(BF16)
    assert got.dtype == BF16 and got.shape == partial.shape
    assert torch.equal(got, want)
    if path == "fp32":
        assert seen == [torch.float32] and not ca.pieces
        return
    if path == "bf16":
        assert seen == [BF16] and not ca.pieces
        return
    assert not seen
    sizes = [p.numel() * p.element_size() for p in ca.pieces]
    assert all(s < _MAX and s % 16 == 0 for s in sizes)
    assert sizes[:-1] == [_MAX - 16] * (len(sizes) - 1)
    assert sum(p.numel() for p in ca.pieces) == partial.numel()
    flat = partial.view(-1)
    start = 0
    for p in ca.pieces:
        assert p.data_ptr() == flat[start:].data_ptr()
        start += p.numel()


def test_wo_b_round_once_whole_message_keeps_custom_path(
    monkeypatch,
) -> None:
    """A message the custom all-reduce takes whole goes to the communicator as
    BF16, never in pieces."""
    ca = _PieceCustomAllreduce(_MAX)
    comm = _piece_comm(ca)
    seen = []

    def all_reduce(t):
        seen.append(t.dtype)
        return t

    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    sm80.wo_b_all_reduce(torch.ones(7, 64, dtype=BF16), True)
    assert seen == [BF16] and not ca.pieces


def test_o_proj_passes_the_round_once_switch(monkeypatch) -> None:
    calls = []

    def wo_b_all_reduce(partial, round_once):
        calls.append(round_once)
        return partial

    monkeypatch.setattr(sm80, "wo_b_all_reduce", wo_b_all_reduce)
    H, K = 5, 16
    layer = _bare_layer(
        n_local_groups=1, o_lora_rank=K, gemm_rs=None,
        wo_a=_Linear(torch.eye(K, dtype=BF16)),
        wo_b=_Linear(torch.ones(H, K, dtype=BF16), tp_size=4),
        _wo_b_round_once=True,
    )
    layer._o_proj(torch.ones(2, 1, K, dtype=BF16), torch.arange(2))
    assert calls == [True]


@pytest.mark.parametrize(
    "tp_size,reduce_results,gemm_rs,reduces",
    [(4, True, None, True), (1, True, None, False), (4, False, None, False),
     (4, True, object(), False)],
)
def test_o_proj_is_reduce_of_wo_b_partial_by_rows(
    monkeypatch, tp_size, reduce_results, gemm_rs, reduces
) -> None:
    """``reduces_wo_b`` says when ``_o_proj`` all-reduces; then ``_o_proj`` is
    ``reduce_wo_b(wo_b_partial(.))``, and both are row-wise, so row slices
    reduced apart concatenate to the whole (the prefill split reduction)."""
    monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", lambda t: t * 3)
    g = torch.Generator().manual_seed(2)
    T, G, R, D, H = 6, 2, 8, 16, 5
    layer = _bare_layer(
        n_local_groups=G, o_lora_rank=R, gemm_rs=gemm_rs,
        wo_a=_Linear((torch.randn(G * R, D, generator=g) / 4).to(BF16)),
        wo_b=_Linear((torch.randn(H, G * R, generator=g) / 4).to(BF16),
                     reduce_results=reduce_results, tp_size=tp_size),
        _wo_b_round_once=False,
    )
    assert layer.reduces_wo_b() is reduces
    if not reduces:
        return
    attn_out = torch.randn(T, G, D, generator=g).to(BF16)
    whole = layer._o_proj(attn_out, torch.arange(T))
    assert torch.equal(whole, layer.reduce_wo_b(layer.wo_b_partial(attn_out)))
    halves = [layer.reduce_wo_b(layer.wo_b_partial(attn_out[s]))
              for s in (slice(0, 2), slice(2, T))]
    assert torch.equal(torch.cat(halves), whole)


def _frexp_exponent(x: torch.Tensor) -> torch.Tensor:
    return torch.frexp(x)[1]


def test_wo_b_fp32_sum_of_bf16_partials_is_order_free_within_2_14() -> None:
    """The basis of the round-once BF16 wo_b message: four BF16 partials whose nonzero
    magnitudes lie within a factor 2^14 of each other have an exact FP32 sum
    (every partial sum is a multiple of the smallest one's last bit and needs
    at most 24 bits), so every summation order (the FP32 ring's, the custom
    all-reduce's rotated one, the reference's) rounds the same exact value
    to BF16 once. Elsewhere orders can differ in the last bit."""
    import itertools

    g = torch.Generator().manual_seed(11)
    rows, cols, world = 256, 512, 4
    # The wo_b partials of the bits check: per-rank, per-row scales 2^-6..2^3.
    parts = torch.stack(
        [
            (
                torch.randn(rows, cols, generator=g)
                * torch.exp2(torch.randint(-6, 4, (rows, 1), generator=g).float())
            ).to(BF16)
            for _ in range(world)
        ]
    )
    exact = parts.double().sum(0)
    e = _frexp_exponent(parts.float().abs())
    nonzero = parts != 0
    e_max = torch.where(nonzero, e, torch.full_like(e, -(2**30))).amax(0)
    e_min = torch.where(nonzero, e, torch.full_like(e, 2**30)).amin(0)
    within = (e_max - e_min) <= 14
    assert within.float().mean() > 0.95
    want = exact.float().to(BF16)
    for order in itertools.permutations(range(world)):
        acc = parts[order[0]].float()
        for r in order[1:]:
            acc = acc + parts[r].float()
        assert torch.equal(acc[within], exact.float()[within])
        assert torch.equal(acc.to(BF16)[within], want[within])
