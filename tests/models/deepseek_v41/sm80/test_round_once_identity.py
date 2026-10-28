# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks that the attention ``wo_b`` sum under
VLLM_SM80_ALLREDUCE_ROUND_ONCE and the MoE output sum under the same switch
are the same once-rounded function
of the ranks' BF16 partials wherever both round once, and where they differ.

Four ranks' partials are reduced through the layers' own entry points
(``DeepseekV41Sm80Attention.reduce_wo_b`` and ``Sm80DeepseekV41MoE.reduce``)
over a fake TP group whose transports compute their real arithmetic on every
rank's partial: the two-stage custom all-reduce (FP32 in the order p, p+1, ...
from the element's owner partition p, rounded once), the host-staged one (FP32
in rank order, rounded once), the copy-engine ring's owner sum (FP32 in rank
order, rounded once), the custom all-reduce in pieces, and NCCL's ring (one
rounding per hop in the message dtype).

    pytest -q tests/models/deepseek_v41/sm80/test_round_once_identity.py
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_sm80.models.deepseek_v41 import sm80_attention as sm80
from vllm_sm80.models.deepseek_v41 import sm80_moe
from vllm_sm80.distributed import ce_ring_all_reduce as ce

from .exact_round import exact_sum, reference_round
from .test_sm80_attention_wiring import _bare_layer
from .test_sm80_moe_allreduce import _bare_moe

BF16 = torch.bfloat16
WORLD = 4


def _partials(rows: int, cols: int = 64, seed: int = 7) -> list[torch.Tensor]:
    """Rank-distinct BF16 partials (the wo_b bits check's per-rank scales
    2^-6..2^3); every other element instead holds a large value, its negation
    and two values 2^-26..2^-30 of it at random ranks, so its FP32 sum keeps
    the small ones only in some orders and the summation order shows in the
    rounded bits."""
    g = torch.Generator().manual_seed(seed)
    n = rows * cols
    base = torch.randn(WORLD, n, generator=g) * torch.exp2(
        torch.randint(-6, 4, (WORLD, n), generator=g).float()
    )
    big = torch.exp2(torch.randint(0, 10, (n,), generator=g).float()) * (
        1 + torch.rand(n, generator=g)
    )
    small = big * torch.exp2(-torch.randint(26, 31, (n,), generator=g).float())
    pattern = torch.stack([big, -big, small, 1.5 * small])
    order = torch.argsort(torch.rand(WORLD, n, generator=g), dim=0)
    values = torch.where(
        torch.arange(n) % 2 == 0, torch.gather(pattern, 0, order), base
    )
    return [v.view(rows, cols).to(BF16) for v in values]


def _rank_order(parts: list[torch.Tensor]) -> torch.Tensor:
    """FP32 sum in rank order, rounded once (host-staged; ring owner sum)."""
    return ce.emulate_owner(parts)


def _two_stage(parts: list[torch.Tensor]) -> torch.Tensor:
    """cross_device_reduce_2stage (and the flags-in-data kernel): the message
    cut into WORLD partitions of 16-byte packs, the last one taking the rest;
    an element of partition p summed in FP32 in the order p, p+1, ... and
    rounded once."""
    flat = [p.reshape(-1).float() for p in parts]
    n = flat[0].numel()
    per_pack = 16 // parts[0].element_size()
    part = n // per_pack // WORLD
    assert part > 0
    owner = (torch.arange(n) // per_pack // part).clamp(max=WORLD - 1)
    out = torch.empty(n)
    for p in range(WORLD):
        acc = flat[p].clone()
        for k in range(1, WORLD):
            acc = acc + flat[(p + k) % WORLD]
        out[owner == p] = acc[owner == p]
    return out.to(parts[0].dtype).view(parts[0].shape)


def _ring(parts: list[torch.Tensor]) -> torch.Tensor:
    """NCCL's ring: one rounding per hop in the message dtype."""
    return ce.emulate(parts).view(parts[0].shape)


class _Group:
    """A four-rank TP group seen from rank 0, holding every rank's partial.

    ``host_max`` / ``ca_max``: the largest message in bytes the host-staged
    path takes / the buffer of the custom all-reduce (0: absent);
    ``preempt``: a backend dispatched before the custom all-reduce is on;
    ``ring``: the copy-engine ring is built and takes eager messages.
    """

    def __init__(self, parts, *, host_max=0, ca_max=0, preempt=False, ring=False):
        self.parts = parts
        self.host_max = host_max
        self.paths: list[str] = []
        self.comm = SimpleNamespace(
            hostshm_comm=SimpleNamespace(should_host_ar=self.host_takes)
            if host_max else None,
            ca_comm=SimpleNamespace(
                disabled=False, max_size=ca_max,
                should_custom_ar=self.custom_takes, all_reduce=self.custom_piece,
            ) if ca_max else None,
            pynccl_comm=None,
            fi_ar_comm=SimpleNamespace(disabled=False) if preempt else None,
        )
        self.ring = SimpleNamespace(
            takes=lambda t: t.dtype in ce.DTYPES and t.is_contiguous(),
            round_once=self.ring_owner_sum, all_reduce=self.ring_all_reduce,
        ) if ring else None

    def ranks(self, t: torch.Tensor) -> list[torch.Tensor]:
        """Every rank's tensor at the place of rank 0's ``t``: a view of its
        partial (whole or a piece), or the FP32 copy of the whole partial."""
        p0 = self.parts[0]
        if t.dtype == p0.dtype and t.untyped_storage().data_ptr() == (
            p0.untyped_storage().data_ptr()
        ):
            off = (t.data_ptr() - p0.data_ptr()) // t.element_size()
            return [p.reshape(-1)[off:off + t.numel()].view(t.shape) for p in self.parts]
        assert t.dtype == torch.float32 and t.shape == p0.shape
        assert torch.equal(t, p0.float())
        return [p.float() for p in self.parts]

    def host_takes(self, t) -> bool:
        return t.numel() * t.element_size() <= self.host_max

    def custom_takes(self, t) -> bool:
        size = t.numel() * t.element_size()
        return size % 16 == 0 and t.is_contiguous() and size < self.comm.ca_comm.max_size

    def custom_piece(self, inp, *, out=None, registered=False):
        self.paths.append("custom_piece")
        out.copy_(_two_stage(self.ranks(inp)))
        return out

    def ring_owner_sum(self, t):
        self.paths.append("ring_owner_sum")
        return _rank_order(self.ranks(t))

    def ring_all_reduce(self, t):
        self.paths.append(f"ring_{t.dtype}")
        return _ring(self.ranks(t))

    def all_reduce(self, t):
        """The communicator's dispatch: host-staged, else the custom
        all-reduce unless a backend before it is on, else NCCL."""
        ranks = self.ranks(t)
        comm = self.comm
        if comm.hostshm_comm is not None and self.host_takes(t):
            self.paths.append(f"host_{t.dtype}")
            return _rank_order(ranks)
        if comm.fi_ar_comm is None and comm.ca_comm is not None and self.custom_takes(t):
            self.paths.append(f"custom_{t.dtype}")
            return _two_stage(ranks)
        self.paths.append(f"nccl_{t.dtype}")
        return _ring(ranks)

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sm80, "get_tp_group", lambda: SimpleNamespace(device_communicator=self.comm)
        )
        monkeypatch.setattr(sm80, "tensor_model_parallel_all_reduce", self.all_reduce)
        monkeypatch.setattr(sm80_moe, "tensor_model_parallel_all_reduce", self.all_reduce)
        monkeypatch.setattr(ce, "_RING", self.ring)


def _pieces_reference(parts, ca_max):
    """Each piece of the custom all-reduce in pieces reduced on its own."""
    numel = (ca_max - 1) // 16 * (16 // parts[0].element_size())
    flat = [p.reshape(-1) for p in parts]
    out = [
        _two_stage([f[s:s + numel] for f in flat])
        for s in range(0, flat[0].numel(), numel)
    ]
    return torch.cat(out).view(parts[0].shape)


def _both_consumers(group, monkeypatch, round_once=True):
    """The wo_b sum and the MoE output sum of rank 0's partial, each through
    its layer's own entry point, with its round-once switch ``round_once``."""
    group.install(monkeypatch)
    layer = _bare_layer(_wo_b_round_once=round_once)
    ffn = _bare_moe()
    sm80_moe._adopt_layer_all_reduce(ffn, round_once=round_once)
    wo_b = layer.reduce_wo_b(group.parts[0])
    wo_b_paths = list(group.paths)
    group.paths.clear()
    moe = ffn.reduce(group.parts[0])
    return wo_b, moe, wo_b_paths, list(group.paths)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


ROWS = 64  # 8 KiB BF16 messages, 16 KiB FP32


@pytest.mark.parametrize(
    "case,kw,wo_b_path,moe_path,reference",
    [
        ("custom_whole", dict(ca_max=1 << 20), ["custom_torch.bfloat16"],
         ["custom_torch.bfloat16"], _two_stage),
        # The wo_b sum sends FP32 where the host-staged path takes the message;
        # it accumulates the same exact FP32 values in the same rank order.
        ("host_staged", dict(host_max=1 << 20, ca_max=1 << 20),
         ["host_torch.float32"], ["host_torch.bfloat16"], _rank_order),
        ("ring_owner_sum", dict(ca_max=1024, ring=True), ["ring_owner_sum"],
         ["ring_owner_sum"], _rank_order),
        ("pieces", dict(ca_max=1024), None, None, None),
    ],
)
def test_wo_b_and_moe_round_once_sums_are_bitwise_identical(
    monkeypatch, case, kw, wo_b_path, moe_path, reference
) -> None:
    """Every transport the served TP4 launch reaches, and the host-staged one:
    the same bits, each the transport's once-rounded sum."""
    parts = _partials(ROWS)
    wo_b, moe, wo_b_paths, moe_paths = _both_consumers(_Group(parts, **kw), monkeypatch)
    if wo_b_path is None:
        assert wo_b_paths == moe_paths
        assert set(wo_b_paths) == {"custom_piece"} and len(wo_b_paths) > 1
        want = _pieces_reference(parts, kw["ca_max"])
    else:
        assert wo_b_paths == wo_b_path and moe_paths == moe_path
        want = reference(parts)
    assert wo_b.dtype == moe.dtype == BF16
    assert torch.equal(_bits(wo_b), _bits(moe))
    assert torch.equal(_bits(wo_b), _bits(want))


def test_where_no_round_once_transport_takes_the_message(monkeypatch) -> None:
    """A backend dispatched before the custom all-reduce (or CUDA graph
    capture of a message too large for it whole): the wo_b sum is the FP32
    message rounded once; the MoE output keeps the runner's BF16 all-reduce,
    which rounds every hop, so the two sums differ only where no round-once
    transport takes the message."""
    parts = _partials(ROWS)
    group = _Group(parts, ca_max=1 << 20, preempt=True)
    wo_b, moe, wo_b_paths, moe_paths = _both_consumers(group, monkeypatch)
    assert wo_b_paths == ["nccl_torch.float32"] and moe_paths == ["nccl_torch.bfloat16"]
    assert torch.equal(_bits(wo_b), _bits(_ring([p.float() for p in parts]).to(BF16)))
    assert torch.equal(_bits(moe), _bits(_ring(parts)))
    assert not torch.equal(_bits(wo_b), _bits(moe))


@pytest.mark.parametrize(
    "case,kw",
    [("custom_whole", dict(ca_max=1 << 20)),
     ("host_staged", dict(host_max=1 << 20, ca_max=1 << 20))],
)
def test_bf16_message_gives_the_fp32_message_bits(monkeypatch, case, kw) -> None:
    """Where the transport takes both messages whole, the round-once BF16 wo_b
    sum is the switch-off sum (the FP32 message rounded once) bit for bit:
    both accumulate the exact FP32 values of the partials in the same order."""
    parts = _partials(ROWS)
    on = _both_consumers(_Group(parts, **kw), monkeypatch)[0]
    off = _both_consumers(_Group(parts, **kw), monkeypatch, round_once=False)[0]
    assert torch.equal(_bits(on), _bits(off))


def test_the_partials_tell_the_orders_apart() -> None:
    """The data above is discriminating: rank order, the two-stage order and
    the correctly rounded exact sum differ somewhere, and NCCL's BF16 ring
    differs from every once-rounded sum."""
    parts = _partials(ROWS)
    exact = reference_round(exact_sum(parts), BF16)
    rank, two, ring = (_bits(f(parts)) for f in (_rank_order, _two_stage, _ring))
    assert not torch.equal(rank, two)
    assert not torch.equal(rank, _bits(exact))
    for once in (rank, two):
        assert not torch.equal(ring, once)
