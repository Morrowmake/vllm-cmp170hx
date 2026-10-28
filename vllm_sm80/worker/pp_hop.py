# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed pipeline-parallel hand-off (V2 model runner).

The default hop (``GroupCoordinator.isend_tensor_dict``) pickles the tensor
metadata, sends it over the gloo CPU group, and then issues one device send
per tensor; the receiver blocks on the gloo receive before it can post its
device receives. A GLM-5.3-Flash boundary carries the mHC residual streams
plus up to four drafter aux states, i.e. up to five device ops and a CPU
round trip per hop and step.

``VLLM_PP_PACKED_HOP=1``: every tensor of the hop is copied into one flat
byte buffer (each segment 16-byte aligned) and sent with a single device op;
the receiver posts one receive and hands out views. The metadata still
travels (keys, per-row shapes, dtypes and the row count) and is checked
against the receiver's own layout.

``VLLM_PP_HOP_NO_METADATA=1`` (with the above): after a first-hop handshake
that exchanges and checks the metadata once, no metadata is sent at all. The
layout is static per boundary (the receiver's persistent receive buffer
describes it) and the row count is the step's scheduled token count, which
both ranks read from the same scheduler output; the sender sends exactly those
rows (the CUDA-graph padding rows are not sent, the receiver keeps whatever its
persistent buffer holds there, and nothing reads padding rows back). If either
side cannot name the row count for a step, both fall back to sending the
metadata for that step (the decision is a pure function of the shared
configuration, so both sides take the same branch).
"""

from collections import deque
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

_ALIGN = 16
_TAG = "__pp_packed_hop__"


@dataclass(frozen=True)
class HopEntry:
    key: str
    row_shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def row_nbytes(self) -> int:
        return prod(self.row_shape) * self.dtype.itemsize


class HopLayout:
    """Byte layout of one hop: entries sorted by key, each segment aligned."""

    def __init__(self, entries: list[HopEntry] | tuple[HopEntry, ...]):
        self.entries = tuple(sorted(entries, key=lambda e: e.key))
        keys = [e.key for e in self.entries]
        assert len(keys) == len(set(keys)), keys

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor]) -> "HopLayout":
        return cls(
            [
                HopEntry(key, tuple(t.shape[1:]), t.dtype)
                for key, t in tensors.items()
                if isinstance(t, torch.Tensor)
            ]
        )

    def signature(self) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        return tuple((e.key, e.row_shape, str(e.dtype)) for e in self.entries)

    def segments(self, rows: int) -> list[tuple[HopEntry, int, int]]:
        out = []
        offset = 0
        for e in self.entries:
            nbytes = rows * e.row_nbytes
            out.append((e, offset, nbytes))
            offset += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
        return out

    def nbytes(self, rows: int) -> int:
        segs = self.segments(rows)
        if not segs:
            return 0
        _, offset, nbytes = segs[-1]
        return (offset + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN

    def pack(self, tensors: dict[str, torch.Tensor], rows: int) -> torch.Tensor:
        keys = {k for k, v in tensors.items() if isinstance(v, torch.Tensor)}
        assert keys == {e.key for e in self.entries}, (sorted(keys), self.signature())
        device = next(iter(tensors.values())).device
        flat = torch.empty(self.nbytes(rows), dtype=torch.uint8, device=device)
        for e, offset, nbytes in self.segments(rows):
            src = tensors[e.key]
            assert tuple(src.shape[1:]) == e.row_shape and src.dtype == e.dtype, (
                e,
                src.shape,
                src.dtype,
            )
            assert src.shape[0] >= rows, (e.key, src.shape[0], rows)
            flat[offset : offset + nbytes].copy_(
                src[:rows].contiguous().reshape(-1).view(torch.uint8)
            )
        return flat

    def unpack(self, flat: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
        assert flat.dtype == torch.uint8 and flat.numel() == self.nbytes(rows)
        return {
            e.key: flat[offset : offset + nbytes].view(e.dtype).view(rows, *e.row_shape)
            for e, offset, nbytes in self.segments(rows)
        }


class PackedHop:
    """One flat device transfer per hop, optionally without per-step metadata.

    ``group`` is the pipeline GroupCoordinator (``ranks``, ``rank_in_group``,
    ``world_size``, ``device_group``, ``isend_object``, ``recv_object``).
    ``dist`` is injectable for tests (defaults to torch.distributed).
    """

    def __init__(self, group: Any, no_metadata: bool, dist: Any = None):
        self.group = group
        self.no_metadata = no_metadata
        self.dist = dist if dist is not None else torch.distributed
        self._sent_first = False
        self._recv_first = False
        # (device handle, flat buffer, metadata handle) kept until drained.
        self._pending: deque[tuple[Any, torch.Tensor, Any]] = deque()

    def _send_metadata_this_step(self, rows_known: bool, first: bool) -> bool:
        return not (self.no_metadata and rows_known and not first)

    def send(
        self,
        tensors: dict[str, torch.Tensor],
        rows: int | None,
        dst: int | None = None,
    ) -> list[Any]:
        """Send ``tensors[k][:rows]`` for every key; ``rows=None`` means all
        rows (and forces metadata for this step). Returns the device handles
        (the caller waits them before overwriting the source buffers)."""
        g = self.group
        if dst is None:
            dst = (g.rank_in_group + 1) % g.world_size
        layout = HopLayout.from_tensors(tensors)
        rows_known = rows is not None
        if rows is None:
            rows = next(iter(tensors.values())).shape[0]
        first = not self._sent_first
        self._sent_first = True
        meta = None
        if self._send_metadata_this_step(rows_known, first):
            meta = g.isend_object((_TAG, layout.signature(), rows), dst=dst)
        flat = layout.pack(tensors, rows)
        handle = self.dist.isend(flat, dst=g.ranks[dst], group=g.device_group)
        if flat.is_cuda:
            flat.record_stream(torch.cuda.current_stream(flat.device))
        self._reap()
        self._pending.append((handle, flat, meta))
        return [handle]

    def recv(
        self,
        layout: HopLayout,
        rows: int | None,
        src: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[Any]]:
        """Post the receive for one hop. ``layout`` is the receiver's own
        (persistent-buffer) layout; ``rows`` the step's scheduled token count
        or None. Returns views into the receive buffer and the handles."""
        g = self.group
        if src is None:
            src = (g.rank_in_group - 1) % g.world_size
        rows_known = rows is not None
        first = not self._recv_first
        self._recv_first = True
        if self._send_metadata_this_step(rows_known, first):
            tag, signature, sent_rows = g.recv_object(src=src)
            if tag != _TAG or tuple(signature) != layout.signature():
                raise RuntimeError(
                    "PP packed hop: the sender's layout does not match this "
                    f"stage's receive buffer: sent {signature}, expected "
                    f"{layout.signature()}"
                )
            if rows_known and sent_rows != rows:
                raise RuntimeError(
                    f"PP packed hop: sender sent {sent_rows} rows, this stage "
                    f"expected {rows}"
                )
            rows = sent_rows
        assert rows is not None
        flat = torch.empty(layout.nbytes(rows), dtype=torch.uint8, device=device)
        handle = self.dist.irecv(flat, src=g.ranks[src], group=g.device_group)
        return layout.unpack(flat, rows), [handle]

    def _reap(self) -> None:
        # Device completion is reliable; drop entries whose device send is done
        # (their metadata send, if any, is then done too: the receiver reads
        # the metadata before it posts the matching device receive).
        while self._pending:
            handle, _, meta = self._pending[0]
            done = getattr(handle, "is_completed", None)
            if done is None or not done():
                break
            if meta is not None:
                meta.wait()
            self._pending.popleft()
