# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only EXL3 header inventory and packed shard validation."""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "F32": 4,
    "I64": 8,
    "F64": 8,
}
# Element sizes of the FP8 dtypes a hybrid checkpoint's second method may
# store (hybrid.py); plain EXL3 checkpoints hold none.
_FP8_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
_MARKERS = {"mcg": (1, 0xCBAC1FED), "mul1": (2, 0x83DCD12D)}
PLE_AUXILIARIES = (
    "head_bias",
    "head_offsets",
    "head_vocab_sizes",
    "layer_multipliers",
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TensorHeader:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_offsets: tuple[int, int]

    @property
    def offset(self) -> int:
        return self.data_start + self.data_offsets[0]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    def read_bytes(self) -> bytes:
        with self.path.open("rb") as file:
            file.seek(self.offset)
            data = file.read(self.nbytes)
        if len(data) != self.nbytes:
            raise ValueError(f"Truncated EXL3 tensor: {self.name}")
        return data


def _read_header(path: Path) -> tuple[dict, int, int]:
    """Return the parsed safetensors header, its payload start and file size."""
    size = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if length > min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        header = json.loads(file.read(length), object_pairs_hook=_unique_object)
    return header, 8 + length, size


def read_metadata(path: Path) -> dict[str, str]:
    metadata = _read_header(path)[0].get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors metadata: {path.name}")
    return metadata


def read_headers(
    path: Path, dtype_bytes: dict[str, int] = _DTYPE_BYTES
) -> dict[str, TensorHeader]:
    header, start, size = _read_header(path)
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, offsets = (entry["dtype"], entry["shape"], entry["data_offsets"])
        if dtype not in dtype_bytes or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"Invalid dtype/shape: {name}")
        if (
            len(offsets) != 2
            or any(type(x) is not int for x in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - start
            or offsets[1] - offsets[0] != math.prod(shape) * dtype_bytes[dtype]
            or start + offsets[1] >= 2**63
        ):
            raise ValueError(f"Invalid tensor extent: {name}")
        tensors[name] = TensorHeader(
            name, path, dtype, tuple(shape), start, tuple(offsets)
        )
    end = 0
    for tensor in sorted(tensors.values(), key=lambda t: t.data_offsets):
        if tensor.data_offsets[0] != end:
            raise ValueError(f"Overlapping or noncontiguous extent: {tensor.name}")
        end = tensor.data_offsets[1]
    if start + end != size:
        raise ValueError(f"Unaccounted safetensors payload: {path.name}")
    return tensors


@dataclass(frozen=True)
class PackedMatrix:
    prefix: str
    padded_in: int
    padded_out: int
    bits_x2: int
    codebook: int
    marker: str

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(
            f"{self.prefix}.{suffix}"
            for suffix in ("trellis", "suh", "svh", self.marker)
        )

    def validate_logical(self, logical_in: int, logical_out: int) -> None:
        if (logical_in + 127) // 128 * 128 != self.padded_in or (
            logical_out + 127
        ) // 128 * 128 != self.padded_out:
            raise ValueError(f"EXL3 logical/padded geometry mismatch: {self.prefix}")


def aligned_shard(logical: int, padded: int, rank: int, size: int) -> slice:
    if size <= 0 or not 0 <= rank < size:
        raise ValueError("Invalid EXL3 shard rank/size")
    if size == 1:
        return slice(0, padded)
    if logical != padded or logical % size or (logical // size) % 128:
        raise ValueError(
            f"EXL3 TP cuts must preserve H128 blocks: {logical=} {padded=} {size=}"
        )
    width = logical // size
    return slice(rank * width, (rank + 1) * width)


def block_shard(
    logical: int, padded: int, rank: int, size: int, rotation: int = 0
) -> slice:
    """A TP shard of whole H128 blocks when the blocks need not divide evenly:
    chunks below ``blocks % size`` take one block more, and rank ``rank``
    takes chunk ``(rank + rotation) % size``. Only for a projection whose
    shard width no other rank or layer depends on (Exl3Layout
    .shards_unevenly; routed experts at tensor parallel size > 1, whose
    rotation is the layer index so the larger chunks move between ranks from
    layer to layer)."""
    if size <= 0 or not 0 <= rank < size or rotation < 0:
        raise ValueError("Invalid EXL3 shard rank/size/rotation")
    if size == 1:
        return slice(0, padded)
    if logical != padded or logical % 128:
        raise ValueError(f"EXL3 TP cuts must preserve H128 blocks: {logical=} {padded=}")
    base, extra = divmod(logical // 128, size)
    if base == 0:
        raise ValueError(f"EXL3 cannot give {size} ranks a block of {logical}")
    chunk = (rank + rotation) % size
    start = chunk * base + min(chunk, extra)
    return slice(start * 128, (start + base + (chunk < extra)) * 128)


class Exl3Manifest:
    """The checkpoint's tensors by name. With ``select_by_index`` (hybrid
    checkpoints) the index's weight_map selects them: only the tensors it
    names, each from the file it names; the files' other tensors are not
    checkpoint tensors (``unselected`` counts them) and FP8 dtypes are
    accepted."""

    def __init__(self, directory: str | Path, select_by_index: bool = False):
        self.directory = Path(directory)
        self.tensors: dict[str, TensorHeader] = {}
        self.matrices: dict[str, PackedMatrix] = {}
        self.unused: dict[str, str] = {}
        self.unselected = 0
        paths = sorted(self.directory.glob("*.safetensors"))
        if not paths:
            raise ValueError("EXL3 requires local safetensors checkpoint files")
        index = self.directory / "model.safetensors.index.json"
        entries = None
        if index.exists():
            entries = json.loads(index.read_text(), object_pairs_hook=_unique_object)[
                "weight_map"
            ]
        elif select_by_index:
            raise ValueError(
                "A hybrid EXL3 checkpoint selects its tensors with "
                "model.safetensors.index.json, which it lacks"
            )
        dtype_bytes = _DTYPE_BYTES
        if select_by_index:
            dtype_bytes = _DTYPE_BYTES | _FP8_DTYPE_BYTES
        for path in paths:
            for name, tensor in read_headers(path, dtype_bytes).items():
                if select_by_index and entries.get(name) != path.name:
                    self.unselected += 1
                    continue
                if name in self.tensors:
                    raise ValueError(f"Duplicate EXL3 source tensor: {name}")
                self.tensors[name] = tensor
        self.tensors = dict(sorted(self.tensors.items()))
        if entries is not None:
            for name, filename in entries.items():
                tensor = self.tensors.get(name)
                if tensor is None or tensor.path.name != filename:
                    raise ValueError(f"EXL3 index/header conflict: {name}")
        self.ple_tables = {}
        for name, tensor in self.tensors.items():
            if not name.endswith(".trellis"):
                continue
            prefix = name.removesuffix(".trellis")
            if prefix.endswith(".ngram_embedding"):
                ple_layout(self.tensors, read_metadata(tensor.path), prefix)
                self.ple_tables[prefix] = tensor
                continue
            self.matrices[prefix] = self._matrix(prefix, tensor)
        packed_keys = {k for matrix in self.matrices.values() for k in matrix.keys}
        for name in self.tensors:
            if name.endswith((".suh", ".svh", ".mcg", ".mul1")):
                if name not in packed_keys:
                    raise ValueError(f"Orphan EXL3 quantization tensor: {name}")
            if (
                name.startswith(("model.visual.blocks.", "visual.blocks."))
                and ".attn.qkv." in name
                and name.endswith((".weight", ".bias"))
            ):
                root, suffix = name.rsplit(".qkv.", 1)
                if all(
                    f"{root}.{part}_proj" in self.matrices for part in ("q", "k", "v")
                ):
                    if suffix == "bias" and not all(
                        f"{root}.{part}_proj.bias" in self.tensors
                        for part in ("q", "k", "v")
                    ):
                        raise ValueError(
                            f"Missing independent vision QKV biases: {root}"
                        )
                    self.unused[name] = "redundant fused vision QKV"

    def expect(self, name: str, dtype: str, shape: tuple[int, ...]):
        tensor = self.tensors.get(name)
        if tensor is None or tensor.dtype != dtype or tensor.shape != shape:
            raise ValueError(f"Missing or incompatible EXL3 tensor: {name}")
        return tensor

    def _matrix(self, prefix: str, trellis: TensorHeader) -> PackedMatrix:
        if trellis.dtype != "I16" or len(trellis.shape) != 3:
            raise ValueError(f"Invalid EXL3 trellis: {prefix}")
        k, n, stride = trellis.shape
        k, n = k * 16, n * 16
        if k <= 0 or n <= 0 or k % 128 or n % 128 or stride % 8:
            raise ValueError(f"Invalid EXL3 tile geometry: {prefix}")
        self.expect(f"{prefix}.suh", "F16", (k,))
        self.expect(f"{prefix}.svh", "F16", (n,))
        markers = [m for m in _MARKERS if f"{prefix}.{m}" in self.tensors]
        if len(markers) != 1:
            raise ValueError(f"Missing/conflicting EXL3 codebook: {prefix}")
        marker = markers[0]
        codebook, value = _MARKERS[marker]
        header = self.expect(f"{prefix}.{marker}", "I32", ())
        if struct.unpack("<I", header.read_bytes())[0] != value:
            raise ValueError(f"Unsupported EXL3 codebook multiplier: {prefix}")
        bits_x2 = stride // 8
        if not (bits_x2 in range(2, 17, 2) or (codebook == 2 and bits_x2 in (3, 5, 7))):
            raise ValueError(f"Unsupported EXL3 bitrate/codebook: {prefix}")
        if f"{prefix}.weight" in self.tensors:
            raise ValueError(f"Conflicting dense and EXL3 weights: {prefix}")
        return PackedMatrix(prefix, k, n, bits_x2, codebook, marker)


@dataclass(frozen=True)
class PLELayout:
    """Geometry of one compressed PLE table, taken from its file headers."""

    trellis: TensorHeader
    # Headers in PLE_AUXILIARIES order.
    auxiliaries: tuple[TensorHeader, ...]
    bits: int
    dim: int
    heads: int
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]

    @property
    def rows(self) -> int:
        return self.trellis.shape[0]

    @property
    def words(self) -> int:
        return self.trellis.shape[1]

    @property
    def row_bytes(self) -> int:
        return self.words * 2


def _metadata_int(metadata: dict[str, str], key: str, prefix: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.isdigit():
        raise ValueError(f"Missing compressed PLE metadata {key}: {prefix}")
    return int(value)


def ple_layout(
    tensors: dict[str, TensorHeader], metadata: dict[str, str], prefix: str
) -> PLELayout:
    """Validate one compressed PLE table and return its layout."""
    trellis = tensors.get(f"{prefix}.trellis")
    if trellis is None or trellis.dtype != "I16" or len(trellis.shape) != 2:
        raise ValueError(f"Invalid compressed PLE table: {prefix}")
    if (
        metadata.get("format") != "exl3_ngram_trellis"
        or metadata.get("version") != "1"
        or metadata.get("codebook") != "mul1"
    ):
        raise ValueError(f"Unsupported compressed PLE format: {prefix}")
    bias = tensors.get(f"{prefix}.head_bias")
    if bias is None or bias.dtype != "F16" or len(bias.shape) != 2:
        raise ValueError(f"Invalid compressed PLE bias: {prefix}")
    heads, dim = bias.shape
    rows, words = trellis.shape
    # K comes from the codec metadata, channels from the bias, the row size
    # from the trellis: one FP16 scale word, then a ring of dim * K bits.
    bits = _metadata_int(metadata, "K", prefix)
    if (
        heads <= 0
        or dim <= 0
        or not 1 <= bits <= 16
        or dim * bits % 16
        or words != 1 + dim * bits // 16
    ):
        raise ValueError(
            f"Compressed PLE rows of {words} words do not hold {dim} "
            f"channels at K={bits}: {prefix}"
        )
    if (
        _metadata_int(metadata, "row_dim", prefix) != dim
        or _metadata_int(metadata, "rows", prefix) != rows
    ):
        raise ValueError(f"Compressed PLE metadata disagrees with headers: {prefix}")
    head_offsets = tensors.get(f"{prefix}.head_offsets")
    head_vocab_sizes = tensors.get(f"{prefix}.head_vocab_sizes")
    for header in (head_offsets, head_vocab_sizes):
        if header is None or header.dtype != "I64" or header.shape != (heads,):
            raise ValueError(f"Invalid compressed PLE head table: {prefix}")
    multipliers = tensors.get(f"{prefix}.layer_multipliers")
    if multipliers is None or multipliers.dtype != "I64" or len(multipliers.shape) != 1:
        raise ValueError(f"Invalid compressed PLE multipliers: {prefix}")
    offsets = struct.unpack(f"<{heads}q", head_offsets.read_bytes())
    sizes = struct.unpack(f"<{heads}q", head_vocab_sizes.read_bytes())
    end = 0
    for offset, size in zip(offsets, sizes):
        if offset != end or size <= 0:
            raise ValueError(f"Invalid compressed PLE head range: {prefix}")
        end += size
    if end > rows or rows - end >= 128:
        raise ValueError(f"Invalid compressed PLE padding: {prefix}")
    auxiliaries = (bias, head_offsets, head_vocab_sizes, multipliers)
    return PLELayout(trellis, auxiliaries, bits, dim, heads, offsets, sizes)


class Exl3LoadLedger:
    def __init__(self, manifest: Exl3Manifest):
        self.manifest = manifest
        self.entries: dict[str, tuple[str, str]] = {}
        for name, reason in manifest.unused.items():
            self.mark(name, "unused", reason)

    def mark(self, name: str, disposition: str, owner: str):
        if name not in self.manifest.tensors:
            raise ValueError(f"Unknown EXL3 source: {name}")
        if name in self.entries:
            raise ValueError(f"EXL3 source consumed more than once: {name}")
        if disposition not in ("consumed", "file-backed", "unused") or not owner:
            raise ValueError("Invalid EXL3 ledger disposition")
        self.entries[name] = disposition, owner

    def finish(self):
        missing = self.manifest.tensors.keys() - self.entries.keys()
        if missing:
            raise ValueError(f"Unaccounted EXL3 sources: {sorted(missing)[:16]}")
