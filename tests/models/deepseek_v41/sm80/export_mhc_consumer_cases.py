# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export byte-linked delayed mHC pairs of point captures into served_cases/1
for the ``mhc_finish_consumer`` family (``mhc_consumer_cases.py``).

A pair is a layer's attention boundary call (``<layer>.hc_attn``) and the
same layer's FFN boundary call (``<layer>.hc_ffn``) of one rank, step, row
count and graph owner, both saved fused calls whose link is byte-exact: the
FFN call's streams, ``post_layer_mix``, ``comb_res_mix`` and ``pre_mix`` are
the attention call's outputs. A pair that is not linked is listed with the
reason and not used.

Captures are given in priority order (``--capture NAME=DIR``, each with
``r<rank>/index.jsonl``): a cell (target or drafter, rows, scheduler phase)
takes its pairs from the first capture with a linked pair for it, at most
:data:`PER_CELL` (:data:`PER_CELL_LARGE` above 1024 rows), round-robin over
the ranks, each rank's pairs in (layer, step, file) order. One case per
cell and judged output, pooling the cell's pairs; its concurrency is the
largest request count among them. The case sources (``<out>/sources/*.pt``)
hold the two calls' paths and hashes, not copies; the case file is written
once and never replaced.

    python tests/models/deepseek_v41/sm80/export_mhc_consumer_cases.py \\
        --capture hybrid=<dir> --capture old=<dir> --out <dir>
    python tests/models/deepseek_v41/sm80/export_mhc_consumer_cases.py --validate <case file>
"""

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path

import torch

OUTPUTS = ("residual", "post", "comb", "layer_input", "pre",
           "post.rp", "comb.rp", "layer_input.rp", "pre.rp")
PER_CELL = 8
PER_CELL_LARGE = 4
LINKS = (("out_residual", "residual"), ("out_post", "post_layer_mix"),
         ("out_comb", "comb_res_mix"), ("out_pre", "pre_mix"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bitwise(x, y) -> bool:
    return x.shape == y.shape and x.dtype == y.dtype and torch.equal(
        x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def candidate_pairs(root: Path, indexes: dict) -> dict:
    """{(role, rows, phase): {rank: [(layer, step, attention record, FFN record)]}}
    of one capture, from its indexes (each index's hash recorded)."""
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for index in sorted(root.glob("r*/index.jsonl")):
        indexes[str(index)] = digest(index)
        rank = int(index.parent.name.removeprefix("r"))
        groups = collections.defaultdict(dict)
        for line in index.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("family") != "mhc" or not rec.get("saved"):
                continue
            layer, sub = rec["prefix"].rsplit(".", 1)
            tail = Path(rec["path"]).name.split("-M", 1)[1]
            groups[(layer, tail)][sub] = rec
        for (layer, tail), recs in sorted(groups.items()):
            if set(recs) != {"hc_attn", "hc_ffn"}:
                continue
            ffn = recs["hc_ffn"]
            role = "target" if layer.startswith("language_model.") else "drafter"
            cells[(role, int(ffn["rows"]), ffn["phase"])][rank].append(
                (layer, int(ffn["step"]), recs["hc_attn"], ffn))
    return cells


def link(producer: dict, consumer: dict) -> str | None:
    """None when the consumer's carried inputs are the producer's outputs."""
    p, c = producer["tensors"], consumer["tensors"]
    if consumer.get("family") != "mhc_delayed" or c.get("x") is None:
        return "the FFN boundary call is not a fused call"
    if producer.get("family") != "mhc_delayed" or p.get("x") is None:
        return "the attention boundary call is not a fused call"
    for out, arg in LINKS:
        if c.get(arg) is None or not bitwise(p[out].reshape(c[arg].shape), c[arg]):
            return f"{arg} is not the attention call's {out}"
    return None


def load(rec: dict) -> dict:
    if digest(Path(rec["path"])) != rec["sha256"]:
        raise ValueError(f"capture hash changed: {rec['path']}")
    obj = torch.load(rec["path"], map_location="cpu", weights_only=True)
    if obj.get("schema") != "lab-point-capture/1":
        raise ValueError(f"wrong capture schema: {rec['path']}")
    return obj


def pick(by_rank: dict, cap: int, unlinked: list, capture: str) -> list[dict]:
    """Up to ``cap`` linked pairs, round-robin over the ranks."""
    queues = {rank: sorted(v, key=lambda q: (q[0], q[1], q[3]["path"]))
              for rank, v in by_rank.items()}
    picked = []
    while len(picked) < cap and any(queues.values()):
        for rank in sorted(queues):
            while queues[rank] and len(picked) < cap:
                layer, step, prec, crec = queues[rank].pop(0)
                p, c = load(prec), load(crec)
                why = link(p, c)
                if why is not None:
                    unlinked.append(dict(capture=capture, producer=prec["path"],
                                         consumer=crec["path"], reason=why))
                    continue
                picked.append(dict(rank=rank, layer=layer, step=step,
                                   owner=c["meta"]["owner"],
                                   requests=len(c["meta"]["query_lens"]),
                                   producer=dict(path=prec["path"], sha256=prec["sha256"]),
                                   consumer=dict(path=crec["path"], sha256=crec["sha256"])))
                break
    return picked


def export(captures: list[tuple[str, Path]], out: Path) -> list[dict]:
    indexes, unlinked = {}, []
    found = [(name, candidate_pairs(root, indexes)) for name, root in captures]
    keys = sorted({key for _, cells in found for key in cells})
    sources_dir = out / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    cases, chosen = [], []
    for key in keys:
        role, rows, phase = key
        cap = PER_CELL_LARGE if rows > 1024 else PER_CELL
        src, picked = None, []
        for name, cells in found:
            if key in cells:
                picked = pick(cells[key], cap, unlinked, name)
                if picked:
                    src = name
                    break
        available = {name: sum(len(v) for v in cells[key].values())
                     for name, cells in found if key in cells}
        chosen.append(dict(cell=list(key), capture=src, pairs=len(picked), available=available,
                           requests=sorted({p["requests"] for p in picked})))
        if not picked:
            continue
        paths = []
        for pair in picked:
            obj = dict(schema="lab-point-capture/1", family="mhc_finish_consumer",
                       meta=dict(pair, capture=src, role=role, rows=rows, phase=phase),
                       tensors={})
            buf = io.BytesIO()
            torch.save(obj, buf)
            data = buf.getvalue()
            path = sources_dir / (hashlib.sha256(data).hexdigest() + ".pt")
            if not path.exists():
                path.write_bytes(data)
            paths.append(str(path))
        for output in OUTPUTS:
            cases.append(dict(
                id=f"mhc_finish_consumer/{output}/{role}/{phase}/M{rows}",
                family="mhc_finish_consumer", output=output,
                phase="prefill" if phase == "prefill" else "decode", scheduler_phase=phase,
                concurrency=max(p["requests"] for p in picked), rows=rows,
                drafter=role == "drafter", capture=src, captured_sources=paths,
                captured_sha256={p: "sha256:" + digest(Path(p)) for p in paths}))
    doc = dict(schema="served_cases/1", cases=cases, cells=chosen, unlinked=unlinked,
               provenance=dict(captures={n: str(r) for n, r in captures}, indexes=indexes),
               scope="byte-linked attention -> FFN boundary pairs of the delayed mHC")
    path = out / "mhc-finish-consumer.json"
    text = json.dumps(doc, indent=1) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace frozen case file {path}")
    if not path.exists():
        path.write_text(text)
    return cases


def validate(path: Path) -> int:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "served_cases/1" or not doc.get("cases"):
        raise ValueError("not a nonempty served_cases/1 case file")
    for index, sha in doc["provenance"]["indexes"].items():
        if digest(Path(index)) != sha:
            raise ValueError(f"changed capture index: {index}")
    ids, seen = set(), {}
    for case in doc["cases"]:
        if case["id"] in ids:
            raise ValueError(f"duplicate case {case['id']}")
        ids.add(case["id"])
        for p in case["captured_sources"]:
            if p not in seen:
                if "sha256:" + digest(Path(p)) != case["captured_sha256"][p]:
                    raise ValueError(f"changed case source: {p}")
                meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
                for role in ("producer", "consumer"):
                    if digest(Path(meta[role]["path"])) != meta[role]["sha256"]:
                        raise ValueError(f"changed capture: {meta[role]['path']}")
                seen[p] = True
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        print(f"validated {validate(args.validate)} cases")
        return
    if not args.capture or not args.out:
        parser.error("--capture (one or more) and --out are required")
    captures = [(n, Path(d)) for n, _, d in (c.partition("=") for c in args.capture)]
    cases = export(captures, args.out)
    print(f"exported {len(cases)} cases")


if __name__ == "__main__":
    main()
