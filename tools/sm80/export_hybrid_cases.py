#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export hybrid capture records to the existing served_cases/1 protocol.

Only actual hybrid records are accepted. A successor link must resolve within the
same rank, scheduler step and graph owner and match the producer view byte-for-byte.
Export never invents activations or reshapes incompatible older checkpoint inputs.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def write_once(path, data):
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace frozen case file {path}")
    path.write_text(text)


def compact_state(obj):
    tensors = dict(obj["tensors"])
    if "snapshot_blocks" not in tensors:
        return obj
    ids = tensors.pop("snapshot_blocks").long()
    name = tensors.pop("snapshot_name")
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    first = torch.stack([(ids == block).nonzero()[0, 0] for block in unique])
    before = tensors[name]
    after_name = "out.ring" if name == "ring" else "out"
    after = tensors[after_name]
    if not same(before[first][inverse], before) or not same(after[first][inverse], after):
        raise ValueError("aliased state blocks disagree in actual capture")
    entries = before.shape[1]
    slots_name = "ring_slots" if name == "ring" else "slots"
    slots = tensors[slots_name].long()
    remap = torch.searchsorted(unique, slots.clamp_min(0) // entries)
    tensors[slots_name] = torch.where(slots >= 0, remap * entries + slots % entries, -1)
    tensors[name], tensors[after_name] = before[first], after[first]
    return dict(obj, tensors=tensors, meta=dict(obj["meta"], observed_blocks=unique.tolist()))


def export(capture, checkpoint, out):
    indexes = sorted(capture.glob("r*/index.jsonl"))
    if not indexes:
        raise ValueError("missing hybrid captures: no rank index")
    records = {}
    raw_rows = []
    missing = []
    for index in indexes:
        for line in index.read_text().splitlines():
            row = json.loads(line)
            if row["family"] != "hybrid":
                continue
            if not row["saved"]:
                missing.append(f"capture cap omitted {index.parent.name}/{row['prefix']}")
                continue
            path = Path(row["path"])
            if sha(path) != row["sha256"]:
                raise ValueError(f"capture hash changed: {path}")
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            meta = obj["meta"]
            key = (meta["rank"], meta["step"], meta["owner"], meta["call_id"])
            if key in records:
                raise ValueError(f"duplicate captured call identity: {key}")
            records[key] = (obj, row)
            raw_rows.append((key, path, row["sha256"]))
    if not records:
        raise ValueError("missing hybrid captures: no hybrid projection records")
    ranks = sorted({key[0] for key in records})
    if ranks != [0, 1, 2, 3]:
        missing.append(f"missing hybrid captures: TP ranks {ranks}")
    if missing:
        raise ValueError("; ".join(missing))
    def parent_keys(key):
        return [(*key[:3], view["call_id"]) for view in
                (records[key][0]["meta"].get("parents") or {}).values()]

    tp_cache = {}
    def crosses_tp(key, visiting=()):
        if key in tp_cache:
            return tp_cache[key]
        if key in visiting:
            raise ValueError("cyclic captured consumer links")
        meta = records[key][0]["meta"]
        if meta.get("tp_boundary"):
            result = True
        elif meta.get("linear") is not None:
            # Each projection can start from its real captured activation;
            # local successor chains do not purport to replay a TP collective.
            result = False
        else:
            result = any(crosses_tp(p, (*visiting, key)) for p in parent_keys(key))
        tp_cache[key] = result
        return result

    out.mkdir(parents=True, exist_ok=True)
    sources = out / "sources"
    sources.mkdir(exist_ok=True)
    written = {}
    visiting = set()

    def freeze(key):
        if key in written:
            return written[key]
        if key in visiting:
            raise ValueError("cyclic captured consumer links")
        visiting.add(key)
        obj, row = records[key]
        meta = dict(obj["meta"])
        parents = {}
        for name, view in (meta.get("parents") or {}).items():
            parent_key = (*key[:3], view["call_id"])
            if parent_key not in records:
                raise ValueError(f"missing hybrid captures: linked producer {parent_key}")
            parent = records[parent_key][0]
            tensor = parent["tensors"][view["output"]]
            expected = tensor.as_strided(view["shape"], view["stride"],
                                         tensor.storage_offset() + view["offset"])
            if not same(expected, obj["tensors"][name]):
                raise ValueError(f"consumer link bytes disagree: {key}/{name}")
            path, digest = freeze(parent_key)
            parents[name] = {"path": str(path), "sha256": digest, "view": view}
        if meta.get("linear") and any(crosses_tp(p) for p in parent_keys(key)):
            meta["captured_parent_sources"] = parents
            parents = {}
        meta["parent_sources"] = parents
        meta["raw_capture_sha256"] = row["sha256"]
        meta["context"] = row["context"]
        path = sources / ("-".join(map(str, key)).replace("/", "_") + ".pt")
        frozen = compact_state(dict(obj, meta=meta))
        meta = frozen["meta"]
        if path.exists():
            prior = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if prior["meta"] != meta or prior["tensors"].keys() != frozen["tensors"].keys():
                raise ValueError(f"frozen source metadata changed: {path}")
            for name, value in frozen["tensors"].items():
                old = prior["tensors"][name]
                if not (same(old, value) if isinstance(value, torch.Tensor) else old == value):
                    raise ValueError(f"frozen source bytes changed: {path}/{name}")
        else:
            torch.save(frozen, path)
        written[key] = path, sha(path)
        visiting.remove(key)
        return written[key]

    pools = {"hybrid-linear": collections.defaultdict(list),
             "hybrid-consumer": collections.defaultdict(list)}
    roles = collections.defaultdict(set)
    local_ops = collections.defaultdict(set)
    dimensions = collections.defaultdict(set)
    deferred = []
    for key, (obj, row) in records.items():
        meta = obj["meta"]
        path, _ = freeze(key)
        linear = meta.get("linear")
        role = "+".join(re.sub(r"\.(\d+)\.", ".*.", p[0]) for p in linear["sources"]) if linear else meta["op"]
        phases = meta["phase"]
        if phases not in ("prefill", "decode", "verify", "draft"):
            raise ValueError(f"unknown captured scheduler phase {phases}")
        dimensions[key[0]].add((phases, row["context"]))
        conc = len(meta.get("query_lens") or [])
        if conc < 1:
            raise ValueError("capture lacks scheduler concurrency")
        phase = "prefill" if phases == "prefill" else "decode"
        base = (role, key[0], meta["rows"], phases, row["context"], conc, phase)
        if linear:
            roles[key[0]].add(role)
            pools["hybrid-linear"][(*base, "out")].append(str(path))
        if crosses_tp(key):
            deferred.append(list(key))
            continue
        frozen_meta = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["meta"]
        if frozen_meta.get("parent_sources"):
            local_ops[key[0]].add(meta["op"])
            for output in meta["outputs"]:
                pools["hybrid-consumer"][(*base, output)].append(str(path))
    for rank in ranks:
        joined = " ".join(roles[rank])
        required_roles = ("wq_a", "wkv", "wq_b", "wo_a", "wo_b", "shared_experts.w1", "shared_experts.w3",
                          "shared_experts.w2", "indexer.wk", "indexer.wq_b", "engram.wkv",
                          "compressor", "main_proj", "head")
        absent = [role for role in required_roles if role not in joined]
        if absent:
            raise ValueError(f"missing hybrid captures: rank {rank} roles {absent}")
        phases = {phase for phase, _ in dimensions[rank]}
        contexts = {context for _, context in dimensions[rank]}
        if not {"prefill", "decode", "verify", "draft"} <= phases or not {"short", "long"} <= contexts:
            raise ValueError(f"missing hybrid captures: rank {rank} phase/context coverage {sorted(dimensions[rank])}")
        if not {"rms_norm", "swiglu", "store_rows", "compress_rows"} <= local_ops[rank]:
            raise ValueError(f"missing hybrid captures: rank {rank} actual local consumers {sorted(local_ops[rank])}")
    if not deferred:
        raise ValueError("missing hybrid captures: actual TP boundaries")
    if not pools["hybrid-consumer"]:
        raise ValueError("missing hybrid captures: no linked actual consumers")
    for name, cases in pools.items():
        family = "hybrid_linear" if name == "hybrid-linear" else "hybrid_linear_consumer"
        rows = []
        for key, paths in sorted(cases.items()):
            role, rank, count, scheduler_phase, context, concurrency, phase, output = key
            cid = f"{role}/r{rank}/M{count}/{scheduler_phase}/{context}/c{concurrency}/{output}"
            # Consumer replay also accepts a linear whose input is another captured
            # producer; freeze a family-specific copy rather than relabel raw input.
            if name == "hybrid-consumer":
                adjusted = []
                for source in paths:
                    obj = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
                    dest = Path(source).with_name("consumer-" + Path(source).name)
                    if not dest.exists():
                        torch.save(dict(obj, family=family), dest)
                    adjusted.append(str(dest))
                paths = adjusted
            rows.append({"id": cid, "family": family, "phase": phase,
                         "scheduler_phase": scheduler_phase, "rows": count,
                         "rank": rank, "context": context, "concurrency": concurrency,
                         "output": output, "captured_sources": sorted(paths)})
        write_once(out / f"{name}.json", {"schema": "served_cases/1",
            "checkpoint": {"hybrid": str(checkpoint)}, "cases": rows})
    write_once(out / "capture-provenance.json", {"schema": "hybrid-capture-provenance/1",
        "raw": [{"identity": list(k), "path": str(p), "sha256": s} for k, p, s in raw_rows],
        "roles": {str(k): sorted(v) for k, v in roles.items()},
        "local_consumer_ops": {str(k): sorted(v) for k, v in local_ops.items()},
        "deferred_tp_consumers": deferred,
        "consumer_scope": "local rank; actual TP snapshots retained for distributed replay",
        "case_counts": {k: len(v) for k, v in pools.items()}})
    return {name: len(cases) for name, cases in pools.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(export(args.capture, args.checkpoint, args.out), sort_keys=True))
    except (ValueError, KeyError, OSError) as exc:
        ap.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
