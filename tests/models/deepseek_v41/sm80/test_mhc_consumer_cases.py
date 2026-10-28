# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The delayed mHC consumer cases on CPU: the export pairs byte-linked
attention -> FFN boundary calls, takes a cell from the first capture that has
a linked pair, lists unlinked pairs, refuses to replace its case file and
validates the hashes; the family refuses an unlinked pair and unknown
outputs. The GPU half runs in the fp64_cases driver (the consumer check)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from . import export_mhc_consumer_cases as E
from . import mhc_consumer_cases as C
from .fp64_cases import mhc_delayed_fp64
from .test_sm80_mhc_fused import make_inputs

HIDDEN = 64


def _call(T: int, seed: int) -> dict:
    a = make_inputs(T, HIDDEN, seed)
    out = mhc_delayed_fp64(a)
    return dict(a, out_residual=out["residual"].to(torch.bfloat16),
                out_post=out["post"].float(), out_comb=out["comb"].float(),
                out_pre=out["pre"].float(), out_layer_input=out["layer_input"].to(torch.bfloat16))


def _linked(T: int, seed: int) -> tuple[dict, dict]:
    p = _call(T, seed)
    c = make_inputs(T, HIDDEN, seed + 100)
    c.update(residual=p["out_residual"], post_layer_mix=p["out_post"],
             comb_res_mix=p["out_comb"], pre_mix=p["out_pre"])
    return p, c


def _save(root: Path, rank: int, prefix: str, tail: str, tensors: dict, rows: int,
          phase: str, step: int) -> dict:
    d = root / f"r{rank}" / "mhc"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prefix}-M{tail}"
    meta = dict(module_prefix=prefix, rank=rank, phase=phase, owner="eager", step=step,
                rows=rows, query_lens=[rows])
    torch.save(dict(schema="lab-point-capture/1", family="mhc_delayed", meta=meta,
                    tensors=tensors), path)
    return dict(family="mhc", point="mhc_delayed", prefix=prefix, rows=rows, phase=phase,
                owner="eager", step=step, saved=True, sha256=E.digest(path), path=str(path))


def _capture(root: Path, cells, broken=()) -> None:
    for rank in (0, 1):
        recs = []
        for layer, rows, phase in cells:
            p, c = _linked(rows, rank)
            if (layer, rows) in broken:
                c = dict(c, pre_mix=c["pre_mix"] + 1)
            tail = f"{rows}-{phase}-eager-s3-0.pt"
            for sub, t in (("hc_attn", p), ("hc_ffn", c)):
                recs.append(_save(root, rank, f"{layer}.{sub}", tail, t, rows, phase, 3))
        (root / f"r{rank}" / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))


def test_export_pairs_links_and_freezes(tmp_path):
    primary, secondary = tmp_path / "primary", tmp_path / "secondary"
    _capture(primary, [("language_model.model.layers.10", 6, "verify"),
                       ("language_model.model.layers.20", 2, "decode")],
             broken={("language_model.model.layers.20", 2)})
    _capture(secondary, [("language_model.model.layers.10", 6, "verify"),
                         ("language_model.model.layers.20", 2, "decode"),
                         ("model.layers.40", 5, "draft")])
    out = tmp_path / "out"
    cases = E.export([("primary", primary), ("secondary", secondary)], out)
    doc = json.loads((out / "mhc-finish-consumer.json").read_text())
    cells = {tuple(c["cell"]): c for c in doc["cells"]}
    assert cells[("target", 6, "verify")]["capture"] == "primary"
    # the primary's pairs of this cell are not linked: the next capture serves it
    assert cells[("target", 2, "decode")]["capture"] == "secondary"
    assert cells[("drafter", 5, "draft")]["capture"] == "secondary"
    assert {u["reason"] for u in doc["unlinked"]} == {"pre_mix is not the attention call's out_pre"}
    assert len(cases) == 3 * len(E.OUTPUTS)
    case = next(c for c in cases if c["id"] == "mhc_finish_consumer/comb.rp/target/verify/M6")
    assert case["output"] == "comb.rp" and case["phase"] == "decode" and case["concurrency"] == 1
    assert len(case["captured_sources"]) == 2  # one pair per rank
    meta = torch.load(case["captured_sources"][0], weights_only=True)["meta"]
    assert meta["rank"] == 0 and meta["capture"] == "primary"
    assert E.validate(out / "mhc-finish-consumer.json") == len(cases)
    # written once: the same export is accepted, a different one refused
    E.export([("primary", primary), ("secondary", secondary)], out)
    with pytest.raises(ValueError, match="refusing to replace"):
        E.export([("secondary", secondary)], out)


def test_family_checks_the_link_and_the_output(tmp_path):
    p, c = _linked(3, 0)
    C.check_link(p, c)
    with pytest.raises(ValueError, match="comb_res_mix"):
        C.check_link(p, dict(c, comb_res_mix=c["comb_res_mix"] * 2))
    src = SimpleNamespace(meta={}, output="mixes")
    with pytest.raises(ValueError, match="not one of"):
        C.mhc_finish_consumer({}, torch.device("cpu"), src)


def test_fp64_chain_feeds_the_producer_into_the_consumer():
    p, c = _linked(3, 1)
    p = C._args(p)
    got = C._fp64(p, c)
    ideal = mhc_delayed_fp64(p)
    want = mhc_delayed_fp64(dict(c, residual=ideal["residual"], post_layer_mix=ideal["post"],
                                 comb_res_mix=ideal["comb"], pre_mix=ideal["pre"]))
    for k in want:
        assert torch.equal(got[k], want[k]), k
