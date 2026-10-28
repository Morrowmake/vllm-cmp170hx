# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The consumer check of the delayed mHC finish v2 on byte-linked captured calls.

A case source (family ``mhc_finish_consumer``, written by
``export_mhc_consumer_cases.py``) names two captured delayed mHC calls of one
step, rank and graph owner: the producer, a layer's attention boundary
(``hc_attn``), and its consumer, the same layer's FFN boundary (``hc_ffn``),
whose carried inputs are the producer's outputs byte for byte (checked
here): the streams (``residual``), ``post_layer_mix``, ``comb_res_mix`` and
the next pre-mix (``pre_mix``). The consumer's sublayer output ``x`` is the
captured one.

Three chains, each producer output fed into the consumer:

- candidate: both calls through ``sm80_mhc_fused`` with the finish v2;
- reference: both through ``sm80_mhc_fused`` with the served finish kernel
  (the pinned shipped incumbent);
- FP64: ``fp64_cases.mhc_delayed_fp64`` of the producer, its unrounded
  outputs into the FP64 consumer; for the ``.rp`` outputs, per chain, at that
  chain's own BF16 rounding points (the producer's FP64 coefficients from the
  chain's own producer streams, the consumer from its own streams).

The judged value is one output of the consumer call (``residual``, ``post``,
``comb``, ``layer_input``, ``pre`` or ``<output>.rp``); row invariance: each
picked row's chain run alone gives the chain's bits.
"""

import collections
import contextlib
import hashlib
from pathlib import Path

import torch

from tests.models.deepseek_v41.sm80 import fp64_cases as F

# producer output -> consumer argument it is carried into
LINKS = (("out_residual", "residual"), ("out_post", "post_layer_mix"),
         ("out_comb", "comb_res_mix"), ("out_pre", "pre_mix"))
# the judged values of the last pairs evaluated: a cell's cases (one per output)
# pool the same pairs, at most 8
_CACHE: collections.OrderedDict = collections.OrderedDict()
_CACHED_PAIRS = 8


def load_source(link: dict) -> dict:
    path = Path(link["path"])
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != link["sha256"].removeprefix("sha256:"):
        raise ValueError(f"captured source hash changed: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("schema") != F.SOURCE_SCHEMA or obj.get("family") != "mhc_delayed":
        raise ValueError(f"{path}: not a captured mhc_delayed call")
    return obj["tensors"]


def check_link(producer: dict, consumer: dict) -> None:
    """The consumer's carried inputs are the producer's outputs, byte for byte."""
    for out, arg in LINKS:
        value = producer[out].reshape(consumer[arg].shape)
        if not F.bits_equal(value, consumer[arg]):
            raise ValueError(f"consumer {arg} is not the producer's {out}")


def _args(t: dict) -> dict:
    return {k: v for k, v in t.items() if not k.startswith("out_")}


@contextlib.contextmanager
def finish(v2: bool):
    from vllm_sm80.models.deepseek_v41.ops import sm80_mhc_fused as fused

    saved = fused.FINISH_KERNEL, fused.FINISH_V2
    fused.FINISH_KERNEL, fused.FINISH_V2 = True, v2
    try:
        yield
    finally:
        fused.FINISH_KERNEL, fused.FINISH_V2 = saved


def _chain(p: dict, c: dict, device, rows: slice, v2: bool) -> tuple[dict, dict, dict]:
    """The producer and the consumer on ``device`` for ``rows``: the producer's
    outputs, the consumer's arguments and outputs (CPU)."""
    with finish(v2):
        pout = {k: v.cpu() for k, v in F._mhc_call(p, device, rows, True).items()}
        cin = {k: (v[rows] if k in F.MHC_TOKEN_ARGS and v is not None else v)
               for k, v in c.items()}
        cin.update(residual=pout["residual"], post_layer_mix=pout["post"],
                   comb_res_mix=pout["comb"], pre_mix=pout["pre"])
        cout = {k: v.cpu() for k, v in F._mhc_call(cin, device, slice(None), True).items()}
    return pout, cin, cout


def _fp64(p: dict, c: dict, chain=None) -> dict:
    """The consumer's outputs in FP64; with ``chain`` (a chain's producer
    outputs, consumer arguments and outputs) at that chain's rounding points."""
    if chain is None:
        ideal = F.mhc_delayed_fp64(p)
        return F.mhc_delayed_fp64(dict(c, residual=ideal["residual"], post_layer_mix=ideal["post"],
                                       comb_res_mix=ideal["comb"], pre_mix=ideal["pre"]))
    pout, cin, cout = chain
    own = F.mhc_delayed_fp64(p, True, pout["residual"])
    return F.mhc_delayed_fp64(dict(cin, post_layer_mix=own["post"], comb_res_mix=own["comb"],
                                   pre_mix=own["pre"]), True, cout["residual"])


def _evaluate(meta: dict, device) -> dict:
    key = (meta["producer"]["path"], meta["consumer"]["path"], str(device))
    if key in _CACHE:
        _CACHE.move_to_end(key)
    else:
        pt, ct = load_source(meta["producer"]), load_source(meta["consumer"])
        check_link(pt, ct)
        p, c = _args(pt), _args(ct)
        cand = _chain(p, c, device, slice(None), True)
        ref = _chain(p, c, device, slice(None), False)
        n = p["residual"].shape[0]

        def alone(v2, chain):
            return all(F.bits_equal(chain[2][o][i:i + 1], _chain(p, c, device, slice(i, i + 1), v2)[2][o])
                       for i in F.picks(n) for o in chain[2])

        def rp(chain):
            return {k: v for k, v in _fp64(p, c, chain).items() if k != "residual"}

        _CACHE[key] = dict(cand=cand[2], ref=ref[2], plain=_fp64(p, c), cand_rp=rp(cand),
                           ref_rp=rp(ref), cand_alone=alone(True, cand),
                           ref_alone=alone(False, ref))
        while len(_CACHE) > _CACHED_PAIRS:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def mhc_finish_consumer(a: dict, device, src) -> F.Call:
    output = src.output
    name, _, mode = (output or "").partition(".")
    if name not in F.MHC_OUTPUTS or mode not in ("", "rp") or (name, mode) == ("residual", "rp"):
        raise ValueError(f"mHC consumer output {output!r} is not one of {F.MHC_OUTPUTS} or <output>.rp")
    e = _evaluate(src.meta, device)
    cand, ref = e["cand"][name], e["ref"][name]
    if mode == "rp":
        exact, ref_exact = e["cand_rp"][name], e["ref_rp"][name]
    else:
        exact, ref_exact = e["plain"][name], None
    return F.Call(cand, ref, exact, e["cand_alone"], ref_exact=ref_exact,
                  ref_row_invariant=e["ref_alone"])
