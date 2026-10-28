# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel call capture (vllm_sm80/capture.py, VLLM_SM80_CAPTURE), CPU
only: off by default; the capture spec's keys, defaults and refusals, and the
banner; eager prefill calls use ordinals or opt-in semantic keys;
decode calls by prefix up to the per-key count; records made while a CUDA
graph is captured are bound to that graph and written after its replays,
with the replaying manager as owner; dummy steps record nothing."""

import importlib
import json
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from vllm_sm80 import capture as cap
from vllm_sm80.kernels.exl3 import capture_points

PREFIX = "layers.3.attn.wq_a"
# The spec the ``on`` fixture writes (dir relative to the spec's directory).
SPEC = dict(
    dir="capture", match=r"layers\.3\.", families=["dense", "moe", "attn", "mhc"],
    rows=[6], per_key=1, prefill_calls=[2], prefill_per_key=0, prefill_min_rows=65,
)


def _toml(settings: dict) -> str:
    # JSON strings, integers and arrays of them are TOML values.
    return "".join(f"{key} = {json.dumps(value)}\n" for key, value in settings.items())


@pytest.fixture
def spec_file(monkeypatch, tmp_path):
    """Write the capture spec SPEC with ``overrides`` (None drops a key) and
    point VLLM_SM80_CAPTURE at it."""
    path = tmp_path / "capture.toml"

    def write(**overrides):
        settings = {k: v for k, v in dict(SPEC, **overrides).items() if v is not None}
        path.write_text(_toml(settings))
        monkeypatch.setenv("VLLM_SM80_CAPTURE", str(path))
        return path

    return write


def test_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_SM80_CAPTURE", raising=False)
    # A retired name never turns the capture on.
    monkeypatch.setenv("VLLM_SM80_CAPTURE_DIR", str(tmp_path))
    assert importlib.reload(cap).active is False
    assert cap.captures("hybrid") is False


def test_spec_defaults(spec_file, tmp_path):
    path = spec_file(**{k: None for k in SPEC if k != "dir"})
    assert path.read_text() == 'dir = "capture"\n'
    assert cap.spec() == cap.Spec(
        path=str(path), dir=str(tmp_path / "capture"),
        families=frozenset({"dense", "moe", "attn"}), match="", rows=frozenset(),
        per_key=2, prefill_match="", prefill_calls=frozenset({1}),
        prefill_per_key=0, prefill_min_rows=65, max_mb=4096,
    )


def test_spec_values(spec_file, tmp_path):
    spec_file(dir=str(tmp_path / "elsewhere"), prefill_match="^mtp\\.", max_mb=12)
    got = cap.spec()
    assert got.dir == str(tmp_path / "elsewhere")  # absolute: as written
    assert (got.families, got.match, got.rows, got.per_key) == (
        frozenset({"dense", "moe", "attn", "mhc"}), r"layers\.3\.", frozenset({6}), 1)
    assert (got.prefill_match, got.prefill_calls, got.prefill_min_rows, got.max_mb) == (
        r"^mtp\.", frozenset({2}), 65, 12)


@pytest.mark.parametrize("overrides,error", [
    (dict(dir=None), "dir is required"),
    (dict(dir=""), "dir is required"),
    (dict(matches="x"), r"unknown keys \['matches'\]"),
    (dict(per_key="2"), "per_key must be int"),
    (dict(per_key=True), "per_key must be int"),
    (dict(rows=[1, "2"]), "rows must be a list of int"),
    (dict(rows="1,2"), "rows must be a list of int"),
    (dict(families="dense,moe"), "families must be a list of str"),
    (dict(families=["dense", "gemm"]), r"unknown families \['gemm'\]"),
    (dict(prefill_per_key=-1), "prefill_per_key must be nonnegative"),
])
def test_spec_refusals_name_the_file(spec_file, overrides, error):
    path = spec_file(**overrides)
    with pytest.raises(ValueError, match=error) as exc:
        cap.spec()
    assert str(exc.value).startswith(f"VLLM_SM80_CAPTURE {path}: ")


def test_captures_reads_the_spec(monkeypatch, spec_file):
    monkeypatch.delenv("VLLM_SM80_CAPTURE", raising=False)
    assert cap.captures("hybrid") is False
    spec_file(families=["attn", "hybrid"])
    assert cap.captures("hybrid") and cap.captures("attn")
    spec_file(families=["attn"])
    assert not cap.captures("hybrid")


def test_banner_names_the_spec_and_directory(on, monkeypatch):
    lines = []
    monkeypatch.setattr(cap.logger, "info", lambda msg, *args: lines.append(msg % args))
    cfg = cap.config()
    assert cap.config() is cfg  # read once
    (line,) = lines
    assert line.startswith(
        f"Kernel call capture active (VLLM_SM80_CAPTURE={cap.spec().path}): "
        f"writing {on}, families attn,dense,mhc,moe, graph rows [6], 1 per key, "
        "prefill calls [2], 0 prefill attempts per semantic key"
    )
    assert "at most 4096 MiB" in line
    assert on.is_dir()


@pytest.fixture
def on(monkeypatch, tmp_path, spec_file):
    spec_file()
    monkeypatch.setattr(cap, "active", True)
    monkeypatch.setattr(cap, "_cfg", None)
    monkeypatch.setattr(cap, "_install", lambda families: None)
    monkeypatch.setattr(cap, "_capturing", lambda: False)
    for name, value in (("_pending", []), ("_counts", {}), ("_prefill_calls", {}),
                        ("_bytes", [0]), ("_weights", {}), ("_seen", {})):
        monkeypatch.setattr(cap, name, value)
    monkeypatch.setattr(
        cap, "_step", {"real": False, "step": 0, "phase": None, "query_lens": []}
    )
    return tmp_path / "capture" / "r0"


def _index(root):
    path = root / "index.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _step(lens, dummy=False, context=()):
    output = SimpleNamespace(
        num_scheduled_tokens=dict(enumerate(lens)),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(num_computed_tokens=list(context)),
    )
    cap.set_step(output, 6, dummy)


def test_prefill_ordinals_and_dummy_steps(on):
    x = torch.randn(80, 4, dtype=torch.bfloat16)
    _step([80], dummy=True)
    assert cap.begin("dense", "exl3_linear", PREFIX, 80) is None
    _step([2048])
    assert cap._step["phase"] == "prefill"
    assert cap.begin("dense", "exl3_linear", PREFIX, 80) is None  # call 1
    rec = cap.begin("dense", "exl3_linear", PREFIX, 80, bits_x2=6)  # call 2
    rec.put(x=x, out=x[:, :2]).done()
    assert cap.begin("dense", "exl3_linear", PREFIX, 64) is None  # too few rows
    assert cap.begin("dense", "exl3_linear", "layers.4.attn.wq_a", 80) is None
    (line,) = _index(on)
    assert line["saved"] and line["phase"] == "prefill" and line["owner"] == "eager"
    obj = torch.load(line["path"], weights_only=True)
    assert obj["schema"] == cap.SCHEMA and obj["family"] == "exl3_linear"
    assert obj["meta"]["prefill_call"] == 2 and obj["meta"]["bits_x2"] == 6
    assert obj["meta"]["query_lens"] == [2048]
    assert torch.equal(obj["tensors"]["x"], x)


def test_decode_per_key(on):
    _step([1, 1])
    cap.begin("dense", "exl3_linear", "layers.9.attn.wq_a", 2)  # not selected, still seen
    assert cap._step["phase"] == "decode"
    for _ in range(3):
        rec = cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2)
        if rec is not None:
            rec.put(x=torch.ones(2, 3)).done()
    assert len(_index(on)) == 1
    _step([1, 1], context=[cap.LONG_CONTEXT, 10])  # a long-context key of its own
    rec = cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2)
    rec.put(x=torch.ones(2, 3)).done()
    assert [x["context"] for x in _index(on)] == ["short", "long"]
    seen = json.loads((on / "rows_seen.json").read_text())
    assert {"family": "dense", "phase": "decode", "rows": 2, "calls": 1} in seen["eager_calls"]
    assert cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 2) is None
    _step([6])
    assert cap._step["phase"] == "verify"
    assert cap.begin("moe", "exl3_moe_out_v2", "model.layers.3.ffn.experts", 6)


class Graph:
    pass


def _graph_record(manager, desc, monkeypatch, prefix="model.layers.3.attn"):
    """A record captured with ``manager``'s graph ``desc``, stored with it."""
    cap._bind(manager)
    monkeypatch.setattr(cap, "_capturing", lambda: True)
    rec = cap.begin("attn", "dsv41_attn_layer", prefix, desc.num_tokens)
    rec.put(q=torch.zeros(desc.num_tokens, 2))
    rec.done()
    manager.graphs[desc] = Graph()
    monkeypatch.setattr(cap, "_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: SimpleNamespace(synchronize=lambda: None))
    return rec


Desc = namedtuple("Desc", "num_tokens num_reqs uniform_token_count max_query_len cg_mode")


def test_graph_records_follow_their_graph(on, monkeypatch):
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    class TargetManager(ModelCudaGraphManager):
        def __init__(self):
            self.graphs = {}

    manager = TargetManager()
    desc = Desc(6, 1, 6, 6, "FULL")
    monkeypatch.setattr(cap, "_capturing", lambda: True)
    assert cap.begin("attn", "dsv41_attn_layer", "model.layers.3.attn", 8) is None
    rec = _graph_record(manager, desc, monkeypatch)
    assert cap._pending == []
    cap._drain(manager, desc)  # no real step yet: nothing written
    assert _index(on) == []
    _step([6])
    rec.tensors["q"].fill_(7)  # what a replay leaves in the graph's copy
    cap._drain(manager, desc)
    cap._drain(manager, desc)  # per key 1
    (line,) = _index(on)
    assert line["owner"] == "TargetManager" and line["phase"] == "verify"
    assert line["step_phase"] == "verify"
    obj = torch.load(line["path"], weights_only=True)
    assert obj["meta"]["graph"]["num_tokens"] == 6
    assert torch.equal(obj["tensors"]["q"], torch.full((6, 2), 7.0))


def test_drafter_graph_in_prefill_step_is_draft_and_bounded(on, monkeypatch):
    """A speculator's graph replayed in a prefill step: phase draft (the
    step's phase in step_phase), per-key bound applied."""

    class DrafterManager:
        def __init__(self):
            self.graphs = {}

    manager = DrafterManager()
    desc = Desc(6, 2, 3, None, "FULL")  # rows in the spec's rows
    _graph_record(manager, desc, monkeypatch, prefix="model.layers.3.attn")
    _step([2048])
    for _ in range(3):
        cap._drain(manager, desc)
    (line,) = _index(on)
    assert (line["phase"], line["step_phase"]) == ("draft", "prefill")
    obj = torch.load(line["path"], weights_only=True)
    assert (obj["meta"]["phase"], obj["meta"]["step_phase"]) == ("draft", "prefill")
    # every replay counts in the row histogram, recorded or not
    _step([1])  # the next step writes the counts
    seen = json.loads((on / "rows_seen.json").read_text())
    assert seen["graph_replays"] == [{"owner": "DrafterManager", "phase": "draft",
                                      "step_phase": "prefill", "rows": 6, "replays": 3}]


def test_moe_point_names_operator(on):
    import functools

    def moe_out_v2(*a, **k):
        pass

    pointers = torch.zeros(2, 3, 3, dtype=torch.int64)
    capture_points.register_moe(pointers, "model.layers.3.ffn.experts")
    _step([1])
    out = torch.zeros(1, 4)
    capture_points.moe(
        functools.partial(moe_out_v2, bits_x2=6), torch.ones(1, 4),
        torch.zeros(1, 6, dtype=torch.int32), torch.ones(1, 6), pointers,
        torch.zeros(2, 3, 4, dtype=torch.int64), torch.arange(4, dtype=torch.int32),
        2304, 10.0, out,
    )
    (line,) = _index(on)
    obj = torch.load(line["path"], weights_only=True)
    assert obj["family"] == "exl3_moe_out_v2"
    assert obj["tensors"]["bits_x2"] == 6 and obj["tensors"]["swiglu_limit"] == 10.0
    assert obj["meta"]["module_prefix"] == "model.layers.3.ffn.experts"
    # The tensor parallel rank and size say which columns of each expert a
    # call held when the routed experts are split (one process: 0 of 1).
    assert (obj["meta"]["tp_rank"], obj["meta"]["tp_size"]) == (0, 1)
    assert obj["meta"]["operator"] == "moe_out_v2"


@pytest.mark.parametrize("name", ["moe_out_v2_fold", "moe_out_v2_try"])
def test_moe_point_of_an_operator_with_v2_bits_is_v2(on, name):
    """moe_out_v2_fold and moe_out_v2_try return moe_out_v2's bits: their calls
    record as exl3_moe_out_v2 (a family the FP64 case driver replays), the
    operator in meta."""
    import functools

    def operator(*a, **k):
        pass

    operator.__name__ = name
    pointers = torch.zeros(2, 3, 3, dtype=torch.int64)
    capture_points.register_moe(pointers, "model.layers.3.ffn.experts")
    _step([1])
    capture_points.moe(
        functools.partial(operator, bits_x2=6), torch.ones(1, 4),
        torch.zeros(1, 6, dtype=torch.int32), torch.ones(1, 6), pointers,
        torch.zeros(2, 3, 4, dtype=torch.int64), torch.arange(4, dtype=torch.int32),
        2304, 10.0, torch.zeros(1, 4),
    )
    (line,) = _index(on)
    obj = torch.load(line["path"], weights_only=True)
    assert obj["family"] == "exl3_moe_out_v2"
    assert obj["meta"]["operator"] == name and obj["tensors"]["bits_x2"] == 6


def test_mhc_proxy_records_call_and_outputs(on):
    """The decoder layer's mHC ops proxy: the attention-side call is named
    hc_attn, the FFN-side hc_ffn; arguments and outputs are recorded and the
    op's outputs pass through unchanged."""
    from vllm_sm80.models.deepseek_v41.sm80_model import _CapturedMhc

    attn_fn, ffn_fn = torch.ones(3), torch.zeros(3)
    outs = tuple(torch.full((2, 1), float(i)) for i in range(6))

    class Ops:
        def mhc_fused_post_pre_delayed(self, *args, **kwargs):
            return outs

        def mhc_post(self):
            return "post"

    layer = SimpleNamespace(attn=SimpleNamespace(prefix="model.layers.3.attn"),
                            hc_attn_fn=attn_fn, hc_ffn_fn=ffn_fn)
    proxy = _CapturedMhc(Ops(), layer)
    assert proxy.mhc_post() == "post"
    _step([1, 1])
    x = torch.randn(2, 4)
    for fn in (attn_fn, ffn_fn):
        got = proxy.mhc_fused_post_pre_delayed(
            x, torch.randn(2, 4, 4), torch.rand(2, 4), torch.rand(2, 16), fn,
            torch.ones(3), torch.zeros(24), 1e-6, 1e-6, 1e-6, 2.0, 20,
            pre_mix=None, norm_weight=torch.ones(4), norm_eps=1e-6,
        )
        assert got is outs
    lines = _index(on)
    assert sorted(x["prefix"] for x in lines) == ["model.layers.3.hc_attn", "model.layers.3.hc_ffn"]
    obj = torch.load(lines[0]["path"], weights_only=True)
    assert obj["family"] == "mhc_delayed" and obj["tensors"]["sinkhorn_repeat"] == 20
    assert torch.equal(obj["tensors"]["x"], x)
    assert torch.equal(obj["tensors"]["out_pre"], outs[4])


@pytest.mark.parametrize("prefix", ["model.layers.3.self_attn.q_a_proj",
                                   "model.layers.3.self_attn.q_proj"])
def test_glm_qwen_plain_linear_capture_keeps_host_materialization(on, prefix):
    """Shared GLM/Qwen EXL3 points have no model-owned deferred state."""
    _step([1, 1])
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.bfloat16)
    out = x[:, :2].clone()
    trellis = torch.zeros(4, dtype=torch.int16)
    cap.register_weight(trellis.data_ptr(), prefix=prefix, out_dtype="bfloat16")
    capture_points.linear(x, trellis, out, torch.matmul)
    (row,) = _index(on)
    obj = torch.load(row["path"], weights_only=True)
    expected = {k: cap._host(v) for k, v in {"x": x, "out": out}.items()}
    assert obj["family"] == "exl3_linear" and row["phase"] == "decode"
    for key, value in expected.items():
        assert obj["tensors"][key].dtype == value.dtype
        assert torch.equal(obj["tensors"][key], value)
    plain = cap.Record("dense", "exl3_linear", prefix, 2, {}).put(x=x, out=out)
    assert not plain.deferred
    assert all(torch.equal(plain.materialize()[k], v) for k, v in expected.items())


@pytest.mark.parametrize("setting", [None, 0])
def test_prefill_default_retains_sparse_ordinals(on, spec_file, setting):
    spec_file(prefill_per_key=setting, prefill_calls=[1, 3])
    _step([2048])
    for ordinal in range(1, 5):
        rec = cap.begin("dense", "exl3_linear", PREFIX, 80)
        assert (rec is not None) == (ordinal in (1, 3))
        if rec is not None:
            rec.put(x=torch.ones(80, 1)).done()
    assert cap.config().prefill_per_key == 0
    assert [torch.load(row["path"], weights_only=True)["meta"]["prefill_call"]
            for row in _index(on)] == [1, 3]
    # Legacy prefill deliberately ignores the ordinary per-key limit of one.
    assert list(cap._counts.values()) == [2]


@pytest.mark.parametrize("family", cap.FAMILIES)
@pytest.mark.parametrize("limit", [1, 2])
def test_prefill_per_key_covers_reached_rows_context_points(on, spec_file, family, limit):
    spec_file(families=[family], prefill_per_key=limit, prefill_calls=[9999],
              prefill_min_rows=1)
    ordinals, calls, expected = {}, {}, []
    for context in (0, cap.LONG_CONTEXT):
        _step([2048], context=[context])
        for rows in (17, 2048):
            for point in ("producer", "consumer"):
                for prefix in (PREFIX, PREFIX + ".other"):
                    for attempt in range(limit + 1):
                        ordinals[prefix] = ordinals.get(prefix, 0) + 1
                        calls[rows] = calls.get(rows, 0) + 1
                        rec = cap.begin(family, point, prefix, rows)
                        assert (rec is not None) == (attempt < limit)
                        if rec is not None:
                            assert rec.meta["prefill_call"] == ordinals[prefix]
                            rec.put(x=torch.full((rows, 1), float(ordinals[prefix]))).done()
                            expected.append((point, prefix, rows, cap._step["context"],
                                             ordinals[prefix]))
    lines = _index(on)
    assert len({row["path"] for row in lines}) == len(expected)
    assert all(row["saved"] and row["phase"] == "prefill" for row in lines)
    actual = []
    for row in lines:
        payload = torch.load(row["path"], weights_only=True)
        ordinal = payload["meta"]["prefill_call"]
        actual.append((row["point"], row["prefix"], row["rows"], row["context"], ordinal))
        assert payload["family"] == row["point"]
        assert payload["meta"]["module_prefix"] == row["prefix"]
        assert torch.equal(payload["tensors"]["x"], torch.full((row["rows"], 1), float(ordinal)))
    assert actual == expected
    assert set(cap._counts.values()) == {limit}
    assert cap._prefill_calls == {(family, prefix): n for prefix, n in ordinals.items()}
    _step([1])  # Flush every reached call, including repeats rejected by the bound.
    seen = json.loads((on / "rows_seen.json").read_text())
    assert sorted(seen["eager_calls"], key=lambda x: x["rows"]) == [
        {"family": family, "phase": "prefill", "rows": rows, "calls": n}
        for rows, n in sorted(calls.items())
    ]


def test_prefill_per_key_preserves_filters_and_actual_phase(on, spec_file):
    spec_file(prefill_per_key=1)
    _step([80], dummy=True)
    assert cap.begin("dense", "projection", PREFIX, 80) is None
    assert cap._prefill_calls == cap._seen == {}
    _step([2048])
    assert cap.begin("dense", "projection", PREFIX, 64) is None
    assert cap.begin("dense", "projection", "layers.4.attn.wq_a", 80) is None
    assert cap._prefill_calls == {}
    with cap.phase_scope("draft"):
        rec = cap.begin("dense", "projection", PREFIX, 80)
        rec.put(x=torch.ones(80, 1)).done()
        assert cap.begin("dense", "projection", PREFIX, 80) is None
    rec = cap.begin("dense", "projection", PREFIX, 80)
    assert rec.meta["prefill_call"] == 1
    rec.put(x=torch.ones(80, 1)).done()
    assert cap.begin("dense", "projection", PREFIX, 80) is None
    assert [row["phase"] for row in _index(on)] == ["draft", "prefill"]
    assert cap._prefill_calls == {("dense", PREFIX): 2}


def test_prefill_per_key_distinguishes_exact_prefixes(on, spec_file):
    spec_file(prefill_per_key=1)
    _step([80])
    prefixes = [PREFIX + ".a/b", PREFIX + ".a_b"]
    for number, prefix in enumerate(prefixes):
        rec = cap.begin("dense", "projection", prefix, 80)
        rec.put(x=torch.full((80, 1), number)).done()
    lines = _index(on)
    assert len({row["path"] for row in lines}) == 2
    for number, row in enumerate(lines):
        payload = torch.load(row["path"], weights_only=True)
        assert payload["meta"]["module_prefix"] == prefixes[number]
        assert torch.equal(payload["tensors"]["x"], torch.full((80, 1), number))


def test_prefill_per_key_write_bound_is_per_owner(on, spec_file):
    spec_file(prefill_per_key=2)
    _step([80])
    records = [cap.begin("attn", "attention", PREFIX, 80) for _ in range(3)]
    for rec in records:
        rec.put(q=torch.ones(80, 1))
    for owner in ("FirstOwner", "SecondOwner"):
        cap._write(records, "prefill", owner, None)
        cap._write(records, "prefill", owner, None)
    lines = _index(on)
    assert [row["owner"] for row in lines] == ["FirstOwner"] * 2 + ["SecondOwner"] * 2
    assert [torch.load(row["path"], weights_only=True)["meta"]["prefill_call"]
            for row in lines] == [1, 2, 1, 2]
    assert set(cap._counts.values()) == {2}


def test_prefill_per_key_quota_refusals_are_bounded_and_counted(on, spec_file):
    spec_file(prefill_per_key=2)
    cfg = cap.config()
    x = torch.ones(80, 1)
    size = x.numel() * x.element_size()
    cfg.max_bytes = 2 * size
    _step([80])
    for point in ("producer", "consumer"):
        for _ in range(2):
            cap.begin("dense", point, PREFIX, 80).put(x=x).done()
        assert cap.begin("dense", point, PREFIX, 80) is None
    lines = _index(on)
    assert [row["saved"] for row in lines] == [True, True, False, False]
    assert all(row["bytes"] == size for row in lines)
    assert all("path" not in row and "sha256" not in row for row in lines if not row["saved"])
    assert cap._bytes == [cfg.max_bytes]
    assert set(cap._counts.values()) == {2}
    assert cap._prefill_calls == {("dense", PREFIX): 6}
    _step([1])
    seen = json.loads((on / "rows_seen.json").read_text())
    assert seen["eager_calls"] == [
        {"family": "dense", "phase": "prefill", "rows": 80, "calls": 6}
    ]


def test_prefill_per_key_rejects_negative_limit_before_hooks(on, spec_file):
    spec_file(prefill_per_key=-1)
    with pytest.raises(ValueError, match="prefill_per_key must be nonnegative"):
        cap.config()
    assert cap._cfg is None and not on.exists()


def test_linked_producer_layout_is_opt_in_and_survives_save_load(on):
    backing = torch.arange(80, dtype=torch.float32).reshape(4, 20)
    source = backing[:, 3:11]
    expected = source.clone()
    _step([1, 1, 1, 1])
    rec = cap.begin("dense", "projection", PREFIX, 4)
    rec.put(x=source)
    rec.put(preserve_layout=True, out=source)
    assert rec.tensors["x"].is_contiguous()
    assert rec.tensors["out"].stride() == source.stride()
    assert rec.tensors["out"].storage_offset() == 0
    assert rec.tensors["out"].untyped_storage().nbytes() == 68 * source.element_size()
    backing.fill_(-1)
    rec.done()
    (row,) = _index(on)
    obj = torch.load(row["path"], weights_only=True, mmap=True)
    assert row["bytes"] == (32 + 68) * source.element_size()
    assert obj["tensors"]["out"].stride() == (20, 1)
    assert torch.equal(obj["tensors"]["out"], expected)
    assert torch.equal(obj["tensors"]["x"], expected)


def test_linked_producer_host_copy_cannot_pack_holes(monkeypatch):
    source = torch.arange(64, dtype=torch.float32).reshape(4, 16)[:, :8]
    rec = cap.Record("hybrid", "hybrid_linear_consumer", PREFIX, 4, {})
    rec.put(preserve_layout=True, out=source)
    # Force the packing behavior of a non-dense device transfer without CUDA;
    # the opt-in path must transfer its flat span, not the strided logical view.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda value: value.clone(memory_format=torch.contiguous_format))
    saved = rec.materialize()["out"]
    assert saved.stride() == source.stride()
    assert saved.untyped_storage().nbytes() == 56 * source.element_size()
    assert torch.equal(saved, source)


def test_linked_producer_byte_quota_counts_storage_span(on):
    source = torch.arange(64, dtype=torch.float32).reshape(4, 16)[:, :8]
    cap.config().max_bytes = source.numel() * source.element_size()
    _step([1, 1, 1, 1])
    cap.begin("dense", "projection", PREFIX, 4).put(preserve_layout=True, out=source).done()
    (row,) = _index(on)
    assert row["bytes"] == 56 * source.element_size()
    assert not row["saved"] and "path" not in row
    assert cap._bytes == [0]
