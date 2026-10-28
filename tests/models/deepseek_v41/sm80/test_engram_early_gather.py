# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scheduling contracts; device and served identity are separate checks."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.engram import Engram
from vllm_sm80.models.deepseek_v41 import sm80_engram_prefetch as P
from vllm_sm80.models.deepseek_v41.sm80_engram_projection import project_kv, project_rows


@pytest.fixture
def streams(monkeypatch):
    log = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            log.append((self.name, "wait_stream", other.name))

        def wait_event(self, event):
            log.append((self.name, "wait_event", event))

    class Event:
        def record(self, stream):
            log.append((stream.name, "record", self))

    main, aux = Stream("main"), Stream("aux")
    current = [main]

    @contextmanager
    def use(stream):
        old, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = old

    monkeypatch.setattr(P, "aux_stream", lambda: aux)
    monkeypatch.setattr(P, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "stream", use)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.Tensor, "record_stream",
                        lambda self, stream: log.append((stream.name, "retain", self)))
    return SimpleNamespace(main=main, aux=aux, current=current, log=log, use=use)


def layer(exl3=True):
    """An Engram whose WKV projection is EXL3 (the auxiliary stream, which owns
    the EXL3 scratch) or, with ``exl3=False``, an ordinary linear (a stream of
    its own)."""
    from vllm_sm80.kernels.exl3.quant import Exl3LinearMethod

    e = Engram.__new__(Engram)
    torch.nn.Module.__init__(e)
    e.use_sequence_parallel = False
    e._batch = None
    e.embed_tokens = SimpleNamespace(dp_size=1)
    method = Exl3LinearMethod.__new__(Exl3LinearMethod) if exl3 else None
    e.wkv = SimpleNamespace(quant_method=method)
    return e


def test_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER", raising=False)
    assert not envs.VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER


def test_gathers_project_once_before_layer_and_join_own_event(monkeypatch, streams):
    def prepare(e, hashes):
        streams.log.append((streams.current[0].name, "lookup", e))

    def project(e, hashes):
        streams.log.append((streams.current[0].name, "gather_project_gather", e))
        return hashes.float() * 3

    monkeypatch.setattr(Engram, "prepare_embeddings", prepare)
    monkeypatch.setattr(P, "project_rows", project)
    a, b = layer(), layer()
    P.EarlyEngram.adopt(a)
    P.EarlyEngram.adopt(b)
    assert a._prefetch_stream is b._prefetch_stream is streams.aux
    for rows in (6, 18, 4, 6):
        streams.log.clear()
        hashes = torch.arange(rows).reshape(rows, 1)
        a.prepare_embeddings(hashes)
        b.prepare_embeddings(hashes + 1)
        assert streams.current[0] is streams.main
        assert [s for s, op, _ in streams.log if op == "gather_project_gather"] == ["aux", "aux"]
        assert not any(op == "wait_event" for _, op, _ in streams.log)
        assert torch.equal(project_kv(a, hashes), hashes.float() * 3)
        assert torch.equal(project_kv(b, hashes + 1), (hashes.float() + 1) * 3)
        assert [(s, obj) for s, op, obj in streams.log if op == "wait_event"] == [
            ("main", a._projection_done), ("main", b._projection_done)]
        assert [(s, obj) for s, op, obj in streams.log if op == "wait_stream"] == [
            ("aux", "main"), ("aux", "main")]
        assert a._early_kv is b._early_kv is None
        assert len([1 for s, op, _ in streams.log if s == "main" and op == "retain"]) == 2


def test_projection_lifetime_and_row_count(monkeypatch, streams):
    monkeypatch.setattr(Engram, "prepare_embeddings", lambda e, h: None)
    monkeypatch.setattr(P, "project_rows", lambda e, h: h.clone())
    e = layer()
    P.EarlyEngram.adopt(e)
    h = torch.zeros(6, 2)
    with pytest.raises(RuntimeError, match="missing"):
        project_kv(e, h)
    e.prepare_embeddings(h)
    with pytest.raises(RuntimeError, match="not consumed"):
        e.prepare_embeddings(h)
    with pytest.raises(RuntimeError, match="missing"):
        project_kv(e, h[:4])
    assert torch.equal(project_kv(e, h), h)
    with pytest.raises(RuntimeError, match="missing"):
        project_kv(e, h)


@pytest.mark.parametrize("attribute,value", [("use_sequence_parallel", True), ("_batch", object())])
def test_refuse_unsupported_projection(streams, attribute, value):
    e = layer()
    setattr(e, attribute, value)
    with pytest.raises(ValueError, match="unbatched TP"):
        P.EarlyEngram.adopt(e)


def test_refuse_missing_aux_and_dp(monkeypatch, streams):
    e = layer()
    e.embed_tokens.dp_size = 2
    with pytest.raises(ValueError, match="TP-only"):
        P.EarlyEngram.adopt(e)
    e.embed_tokens.dp_size = 1
    monkeypatch.setattr(P, "aux_stream", lambda: None)
    with pytest.raises(ValueError, match="registered"):
        P.EarlyEngram.adopt(e)


def test_ordinary_projection_runs_on_a_stream_of_its_own(monkeypatch, streams):
    own = type(streams.main)("own")
    made = []

    def new_stream(device=None):
        made.append(device)
        return own

    monkeypatch.setattr(P, "_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", new_stream)
    a, b = layer(exl3=False), layer(exl3=False)
    P.EarlyEngram.adopt(a)
    P.EarlyEngram.adopt(b)
    # one stream per device, shared by the Engram layers, not the aux stream
    assert a._prefetch_stream is b._prefetch_stream is own
    assert made == [0]
    # an EXL3 projection keeps the auxiliary stream
    c = layer()
    P.EarlyEngram.adopt(c)
    assert c._prefetch_stream is streams.aux


def test_refuse_microbatching():
    model = SimpleNamespace(vllm_config=SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True)))
    with pytest.raises(ValueError, match="microbatching"):
        P.adopt_early_engram(model)


def test_registered_aux_uses_distinct_exl3_scratch(monkeypatch, streams):
    from vllm_sm80.kernels.exl3 import ops
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace

    resources = {}

    def get_resource(key, factory):
        if key not in resources:
            resources[key] = factory()
        return resources[key]

    monkeypatch.setattr(ops, "LINEAR_SCRATCH_ELEMENTS", 16)
    monkeypatch.setattr(torch_utils, "aux_stream", lambda: streams.aux)
    monkeypatch.setattr(torch_utils, "current_stream", lambda: streams.current[0])
    monkeypatch.setattr(workspace, "current_workspace_manager", lambda: SimpleNamespace(
        get_persistent_resource=get_resource))
    main = ops._linear_scratch(torch.device("cpu"))
    with streams.use(streams.aux):
        early = ops._linear_scratch(torch.device("cpu"))
        shared = ops._linear_scratch(torch.device("cpu"))
    assert early is shared and early.data_ptr() != main.data_ptr()
    assert ops._linear_scratch(torch.device("cpu")) is main


@pytest.mark.parametrize("mode_name", ["PIECEWISE", "FULL_AND_PIECEWISE"])
def test_refuse_piecewise_stream_forks(mode_name):
    from vllm.config import CUDAGraphMode

    model = SimpleNamespace(vllm_config=SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=False),
        compilation_config=SimpleNamespace(cudagraph_mode=getattr(CUDAGraphMode, mode_name))))
    with pytest.raises(ValueError, match="piecewise"):
        P.adopt_early_engram(model)


@pytest.mark.parametrize("batched", [False, True])
def test_ordinary_projection_keeps_original_input_and_dtype(batched):
    hashes = torch.arange(6).reshape(3, 2)
    embedded = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
    seen = []
    def wkv(rows):
        seen.append(rows)
        return rows[:, :2]
    e = SimpleNamespace(_batch=None, embed=lambda h: embedded, wkv=wkv)
    if batched:
        e._batch = SimpleNamespace(wkv_input=lambda owner, rows: embedded.flatten(-2))
    want = embedded.flatten(-2)[:, :2]
    assert torch.equal(project_kv(e, hashes), want)
    assert seen[0].dtype == embedded.dtype and seen[0].data_ptr() == embedded.data_ptr()
    assert torch.equal(project_rows(e, hashes), want)
