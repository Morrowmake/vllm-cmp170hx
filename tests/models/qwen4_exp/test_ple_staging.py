# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of Qwen4Exp PLE storage selection and per-step row staging
(``vllm_sm80.models.qwen4_exp.ple_staging``)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.compilation import CUDAGraphMode
from vllm_sm80.models.qwen4_exp import ple_staging
from vllm_sm80.models.qwen4_exp.mmap_ple import Qwen4ExpPLEMmapEmbedding
from vllm_sm80.plugin import register_fork_env_vars

register_fork_env_vars()


@pytest.fixture
def mmap_switch(monkeypatch):
    def set_switch(on: bool) -> None:
        monkeypatch.setenv("VLLM_SM80_PLE_MMAP", "1" if on else "0")

    return set_switch


# ---- storage selection and loader configuration ------------------------------


def test_embedding_cls_follows_quantization_then_switch(mmap_switch):
    class Default: ...

    class Quantized: ...

    mmap_switch(False)
    assert ple_staging.embedding_cls(None, Default) is Default
    quant = SimpleNamespace(ple_embedding_cls=Quantized)
    assert ple_staging.embedding_cls(quant, Default) is Quantized
    mmap_switch(True)
    assert ple_staging.embedding_cls(None, Default) is Qwen4ExpPLEMmapEmbedding
    with pytest.raises(ValueError, match="brings its own PLE storage"):
        ple_staging.embedding_cls(quant, Default)


def _config(
    *,
    ple_layer_ids=(2,),
    quantization=None,
    tp=1,
    load_format="auto",
    strategy=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_layer_ids=list(ple_layer_ids)),
            quantization=quantization,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        load_config=SimpleNamespace(
            load_format=load_format, safetensors_load_strategy=strategy
        ),
    )


def test_configure_selects_the_lazy_safetensors_loader(mmap_switch):
    mmap_switch(False)
    untouched = _config()
    ple_staging.configure(untouched)
    assert untouched.load_config.load_format == "auto"
    mmap_switch(True)
    config = _config()
    ple_staging.configure(config)
    assert config.load_config.load_format == "safetensors"
    assert config.load_config.safetensors_load_strategy == "lazy"
    explicit = _config(load_format="safetensors", strategy="lazy")
    ple_staging.configure(explicit)


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"ple_layer_ids": ()}, "PLE layers"),
        ({"quantization": "exl3"}, "EXL3"),
        ({"tp": 2}, "tensor_parallel_size=1"),
        ({"load_format": "dummy"}, "safetensors loader"),
        ({"strategy": "eager"}, "lazy"),
    ],
)
def test_configure_rejects_what_mmap_cannot_serve(mmap_switch, fields, match):
    mmap_switch(True)
    with pytest.raises(ValueError, match=match):
        ple_staging.configure(_config(**fields))


# ---- per-step staging ------------------------------------------------------------


class _Storage:
    """A staging PLE storage that records the runner-side calls."""

    staging_enabled = True

    def __init__(self, staged_modes=None):
        self.staged_modes = staged_modes
        self.calls: list[tuple] = []

    def stages(self, cg_mode) -> bool:
        return self.staged_modes is None or cg_mode in self.staged_modes

    def finalize_staging(self) -> None:
        self.calls.append(("finalize",))

    def zero_rows(self, tokens: int) -> None:
        self.calls.append(("zero", tokens))

    def prepare(self, ids: torch.Tensor, padded: int) -> None:
        self.calls.append(("prepare", ids.clone(), padded))


class _NGram(nn.Module):
    """The parts of Qwen4ExpNGramEmbedding that staging uses."""

    def __init__(self, storage, capacity=8, heads=2):
        super().__init__()
        self.ngram_embedding = storage
        self._prefetch_ids = torch.full((capacity, heads), -1, dtype=torch.long)
        self.inputs: list[tuple] = []

    def compute_ngram_ids(self, input_ids, query_start_loc, ngram_context, output):
        self.inputs.append((input_ids.tolist(), query_start_loc, ngram_context))
        output.copy_(input_ids.unsqueeze(1).expand_as(output) * 10)
        return output


def _model(*storages):
    model = nn.Module()
    model.layers = nn.ModuleList(_NGram(storage) for storage in storages)
    return model


def _batch(req_ids, num_tokens, padded):
    return SimpleNamespace(
        req_ids=req_ids,
        num_tokens=num_tokens,
        num_tokens_after_padding=padded,
        input_ids=torch.arange(1, padded + 1),
    )


_INPUTS = {"query_start_loc": "qsl", "ngram_context": "ctx"}


def test_staged_modules_are_found_once_per_model():
    always, unstaged = _Storage(), SimpleNamespace(staging_enabled=False)
    model = _model(always)
    model.other = nn.Module()
    model.other.ngram_embedding = unstaged
    assert [m.ngram_embedding for m in ple_staging.staged_modules(model)] == [always]
    ple_staging.finalize(model)
    assert always.calls == [("finalize",)]


def test_stage_rows_prepares_the_storages_this_step_stages():
    always, full_only = _Storage(), _Storage({CUDAGraphMode.FULL})
    model = _model(always, full_only)
    states = SimpleNamespace(req_id_to_index={"a": 0, "b": 1})
    batch = _batch(["a", "b"], num_tokens=3, padded=4)

    ple_staging.stage_rows(
        model, batch, states, _INPUTS, CUDAGraphMode.PIECEWISE, dummy_run=False
    )
    assert full_only.calls == []
    (call,) = always.calls
    assert call[0] == "prepare" and call[2] == 4
    assert call[1].tolist() == [[10, 10], [20, 20], [30, 30]]
    assert model.layers[0].inputs == [([1, 2, 3], "qsl", "ctx")]

    ple_staging.stage_rows(
        model, batch, states, _INPUTS, CUDAGraphMode.FULL, dummy_run=False
    )
    assert [c[0] for c in full_only.calls] == ["prepare"]


def test_stage_rows_zeroes_dummy_and_unregistered_batches():
    always, full_only = _Storage(), _Storage({CUDAGraphMode.FULL})
    model = _model(always, full_only)
    states = SimpleNamespace(req_id_to_index={"a": 0})
    for batch, dummy in (
        (_batch(["a"], 1, 2), True),
        (_batch(["profile_0"], 1, 2), False),
    ):
        ple_staging.stage_rows(
            model, batch, states, _INPUTS, CUDAGraphMode.NONE, dummy_run=dummy
        )
    # Every staging storage, whatever the step's mode: buffers exist before
    # capture and no previous request's rows or failures carry over.
    assert always.calls == full_only.calls == [("zero", 2), ("zero", 2)]


def test_stage_rows_rejects_a_partly_unknown_batch():
    model = _model(_Storage())
    states = SimpleNamespace(req_id_to_index={"a": 0})
    with pytest.raises(RuntimeError, match="unknown active request"):
        ple_staging.stage_rows(
            model,
            _batch(["a", "x"], 2, 2),
            states,
            _INPUTS,
            CUDAGraphMode.NONE,
            dummy_run=False,
        )


def test_stage_rows_without_staging_storages_is_a_no_op():
    model = nn.Module()
    ple_staging.stage_rows(
        model, None, None, _INPUTS, CUDAGraphMode.FULL, dummy_run=False
    )


@pytest.mark.parametrize(
    "mode,capturing,expected",
    [
        (CUDAGraphMode.FULL, False, True),
        (CUDAGraphMode.NONE, True, True),
        (CUDAGraphMode.NONE, False, False),
        (CUDAGraphMode.PIECEWISE, True, False),
        (None, False, False),
    ],
)
def test_forward_is_staged_for_full_replay_and_capture(
    monkeypatch, mode, capturing, expected
):
    import vllm.forward_context as forward_context

    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: mode is not None
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    assert ple_staging.forward_is_staged() is expected
