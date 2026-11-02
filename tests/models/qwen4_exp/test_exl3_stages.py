# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the stage-local model changes an EXL3 checkpoint gets under
pipeline parallelism (``sm80_model.adopt_exl3_stages``,
``sm80_model.adopt_pp_failed_tokens``)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm_sm80.models.qwen4_exp import sm80_model


def _pp_config(*, pp=2, quantization="exl3", spec_method="mtp", ple=(2,)):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        model_config=SimpleNamespace(
            quantization=quantization,
            hf_text_config=SimpleNamespace(ple_layer_ids=list(ple)),
        ),
        speculative_config=(
            None if spec_method is None else SimpleNamespace(method=spec_method)
        ),
    )


def _text_model(*, shares=True):
    lm = nn.Module()
    lm.config = SimpleNamespace(hidden_size=8)
    lm.quant_config = SimpleNamespace(mtp_shares_target_embeddings=lambda c: shares)
    lm.model = nn.Module()
    lm.model.vocab_size = 16
    lm.model.embed_tokens = PPMissingLayer()
    lm.model.hyper_connection_mixer = None
    return lm


@pytest.fixture
def stage(monkeypatch):
    def set_stage(is_last: bool) -> None:
        monkeypatch.setattr(
            sm80_model, "get_pp_group", lambda: SimpleNamespace(is_last_rank=is_last)
        )

    return set_stage


def test_exl3_first_stage_accounts_for_the_remote_mixer(stage):
    stage(False)
    lm = _text_model()
    sm80_model.adopt_exl3_stages(lm, _pp_config())
    assert isinstance(lm.model.hyper_connection_mixer, PPMissingLayer)
    assert isinstance(lm.model.embed_tokens, PPMissingLayer)


def test_exl3_last_stage_keeps_the_embedding_its_drafter_shares(stage, monkeypatch):
    built = []

    def embedding(vocab_size, hidden_size):
        built.append((vocab_size, hidden_size))
        return nn.Identity()

    monkeypatch.setattr(sm80_model, "VocabParallelEmbedding", embedding)
    stage(True)
    lm = _text_model()
    sm80_model.adopt_exl3_stages(lm, _pp_config())
    assert built == [(16, 8)]
    # A drafter with its own embedding, or none at all, takes none from here.
    for lm, config in (
        (_text_model(shares=False), _pp_config()),
        (_text_model(), _pp_config(spec_method=None)),
    ):
        sm80_model.adopt_exl3_stages(lm, config)
        assert isinstance(lm.model.embed_tokens, PPMissingLayer)
    assert len(built) == 1


@pytest.mark.parametrize(
    "config", [_pp_config(pp=1), _pp_config(quantization="auto-round")]
)
def test_other_layouts_keep_upstream_stage_modules(stage, config):
    stage(True)
    lm = _text_model()
    sm80_model.adopt_exl3_stages(lm, config)
    assert lm.model.hyper_connection_mixer is None
    assert isinstance(lm.model.embed_tokens, PPMissingLayer)


def _empty_tensors(batch_size, dtype, device):
    return IntermediateTensors(
        {"hidden_states": torch.zeros(batch_size, 4, dtype=dtype, device=device)}
    )


def test_pp_failed_tokens_field_joins_the_intermediate_tensors():
    model = nn.Module()
    model.make_empty_intermediate_tensors = _empty_tensors
    sm80_model.adopt_pp_failed_tokens(model, _pp_config())
    assert model.pp_failed_tokens
    tensors = model.make_empty_intermediate_tensors(3, torch.float32, "cpu")
    assert set(tensors.tensors) == {"hidden_states", "failed_tokens"}
    assert tensors["failed_tokens"].dtype == torch.bool
    assert not tensors["failed_tokens"].any()


@pytest.mark.parametrize(
    "config",
    [_pp_config(pp=1), _pp_config(quantization=None), _pp_config(ple=())],
)
def test_pp_failed_tokens_field_only_for_exl3_ple_pipelines(config):
    model = nn.Module()
    model.make_empty_intermediate_tensors = _empty_tensors
    sm80_model.adopt_pp_failed_tokens(model, config)
    assert not getattr(model, "pp_failed_tokens", False)
    assert model.make_empty_intermediate_tensors is _empty_tensors
