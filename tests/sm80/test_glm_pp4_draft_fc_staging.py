# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter fc fold of GLM-5.3-Flash at PP4 (VLLM_GLM5_PP_FOLD_DRAFT_FC)
under the FULL draft CUDA graph, on the last stage where the drafter runs.

The FULL graph combines its context rows from a fixed staging buffer that the
speculator sizes for the concatenated aux states (5 x 4096 here). With the fold
the last stage's aux input is the finished fc output: one fp32
[tokens, hidden_size] tensor. Capture runs the graph's context step on the
staged rows, and a replay must give the drafter the rows the eager combine
gives: the fp32 fc output rounded once to the model dtype.

The speculator is built on CPU from the deployed checkpoint configs (TP1, PP4,
last rank, FULL decode graphs) and the drafter on meta: the folded combine
reads only the fc geometry and dtype.

Run in a fresh process with CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.models.test_native_checkpoint_config_meta import (  # noqa: F401
    _parallel_groups,
    _registered_model_cls,
    _vllm_config,
    meta_platform,
)
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm_sm80.spec_decode.aux_fc_fold import mark_drafter_aux_fc_folded
from vllm.utils.torch_utils import set_default_torch_dtype

_PP = 4


def _last_stage_speculator(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    _parallel_groups(monkeypatch, 1, 0, _PP, _PP - 1)
    config = _vllm_config("glm53_flash", 1, _PP, with_dflash=True)
    config.compilation_config = CompilationConfig(
        mode=0, cudagraph_mode="FULL_AND_PIECEWISE", custom_ops=["all"]
    )
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        spec = DFlash2Speculator(config, torch.device("cpu"))
        with torch.device("meta"):
            spec.model = _registered_model_cls("DFlash2DraftModel")(
                vllm_config=config, prefix="draft_model"
            )
    # The anchor and the context K/V store follow the combine and need the
    # attention state; only the combined rows are under test.
    monkeypatch.setattr(spec, "prepare_context_anchor", lambda *args: None)
    monkeypatch.setattr(spec, "_precompute_context_kv", lambda *args: None)
    spec.target_input_buffers = SimpleNamespace(query_start_loc=None)
    return spec


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [(8, 64), (8, 20), (2, 40)],
    ids=["all-graph-rows", "fewer-tokens", "rows-past-graph"],
)
def test_captured_context_step_matches_eager_fold(
    monkeypatch, meta_platform, num_reqs, num_tokens
):
    spec = _last_stage_speculator(monkeypatch)
    mark_drafter_aux_fc_folded(spec)
    hidden = spec.model.model.fc.output_size
    num_context = spec._num_graph_context_tokens(num_reqs)

    # Capture: the context step on the staged rows before any input arrives.
    spec._prepare_graph_forward(spec.max_num_reqs, None)

    torch.manual_seed(0)
    fc_out = 8 * torch.randn(num_tokens, hidden, dtype=torch.float32)
    last = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16)
    spec._store_hidden_states(num_tokens, last, [fc_out])
    eager = spec.hidden_states[:num_tokens].clone()
    assert torch.equal(eager, fc_out.to(torch.bfloat16))

    spec.hidden_states.fill_(float("nan"))
    spec._store_hidden_states(num_tokens, last, [fc_out], num_context)
    spec._prepare_graph_forward(num_reqs, None)
    assert torch.equal(spec.hidden_states[:num_tokens], eager)
