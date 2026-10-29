# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FP32 outputs of the sm_80 DeepSeek-V4.1 port against the reference
formulas in FP64 (CPU).

The reference (DeepSeek's inference/model.py) computes three projections in
FP32: the LM head (``ParallelHead``: an FP32 weight applied to ``x.float()``),
the ratio-2 compressor's ``wkv`` and ``wgate`` (FP32 linears on ``x.float()``,
FP32 softmax pooling) and the DSpark Markov head (an FP32 ``ParallelHead``
whose logits are added to the FP32 logits). Each numeric test runs the port's
path and checks that its error against the formula evaluated in FP64 is no
larger than the reference's own FP32 evaluation (mean 1.10x, max 1.25x), and
that the BF16 rounding the port did before is worse.

The EXL3 operator is replaced by an FP32 GEMM of the dequantised weight, so
the head test covers everything after the kernel: the output dtype the layer
asks for, the operator's output buffer, the logits processor and the
vocabulary crop. The kernel's FP32 store is checked against FP64 on the GPU
(tests/kernels/quantization/test_exl3_kernels.py).

    pytest -q tests/models/deepseek_v41/sm80/test_fp32_outputs.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.model_executor.layers.test_exl3_layers import (
    _matrix_entries,
    _tp,
    _write_sparse,
)
from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm_sm80.kernels.exl3 import ops as exl3_ops
from vllm_sm80.kernels.exl3 import quant as exl3_quant

BF16, FP32, FP64 = torch.bfloat16, torch.float32, torch.float64
ARCH = "DeepseekV41ForCausalLM"
MEAN_BOUND, MAX_BOUND = 1.10, 1.25


def _errors(actual: torch.Tensor, exact: torch.Tensor) -> tuple[float, float]:
    error = (actual.double() - exact).abs()
    return error.mean().item(), error.max().item()


def _no_worse(actual, reference, exact) -> None:
    """``actual``'s error against ``exact``: at most MEAN_BOUND (mean) and
    MAX_BOUND (max) times ``reference``'s."""
    mean, worst = _errors(actual, exact)
    ref_mean, ref_worst = _errors(reference, exact)
    assert mean <= MEAN_BOUND * ref_mean, (mean, ref_mean)
    assert worst <= MAX_BOUND * ref_worst, (worst, ref_worst)


def _bf16_weight(g, rows, cols, scale) -> torch.Tensor:
    """A BF16 checkpoint weight (the reference promotes it to FP32 exactly)."""
    return (torch.randn(rows, cols, generator=g, dtype=FP64) * scale).to(BF16)


# -- the model wiring (first: its meta construction needs CUDA uninitialised) --


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def test_the_model_wires_the_fp32_outputs(
    monkeypatch,
    meta_platform,  # noqa: F811
    checkpoint,
):
    """With the plugin's head dtype: the target's EXL3 head stores FP32 and
    both models' processors serve it; every pooling (ratio-2) compressor
    projects in FP32, every ratio-1 one in BF16."""
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm_sm80.models.deepseek_v41.sm80_attention import (
        DeepseekV41Sm80Attention,
    )
    from vllm_sm80 import plugin

    plugin._register_exl3()
    monkeypatch.setitem(MODELS_CONFIG_MAP, ARCH, MODELS_CONFIG_MAP[ARCH])
    plugin.register_hf_config_defaults()
    construct.set_groups(monkeypatch, 1, 0)
    construct.stub_engram_tables(monkeypatch, 1)
    target_config, draft_config = construct.vllm_configs(str(checkpoint), 1)
    # load_dspark_model builds the drafter under the target's model config;
    # here under its copy, which the hook gives the same head dtype.
    for config in (target_config, draft_config):
        MODELS_CONFIG_MAP[ARCH].verify_and_update_model_config(config.model_config)
        assert config.model_config.head_dtype == FP32
    target = construct.build_model(target_config)
    draft = construct.build_model(draft_config)

    language = target.language_model
    assert language.lm_head.quant_method.out_dtype == FP32
    for processor in (language.logits_processor, draft.logits_processor):
        assert type(processor) is exl3_quant.Exl3HeadLogitsProcessor
        assert processor.head_dtype == FP32
    ratios = {}
    for model in (target, draft):
        for name, module in model.named_modules():
            compressor = getattr(module, "compressor", None)
            if not isinstance(module, DeepseekV41Sm80Attention) or compressor is None:
                continue
            method = compressor.fused_wkv_wgate.quant_method
            assert isinstance(method, exl3_quant.Exl3LinearMethod), name
            ratios[name] = (compressor.compress_ratio, method.out_dtype)
    assert {r for r, _ in ratios.values()} == {1, 2}
    for name, (ratio, out_dtype) in ratios.items():
        assert out_dtype == (FP32 if ratio > 1 else BF16), name


# -- the plugin's head dtype ------------------------------------------------


def test_plugin_defaults_the_head_dtype_to_fp32(monkeypatch):
    from transformers import PretrainedConfig

    from vllm.config.model import _get_head_dtype
    from vllm.model_executor.models import config as models_config
    from vllm_sm80 import plugin

    registered = models_config.MODELS_CONFIG_MAP[ARCH]
    upstream = getattr(registered, "replaced_hook", registered)
    monkeypatch.setitem(models_config.MODELS_CONFIG_MAP, ARCH, upstream)
    plugin.register_hf_config_defaults()
    hook = models_config.MODELS_CONFIG_MAP[ARCH]
    assert hook.replaced_hook is upstream and issubclass(hook, upstream)
    plugin.register_hf_config_defaults()
    assert models_config.MODELS_CONFIG_MAP[ARCH] is hook

    def head_dtype(**attrs):
        hf = PretrainedConfig(**attrs)
        hook.verify_and_update_model_config(
            SimpleNamespace(hf_config=hf, hf_text_config=hf)
        )
        return _get_head_dtype(config=hf, dtype=BF16, runner_type="generate")

    assert head_dtype() == FP32
    # A checkpoint or --hf-overrides value wins.
    assert head_dtype(head_dtype="model") == BF16


# -- (1) the LM head ---------------------------------------------------------

HIDDEN, VOCAB, PADDED_VOCAB = 256, 600, 640


@pytest.fixture
def exl3_head_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(8, 0),
    )
    exl3_quant._require_sm80.cache_clear()
    _write_sparse(
        tmp_path / "model.safetensors",
        _matrix_entries("lm_head", HIDDEN, PADDED_VOCAB, 12),
    )
    config = exl3_quant.Exl3Config()
    config.maybe_update_config(str(tmp_path))
    _tp(monkeypatch, 0, 1)
    yield config
    exl3_quant._require_sm80.cache_clear()


def _exl3_head(monkeypatch, config, head_dtype):
    """An EXL3 LM head built while the model's head dtype is ``head_dtype``."""
    import vllm.config

    current = SimpleNamespace(
        load_config=SimpleNamespace(load_format="exl3"),
        model_config=SimpleNamespace(head_dtype=head_dtype, multimodal_config=None),
    )
    with monkeypatch.context() as patch:
        patch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: current)
        return ParallelLMHead(
            VOCAB, HIDDEN, params_dtype=BF16, quant_config=config, prefix="lm_head"
        )


def test_the_exl3_head_refuses_other_dtypes(exl3_head_config, monkeypatch):
    with pytest.raises(ValueError, match="BF16 or FP32"):
        _exl3_head(monkeypatch, exl3_head_config, torch.float16)


@pytest.fixture
def fp32_operator(monkeypatch):
    """``torch.ops.vllm.exl3_linear`` on CPU: the operator's Python body
    (ops._linear: output buffer, row chunks, scratch) around an FP32 GEMM of
    the dequantised weight ``state.weight``, which records the outputs it
    stores (``state.stored``)."""
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace

    monkeypatch.setattr(torch_utils, "aux_stream", lambda: None)
    monkeypatch.setattr(workspace, "_manager", None)
    workspace.init_workspace_manager(torch.device("cpu"))
    state = SimpleNamespace(weight=None, stored=[])

    def native(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, ws
    ):
        out.copy_(F.linear(x.float(), state.weight[:logical_out].float()))
        state.stored.append(out.clone())

    monkeypatch.setattr(exl3_ops, "linear_out", native)
    monkeypatch.setattr(torch.ops.vllm, "exl3_linear", exl3_ops._linear)
    yield state
    workspace.reset_workspace_manager()


def _processor(head_dtype, vocab=VOCAB) -> exl3_quant.Exl3HeadLogitsProcessor:
    processor = LogitsProcessor(vocab)
    processor.head_dtype = head_dtype
    exl3_quant.Exl3HeadLogitsProcessor.adopt(processor)
    return processor


def test_lm_head_logits_are_the_fp32_gemm(
    exl3_head_config, monkeypatch, fp32_operator, default_vllm_config
):
    """FP32 logits from the EXL3 head, unrounded through the processor: the
    reference's F.linear(x.float(), W) of the same dequantised weight."""
    g = torch.Generator().manual_seed(1)
    # Dequantised EXL3 weights are not BF16 numbers; the FP32 GEMM takes
    # them in FP32.
    weight = _bf16_weight(g, PADDED_VOCAB, HIDDEN, 0.05).double()
    weight = (
        weight + torch.randn(weight.shape, generator=g, dtype=FP64) * 1e-4
    ).float()
    fp32_operator.weight = weight
    x = (torch.randn(5, HIDDEN, generator=g) * 3).to(BF16)

    head = _exl3_head(monkeypatch, exl3_head_config, FP32)
    head.quant_method.row_chunks = [2]  # several launches
    logits = _processor(FP32)(head, x)

    # The operator's FP32 stores reach the caller unchanged.
    stored = fp32_operator.stored
    assert [s.dtype for s in stored] == [FP32] * 3
    assert logits.dtype == FP32 and torch.equal(logits, torch.cat(stored))
    official = F.linear(x.float(), weight)[:, :VOCAB]
    exact = x.double() @ weight[:VOCAB].double().T
    _no_worse(logits, official, exact)
    before = official.to(BF16)  # the BF16 head the port had
    assert _errors(before, exact)[0] > 100 * _errors(logits, exact)[0]


def test_lm_head_dtype_mismatch_is_refused(
    exl3_head_config, monkeypatch, fp32_operator, default_vllm_config
):
    head = _exl3_head(monkeypatch, exl3_head_config, BF16)
    with pytest.raises(ValueError, match="head dtype"):
        _processor(FP32)(head, torch.zeros(1, HIDDEN, dtype=BF16))


# -- (2) the ratio-2 compressor ----------------------------------------------


def _official_compress(x, wkv, wgate, norm_weight, eps, dtype):
    """model.py Compressor.forward at ratio 2 over whole groups (start_pos 0):
    FP32 projections of x.float(), FP32 softmax pooling, rounded to the
    activation dtype and normalised (RMSNorm). ``dtype`` FP64 evaluates the
    same formula exactly (no rounding)."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    xf = x.to(dtype)
    kv, score = F.linear(xf, wkv.to(dtype)), F.linear(xf, wgate.to(dtype))
    kv, score = kv.unflatten(0, (-1, 2)), score.unflatten(0, (-1, 2))
    pooled = (kv * score.softmax(dim=1)).sum(dim=1)
    if dtype == FP64:
        var = pooled.square().mean(-1, keepdim=True)
        return norm_weight.double() * (pooled * torch.rsqrt(var + eps))
    return core.rms_norm(pooled.to(x.dtype), norm_weight, eps)


def _port_compress(kv_score, norm_weight, eps, d):
    """sm80_attn_core.compress_rows on one request whose chunk is the whole
    prompt; the closing rows' latents."""
    from vllm_sm80.models.deepseek_v41.ops import sm80_attn_core as core

    T = kv_score.shape[0]
    pos = torch.arange(T)
    capacity = 2
    ring = torch.zeros(2, capacity, 2 * d)
    closes, latent = core.compress_rows(
        kv_score,
        norm_weight,
        ring,
        pos,
        torch.zeros(T, dtype=torch.int32),
        torch.tensor([0, T]),
        pos,
        capacity + pos % capacity,
        2,
        d,
        eps,
    )
    assert torch.equal(closes, (pos + 1) % 2 == 0)
    return latent[closes]


def test_ratio2_compressor_projects_in_fp32():
    g = torch.Generator().manual_seed(2)
    T, hidden, d, eps = 64, 256, 64, 1e-6
    x = torch.randn(T, hidden, generator=g).to(BF16)
    wkv = _bf16_weight(g, d, hidden, 0.1)
    # Gate scores of several units: their rounding moves the pooling weights.
    wgate = _bf16_weight(g, d, hidden, 0.3)
    norm_weight = (1 + 0.1 * torch.randn(d, generator=g)).to(BF16)

    # The layer's projection: the EXL3 linear's FP32 store (here the FP32
    # GEMM of the dequantised weights), as _sm80_projections passes it on.
    kv_score = torch.cat(
        [F.linear(x.float(), wkv.float()), F.linear(x.float(), wgate.float())], -1
    )
    port = _port_compress(kv_score, norm_weight, eps, d)
    official = _official_compress(x, wkv, wgate, norm_weight, eps, FP32)
    exact = _official_compress(x, wkv, wgate, norm_weight, eps, FP64)
    assert port.dtype == BF16 and torch.equal(port, official)
    _no_worse(port, official, exact)
    # The BF16 projection the port had rounds kv and the gate scores first.
    before = _port_compress(kv_score.to(BF16).float(), norm_weight, eps, d)
    assert _errors(before, exact)[0] > _errors(port, exact)[0]


# -- (3) the DSpark Markov head ----------------------------------------------


def test_markov_bias_is_an_fp32_head_on_fp32_logits(default_vllm_config):
    """The drafter's processor (head dtype FP32) runs the unquantized Markov
    head as the reference's FP32 ParallelHead, F.linear(embed.float(), W),
    and the bias adds to the FP32 base logits in FP32."""
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead

    g = torch.Generator().manual_seed(3)
    vocab, rank, rows = 640, 128, 6
    markov = DSparkMarkovHead(vocab, vocab, rank, prefix="markov_head")
    markov.markov_w1.weight.data = _bf16_weight(g, vocab, rank, 1.0)
    markov.markov_w2.weight.data = _bf16_weight(g, vocab, rank, 0.1)
    tokens = torch.randint(0, vocab, (rows,), generator=g)
    base = torch.randn(rows, vocab, generator=g) * 4  # the FP32 head's logits

    embed = markov.embed(tokens)
    bias = markov.bias(embed, _processor(FP32, vocab))
    official = F.linear(embed.float(), markov.markov_w2.weight.float())
    assert bias.dtype == FP32 and torch.equal(bias, official)
    exact = embed.double() @ markov.markov_w2.weight.double().T
    _no_worse(bias, official, exact)
    assert (
        _errors(markov.bias(embed, _processor(BF16, vocab)), exact)[0]
        > 100 * (_errors(bias, exact)[0])
    )
    logits = base + bias
    assert logits.dtype == FP32
    _no_worse(logits, base + official, base.double() + exact)
