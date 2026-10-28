# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the sm_80 DeepSeek-V4.1 target and DSpark drafter for one TP/EP rank
on the meta device, from an EXL3 checkpoint directory, without engine
post-init or process groups (the pattern of
tests/models/test_native_checkpoint_config_meta.py)."""

import copy
from dataclasses import dataclass

import torch

from tests.models.test_native_checkpoint_config_meta import _runtime_defaults
from vllm.config import (
    CompilationConfig,
    LoadConfig,
    ModelConfig,
    MultiModalConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.engram import EngramConfig
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.utils.torch_utils import set_default_torch_dtype


@dataclass
class _Group:
    """Rank metadata only; no process group or collective is initialized."""

    world_size: int = 1
    rank_in_group: int = 0
    device_communicator: object = None

    @property
    def is_first_rank(self):
        return self.rank_in_group == 0

    @property
    def is_last_rank(self):
        return self.rank_in_group == self.world_size - 1

    @property
    def device_group(self):
        return self

    def size(self):
        return self.world_size

class _NoStream:
    def __init__(self, *args, **kwargs):
        pass



def set_groups(monkeypatch, tp_size: int, tp_rank: int) -> None:
    """TP ranks with whole-expert EP over the same ranks (DP 1, PP 1)."""
    import vllm.distributed.parallel_state as ps

    tp = _Group(tp_size, tp_rank)
    single = _Group()
    for name, group in {
        "_TP": tp,
        "_ETP": tp,
        "_EP": tp,
        "_PP": single,
        "_DP": single,
        "_DCP": single,
        "_PCP": single,
        "_WORLD": tp,
    }.items():
        monkeypatch.setattr(ps, name, group)
    # The backbone and attention create side streams; construction runs none.
    monkeypatch.setattr(torch.cuda, "Stream", _NoStream)


def _model_config(directory: str, hf_config, runner_type: str) -> ModelConfig:
    config = _runtime_defaults(
        ModelConfig,
        model=directory,
        tokenizer=directory,
        hf_config=hf_config,
        hf_text_config=get_hf_text_config(hf_config),
        dtype=torch.bfloat16,
        max_model_len=2048,
        original_max_model_len=2048,
        runner_type=runner_type,
        convert_type="none",
        multimodal_config=None,
        quantization="exl3",
        skip_tokenizer_init=True,
    )
    config.model_arch_config = config.get_model_arch_config()
    if runner_type == "generate":
        # EXL3 runs vision towers data-parallel (--mm-encoder-tp-mode data).
        config.multimodal_config = MultiModalConfig(mm_encoder_tp_mode="data")
    return config


def _quant_config(directory: str, hf_config):
    from vllm_sm80.kernels.exl3.quant import Exl3Config

    quant = Exl3Config.from_config(hf_config.quantization_config)
    quant.maybe_update_config(directory, hf_config=hf_config)
    return quant


def vllm_configs(directory: str, tp_size: int) -> tuple[VllmConfig, VllmConfig]:
    """The target's config and the DSpark drafter's (its own EXL3 config,
    as load_dspark_model gives it)."""
    hf_config = get_config(directory, trust_remote_code=False)
    target = _model_config(directory, hf_config, "generate")
    draft_hf = copy.deepcopy(hf_config)
    draft_hf.architectures = ["DSparkV41DraftModel"]
    draft = _model_config(directory, draft_hf, "draft")
    parallel = _runtime_defaults(
        ParallelConfig,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
    )
    scheduler = SchedulerConfig(
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        is_encoder_decoder=False,
    )
    speculative = _runtime_defaults(
        SpeculativeConfig,
        method="dspark",
        num_speculative_tokens=4,
        target_model_config=target,
        target_parallel_config=parallel,
        draft_model_config=draft,
        draft_parallel_config=parallel,
    )

    def build(model_config, quant):
        config = _runtime_defaults(
            VllmConfig,
            model_config=model_config,
            parallel_config=parallel,
            scheduler_config=scheduler,
            compilation_config=CompilationConfig(
                mode=0, cudagraph_mode="NONE", custom_ops=["all"]
            ),
            load_config=LoadConfig(load_format="exl3"),
            quant_config=quant,
            speculative_config=speculative,
            # Device tables: the meta construction pins no host memory (the
            # sm_80 Engram storage replaces the table either way).
            engram_config=EngramConfig(cpu_offload=False),
        )
        config.kernel_config.enable_jit_warmup = False
        config.attention_config.sparse_mla_prefill_workspace_factor = 5
        return config

    return (
        build(target, _quant_config(directory, hf_config)),
        build(draft, _quant_config(directory, draft_hf)),
    )


def build_model(config: VllmConfig) -> torch.nn.Module:
    """The registered (fork) class of the config's architecture, built."""
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_sm80 import plugin

    plugin.register_models()
    architecture = config.model_config.hf_config.architectures[0]
    model_cls = ModelRegistry._try_load_model_cls(architecture)
    assert model_cls is not None, architecture
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        return model_cls(vllm_config=config, prefix="")


class _EngramTable(torch.nn.Module):
    """A parameter-free Engram table, as the sm_80 Engram storage builds
    (its rows come from the base checkpoint, not from the EXL3 directory)."""

    def __init__(self, n_hash_cols: int, tp_size: int):
        super().__init__()
        self.dp_size = 1
        self.tp_size = tp_size
        self.n_hash_cols = n_hash_cols
        self.part_n_hash_cols = n_hash_cols // tp_size
        self.cpu_offload = True


def stub_engram_tables(monkeypatch, tp_size: int) -> None:
    """Build Engram layers with :class:`_EngramTable` tables: the sm_80
    storage opens the base checkpoint's table files, which no construction
    test has."""
    from vllm.models.deepseek_v41.common import engram

    def create(self, layout, layer_hash_index):
        return _EngramTable(
            (layout.max_ngram_size - 1) * layout.n_heads, tp_size
        )

    monkeypatch.setattr(engram.Engram, "_create_embedding", create)
    # The n-gram hash state loads the tokenizer: runtime state, not weights.
    from vllm.models.deepseek_v41.nvidia import model

    monkeypatch.setattr(model, "NgramHashState", lambda *args, **kwargs: None)
