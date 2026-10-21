# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build EXL3 Qwen4Exp target/draft layouts on CPU from checkpoint headers.

Uses the real checkpoint config and EXL3 tensor headers without weight data.
Two gloo ranks construct TP2+EP target/draft models and a vision-enabled PP2
target on the meta device. The parent checks TP weight ownership and PP vision
stage construction. Set ``VLLM_TEST_EXL3_QWEN_CHECKPOINT`` to run it.
"""

import json
import os
import socket
import traceback
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

CHECKPOINT = os.environ.get("VLLM_TEST_EXL3_QWEN_CHECKPOINT")
TP = 2
QSA_MEMBERS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")


def _params_on_meta(module, name, param, register):
    register(module, name, param)
    if param is not None and param.device.type != "meta":
        moved = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
        moved.__dict__.update(param.__dict__)
        module._parameters[name] = moved


def _summary(model):
    from vllm.models.qwen4_exp.nvidia.qsa import (
        Qwen4ExpQSAAttention,
        Qwen4ExpQSAQKVIndexerLinear,
    )
    from vllm.sm80.kernels.exl3.quant import (
        Exl3LinearMethod,
        Exl3MoEMethod,
    )

    def span(part):
        return [part.start, part.stop]

    linears, experts, qsa = [], {}, []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, Exl3LinearMethod):
            linears += [
                {
                    "module": name,
                    "type": type(module).__name__,
                    "matrix": w.matrix.prefix,
                    "padded": [w.matrix.padded_in, w.matrix.padded_out],
                    "input": span(w.input_slice),
                    "outputs": [span(s) for s in w.output_slices],
                    "logical_out": w.logical_out,
                }
                for w in module.exl3_weights
            ]
        elif isinstance(method, Exl3MoEMethod):
            local = {
                w.matrix.prefix: name for expert in module.exl3_experts for w in expert
            }
            experts[name] = {
                "local": sorted(local),
                "nonlocal": sorted({k.rsplit(".", 1)[0] for k in method.nonlocal_keys}),
            }
        if isinstance(module, Qwen4ExpQSAAttention):
            qsa.append(
                {
                    "module": name,
                    "qkv_type": type(module.qkv_proj).__name__,
                    "fused": isinstance(module.qkv_proj, Qwen4ExpQSAQKVIndexerLinear),
                    "indexer_linear": hasattr(module.indexer, "index_qk_proj"),
                    "split": [
                        2 * module.q_size + 2 * module.kv_size,
                        getattr(module, "index_qk_size", 0),
                    ],
                }
            )
    return {"linears": linears, "experts": experts, "qsa": qsa}


def _build(
    rank,
    port,
    checkpoint,
    out,
    tensor_parallel_size,
    pipeline_parallel_size,
    enable_expert_parallel,
    build_draft,
):
    os.environ.update(
        HF_HUB_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        VLLM_DISABLE_SHARED_EXPERTS_STREAM="1",
    )
    if pipeline_parallel_size > 1:
        os.environ["VLLM_PP_LAYER_PARTITION"] = "26,22"
    else:
        os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
    try:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.platforms import current_platform
        from vllm.platforms.interface import DeviceCapability
        from vllm.utils.torch_utils import set_default_torch_dtype

        # Construction only: no CUDA device, NCCL or weight data.
        platform = type(current_platform)
        platform.get_device_capability = classmethod(
            lambda cls, device_id=0: DeviceCapability(8, 0)
        )
        platform.current_device = classmethod(lambda cls: torch.device("meta"))
        platform.get_device_communicator_cls = classmethod(
            lambda cls: (
                "vllm.distributed.device_communicators."
                "base_device_communicator.DeviceCommunicatorBase"
            )
        )
        register = torch.nn.Module.register_parameter
        torch.nn.Module.register_parameter = lambda module, name, param: (
            _params_on_meta(module, name, param, register)
        )
        engine_args = {
            "model": checkpoint,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "enable_expert_parallel": enable_expert_parallel,
            "distributed_executor_backend": "external_launcher",
            "max_model_len": 32768,
            "max_num_batched_tokens": 3460,
            "enforce_eager": True,
            "load_format": "exl3",
            "mm_encoder_tp_mode": "data",
        }
        if build_draft:
            engine_args["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": 3,
            }
        vllm_config = EngineArgs(**engine_args).create_engine_config()
        assert vllm_config.quant_config.get_name() == "exl3"
        result = {}
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=tensor_parallel_size * pipeline_parallel_size,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(
                tensor_parallel_size,
                pipeline_parallel_size,
            )
            with set_default_torch_dtype(vllm_config.model_config.dtype):
                target = initialize_model(vllm_config=vllm_config)
                result["target"] = _summary(target)
                result["target"]["layer_range"] = [
                    target.language_model.model.start_layer,
                    target.language_model.model.end_layer,
                ]
                result["target"]["vision"] = {
                    "language_model_only": target.language_model_only,
                    "uses_data_parallel": target.use_data_parallel,
                    "parameter_count": sum(
                        parameter.numel() for parameter in target.visual.parameters()
                    ),
                }
                if build_draft:
                    spec = vllm_config.speculative_config
                    assert spec is not None
                    draft_config = spec.apply_draft_overrides(vllm_config)
                    draft = initialize_model(
                        vllm_config=draft_config,
                        model_config=spec.draft_model_config,
                    )
                    result["draft"] = _summary(draft)
        Path(out, f"rank{rank}.json").write_text(json.dumps(result))
    except BaseException:
        Path(out, f"rank{rank}.error").write_text(traceback.format_exc())
        raise


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_build(
    tmp_path_factory,
    name,
    *,
    tensor_parallel_size,
    pipeline_parallel_size,
    enable_expert_parallel,
    build_draft,
):
    if CHECKPOINT is None:
        pytest.skip("VLLM_TEST_EXL3_QWEN_CHECKPOINT is not set")
    if not Path(CHECKPOINT, "quantization_config.json").is_file():
        pytest.fail(f"Not an EXL3 checkpoint: {CHECKPOINT}")
    out = tmp_path_factory.mktemp(name)
    context = mp.get_context("spawn")
    port = _free_port()
    world_size = tensor_parallel_size * pipeline_parallel_size
    procs = [
        context.Process(
            target=_build,
            args=(
                rank,
                port,
                CHECKPOINT,
                str(out),
                tensor_parallel_size,
                pipeline_parallel_size,
                enable_expert_parallel,
                build_draft,
            ),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=900)
    errors = sorted(out.glob("rank*.error"))
    if errors:
        pytest.fail("\n".join(e.read_text() for e in errors))
    for proc in procs:
        if proc.is_alive():
            proc.kill()
            pytest.fail("EXL3 construction timed out")
        assert proc.exitcode == 0
    return [
        json.loads((out / f"rank{rank}.json").read_text()) for rank in range(world_size)
    ]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ranks = _run_build(
        tmp_path_factory,
        "exl3_build",
        tensor_parallel_size=TP,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        build_draft=True,
    )
    from vllm.sm80.kernels.exl3.utils import Exl3Manifest

    return Exl3Manifest(CHECKPOINT, require_qwen_aux=True), ranks


@pytest.fixture(scope="module")
def pp2_vision_built(tmp_path_factory):
    return _run_build(
        tmp_path_factory,
        "exl3_pp2_vision",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        build_draft=False,
    )


def test_pp2_meta_construction_keeps_vision_on_first_stage(pp2_vision_built):
    first_stage, second_stage = pp2_vision_built
    assert first_stage["target"]["layer_range"] == [0, 26]
    assert second_stage["target"]["layer_range"] == [26, 48]
    vision = first_stage["target"]["vision"]
    assert vision["language_model_only"] is False
    assert vision["uses_data_parallel"] is True
    assert vision["parameter_count"] > 0


@pytest.mark.parametrize("role", ["target", "draft"])
def test_qsa_projects_q_k_v_and_the_replicated_indexer_in_one_layer(built, role):
    _, ranks = built
    for rank, result in enumerate(ranks):
        layers = result[role]["qsa"]
        assert layers, f"{role} has no QSA attention"
        for attn in layers:
            assert attn["fused"] and not attn["indexer_linear"], attn
            weights = [
                w
                for w in result[role]["linears"]
                if w["module"] == attn["module"] + ".qkv_proj"
            ]
            assert [w["matrix"].rsplit(".self_attn.", 1)[1] for w in weights] == list(
                QSA_MEMBERS
            )
            q, k, v, index = weights
            assert q["outputs"] == [
                [rank * q["logical_out"], (rank + 1) * q["logical_out"]]
            ]
            assert index["outputs"] == [[0, index["padded"][1]]]
            assert attn["split"] == [
                q["logical_out"] + k["logical_out"] + v["logical_out"],
                index["logical_out"],
            ]


def _owner_keys(manifest, role):
    """Packed matrices the role's loader must find a runtime owner for."""
    keys = set()
    for prefix in manifest.matrices:
        is_mtp = prefix.startswith("mtp.")
        if (role == "draft") == is_mtp:
            keys.add(prefix)
    return keys


@pytest.mark.parametrize("role", ["target", "draft"])
def test_every_packed_matrix_has_one_owner_and_ranks_tile_it(built, role):
    manifest, ranks = built
    expected = _owner_keys(manifest, role)
    coverage, local_experts = defaultdict(list), defaultdict(list)
    for rank, result in enumerate(ranks):
        owners = defaultdict(list)
        for w in result[role]["linears"]:
            owners[w["matrix"]].append(w["module"])
            coverage[w["matrix"]].append(w)
        for module, assignment in result[role]["experts"].items():
            for prefix in assignment["local"] + assignment["nonlocal"]:
                owners[prefix].append(module)
            for prefix in assignment["local"]:
                local_experts[prefix].append(rank)
        duplicated = {k: v for k, v in owners.items() if len(v) != 1}
        assert not duplicated, f"rank {rank} {role}: {duplicated}"
        assert set(owners) == expected, (
            f"rank {rank} {role}: unowned {sorted(expected - set(owners))[:8]}, "
            f"foreign {sorted(set(owners) - expected)[:8]}"
        )
    # Expert parallelism places every expert on exactly one rank.
    experts = {p for p in expected if ".mlp.experts." in p}
    assert set(local_experts) == experts
    assert all(len(r) == 1 for r in local_experts.values())
    for prefix, parts in coverage.items():
        assert len(parts) == TP, (prefix, parts)
        padded_in, padded_out = parts[0]["padded"]
        inputs = {tuple(p["input"]) for p in parts}
        outputs = {tuple(map(tuple, p["outputs"])) for p in parts}
        # Either the ranks split one dimension into disjoint H128 shards or
        # every rank holds the whole matrix.
        if len(inputs) == 1:
            assert inputs == {(0, padded_in)}, prefix
            split, length = [p["outputs"] for p in parts], padded_out
        else:
            assert outputs == {((0, padded_out),)}, prefix
            split, length = [[p["input"]] for p in parts], padded_in
        counts = [0] * length
        for spans in split:
            for start, stop in spans:
                for index in range(start, stop):
                    counts[index] += 1
        assert len(set(counts)) == 1 and counts[0] in (1, TP), (prefix, split)
