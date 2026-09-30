# Quantization

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices.

!!! tip
    To get started with quantization, see [LLM Compressor](llm_compressor/README.md), a library for optimizing models for deployment with vLLM that supports FP8, INT8, INT4, and other quantization formats.

The following are the supported quantization formats for vLLM:

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [Online Quantization](online.md)
- [AMD Quark](quark.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT Encoder Attention](fp8_vit_attn.md)

## Optional compiled Marlin MoE on SM 8.0

The optional `vllm._ampere_marlin_C` extension provides GLM-5.3-Flash W4A16
MoE kernels for both TP4 and PP4 in one binary. Switching layouts needs a
restart and warmup, not a rebuild. It does not requantize weights.

Builds are opt-in with `VLLM_BUILD_AMPERE_MARLIN=1`. A CUDA toolkit and the
matching PyTorch installation are required; a GPU is not required to build
the SM 8.0 kernels. For an existing source installation:

```bash
python csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py --out vllm
```

The standalone command also accepts `--build-dir DIR` and `--verbose`.
It writes `_ampere_marlin_C.abi3.so`. Runtime never compiles CUDA source.

Both runtime flags default to `0` and are independent:

| Flag | Guarded regime |
| --- | --- |
| `VLLM_GLM5_MARLIN_DECODE_CUDA=1` | TP4 intermediate width 512, 1–32 tokens; PP4 width 2048, exactly 4 or 8 tokens |
| `VLLM_GLM5_MARLIN_PREFILL_CUDA=1` | PP4 intermediate width 2048, 384–2304 tokens inclusive |

`VLLM_GLM5_MARLIN_DECODE_VARIANT` selects the decode reduction order: `orig`
(default) splits the first projection four ways along K into fixed-order fp32
partials, which is faster but sums in a different order from the released
kernels, so decoded text can differ while staying within their accuracy
bounds; `exact` keeps the released Marlin summation order. Both are in the same
library; the startup log names the active variant.

All paths require SM 8.0, bf16 activations, 288 local/global experts, top-8
routing, hidden width 4096, uint4b8 group-128 weights, and SiLU with clamp
10.0. Expert maps, LoRA, activation quantization, zero points, biases,
global scales and router weighting on input are excluded. Decode also
retains the released path when torch-based deterministic alignment is
selected. Unsupported shapes retain the released kernels, including the
TP4 Python split-align/cost-cover prefill improvements when enabled.
The compiled prefill flag can enable its supported PP4 split path without
requiring the older Python prefill flag; an explicitly configured Python
prefill minimum remains respected.
PP4 decode batches other than 4/8 tokens and prefills above 2304 tokens keep
the released path. TP4 prefill never selects the optional compiled kernel.

The decode schedule retains routing and padding semantics, slot-order
summation, shared-expert handling and final reduction. The prefill schedule
retains split alignment, its cost-optimal cover and activation/sum behavior,
using wide tiles only for 32/48/64-row lists. Optional scratch is reserved
per device and stream before CUDA-graph capture, including PP workers.
Decode scratch covers 32 tokens for TP4 and only 8 for PP4. Excluded widths
do not reserve optional scratch: TP4 prefill uses only its released buffers.
Fully masked graph padding rows remain don't-care, as with released kernels.

With both flags off, the optional extension is not imported or loaded.
Explicitly enabling either flag validates the binary at worker startup,
before model loading/serving. A missing binary, mismatched PyTorch/C++ ABI
or CUDA major version, or missing/incompatible operators raises an actionable
`RuntimeError`; it never silently disables the requested extension. CUDA
toolkit minor versions within the same major are accepted. Rebuild against
the current environment or set both flags to `0` and restart to recover.

## Selecting Linear Backends per Quantization

`--linear-backend` selects one backend for all quantized linear layers. For
mixed-precision models that use more than one linear quantization scheme, use
`linear_backend_per_quant` to override the backend for individual schemes:

```bash
vllm serve <model> \
  --linear-backend cutlass \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'
```

Here, NVFP4 W4A16 linear layers use Humming, while all other quantized linear
layers use CUTLASS. Per-quantization overrides take precedence over
`--linear-backend`; schemes without an override continue to use the global
setting, including automatic selection when it is `auto`.

## Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Implementation            | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- ✅︎ indicates that the quantization method is supported on the specified hardware.
- ❌ indicates that the quantization method is not supported on the specified hardware.
- All Intel Gaudi quantization support has been migrated to [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi).
- *Turing does not support Marlin MXFP4.

!!! note
    For information on quantization support on Google TPU, please refer to the [TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) documentation.

!!! note
    This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

    For the most up-to-date information on hardware support and quantization methods, please refer to [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) or consult with the vLLM development team.

## Out-of-Tree Quantization Plugins

vLLM supports registering custom, out-of-tree quantization methods using the `@register_quantization_config` decorator. This allows you to implement and use your own quantization schemes without modifying the vLLM codebase.

### Registering a Custom Quantization Method

To register a custom quantization method, create a class that inherits from `QuantizationConfig` and decorate it with `@register_quantization_config`. The `get_quant_method` dispatches to the appropriate quantize method based on the layer type:

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """Custom quantization config."""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability, -1 for no restriction
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Config files to search for in model directory
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # Create config from model's quantization config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        # NOTE: you only need to implement methods you care about
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### Required QuantizationConfig Methods

Your custom `QuantizationConfig` subclass must implement these abstract methods:

| Method | Description |
| ------ | ----------- |
| `get_name()` | Returns the name of the quantization method |
| `get_supported_act_dtypes()` | Returns list of supported activation dtypes (e.g., `torch.float16`) |
| `get_min_capability()` | Returns minimum GPU compute capability (e.g., 80 for Ampere, -1 for no restriction) |
| `get_config_filenames()` | Returns list of config filenames to search for in model directory |
| `from_config(config)` | Class method to create config from model's quantization config dict |
| `get_quant_method(layer, prefix)` | Returns the quantization method for a given layer, or `None` to skip |

### Implementing a Quantized Linear Method

For linear layers, return a `QuantizeMethodBase` subclass from `get_quant_method`. You can extend `UnquantizedLinearMethod` as a starting point:

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """Custom quantization method for linear layers."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # Create quantized weights for the layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Apply custom quantization logic here
        ...
```

### Implementing a Quantized MoE Method

For Mixture of Experts (MoE) models, return a `FusedMoEMethodBase` subclass from `get_quant_method`. You can use `UnquantizedFusedMoEMethod` to skip MoE quantization:

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """Custom quantization method for MoE layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create quantized weights for the MoE layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Apply MoE computation with quantized weights
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Return the MoE quantization configuration
        ...
```

See existing implementations like `Fp8MoEMethod` in `vllm/model_executor/layers/quantization/fp8.py` for reference.

### Using the Plugin

Once registered, you can use your custom quantization method with vLLM:

```python
# Register your quantization method (import the module containing your config)
import my_quant_plugin

from vllm import LLM

# Use the custom quantization method
llm = LLM(model="your-model", quantization="my_quant")
```

For more information on the plugin system, see the [Plugin System documentation](../../design/plugin_system.md).
