# File-backed PLE

`VLLM_SM80_PLE_MMAP=1` (default `0`) keeps a Qwen4Exp checkpoint's BF16 or FP16
PLE table in its safetensors files instead of loading it into host or device
memory. The Qwen3.8-Flash-Next W4A16 AutoRound checkpoint keeps PLE out of
its INT4 quantization: its 320,001,536 × 160 BF16 table is about 102.4 GB,
more than the host RAM of a typical CMP 170HX machine.

With the switch set, each step gathers only the rows its tokens select from
the memory-mapped checkpoint shards into a persistent GPU buffer before the
forward (`vllm_sm80/models/qwen4_exp/mmap_ple.py`, staged by
`vllm_sm80/models/qwen4_exp/ple_staging.py`). Captured forwards read that
buffer, so CUDA graph replay, FULL graphs included, performs no file reads or
host transfers. Dummy and profiling batches zero the rows.

The switch selects the safetensors loader with `safetensors_load_strategy`
`lazy`, which keeps the shards mapped. Startup refuses any other combination
instead of falling back:

- a model without PLE layers;
- a checkpoint whose quantization brings its own PLE storage, such as EXL3,
  which reads its compressed rows from the SSD (see
  [EXL3](quantization/exl3.md#compressed-ple-placement));
- a tensor-parallel size other than 1, or sequence-parallel PLE;
- a load format other than `auto` or `safetensors`, or a non-lazy safetensors
  load strategy.

Without the switch, upstream's PLE storage is unchanged.

## Qwen3.8-Flash-Next W4A16 at PP2

The W4A16 AutoRound checkpoint has been validated at
`--tensor-parallel-size 1 --pipeline-parallel-size 2` with native MTP, vision,
prefix caching and FULL_AND_PIECEWISE CUDA graphs. Every PLE layer must belong
to the first pipeline stage; `26,22` is the tested partition of the 48 layers.
The measured configuration:

```bash
VLLM_SM80_PLE_MMAP=1 \
VLLM_PP_LAYER_PARTITION=26,22 \
VLLM_PP_HOP_NO_METADATA=1 VLLM_PP_PACKED_HOP=1 \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
VLLM_SM80_ALLREDUCE=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0 \
VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1 \
VLLM_SM80_TOPK_CANONICAL_TIES=1 VLLM_SM80_TOPK_SORTED=1 \
VLLM_THIN_GEMM=1 VLLM_THIN_GEMM_GLUON=1 \
vllm serve /path/to/Qwen3.8-Flash-Next-W4A16-AutoRound \
  --served-model-name qwen3.8-flash-next-w4a16-pp2 \
  --distributed-executor-backend mp --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --enable-expert-parallel --mm-encoder-tp-mode data \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --block-size 1680 --mamba-ssm-cache-dtype float32 \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --long-prefill-token-threshold 0 --enable-prompt-tokens-details \
  --enable-prefix-caching --prefix-match-unit 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}'
```

`--prefix-match-unit 4` is the measured value, not a requirement; it must
divide the resolved prefix-cacheable block sizes. The EXL3 checkpoint's PP2
configuration is in [EXL3](quantization/exl3.md#experimental-pp2).
