# EXL3

EXL3 loads the checkpoint's packed trellis matrices and original FP16 scales without requantizing them. Linear and whole-expert MoE operators accept and return BF16 tensors, with FP32 transforms and accumulation. There is no FP16 activation adapter, persistent decoded-weight cache, or dependency on the standalone ExLlamaV3 runtime.

Each GPU block decodes a 16-row slab of trellis tiles once into shared memory and applies it to a block of up to 128 rows; routed experts are grouped by local expert on the device every step, so each active expert is decoded once per step and block of rows. Every output element is still one FP32 fused multiply-add chain over the input channels in increasing order, so a row's result does not depend on how many rows share the launch or which block shape was chosen. The extension keeps the original scalar kernels as test-only reference operators (`linear_out_reference`, `moe_out_reference`) that the fast operators must match bit for bit.

`VLLM_EXL3_MOE_V2=1` (default `0`) runs the routed experts of MoE calls with at most 64 rows on the tensor cores (`csrc/exl3/moe_v2.cu`, operator `moe_out_v2`), for two configurations only (`ops.MOE_V2_CONFIGS`), on SM 8.0 with BF16 activations and codebook 2: hidden size 2560, expert intermediate size 640, top-10 routing with 256 of 512 experts local, every expert matrix at `bits_x2` 10; and hidden size 5120, intermediate size 2304, top-6 routing with 96 of 384 whole experts local (expert parallel size 4), every expert matrix of a layer at one `bits_x2`, 6 or 4. Each warp decodes its own trellis tiles straight into `mma.m16n8k16` FP16 operands (decoded weights are exact FP16 values) and multiplies them with the activations as two FP16 parts, a high part and the rounded remainder (about 22 significant bits together, not the full FP32 value), after an exact scaling by a power of two per row (per row and 640-column group for the gate and up inputs of hidden-2560 calls of at least 160 routed assignments), starting every 16-deep product from zero and summing in FP32 in a fixed order; the input transform, activation and combine run in FP64 and round once. Results are deterministic and independent of the other rows, but differ from the default kernels in the last bits; at hidden size 5120 every call uses one K-slice layout, so a row's result also does not depend on the call's row count. Any other layer, and every call with more rows, keeps `moe_out`; a startup line per bit width says the path is active, or one says why the gate is closed.

`VLLM_EXL3_MOE_V2_DRAFT=1` (default `0`) runs the DSpark drafter's routed experts of DeepSeek-V4.1-Flash (top-3 of 128 experts, every expert matrix at `bits_x2` 8, codebook 2) on `moe_out_v2` for MoE calls of at most 64 rows on SM 8.0, independently of `VLLM_EXL3_MOE_V2` (`ops.MOE_V2_DRAFT_CONFIGS`). At expert parallel size 4 (intermediate 2304, 32 whole experts local) the calls take stages of their own (`moe_v2.cu` `DraftTiling`): one gate/up scale per 640 columns, one down scale per K slice, row tiles of up to 16 rows whose low FP16 part (tiles of 5 to 16 rows) is summed in the MMA's C operand. The drafter's experts split across tensor-parallel ranks (intermediate 640 or 512, all 128 local) keep `moe_out`. These calls are never folded (`VLLM_EXL3_MOE_V2_FOLD` does not apply). A row's result depends on whether its tile holds 1..4 rows or more; the diagnostic `VLLM_EXL3_MOE_V2_DRAFT_FORCE_ROWS` (0, 8 or 16) gives every tile at least that tile's kernel.

`VLLM_EXL3_MOE_PREFILL_TC=1` (default `0`) runs the routed experts of MoE calls of 65 to 3,460 rows (one prefill chunk at the validated `--max-num-batched-tokens 3460`) in prefill steps on the tensor cores (`csrc/exl3/moe_prefill.cu`), for the hidden-2560 configuration `VLLM_EXL3_MOE_V2` covers (`ops.MOE_PREFILL_CONFIGS`) when every local expert's trellis starts at a 16-byte-aligned address, at tensor parallel size 2 without pipeline parallelism, and not with drafters that use parallel drafting slots (DFlash, DSpark). A step is a prefill step when the scheduler gave some request more than 1 + `num_speculative_tokens` tokens; the row count of a call never decides it, so decode steps of any size keep their kernels. The routed rows of each expert are grouped into 64-row tiles, and each tile's GEMMs run `mma.m16n8k16` FP16 on exactly decoded weights against the same high/low FP16 split of the activations, scaled by a power of two per row and projection, with every 16-deep product starting from zero and summed in FP32 in a fixed order. The input, activation and per-route output transforms are `moe_out`'s FP32 operations; the routes' weighted sum runs in FP64 and rounds once. Results are deterministic and independent of the other rows, but differ from `moe_out` in the last bits. The MoE workspace grows to fit the operator (at 3,460 rows per call, 846 MiB instead of 508 MiB per GPU). Calls of at most 64 rows keep `moe_out_v2` or `moe_out`, larger calls keep `moe_out`, and prefill steps padded to a captured CUDA graph size replay the kernels captured for decode; a startup line says whether the path is active or why the gate is closed.

`VLLM_DEEPSEEK_V41_MOE_PREFILL_TC=1` (default `0`) runs the routed experts of DeepSeek-V4.1-Flash MoE calls of 65 to 2,048 rows in prefill steps on the tensor cores (`csrc/exl3/moe_prefill_v2.cu`, `moe_out_prefill_v2`): `moe_out_prefill` generalised to hidden 5120, intermediate 2304 and every routed matrix of a layer at `bits_x2` 6 or 4 with codebook 2, at tensor parallel size 4 without pipeline parallelism, with 16-byte-aligned trellis bases, and, with a drafter that drafts in parallel slots, only when no decode step can carry 65 rows. Calls run in row chunks of 512 rows, so its workspace fits in the MoE workspace of 512-row `moe_out` chunks. Decode steps, graph captures, calls of at most 64 rows and larger calls keep their kernels. Results differ from `moe_out` in the last bits.

`VLLM_EXL3_EXPERTS_CONTIGUOUS=1` (default `0`) allocates the trellis, `suh` and `svh` of a MoE layer's local expert matrices as views into one device buffer per layer and tensor kind instead of one allocation per tensor. Each view starts at a 512-byte-aligned offset, the alignment of a caching-allocator block, so the kernels, the per-expert pointer table and the loaded bytes are the same; only the allocator stops leaving holes between the tensors. With routed-expert trellis tensors of 1 to 10 MiB the PyTorch caching allocator serves each from a 20 MiB segment, so several MiB of each segment can stay unusable (about 3.7 GiB per GPU for DeepSeek-V4.1-Flash 2.9 bpw at expert parallel size 4).

The `moe_routed_gemm/exl3_chunk_rows` row of the sm80 tuning table (`vllm_sm80/tuning/`, keyed by the hidden size; without a row `0`, whole calls) set to N runs `moe_out` on calls of more than N rows in row chunks of at most N, so its workspace is sized for N rows instead of the most rows one call can dispatch (`--max-num-batched-tokens` per rank). `moe_out` computes every row from its own input and routes alone, so the output bits do not change; larger prefill calls launch once per chunk. `moe_out_v2` and `moe_out_prefill` calls are not chunked.

EXL3 runs on the extension `vllm_sm80._exl3_C`, built for SM 8.0 only; selecting EXL3 on any other GPU is an error. The vLLM build does not compile it: build it ahead of time with `python csrc/exl3/build_standalone.py --out vllm` from the source tree, with the PyTorch that serves (CUDA-enabled PyTorch, a matching CUDA toolkit, a C++20 compiler and Ninja; no GPU needed). Serving never compiles this extension. Its operators are registered through the non-stable `TORCH_LIBRARY` API, so loading refuses a library whose recorded PyTorch version, CUDA major version or C++ ABI differs from the running PyTorch. A successful build, CPU helper check or Meta dispatch is not GPU numerical, graph, quality or capacity validation. Validate those paths before deployment.

## Enablement and checkpoint loading

EXL3 is registered by the fork's vLLM plugin (`vllm_sm80/plugin.py`, entry point `sm80` of the `vllm-sm80` distribution in `plugins/sm80/`): the quantization method `exl3`, which a checkpoint whose `quant_method` is `exl3` selects, and the load format `exl3`. Without the plugin such a checkpoint is an unknown quantization method at startup, never an unquantized model.

An EXL3 checkpoint always loads with EXL3; there is no switch and no fallback to another quantization method. A missing or incompatible compiled extension is also an error. EXL3 needs `--load-format exl3`, and models with a vision tower also need `--mm-encoder-tp-mode data`; building the model without them is an error. Off the CUDA platform, model configuration refuses EXL3 checkpoints.

```bash
vllm serve ./exl3-checkpoint \
  --quantization exl3 --load-format exl3 --dtype bfloat16
```

The checkpoint must already be a local safetensors directory. Loading inventories actual file headers, verifies packed geometry/codebook markers and accounts for consumed, file-backed and explicitly unused tensors; the counts are logged per model. The inventory is released after the last model that uses it is loaded: the target, or the MTP draft when a speculative draft is configured (the draft is built with the target's configuration). An index cannot hide auxiliary files or override their headers. Integer rates from 1 through 8 bits are supported; 1.5-, 2.5- and 3.5-bit formats require the `mul1` codebook.

Tensor-parallel cuts must preserve complete 128-channel transform blocks. Routed experts use whole-expert expert parallelism rather than splitting each expert's transform. Pipeline parallelism other than one and LoRA are rejected. Ordinary embeddings and active non-PLE weights remain on the GPU; general CPU weight offload is not supported by this path.

Dense layers share one fixed 32 MiB FP32 scratch per GPU, plus one for the shared-expert stream, and process rows in chunks that fit it, so neither batch size nor graph capture grows it. The routed-expert scratch is sized for the rows the selected dispatch delivers: with tensor and expert parallelism but no data parallelism, each rank's own tokens.

`VLLM_EXL3_DENSE_PREFILL_TC=1` (default `0`) runs dense launches of prefill steps on the tensor cores through `linear_out_prefill` (`csrc/exl3/dense_prefill_tc.cu`) on SM 8.0 with 74 multiprocessors: launches of more than 64 rows with at least 16 output tiles of 128 x 128, for the nine matrices of one rank of Qwen3.8-Flash-Next at tensor parallel size 2 (`bits_x2` 10, 12 or 14 and the mul1 codebook), with pipeline parallel size 1 and either no drafter or an MTP or EAGLE drafter without parallel drafting. A step is a prefill step when the scheduler gave some request more than 1 + `num_speculative_tokens` tokens (`vllm_sm80/kernels/exl3/step_phase.py`), so decode steps of any size, graph captures and launches that `VLLM_EXL3_DENSE_SPLITK` splits keep their current kernels. The input is transformed as `linear_out` transforms it, scaled per 128-value block by a power of two and split into an integer FP16 part and an FP16 remainder; the weight is split into its 2^-7-grid part and the rest, so the dominant product is exact on the tensor cores, and the per-step sums are accumulated in FP32 in k order. When the last wave of tiles would leave multiprocessors idle, its tiles' k range is split over blocks; the block that finishes a tile last adds the partial sums in k order, so results are deterministic. They differ from `linear_out` in the last bits, and a row's result may depend on how many rows share its launch and on its position in it. The partial tiles and their counters live in the existing workspace and the row chunks are unchanged. The kernels are loaded once at startup. `VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE=whole` (diagnostic) disables the k split. A startup line says whether the path is active or why the gate is closed.

`VLLM_EXL3_DENSE_SPLITK=1` (default `0`) runs dense launches of the tiny and small block shapes with at most 1,024 blocks, for matrices stored with `bits_x2` 10, 12 or 14 and the mul1 codebook, through `linear_out_splitk` (`csrc/exl3/dense_splitk.cu`) on SM 8.0. Its blocks have separate decode warps that fill a ring of decoded chunks ahead of the FMA threads, and K is split into two halves whose FP32 sums are added atomically onto a zeroed result. Within a half, each chunk's FMA chain starts from zero and the chunk sums are added in order; the output Hadamard transform and the `svh` scaling run in FP64 and round once. Results are deterministic but differ from `linear_out` in the last bits, and a row's result may depend on how many rows share its launch. The workspace and the row chunks are unchanged. Every other launch (larger grids such as the LM head's, the medium and large blocks of prefill, other formats) keeps `linear_out`; a startup line says whether the path is active or why the gate is closed.

`VLLM_EXL3_DENSE_MMA=1` (default `0`, needs `VLLM_EXL3_DENSE_SPLITK=1`) runs every dense launch of 1 to 32 rows of the validated matrices (the dense matrices of one Qwen3.8-Flash-Next rank at tensor parallel size 2, the LM head included: `bits_x2` 10, 12 or 14, codebook 2, a 16-byte aligned trellis, and an output large enough to hold the per-block scale exponents until it is written) on tensor-core blocks inside `linear_out_splitk` (`csrc/exl3/dense_mma.cu`), on SM 8.0 with 74 multiprocessors at tensor parallel size 2 and pipeline parallel size 1. Each 128-column block of each activation row is scaled by a power of two and split into FP16 high and low halves; the trellis is decoded in registers straight into `mma.sync` fragments, and each k16 slice's FP32 products are added, times the block's inverse scale, into an FP32 sum. Wide matrices such as the LM head run four 16-column tiles per warp over all of K; the others run two tiles per warp and split K over up to 8 warps of a block and 2 blocks, the most parts whose blocks fit on the SMs at once. The parts are added in a fixed order, and two blocks add atomically onto a zeroed result, so results are deterministic; they differ from `linear_out` in the last bits, and a row's result may depend on the row count of its launch. The output transform is the split-K path's FP64 one, and the workspace and row chunks are unchanged. Launches of more than 32 rows (drafter `fc` at 48 to 128 rows, prefill) and every other matrix keep the split-K operator's path. The switch is set once per process before any launch; it logs whether the path is active or why the gate is closed. `VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE=MT,KW,SPLITS` (diagnostic, unset by default) forces one plan on every launch the blocks take.

## Qwen3.8-Flash-Next

The model integration preserves packed constituents of fused projections, replicates shared experts when their widths cannot be legally split, retains full indexer keys, and uses data-parallel vision encoding. Sparse attention computes Q/gate, K, V and the indexer Q/K in one fused projection; Q/gate splits across ranks, K/V follow the key-value heads, and every rank holds the whole indexer matrix. Native MTP shares the target embedding and head when the checkpoint does not provide dedicated ones. The loader requires the compressed ngram table and MTP hyper-connection mixer patch in addition to the main shards.

For the two-card configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_EXL3_PLE_PLACEMENT=ssd \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve ./qwen-exl3-checkpoint \
  --quantization exl3 --load-format exl3 --mm-encoder-tp-mode data \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --engram-config '{"cpu_offload":false}' \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The MTP count is configurable; this example is not a measured optimum. Keep vision enabled. Validate eager execution before graph replay and test the native context independently of a shorter diagnostic boot. `tests/models/qwen4_exp/test_exl3_construction.py` builds the target and MTP draft at two-rank expert-parallel shapes on the CPU from the checkpoint's configuration and tensor headers when `VLLM_TEST_EXL3_QWEN_CHECKPOINT` names the checkpoint directory.

The text config field `moe_router_dtype="float32"` selects the existing `GateLinear` BF16-by-BF16 GEMM with FP32 accumulation and FP32 router logits, without changing the stored router weights. The checkpoint does not set it, so the router returns logits in the model dtype; set it with `--hf-overrides '{"text_config": {"moe_router_dtype": "float32"}}'`. The model logs the selected precision at construction. The field applies to both the target and the native MTP router: the MTP layers read it from the target config, because dict `--hf-overrides` do not reach the draft model config.

The text config field `mtp_hidden_norm="per_stream"` computes a separate RMS for each native MTP hidden stream. It preserves the checkpoint's affine weight for each channel and applies the Gemma `(1 + weight)` scaling in FP32 before returning BF16. The default `"flat"` (field absent) retains one RMS over the entire concatenated hidden stack; any other value is refused at construction. Set it with `--hf-overrides '{"text_config": {"mtp_hidden_norm": "per_stream"}}'`; it is read from the target config. The draft model logs the selected normalization. These are precision/normalization comparison paths, not validated default changes.

## Compressed PLE placement

`VLLM_EXL3_PLE_PLACEMENT` accepts `ssd` (default), `ram` or `device`. Placement is explicit and logged; RAM/device modes are comparisons, not automatic fallbacks.

- `ssd`: retain the file descriptor and read only requested owned rows with buffered positional reads into bounded reusable pinned staging. A failed or truncated row read is retried up to `VLLM_EXL3_PLE_READ_RETRIES` times (default 3, backoff 1 ms doubling); see below for a read that still fails.
- `ram`: explicitly pin the compressed rank shard and expose its accelerator view.
- `device`: explicitly load the compressed rank shard to the GPU.

All modes use the same GPU row decoder. Its layout comes from the PLE file: `K` from the codec metadata, the channel count from the head bias, and the row size (one FP16 scale word plus a ring of `channels * K` bits) from the trellis. Ownership and head padding are masked before bias; only decoded BF16 rows participate in the embedding reduction. Repeated IDs preserve request order. Original auxiliary tensors stay GPU-resident.

SSD reads occur at eager Python breaks between CUDA graph segments: ID completion, CPU reads, H2D readiness, decode and output-copy completion have explicit ordering. This is not a captured CUDA host callback. Staging cannot be reused until the previous output consumer has completed. Warm stable workspaces before capture; growth after capture locking is rejected.

A row read that still fails after its retries fails only the requests that use the row; the engine keeps serving the rest. The reader stages the row with an FP16 NaN scale, so the decoder emits a NaN row. NaN survives the tensor-parallel sum of row shards, so every rank sees which tokens failed, whichever rank owns the row. Those tokens' PLE embeddings are zeroed before the forward continues, so NaN cannot reach other requests. The model runner reports the requests that own the tokens. The scheduler finishes them with `FINISHED_ERROR` (finish reason `error`) and drops from the prefix cache the KV blocks they computed from that step on. It also fails any running request that already reused those blocks, so no request is served from a wrong embedding. The worker logs the row and the OS error; the scheduler logs the request IDs. The reader stays usable, and later requests read the row again. Errors while reading the table or its auxiliaries during startup still fail startup. `ram` and `device` read the table only at startup.

The PLE file is opened with `posix_fadvise(POSIX_FADV_RANDOM)`, so the kernel does not read ahead around requested rows. Buffered reads still hit the page cache, so warm runs mostly read RAM. Set `VLLM_EXL3_PLE_COLD=1` to call `POSIX_FADV_DONTNEED` on the table at setup (no root needed; pages that another process has mapped stay resident). Page-cache residency of the table (via `mincore`) is logged at setup, after the cold drop and at process exit. `Exl3PLEEmbedding.page_cache_residency()` returns `(resident, total)` pages for sampling between runs.

## Tensor-parallel routed experts

With `--tensor-parallel-size` above 1 and without `--enable-expert-parallel`,
every rank holds every routed expert, cut into whole 128-column blocks of the
expert intermediate (the EXL3 Hadamard block). The blocks go to the ranks in
chunks as even as they allow, the larger chunks first, and rank r of layer L
holds chunk (r + L) mod the tensor parallel size, so the larger chunks move
between ranks from layer to layer (the layer index is the one in vLLM's module
name; an MTP drafter's layers follow the target's). DeepSeek-V4.1-Flash's
2304-wide experts give 640, 640, 512 and 512 columns at tensor parallel size
4: each rank holds 640 in 20 of the 40 target layers. Gate and up keep all
inputs and the chunk's output columns (`suh` whole, `svh` of the chunk); down
keeps the chunk's input rows (`suh` of the chunk) and all outputs (`svh`
whole). The native operators run at the rank's width. Routing is unchanged:
each rank routes the same rows over all experts with the identity expert map,
and the MoE's tensor-parallel all-reduce adds the ranks' partial outputs, so
final MoE bits need not match whole-expert EP. The startup log names each
rank's widths. Not supported with data, context or sequence parallelism,
EPLB or all-to-all dispatch. Whether the decode and prefill operators take a
rank width is up to their gates
(`ops.MOE_V2_CONFIGS`, `ops.MOE_PREFILL_V2_CONFIGS`); `moe_out` takes any.
The block-aligned split rotated per layer follows the idea of the
DeepSeek-V4.1-Flash CMP 170HX recipes of GNmac-lab and Curtis White
(CurtisAccelerate); no code is taken from them.

## Third-party material

Codec adaptations preserve the upstream MIT grant and modification credit in [`plugins/sm80/licenses/exl3`](../../../plugins/sm80/licenses/exl3). The fork's plugin distribution (`plugins/sm80`, which every installation that runs EXL3 needs) ships that `LICENSE` and `NOTICE` as its license files: in the sdist, and in the `.dist-info/licenses/` directory of its wheel and of every installed copy.

The hidden-5120 shape of `moe_out_v2` follows Curtis White's Apache-2.0 port of that operator; its credit is in the same `NOTICE`.

## DeepSeek switch control

`VLLM_DEV_AB_SWITCH` is empty by default. When explicitly configured, the
worker captures both implementations before serving and switches only while
requests are drained. Available DeepSeek selectors are
`dsv41_fused_attn_decode`, `dsv41_attn_pipeline`, `dsv41_moe_prefill_tc`,
`dsv41_fused_mhc`, `dsv41_fused_mhc_finish`, `dsv41_fused_mhc_finish_v2`,
`exl3_moe_v2`, `exl3_moe_v2_draft` (the drafter's experts only), `exl3_dense_splitk`, and
`exl3_dense_prefill_tc`. The mHC selectors change the loaded decoder operation
module or its cached finish dispatch; capture wrappers retain their ownership.
The dense-prefill selector changes the cached dispatcher, not its decode bound,
workspace, hardware gate, or schedule.

MoE V2, MoE-prefill and dense-prefill selection require their feature enabled at
construction so that the existing workspace and warmup have completed.
Selecting the fused mHC finish requires fused mHC enabled at construction.
Selecting the mHC finish v2 also requires the fused mHC finish enabled at
construction; switching it on compiles its kernels before its graphs are captured.
Attention pipeline selection reserves both implementations' scratch before
profiling. None of these selectors changes the speculative-width policy,
graph-size ladder, checkpoint, or loaded expert placement.

Hybrid checkpoint selection, true per-K
DSpark construction, reduction transports/output dtypes, early Engram gathers,
and draft-only shard argmax require a new process. Environment changes alone
cannot update their constructed modules, storage, graph geometry, or cached
sampler configuration. Numerical and model-quality qualification remains
separate from successful switch control.
