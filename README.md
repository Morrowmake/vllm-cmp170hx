<h1 align="center">vLLM for the NVIDIA CMP 170HX</h1>

<p align="center">
  <img alt="target" src="https://img.shields.io/badge/target-Ampere%20sm__80-76b900?style=flat">
  &nbsp;
  <img alt="base" src="https://img.shields.io/badge/upstream-vLLM%200.30.1%20dev-4b32c3?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue?style=flat">
</p>

This is a fork of [vLLM](https://github.com/vllm-project/vllm) for the
**NVIDIA CMP 170HX** (Ampere, sm_80), built to serve models fast on one card
or several: tensor-parallel and pipeline-parallel layouts across cards, a
full-precision KV cache and repeatable single-request outputs.
**GLM-5.3-Flash** is the first supported model; more will follow.

The fork has two layers:

- **General work** for sm_80 and PCIe-connected cards: kernels and attention
  backends, all-reduce through host memory or PCIe, KV accounting,
  determinism, pipeline-parallel scheduling, and an optional compiled Marlin
  extension.
- **Model-specific work**, gated by model family: it lives in that family's
  model code or behind its own switches, and its kernels check the exact
  shapes they were built for, handing anything else back to upstream's path.

Every performance feature sits behind an environment variable and is **off in
the code by default**. Each supported model has a recipe repository whose
launcher switches on the tested set. `0` turns a feature off again, with no rebuild.
Variables carry the prefix of the model they were first built for
(`VLLM_GLM5_*`), including some general ones.

## Using it: take a recipe

Each model is served through its own recipe, which pins an exact fork commit,
ships the engine as a container image, downloads the weights and starts an
OpenAI-compatible server. Questions about running a model belong in its
recipe's issues.

| Model | Recipe |
|---|---|
| GLM-5.3-Flash | [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) |

## Supported models

### GLM-5.3-Flash

320B MoE, W4A16, 262,144-token context, DFlash2 speculative decoding.
Upstream's sparse-attention path needs Hopper; this fork adds the sm_80 path
that makes the model run at all, then makes it fast.

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe && ./start.sh     # LAYOUT=pp4 for pipeline-parallel 4
```

From the recipe's [release 1.5.0 results](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe#results)
(DFlash2 at k=3, 180 W per card; methods and caveats are there):

| | TP4 (default) | PP4 (`LAYOUT=pp4`) |
|---|---:|---:|
| Streaming decode, 1 user, structured | **267.3 tok/s** | 141.8 tok/s |
| Decode, 8 users, aggregate, structured | **758.9 tok/s** | 603.0 tok/s |
| Cold prefill | 2,657 tok/s | **6,575 tok/s** |
| KV pool at 262,144 context | 1,176,646 tokens | **2,334,498 tokens** |

Quality with DFlash2 (release 1.4.3, fixed batches): HumanEval 162/164 under
TP4 and 163/164 under PP4; GSM8K 1,281/1,319 and 1,284/1,319.

GLM-specific switches (the recipe's
[kill switch table](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe#kill-switches)
says which layout each serves):

- Fused decode kernels for mHC mixing, MoE routing and alignment, and KDA
  decode — `VLLM_GLM5_DECODE_KERNELS`; second generation `VLLM_GLM5_DECODE_MOE_ROUTE_V2`,
  `VLLM_GLM5_DECODE_MHC_V2`, `VLLM_GLM5_DECODE_KDA_V2`, `VLLM_GLM5_DECODE_IDX_GLUE`;
  fused decode prologue — `VLLM_GLM5_PROLOGUE_FUSE`.
- Prefill kernels (mHC projection, sparse attention) — `VLLM_GLM5_PREFILL_KERNELS`,
  `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD`; at TP4 shapes `VLLM_GLM5_TP4_KDA_PREFILL`,
  `VLLM_GLM5_TP4_MARLIN_PREFILL`; at PP4 shapes `VLLM_GLM5_PP_KDA_PREFILL`,
  `VLLM_GLM5_PP_SPARSE_MLA_PREFILL`, `VLLM_GLM5_PP_MARLIN_PREFILL`.
- Prefill all-reduces overlapped with the MoE (TP4) — `VLLM_GLM5_PREFILL_OVERLAP`;
  batch-sharded logits and sampling (TP4) — `VLLM_GLM5_LOCAL_LOGITS`.
- Drafter: selector tables split across cards — `VLLM_GLM5_DRAFTER_SELECTOR_SHARD`;
  position table sized to the context — `VLLM_GLM5_DRAFTER_ROPE_FIT`; folded
  input projection under PP4 — `VLLM_GLM5_PP_FOLD_DRAFT_FC`.
- Always on: a rejected draft can no longer overwrite the tail of the
  sparse-attention key pool.

## General features

**sm_80 kernels and backends.** A sparse-MLA attention backend, an sm_80 path
for the attention indexer and its FP8 stores, and sm_80 key-pool compression
(always on). A retuned sparse-attention decode schedule, on in the engine
(`VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY=1` restores the old one). Thin-batch BF16
GEMMs tuned per shape, up to 32 rows — `VLLM_GLM5_THIN_GEMM`. MoE shared
experts that overlap the routed experts — `VLLM_GLM5_SHARED_EXPERT_REORDER`.

**All-reduce over PCIe.** For cards without GPU peer access, small all-reduces
take one round trip through shared host memory — `VLLM_GLM5_HOST_ALLREDUCE`.
Where peer access is granted, a PCIe custom all-reduce runs in device memory
instead — `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE`, kernel chosen by
`VLLM_CUSTOM_ALLREDUCE_ALGO`; it falls back to the host path on its own.

**KV accounting.** The KV reserve covers what prefill chunks in flight really
hold, so the reported pool is one the server can fill —
`VLLM_KV_MAMBA_INFLIGHT_STATES`, `VLLM_KV_SWA_INFLIGHT_SCRATCH`. Indexer decode
tables sized by the decode rows — `VLLM_GLM5_INDEXER_DECODE_ROWS`; indexer
gather workspace clamp, on in the engine — `VLLM_GLM5_INDEXER_GATHER_CLAMP`.
`VLLM_GLM5_MEM_ATTRIBUTION` logs where device memory goes at start-up.

**Determinism.** A request on its own returns the same tokens and
log-probabilities on every run and every install: fixed-order MoE block
alignment (`VLLM_GLM5_DETERMINISTIC_MOE_ALIGN`), CUDA-graph padding kept out of
the MoE (`VLLM_GLM5_MOE_MASK_PADDING`), indexer top-k consistent on ties and in
a fixed order (`VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED`,
`VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS`), pinned linear-attention prefill
configurations (`VLLM_GLM5_FLA_PIN_AUTOTUNE`).

**Pipeline-parallel and scheduling.** Decodes spread over every in-flight
micro-batch — `VLLM_PP_SPREAD_DECODES`; one packed, metadata-free hand-off
between stages — `VLLM_PP_PACKED_HOP`, `VLLM_PP_HOP_NO_METADATA`; drafter
synchronisation — `VLLM_PP_SPLIT_DRAFT_EVENT`; the drafter's final step on a
chosen stage — `VLLM_PP_DRAFT_TAIL_STAGE` (`-1` off). Fair prefill, where long
prompts yield to running decodes — `--prefill-chunk-with-decodes N`. An
acceptance-adaptive draft count, off unless set — `adaptive_k` in
`--speculative-config` ([docs](docs/features/speculative_decoding/adaptive_k.md)).

**Optional compiled Marlin.** One sm_80 library, `vllm._ampere_marlin_C`, for
both layouts: `VLLM_GLM5_MARLIN_DECODE_CUDA` (eligible small batches) and
`VLLM_GLM5_MARLIN_PREFILL_CUDA` (PP4). Requesting either without a compatible
library fails at startup rather than falling back silently. The decode kernels
come in two reduction orders, chosen by `VLLM_GLM5_MARLIN_DECODE_VARIANT`:
`orig` (default; splits the first projection along K, faster, changes the fp32
summation order) and `exact` (the released Marlin summation order).

**Correctness fixes (always on).** 64-bit KV row offsets in the
sparse-attention kernels and a vocabulary clamp in the sampler kernels.

Every custom kernel on a default path is checked on real captured inputs
against a 64-bit reference, side by side with the code it replaces, and must
be at least as accurate.

## Upstream base

The fork sits on upstream `main` at
[`e55d076f89`](https://github.com/vllm-project/vllm/commit/e55d076f89fd01a0538a3e496d8ff20bf7980100)
(2026-09-25), the vLLM **0.30.1** development line (`git describe`:
`v0.30.1rc0-181-ge55d076f8`). `main` on this fork stays a clean mirror of
upstream; general fixes are offered back upstream as pull requests.

## Branches and releases

The work is one branch, today `ampere-glm53`. When the second model lands it
will be renamed to a model-neutral name; GitHub redirects the old name.

A recipe pins an exact fork commit, never the branch. The branch moves to a
newer upstream from time to time; every commit a recipe release pins is kept
under a tag `<recipe>-<version>` (for GLM-5.3-Flash, `glm53-recipe-<version>`),
so older releases keep installing. Release notes live in each recipe's
changelog, e.g. [GLM-5.3-Flash](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/main/CHANGELOG.md).

## Building from source

A recipe's native install (`RUNTIME=native ./install.sh`) is the reference.
In short, on Python 3.12 with torch 2.13.0 (CUDA 13.0):

```bash
git clone -b ampere-glm53 https://github.com/Morrowmake/vllm-cmp170hx.git && cd vllm-cmp170hx
VLLM_USE_PRECOMPILED=1 uv pip install -e .    # upstream's precompiled extensions carry sm_80
uv pip install -r requirements/cuda.txt
# optional compiled Marlin, built on its own without rebuilding the engine:
VLLM_BUILD_AMPERE_MARLIN=1 python csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py --out vllm
```

The optional library needs a CUDA toolkit (12.8 or newer), a C++20 compiler and
Ninja; no GPU is needed to build it. `VLLM_BUILD_AMPERE_MARLIN=1` cannot be
combined with `VLLM_USE_PRECOMPILED=1` in one install. Each recipe names the
upstream wheel that supplies the precompiled extensions for its pin.

## Licence and credit

Apache-2.0, as upstream ([LICENSE](LICENSE)). This fork is a layer on the work
of the [vLLM project](https://github.com/vllm-project/vllm) and its
contributors; thank you. Upstream's own README is kept at
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
