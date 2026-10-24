<!-- Modified by Morrowmake: CMP 170HX presentation and source attribution. -->

<h1 align="center">vLLM for the NVIDIA CMP 170HX</h1>

<p align="center">
  <img alt="target" src="https://img.shields.io/badge/target-Ampere%20sm__80-76b900?style=flat">
  &nbsp;
  <img alt="base" src="https://img.shields.io/badge/upstream-vLLM%200.32.0%20dev-4b32c3?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue?style=flat">
</p>

This is a fork of [vLLM](https://github.com/vllm-project/vllm) for the
**NVIDIA CMP 170HX** (Ampere, sm_80), built to serve models fast on one card
or several: tensor-parallel and pipeline-parallel layouts across cards, a
full-precision KV cache and repeatable single-request outputs.
Two models are supported today: **GLM-5.3-Flash** on four cards and
**Qwen3.8-Flash-Next** on two.

The fork has two layers:

- **General work** for sm_80 and PCIe-connected cards: kernels and attention
  backends, EXL3 quantised kernels, all-reduce through host memory or PCIe,
  KV accounting, determinism, pipeline-parallel scheduling, and an optional
  compiled Marlin extension.
- **Model-specific work**, gated by model family: it lives in that family's
  model code or behind its own switches, and its kernels check the exact
  shapes they were built for, handing anything else back to upstream's path.

The fork's own code lives in the `vllm/sm80/` package and in each model
family's `nvidia/` directory under `vllm/models/`. vLLM loads it through the
`sm80` plugin (the `vllm-sm80-plugin` distribution in `plugins/sm80/`), which
registers the fork's platform, layers and model classes; upstream files carry
only small hooks into it. The engine stops at startup if the plugin is missing,
rather than quietly running upstream's path.

Every performance feature sits behind an environment variable and is **off in
the code by default**. Each supported model has a recipe repository whose
launcher switches on the tested set. `0` turns a feature off again, with no rebuild.
Switches are named after what they control: shared sm_80 kernel paths use
`VLLM_AMPERE_*`, other shared features carry the feature's name
(`VLLM_CUSTOM_ALLREDUCE_*`, `VLLM_EXL3_*`, `VLLM_DFLASH_*`, `VLLM_MOE_*`,
`VLLM_PP_*`, `VLLM_THIN_GEMM*`, …), and only code for one model family keeps
that family's prefix (`VLLM_GLM5_*`, `VLLM_QWEN4_*`). All of them are declared
in `vllm/sm80/envs.py`, together with every earlier name that was renamed or
removed; setting a retired name logs a warning that names its replacement and
is otherwise ignored.

## Using it: take a recipe

Each model is served through its own recipe, which pins an exact fork commit,
ships the engine as a container image, downloads the weights and starts an
OpenAI-compatible server. Both recipes start with one command from a fresh
clone, `./start.sh` or `docker compose up`, and run only as a container.
Questions about running a model belong in its recipe's issues.

| Model | Cards | Recipe |
|---|---|---|
| GLM-5.3-Flash | 4 | [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) |
| Qwen3.8-Flash-Next | 2 | [Morrowmake/qwen38-flash-next-cmp170hx-recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe) |

## Community recipes

Other people build recipes on this fork too: for other models, card counts or
settings. Each one lives in its author's own repository and pins an exact fork
version. To add yours, see [COMMUNITY_RECIPES.md](COMMUNITY_RECIPES.md).

Community recipes are made and maintained by their authors; Morrowmake has not
tested them.

| Recipe | Model | Hardware/layout | Fork version | Author |
|---|---|---|---|---|
| [mimo-v26-flash-cmp170hx-recipe](https://github.com/ElectroHobo/mimo-v26-flash-cmp170hx-recipe) | MiMo-V2.6-Flash-RL (NVFP4) | 4× CMP 170HX, TP=4 | `18305948acba3f4c4d912cdd56f2202d2ebd94e8` | @ElectroHobo |
| [qwen38-flash-next-w4a16-2x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe) | Qwen3.8-Flash-Next (W4A16, DFlash 2 drafter) | 2× CMP 170HX, TP=2 | `18305948acba3f4c4d912cdd56f2202d2ebd94e8` | @ElectroHobo |

## Supported models

### GLM-5.3-Flash

320B MoE, W4A16, 262,144-token context, DFlash2 speculative decoding, on four
cards. Upstream's sparse-attention path needs Hopper; this fork adds the sm_80
path that makes the model run at all, then makes it fast.

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe && ./start.sh     # LAYOUT=pp4 for pipeline-parallel 4
```

From the recipe's [release 1.8.0 results](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/main/docs/results.md)
(DFlash2, peer-to-peer on, 74 SMs and 180 W per card, PCIe x16 links; methods
and caveats are there):

| | TP4 (default) | PP4 (`LAYOUT=pp4`) |
|---|---:|---:|
| Streaming decode, 1 user, structured | **up to 482.6 tok/s** | up to 232.4 tok/s |
| Decode, 8 users, aggregate, structured | **up to 943.7 tok/s** | up to 664.1 tok/s |
| Cold prefill | up to 3,161 tok/s | **up to 7,419 tok/s** |
| KV pool at 262,144 context | 1,203,765 tokens | **2,088,960 tokens** |

Quality (release 1.8.0, TP4, scored on the answer after the reasoning):
HumanEval 151/164, GSM8K 1,286/1,319.

GLM-specific switches (the recipe's
[engine switches](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/main/docs/engine-switches.md)
page says which layout each serves):

- Fused decode kernels for mHC mixing, MoE routing and alignment, and the KDA
  decode step (gate projections, convolution, delta rule and gated norm in one
  kernel, one KDA state per request under TP4) — `VLLM_AMPERE_DECODE_KERNELS`;
  its TP4 verify (the commit fold's included) with at most 16 rows per warp,
  less stack and more KV cache, bitwise the same
  (not yet validated, off unless set) — `VLLM_AMPERE_KDA_SMALL_STACK`;
  the KDA step for PP4 decode batches beyond one short request (where it
  measures faster) — `VLLM_GLM5_DECODE_KDA_WIDE`; second
  generation `VLLM_MOE_DECODE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2`,
  `VLLM_AMPERE_DECODE_IDX_GLUE`; under TP4, the KDA state commit after
  sampling folded into the next decode step, same state bit for bit
  (not yet validated, off unless set) — `VLLM_AMPERE_KDA_FOLD`.
  A diagnostic, never for serving: under TP4, one KDA state per draft
  position instead of one per request (the reference without state
  recovery, as wherever that path's limits are not met) —
  `VLLM_AMPERE_KDA_RECOVER_OFF`.
- Prefill kernels (mHC projection, sparse attention) — `VLLM_GLM5_PREFILL_KERNELS`,
  `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD`; at TP4 shapes `VLLM_AMPERE_KDA_PREFILL_NARROW`,
  `VLLM_AMPERE_MARLIN_PREFILL_NARROW`; at PP4 shapes `VLLM_AMPERE_KDA_PREFILL_WIDE`,
  `VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE`, `VLLM_AMPERE_MARLIN_PREFILL_WIDE`.
- Batch-sharded logits and sampling (TP4) — `--enable-batch-sharded-sampling`.
- Drafter: selector tables split across cards — `VLLM_GLM5_DRAFTER_SELECTOR_SHARD`;
  position table sized to the context — `VLLM_GLM5_DRAFTER_ROPE_FIT`; folded
  input projection under PP4 — `VLLM_GLM5_PP_FOLD_DRAFT_FC`.
- Always on: a rejected draft can no longer overwrite the tail of the
  sparse-attention key pool.

### Qwen3.8-Flash-Next

EXL3 5.05 bpw weights, BF16 KV cache, 262,144-token context and native MTP
speculative decoding, on two cards: tensor-parallel 2 with expert parallelism,
whole routed experts on each card. Every weight sits on the GPUs except the
compressed n-gram table, read from the checkpoint on an SSD while decoding.

```bash
git clone https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe.git
cd qwen38-flash-next-cmp170hx-recipe && ./start.sh
```

From the recipe's [release 1.0.0 results](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe#headline-numbers)
(MTP with 3 draft tokens, peer-to-peer on, 74 SMs and 180 W per card, PCIe x16
links; methods and caveats are there):

| | TP2 |
|---|---:|
| Streaming decode, 1 user, structured | **up to 152.8 tok/s** |
| Decode, 8 users, aggregate, code | **up to 674.7 tok/s** |
| Cold prefill | up to 2,651 tok/s |
| KV pool at 262,144 context | 1,044,325 tokens |

Quality (release 1.0.0): HumanEval 157/164, GSM8K 1,289/1,319.

The EXL3 kernels written for this card — MoE kernels for decode and prefill, a
split-K dense kernel with tensor-core blocks for small decode launches, a
tensor-core dense kernel for prefill and a native n-gram table read — are
switched by `VLLM_EXL3_*`; the model's sparse-attention top-k tie order by
`VLLM_QWEN4_QSA_TOPK_CANONICAL`. The recipe's
[engine switches](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe#engine-switches)
table lists the set it turns on.

## General features

**sm_80 kernels and backends.** A sparse-MLA attention backend, an sm_80 path
for the attention indexer and its FP8 stores, and sm_80 key-pool compression
(always on). A retuned sparse-attention decode schedule. Thin-batch BF16
GEMMs tuned per shape, up to 32 rows — `VLLM_THIN_GEMM`. MoE shared
experts that overlap the routed experts — `VLLM_SHARED_EXPERT_REORDER`.

**All-reduce over PCIe.** For cards without GPU peer access, small all-reduces
take one round trip through shared host memory — `VLLM_HOST_ALLREDUCE`.
Where peer access is granted, a PCIe custom all-reduce runs in device memory
instead — `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`, kernel chosen by
`VLLM_CUSTOM_ALLREDUCE_ALGO`; it falls back to the host path on its own.

**KV accounting.** The KV reserve covers what prefill chunks in flight really
hold, so the reported pool is one the server can fill —
`VLLM_KV_MAMBA_INFLIGHT_STATES`.
`VLLM_MEM_ATTRIBUTION` logs where device memory goes at start-up.

**Determinism.** A request on its own returns the same tokens and
log-probabilities on every run and every install: fixed-order MoE block
alignment (`VLLM_MOE_DETERMINISTIC_ALIGN`), CUDA-graph padding kept out of
the MoE (`VLLM_MOE_MASK_PADDING`), indexer top-k consistent on ties and in
a fixed order (`VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED`,
`VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS`), pinned linear-attention prefill
configurations (`VLLM_FLA_PIN_AUTOTUNE`).

**Pipeline-parallel and scheduling.** Decodes spread over every in-flight
micro-batch — `VLLM_PP_SPREAD_DECODES`; one packed, metadata-free hand-off
between stages — `VLLM_PP_PACKED_HOP`, `VLLM_PP_HOP_NO_METADATA`; drafter
synchronisation — `VLLM_PP_SPLIT_DRAFT_EVENT`; the drafter's final step on a
chosen stage — `VLLM_PP_DRAFT_TAIL_STAGE` (`-1` off). Fair prefill, where long
prompts yield to running decodes — `--prefill-chunk-with-decodes N`. A
load-adaptive draft count, off unless set — `adaptive_k` in
`--speculative-config` ([docs](docs/features/speculative_decoding/adaptive_k.md)).

**Optional compiled Marlin.** One sm_80 library, `vllm._ampere_marlin_C`, for
both layouts: `VLLM_AMPERE_MARLIN_DECODE_CUDA` (eligible decode batches). Requesting
it without a compatible library fails at startup rather than falling back
silently. The decode kernels split the first projection along K: faster, and
the fp32 summation order differs from the released Marlin kernels. Under PP4
the split-block prefill (`VLLM_AMPERE_MARLIN_PREFILL_WIDE`) also runs on the
library's (64,512) tiles with a one-HFMA2 dequant —
`VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED` (engine default off, on in the PP4
serving defaults), the released kernels without the library; TP4 prefill uses
the released kernels.

**Correctness fixes (always on).** 64-bit KV row offsets in the
sparse-attention kernels and a vocabulary clamp in the sampler kernels.

Every custom kernel on a default path is checked on real captured inputs
against a 64-bit reference, side by side with the code it replaces, and must
be at least as accurate.

**Removed opt-in paths.** Draft-confidence skipping and its fitting logs, depth-two
verification, the full canonical top-k serving override and the separate tie-repair
fallback, cross-layer prefill overlap, replicated input embeddings, and the legacy
sparse-attention decode schedule are no longer supported. Their released defaults
were disabled; the shipped adaptive acceptance tracking and depth-three-through-seven
policy, fused top-k tie fix and sorting, sharded embeddings, and current
sparse-attention schedule remain unchanged. The within-layer prefill overlap is
removed; TP4 prefill all-reduces run unsplit, as upstream. The canonical top-k reference
is retained only in tests.

## Upstream base

The fork sits on upstream `main` at
[`4e0681ed56`](https://github.com/vllm-project/vllm/commit/4e0681ed5654c1d8afb3c104d4381afd08529de8)
(2026-10-14), the vLLM **0.32.0** development line (`git describe`:
`v0.32.0rc0`, plus the fork's commits). `main` on this fork stays a clean
mirror of upstream; general fixes are offered back upstream as pull requests.

## Branches and releases

The work is one branch, `ampere`, shared by every model. The earlier
`ampere-glm53` branch is retired.

A recipe pins an exact fork commit, never the branch. The branch moves to a
newer upstream from time to time; every commit a recipe release pins is kept
under a tag `<model>-recipe-<version>`, so older releases keep installing:

| Recipe release | Fork tag | Commit |
|---|---|---|
| GLM-5.3-Flash 1.8.0 | `glm53-recipe-1.8.0` | [`819146023f`](https://github.com/Morrowmake/vllm-cmp170hx/commit/819146023facb31c04f1f8512e52464835a10d40) |
| Qwen3.8-Flash-Next 1.0.0 | `qwen38-recipe-1.0.0` | [`18305948ac`](https://github.com/Morrowmake/vllm-cmp170hx/commit/18305948acba3f4c4d912cdd56f2202d2ebd94e8) |

Release notes live in each recipe's changelog:
[GLM-5.3-Flash](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/main/CHANGELOG.md),
[Qwen3.8-Flash-Next](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe/blob/main/CHANGELOG.md).

## Building from source

The recipes serve the engine only as a container image; each recipe's
`docker/README.md` shows how that image is built and is the reference. In
short, on Python 3.12 with torch 2.13.0 (CUDA 13.0):

```bash
git clone -b ampere https://github.com/Morrowmake/vllm-cmp170hx.git && cd vllm-cmp170hx
VLLM_USE_PRECOMPILED=1 uv pip install -e .    # upstream's precompiled extensions carry sm_80
uv pip install -r requirements/cuda.txt
uv pip install --no-deps plugins/sm80          # the plugin entry points; required
# optional compiled Marlin, built on its own without rebuilding the engine:
VLLM_AMPERE_MARLIN_BUILD=1 python csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py --out vllm
```

The optional library needs a CUDA toolkit (12.8 or newer), a C++20 compiler and
Ninja; no GPU is needed to build it. `VLLM_AMPERE_MARLIN_BUILD=1` cannot be
combined with `VLLM_USE_PRECOMPILED=1` in one install. Each recipe names the
upstream wheel that supplies the precompiled extensions for its pin.

## Licence and credit

Apache-2.0, as upstream ([LICENSE](LICENSE)). This fork is a layer on the work
of the [vLLM project](https://github.com/vllm-project/vllm) and its
contributors; thank you. Upstream's own README is kept at
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

Some implementations are adapted from other projects; these are not all
independently authored patches. [Third-party sources and idea credits](docs/THIRD_PARTY.md)
identify those adaptations, their licences and the ideas that informed our
changes. Original component copyright and licence notices remain applicable;
the adapted ExLlamaV3 code's MIT notice ships with the `plugins/sm80`
distribution.

[NOTICE](NOTICE) states the attribution that goes with redistributions of this
fork. If this work helps you, please credit Morrowmake and link this
repository; citation metadata is in [CITATION.cff](CITATION.cff).
