<!-- Modified by Morrowmake: CMP 170HX presentation and source attribution. -->

# vLLM for the NVIDIA CMP 170HX

This repository makes [vLLM](https://github.com/vllm-project/vllm) run well on the
**NVIDIA CMP 170HX** and other sm_80 (GA100-class) cards. vLLM itself stays
unmodified upstream code; our work comes as two pieces on top of it:

- **The `vllm-sm80` plugin** (Python package `vllm_sm80`): our
  kernels and attention backends, the EXL3 and Marlin compiled extensions, the
  model classes for the models we serve, and the launcher. vLLM loads it through
  its standard plugin entry points at startup. If the plugin is missing, the
  engine stops instead of quietly running upstream's path.
- **A short patch queue** on an exact upstream commit (branch
  `queue/<first 10 characters of the upstream commit>`): one commit per upstream
  file we still have to touch, each named `queue: <path>`. Most are hooks of a few
  lines that call into the plugin; a few are upstream bug fixes we carry until
  upstream has them.

**An engine is three pins:** an upstream vLLM commit, a queue commit on top of it,
and a plugin commit. Every speed feature sits behind an environment variable that
is off in the code by default (all declared in `vllm_sm80/envs.py`); each model's
recipe turns on its tested set.

## Models

Each model has a recipe repository that ships its engine as a container image,
downloads the weights and starts an OpenAI-compatible server with one command.
Questions about running a model belong in its recipe's issues.

| Model | Cards | Current release | Recipe | Engine pins |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 4 | 1.0.2 | [dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) | [`dsv41-flash-recipe-1.0.2-queue`](https://github.com/Morrowmake/vllm-cmp170hx/tree/dsv41-flash-recipe-1.0.2-queue), [`dsv41-flash-recipe-1.0.2-plugin`](https://github.com/Morrowmake/vllm-cmp170hx/tree/dsv41-flash-recipe-1.0.2-plugin) |
| GLM-5.3-Flash | 4 | 1.8.0 | [glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) | [`glm53-recipe-1.8.0`](https://github.com/Morrowmake/vllm-cmp170hx/tree/glm53-recipe-1.8.0) (`ampere` line) |
| Qwen3.8-Flash-Next | 2 | 1.0.0 | [qwen38-flash-next-cmp170hx-recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe) | [`qwen38-recipe-1.0.0`](https://github.com/Morrowmake/vllm-cmp170hx/tree/qwen38-recipe-1.0.0) (`ampere` line) |

## Pins

The pin tags are the authoritative code of each release. A recipe release tags
its engine here, so the code never goes away:

- `<model>-recipe-<version>-queue` on the queue commit. Its upstream commit is
  the queue branch's base, and the tag message names all three pins.
- `<model>-recipe-<version>-plugin` on the plugin commit.

Releases made before the plugin layout pin one commit of the frozen `ampere`
line instead, under the tag `<model>-recipe-<version>`.

## Build an engine from its pins

The recipes' container images are the supported way to run. To build the same
engine from source (Python 3.12, torch 2.14.0, CUDA toolkit 12.8 or newer for the
compiled extensions), here for DeepSeek-V4.1-Flash 1.0.2:

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
git clone https://github.com/Morrowmake/vllm-cmp170hx.git engine && cd engine
git checkout dsv41-flash-recipe-1.0.2-queue        # upstream + the patch queue
git tag -n1 dsv41-flash-recipe-1.0.2-queue         # names the upstream commit
# upstream's prebuilt extensions for that exact upstream commit (they include sm_80)
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=330c5bffeeb48725c0502a351e8ead2b54bd81eb \
  uv pip install -e .
git worktree add ../plugin dsv41-flash-recipe-1.0.2-plugin && cd ../plugin
uv pip install --no-deps -e .                      # the vllm-sm80 plugin
python csrc/exl3/build_standalone.py --out vllm_sm80                                 # EXL3 kernels
VLLM_AMPERE_MARLIN_BUILD=1 python csrc/ampere_marlin/build_standalone.py --out vllm_sm80  # optional Marlin
```

The two extension builds need no GPU. For another release, use its two tags and
the upstream commit its `-queue` tag names. An `ampere`-line release is a single
checkout of its tag.

## Branches

| Branch | What it holds |
|---|---|
| `plugin` (default) | the plugin as last released, with this README and the notices: `vllm_sm80/`, its tests, the native sources in `csrc/`, `FORK.md` (a generated map of every file and every queue patch) |
| `plugin-dev` | plugin development; may be ahead of every release |
| `queue/<upstream10>` | the patch queue for one upstream commit. A newer upstream gets a new queue branch; older ones stay. |
| `main` | a plain mirror of upstream vLLM |
| `ampere` | **frozen**: the earlier single-tree fork. Kept so that old releases and links keep working; it takes no more changes. |

## Community recipes

Community recipes are welcome in your own repository: pin our release tags, tag the repository `cmp170hx`, and keep our NOTICE (Apache-2.0 section 4(d)).

## Contributing

| You want to change | Branch to target |
|---|---|
| sm_80 kernels, attention backends, model code we own, the launcher, the fork's tests | `plugin-dev` |
| a hook or a carried fix in a vLLM file (a file that exists upstream) | the newest `queue/<upstream10>`: one commit per file, titled `queue: <path>`, amending that file's commit if it has one |
| a general vLLM fix or feature that is not sm_80-specific | upstream [vllm-project/vllm](https://github.com/vllm-project/vllm) first; we pick it up when we move upstream |
| a recipe for another model, card count or layout | your own repository (Community recipes) |

`ampere` takes no pull requests any more. If your change was written against it,
the paths move like this:

| Path on `ampere` | Path now |
|---|---|
| `vllm/sm80/<path>` | `vllm_sm80/<path>` on `plugin-dev` (imports `vllm.sm80` become `vllm_sm80`) |
| our own files in `vllm/models/<model>/nvidia/` (for example `exl3_ple.py`, `sm80_kda.py`, `ops/kda_prefill.py`) | `vllm_sm80/models/<model>/` on `plugin-dev` |
| `csrc/libtorch_stable/moe/ampere_marlin/` | `csrc/ampere_marlin/` on `plugin-dev` |
| `csrc/exl3/` | `csrc/exl3/` on `plugin-dev` |
| `plugins/sm80/pyproject.toml`, `plugins/sm80/licenses/` | `pyproject.toml`, `licenses/` on `plugin-dev` |
| fork-only tests (`tests/sm80/`, `tests/kernels/test_ampere_*`, …) | same path on `plugin-dev` |
| a file that exists in upstream vLLM (for example `vllm/v1/worker/gpu/model_runner.py`, `vllm/models/qwen4_exp/nvidia/model.py`) | the newest `queue/<upstream10>` |
| per-layout switches (`VLLM_GLM5_PP_*`, `VLLM_GLM5_TP4_*`) | one switch per operation (`VLLM_SM80_*`); the layer's shape picks the kernel. Old names are listed in `vllm_sm80/envs.py` with their replacement. |

`FORK.md` lists every file and every queue patch. New speed features
come behind a switch that is off by default, with a test that runs without a GPU.

## Licence and credit

Apache-2.0, as upstream ([LICENSE](LICENSE)). This work is a layer on the
[vLLM project](https://github.com/vllm-project/vllm) and its contributors; thank
you. Adapted code and idea credits are in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md);
the adapted ExLlamaV3 code's MIT notice ships in `licenses/exl3/`. [NOTICE](NOTICE)
states the attribution for redistributions; citation metadata is in
[CITATION.cff](CITATION.cff). If you use or build on this, please credit
Morrowmake (see [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff)).
