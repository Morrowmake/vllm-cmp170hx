<!-- Modified by Morrowmake: CMP 170HX presentation and source attribution. -->

# vLLM for the NVIDIA CMP 170HX

This repository makes [vLLM](https://github.com/vllm-project/vllm) run well on the
**NVIDIA CMP 170HX** and other sm_80 (GA100-class) cards. It is not a vLLM fork in
the usual sense. vLLM itself stays unmodified upstream code, and our work comes as
two pieces on top of it:

- **The `vllm-sm80` plugin** (this branch, Python package `vllm_sm80`). It holds
  our kernels and attention backends, the EXL3 and Marlin compiled extensions, the
  model classes for the models we serve, and the launcher. vLLM loads it through
  its standard plugin entry points at startup. If the plugin is missing, the
  engine stops instead of quietly running upstream's path.
- **A short patch queue** on an exact upstream commit: the branch
  `queue/<first 10 characters of the upstream commit>`. It has one commit per
  upstream file we still have to touch, each named `queue: <path>`. Most are hooks
  of a few lines that call into the plugin; a few are upstream bug fixes we carry
  until upstream has them. Today that is 62 commits.

An engine is these three pins put together: an upstream vLLM commit, a queue
commit and a plugin commit.

Three models are supported: **GLM-5.3-Flash** and **DeepSeek-V4.1-Flash** on four
cards, and **Qwen3.8-Flash-Next** on two. Every speed feature sits behind an
environment variable that is off in the code by default (all declared in
`vllm_sm80/envs.py`); each model's recipe turns on its tested set.

## Branches

| Branch | What it holds |
|---|---|
| `plugin` (default) | the plugin: `vllm_sm80/`, its tests, the native sources in `csrc/`, `FORK.md` (a generated map of every file and every queue patch) |
| `queue/330c5bffee`, `queue/165213f911`, … | the patch queue, one branch per upstream commit it applies to. A newer upstream gets a new queue branch; older ones stay. |
| `main` | a plain mirror of upstream vLLM |
| `ampere` | **frozen** at `12d6ed3ce0`: the earlier single-tree fork. Kept so that old releases and links keep working; it takes no more changes. |

## Use a model: take a recipe

Each supported model has a recipe repository that pins its engine, ships it as a
container image, downloads the weights and starts an OpenAI-compatible server
with one command. Questions about running a model belong in its recipe's issues.

| Model | Cards | Recipe |
|---|---|---|
| GLM-5.3-Flash | 4 | [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) |
| Qwen3.8-Flash-Next | 2 | [Morrowmake/qwen38-flash-next-cmp170hx-recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe) |
| DeepSeek-V4.1-Flash | 4 | [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) |

## Which commits a model uses (pins)

Every model pins its own three commits, and models move to a newer upstream one
at a time. A recipe release names its pins in its changelog and tags them here,
so they never go away:

- `<model>-recipe-<version>-queue` on the queue commit (its upstream commit is the
  queue branch's base), and
- `<model>-recipe-<version>-plugin` on the plugin commit.

Releases made before the move pin one commit of the frozen `ampere` line instead,
under the tag `<model>-recipe-<version>`:

| Recipe release | Tag | Commit |
|---|---|---|
| GLM-5.3-Flash 1.8.0 | `glm53-recipe-1.8.0` | [`819146023f`](https://github.com/Morrowmake/vllm-cmp170hx/commit/819146023facb31c04f1f8512e52464835a10d40) |
| Qwen3.8-Flash-Next 1.0.0 | `qwen38-recipe-1.0.0` | [`18305948ac`](https://github.com/Morrowmake/vllm-cmp170hx/commit/18305948acba3f4c4d912cdd56f2202d2ebd94e8) |
| DeepSeek-V4.1-Flash 1.0.0 | `dsv41-flash-recipe-1.0.0` | [`a0c57b5237`](https://github.com/Morrowmake/vllm-cmp170hx/commit/a0c57b52376efd67c57735bc949d4bafa22ae29e) |

The development line is upstream `330c5bffee` + `queue/330c5bffee` + `plugin`.
The newest upstream we have tested is `165213f911` + `queue/165213f911` + `plugin`.

## Assemble an engine yourself

The recipes' container images are the supported way to run. To build the same
engine from source (Python 3.12, torch 2.14.0, CUDA toolkit 12.8 or newer for the
compiled extensions), here for upstream `330c5bffee`:

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate   # a fresh environment for the engine
git clone https://github.com/Morrowmake/vllm-cmp170hx.git engine && cd engine
git checkout queue/330c5bffee        # upstream 330c5bffee plus the patch queue
# upstream's prebuilt extensions for that exact upstream commit (they include sm_80)
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=330c5bffeeb48725c0502a351e8ead2b54bd81eb \
  uv pip install -e .
git worktree add ../plugin plugin && cd ../plugin
uv pip install --no-deps -e .                      # the vllm-sm80 plugin
python csrc/exl3/build_standalone.py --out vllm_sm80                                 # EXL3 kernels
VLLM_AMPERE_MARLIN_BUILD=1 python csrc/ampere_marlin/build_standalone.py --out vllm_sm80  # optional Marlin
```

The two extension builds need no GPU. For a recipe's exact engine, check out its
`-queue` and `-plugin` tags instead of the branch heads, and use the queue's base
as the wheel commit.

## Community recipes

Community recipes are welcome in your own repository: pin our commits, tag the repository `cmp170hx`, and keep our NOTICE (Apache-2.0 section 4(d)).

## Contributing

| You want to change | Branch to target |
|---|---|
| sm_80 kernels, attention backends, model code we own, the launcher, the fork's tests | `plugin` |
| a hook or a carried fix in a vLLM file (a file that exists upstream) | the newest `queue/<upstream10>`: one commit per file, titled `queue: <path>`, amending that file's commit if it has one |
| a general vLLM fix or feature that is not sm_80-specific | upstream [vllm-project/vllm](https://github.com/vllm-project/vllm) first; we pick it up when we move upstream |
| a recipe for another model, card count or layout | your own repository (Community recipes) |

`ampere` takes no pull requests any more. If your change was written against it,
the paths move like this:

| Path on `ampere` | Path now |
|---|---|
| `vllm/sm80/<path>` | `vllm_sm80/<path>` on `plugin` (imports `vllm.sm80` become `vllm_sm80`) |
| our own files in `vllm/models/<model>/nvidia/` (for example `exl3_ple.py`, `sm80_kda.py`, `ops/kda_prefill.py`) | `vllm_sm80/models/<model>/` on `plugin` |
| `csrc/libtorch_stable/moe/ampere_marlin/` | `csrc/ampere_marlin/` on `plugin` |
| `csrc/exl3/` | `csrc/exl3/` on `plugin` |
| `plugins/sm80/pyproject.toml`, `plugins/sm80/licenses/` | `pyproject.toml`, `licenses/` on `plugin` |
| fork-only tests (`tests/sm80/`, `tests/kernels/test_ampere_*`, …) | same path on `plugin` |
| a file that exists in upstream vLLM (for example `vllm/v1/worker/gpu/model_runner.py`, `vllm/models/qwen4_exp/nvidia/model.py`) | the newest `queue/<upstream10>` |
| per-layout switches (`VLLM_GLM5_PP_*`, `VLLM_GLM5_TP4_*`) | one switch per operation (`VLLM_SM80_*`); the layer's shape picks the kernel. Old names are listed in `vllm_sm80/envs.py` with their replacement. |

`FORK.md` on `plugin` lists every file and every queue patch. New speed features
come behind a switch that is off by default, with a test that runs without a GPU.

## Licence and credit

Apache-2.0, as upstream ([LICENSE](LICENSE)). This work is a layer on the
[vLLM project](https://github.com/vllm-project/vllm) and its contributors; thank
you. Adapted code and idea credits are in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md);
the adapted ExLlamaV3 code's MIT notice ships in `licenses/exl3/`. [NOTICE](NOTICE)
states the attribution for redistributions; citation metadata is in
[CITATION.cff](CITATION.cff). If you use or build on this, please credit
Morrowmake (see [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff)).
