# Load-Adaptive Draft Count

## Why

A deep draft pays off when few requests run: the verification rows are cheap
and every accepted draft saves a whole step. Under load the same depth spends
most of each step verifying drafts that are rejected. The best draft count
therefore follows the number of requests in the server, and within that, how
predictable the running text is.

[Dynamic SD](dynamic_speculative_decoding.md) picks K from the *scheduled batch
size* and sets the next draft count from it. [Adaptive
verification](adaptive_verification.md) adapts to content, but trims the
verification request *on the device*, which some attention backends (e.g.
`DeepseekV32IndexerBackend`) cannot support.

`adaptive_k` picks the count each step from the number of requests the server
holds (running, waiting and KV-holding), optionally refines it by recent
per-draft acceptance, and can ask the drafter to draft only that width. All of
it runs on the scheduler's CPU thread: nothing is trimmed on the device, so it
composes with any attention backend, and the per-step choice comes out of the
same `num_spec_tokens_to_schedule` path Dynamic SD already uses.

## `--speculative-config` schema

```bash
--speculative-config '{
    "method": "dflash",
    "model": "<drafter>",
    "num_speculative_tokens": 7,
    "adaptive_k": {
      "by_load": [7, 5, 3],
      "draft_by_load": true,
      "accept": {
        "costs": [1.0, 1.1009, 1.1631, 1.2214, 1.2718],
        "costs_multi": [1.0, 1.1132, 1.1958, 1.2214, 1.2718],
        "hysteresis": 0.03
      }
    }
  }'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `by_load` | required | Draft count for 1, 2, ... requests in the server; the last entry covers every larger number. No entry may exceed `num_speculative_tokens`, which sizes the drafter's buffers and the widest captured CUDA graph. |
| `accept` | off | Acceptance-aware depth within the load width; see below. |
| `draft_by_load` | `false` | The drafter drafts only the width the next step verifies (greedy DFlash2 only; otherwise it keeps its full block). |
| `log_interval` | `0` (off) | Emit a chosen-k histogram every N steps at INFO. |

`accept` is a mapping:

| Key | Default | Meaning |
| --- | --- | --- |
| `costs` | required | Relative step cost of verifying each count from the shallowest to the deepest `by_load` entry (one positive value per count). |
| `costs_multi` | `costs` | The same for steps of two or more requests. |
| `hysteresis` | `0.03` | A new depth must beat the request's last depth by this fraction. |
| `prior` | `0.75` | Per-draft acceptance a new request starts from. |
| `decay` | `0.9` | Decay on each request's accepted and rejected counts. |

`adaptive_k` is off by default, and is mutually exclusive with
`num_speculative_tokens_per_batch_size` and `enable_adaptive_verification` —
all three decide the same number. It is also disabled (with a warning) under data
parallelism, where ranks choosing different counts would deadlock. A mapping
without `by_load` is rejected.

With the key absent the feature is off and every code path is the one that ran
before it existed: the scheduler's per-step counts are both initialised to
`num_speculative_tokens` and never reassigned, no extra CUDA graphs are
captured. Draft proposal uses upstream's `num_speculative_tokens` argument;
without the policy it retains the configured count. Block-width drafters store
only the returned prefix in request state, while autoregressive drafters may
retain full-width buffers; only the scheduled draft width reaches the scheduler.

## The rule

```text
load_width = by_load[min(num_requests, len(by_load)) - 1]
k_step     = load_width                         (without accept)
k_step     = argmax_k<=load_width  sum_r E[tokens | p_r, k] / cost[k]   (with accept)
```

With `accept`, each request's per-draft acceptance `p_r` is estimated from its
own decayed counts of accepted drafts and rejections, shrunk towards the fixed
`prior`. `E[tokens | p, k] = (1 - p^(k+1)) / (1 - p)` is the expected tokens a
step commits at depth `k` (bonus token included). A request's depth depends only
on its own history and on the batch it runs in, so an identical request run
alone repeats exactly whatever ran before it.

Two counts are tracked per step, and they are not always the same number:

* `cur_num_spec_tokens` -- what this step *verifies*. It is capped by the
  drafts already on hand, because every request in a uniform decode batch must
  present the same number, so the batch minimum wins and longer draft lists are
  truncated.
* `num_spec_tokens_to_schedule` -- what the drafter is asked to *produce* for
  the next step: the full block, or with `draft_by_load` the chosen count.

Keeping them separate is what lets the depth climb back when load drops: a step
verifies the drafts it has while asking for more, and gets them on the next
step.

## What is not evidence

The scheduler pads a request re-entering decode with `[-1]` placeholder drafts
so the batch stays uniform and inside its captured graph. Those placeholders are
rejected by construction, so the policy is told to skip that step rather than
record a fabricated zero-acceptance sample. Grammar-invalidated drafts are
discounted the same way, by subtracting `num_invalid_spec_tokens`.

## CUDA graphs

Every count the policy may pick gets its own uniform-decode capture (query
length `k + num_sampled_tokens_per_step`), alongside the ordinary capture
sizes. Without `accept` those are the `by_load` entries; with it, every count
from the shallowest to the deepest entry. A count deeper than the last `by_load`
entry runs only while few requests are in the server, so its graphs are captured
only up to that request count. A count without a graph would fall back to eager
silently, so startup logs

```text
Adaptive SD: decode CUDA graphs captured for draft counts [3, 4, 5, 6, 7].
```

and warns instead if any count is missing. A uniform decode batch that misses a
graph at runtime is also reported, with the shape that missed.

## Drafter support

The count applies to **verification**: a step verifies a prefix of the drafts
already on hand and discards the tail.

* **Autoregressive drafters (`mtp`, `eagle`, `eagle3`, `draft_model`)** propose
  into a fixed-width buffer, so any count in `1..num_speculative_tokens` is legal.
* **`dflash`** emits its whole block in one fixed-shape pass. Without
  `draft_by_load` its block stays at `num_speculative_tokens` and only the
  verification width varies, so the drafting cost does not shrink with the
  count. With `draft_by_load` (greedy DFlash2) the drafter keeps one graph set
  per width and drafts the width the next step verifies, at the price of a
  one-step lag when the load drops.
* **`dspark`** defaults to a fixed-width draft followed by truncation. Opt in
  with `VLLM_DSPARK_PER_K_WIDTH=1` to draft each positive K from
  `num_speculative_tokens_per_batch_size` (or `adaptive_k.draft_by_load`) at
  its actual width. For example, `[[1,3,5],[4,8,3]]` drafts three query rows
  per request at batch sizes four through eight for anchor-sampling models,
  instead of five. This switch requires `sample_from_anchor=True`; fill-in
  models are refused before device allocation because their graph descriptors
  are not yet restricted to the selected width. Zero-K context-only steps and
  the target verification/acceptance policy are unchanged.
  Per-width anchor tensors and graph managers are created before capture;
  switching widths allocates no new device storage. Logits keep their
  maximum-width request/column layout for probabilistic rejection sampling.
  Capture sizes remain the configured **token** sizes; uncovered shapes run
  eagerly. Keep the capture ladder unchanged when comparing this switch so
  the baseline dispatch does not change. The width-five path retains its
  original layout, graph descriptors and sampling order. A non-causal backbone can
  produce different narrow-block proposals; compare acceptance and end-to-end
  output quality rather than assuming prefix identity with a wider block.

## Observability

With `log_interval` set, the scheduler periodically logs the chosen-k histogram:

```text
Adaptive SD: 200 steps, mean k=4.71, per-draft acceptance observed=0.812 (prior 0.75), k=3:41(20%) k=5:12(6%) k=7:147(74%)
```

Acceptance itself is reported as usual by the `SpecDecoding` metrics logger and
by [acceptance metrics](acceptance_metrics.md).
