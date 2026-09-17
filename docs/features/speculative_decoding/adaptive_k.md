# Acceptance-Adaptive Draft Count

## Why

[Dynamic SD](dynamic_speculative_decoding.md) picks K from the *batch size*, which
captures how expensive verification is but says nothing about whether the drafts
are any good. Acceptance is strongly content-dependent: the same deployment may
sustain three accepted drafts per step on prose and barely one on code or
arithmetic, and the K that wins on one loses on the other.

[Adaptive verification](adaptive_verification.md) adapts to content, but trims the
verification request *on the device*, which some attention backends (e.g.
`DeepseekV32IndexerBackend`) cannot support.

`adaptive_k` sits between the two: it keeps an exponential moving average of how
many drafts each running request has recently had accepted and picks the draft
count for the next step from it, entirely on the scheduler's CPU thread. Nothing
is trimmed on the device, so it composes with any attention backend, and the
per-step choice comes out of the same `num_spec_tokens_to_schedule` path Dynamic
SD already uses.

## `--speculative-config` schema

```bash
--speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 4,
    "adaptive_k": {"min": 1, "max": 4, "ema": 0.8, "log_interval": 200}
  }'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `min` / `max` | `1` / `num_speculative_tokens` | Bounds on the chosen count. `max` may not exceed `num_speculative_tokens`, which sizes the drafter's buffers and the widest captured CUDA graph. |
| `ema` | `0.8` | Decay on the per-request acceptance average. Higher is smoother and slower to react. |
| `margin` | `1.0` | Added to the EMA before rounding, so a request whose drafts are all accepted reaches for one more next step. |
| `quantile` | `0.5` | Which quantile of the batch's per-request choices becomes the step's count. `0.0` is the batch minimum; see below for why that is not the default. |
| `allowed` | every count in `1..num_speculative_tokens` | Explicit list of permitted counts, e.g. `[2, 4, 7]`. Decode CUDA graphs are captured for exactly this set. |
| `log_interval` | `0` (off) | Emit a chosen-k histogram every N steps at INFO. |

The rule is

```text
k_request = clamp(round(ema_accepted_drafts + margin), min, max)
k_step    = snap_to_allowed(quantile(k_request over the batch))
```

`adaptive_k` is off by default, and is mutually exclusive with
`num_speculative_tokens_per_batch_size` and `enable_adaptive_verification` —
all three decide the same number. It is also disabled (with a warning) under data
parallelism, where ranks choosing different counts would deadlock.

With the keys absent the feature is off and every code path is the one that ran
before it existed: the scheduler's per-step counts are both initialised to
`num_speculative_tokens` and never reassigned, no extra CUDA graphs are
captured, and the drafter is never handed a `num_steps` override.

## How k climbs back

Acceptance is a *right-censored* observation: a step that offered k drafts and
had all k accepted only proves acceptance is at least k, never how much more.
Averaging the observed value as if it were the truth makes every width its own
self-fulfilling equilibrium, because the ceiling that k imposes is what the
average then measures. Measured on GLM-5.3-Flash / DFlash2 that is not a
theoretical worry: with `max: 7` the count latched at k=2 for 87-98% of steps
and gave up 16% of aggregate throughput against a fixed k=3.

So a censored sample is recorded as `k + margin` rather than `k`, and `margin`
is added again when the count is chosen. The two do different jobs: the first
repairs the *estimate* (acceptance was at least this, probably more), the
second sets the *operating point* (aim one draft beyond what you expect to have
accepted, so the last one is the one that gets rejected). The steady state is
the width at which about one draft per step is rejected, reached by ratcheting
a notch per EMA round trip -- roughly `1/(1-ema)` steps, about 5 at the
default.

Two counts are therefore tracked per step, and they are not the same number:

* `cur_num_spec_tokens` -- what this step *verifies*. It is capped by the
  drafts already on hand, because every request in a uniform decode batch must
  present the same number, so the batch minimum wins and longer draft lists are
  truncated.
* `num_spec_tokens_to_schedule` -- what the drafter is asked to *produce* for
  the next step. This is the policy's choice, uncapped by what is on hand.

Keeping them separate is what lets k recover: a step verifies the two drafts it
has while asking for three, and gets three on the next step. Collapsing them
would latch k at whatever width a bad patch of content drove it to.

## What is not evidence

The scheduler pads a request re-entering decode with `[-1]` placeholder drafts
so the batch stays uniform and inside its captured graph. Those placeholders are
rejected by construction, so the policy is told to skip that step rather than
record a fabricated zero-acceptance sample. Grammar-invalidated drafts are
discounted the same way, by subtracting `num_invalid_spec_tokens`.

## Why the count is per step, not per request

Decode runs as a uniform batch under full CUDA graphs: every request in a step
verifies the same number of drafts, because the graph is captured for a fixed
`num_tokens = batch_size x (k + 1)`. So `adaptive_k` picks one count for the
whole step. At concurrency 1 that is the same thing as per-request. At higher
concurrency the count is the `quantile` of the batch's per-request choices.

The default is `0.5`, the median, not the minimum. The minimum looks like the
safe choice -- no request is ever over-drafted -- but it is a minimum over a
sample that grows with the batch, so it drifts down as concurrency rises even
when acceptance does not. Measured on GLM-5.3-Flash W4A16 with the DFlash2
drafter (4x CMP 170HX, TP4, `max=7`), `quantile: 0.0` chose k=2 for 87-98% of
steps at concurrency 4-8 and gave up 16% of aggregate throughput against a
fixed k=3. Over-drafting one straggler wastes a verification row; under-drafting
the whole batch wastes a step. Lower it if verification rows are the scarce
resource in your deployment.

## CUDA graphs

Every count in `allowed` gets its own uniform-decode capture (query length
`k + num_sampled_tokens_per_step`), alongside the ordinary capture sizes.
A count without a graph would fall back to eager silently, so startup logs

```text
Adaptive SD: decode CUDA graphs captured for draft counts [1, 2, 3, 4].
```

and warns instead if any count is missing. A uniform decode batch that misses a
graph at runtime is also reported, with the shape that missed. When restricting
`allowed`, make sure `cudagraph_capture_sizes` still yields multiples of each
`k + 1` — e.g. DFlash2 with `"allowed": [2, 4, 7]` needs multiples of 3, 5 and 8.

## Drafter support

The count applies to **verification**: a step verifies a prefix of the drafts
already on hand and discards the tail.

* **Autoregressive drafters (`mtp`, `eagle`, `eagle3`, `draft_model`)** propose
  into a fixed-width buffer, so any count in `1..num_speculative_tokens` is legal.
* **`dflash`** emits its whole block in one fixed-shape pass whose own CUDA graph
  must not change. Its block size therefore stays at `num_speculative_tokens`;
  only the verification width varies. Use `allowed` to restrict the set.

Because the drafter always proposes the full width, `max` should be set to the
count you would otherwise have run fixed: the drafting cost does not shrink when
k does, only the verification rows do.

## Observability

With `log_interval` set, the scheduler periodically logs the chosen-k histogram:

```text
Adaptive SD: 200 steps, mean k=2.71, accepted-draft prior=1.83, k=1:12(6%) k=2:41(20%) k=3:147(74%)
```

Acceptance itself is reported as usual by the `SpecDecoding` metrics logger and
by [acceptance metrics](acceptance_metrics.md).
