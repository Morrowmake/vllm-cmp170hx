# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request DSpark verification depth: shared state and depth policy.

With ``VLLM_DSPARK_ADAPTIVE_DEPTH=<policy JSON>`` each verify step keeps, per
request, only the first ``live_len`` drafts the depth policy makes worth
verifying, and the rows past them (rows ``live_len + 1 .. K`` of the
request's ``K + 1`` verify rows) take no routed expert in the target's MoE
layers. The rejection sampler treats those drafts as rejected, so a request
emits at most ``live_len`` drafts plus the target's token from row
``live_len``: rows ``0 .. live_len`` are computed exactly as without the mask,
and greedy output is the unmasked output truncated at ``live_len``.

Pieces (each in its own module, all reading the one ``RowMaskState``):

* the draft step writes the confidence head's per-position outputs (and,
  when the policy uses them, the draft logits' top-1 probability and
  entropy), keyed by request-state slot, into ``conf_by_slot``
  (``stats_by_slot``);
* before the target forward of a uniform speculative decode step, a kernel
  turns them and the cost table into ``live_len`` (batch order) and
  ``row_live`` (token order); any other step leaves every row live;
* the target MoE routing gives rows with ``row_live == 0`` the expert id -1,
  which the EXL3 MoE kernels skip on every rank;
* the rejection sampler rejects the drafts past ``live_len``.

The policy file (schema ``dspark-depth-policy/1``)::

    {"schema": "dspark-depth-policy/1", "max_depth": 7,
     "features": ["bias", "logit_conf", "logit_top1", "ent"],
     "weights": [[w_1 per feature], ..., [w_max_depth per feature]],
     "costs": {"<batch>": {"<k>": ms, ...}, ...}}

Draft position i (1-based) has the estimated acceptance probability
``p_i = sigmoid(sum_f weights[i-1][f] * x_f)`` over the listed features:
``bias`` (1), ``logit_conf`` (log-odds of the confidence head's output),
``logit_top1`` (log-odds of the draft logits' top-1 probability) and ``ent``
(their entropy, nats), both of the full-vocabulary draft logits the greedy
sampler takes (Markov bias included). Probabilities are clamped to
[1e-6, 1 - 1e-6] before the log-odds. The raw confidence rule is the policy
with weights [0, 1] over [bias, logit_conf]. Other keys are ignored.

Decision per request, over k = 0 .. K (the step's verified width): at one
request, or when the file has no ``tau``, ``argmax_k E(k) / cost_r(k)``
(see ``dspark_row_mask_kernel``); with ``"tau": {"<batch>": t}`` a step of
b >= 2 requests keeps the largest k with ``prod_{j <= k} p_j >= tau_b``
(a request's extra row delays every request of the step, which its own
E(k) does not price). Batch sizes ``tau`` does not list are interpolated
linearly between the nearest listed ones (the nearest one outside them).

``costs`` gives, per batch size b, ``T_b(k)``: the step time in ms when
every request of a b-request uniform decode batch verifies k drafts; the
listed k of a batch must be 0 .. m without gaps, positive and nondecreasing.
Batch sizes not listed are interpolated linearly between the nearest listed
ones on the k both list (the nearest one outside them). Every batch size the
speculative config's per-batch-size draft counts reach must have ``T_b(k)``
for k = 0 .. K(b).

An optional ``host_width`` block has the scheduler choose the step's
verified width K from each request's ``p_i`` of an earlier round
(``dspark_host_width``); the decision kernel then also returns every
request's ``p_i`` for the block's widest K.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA = "dspark-depth-policy/1"
SWITCH = "VLLM_DSPARK_ADAPTIVE_DEPTH"
# The kernel's feature columns; a policy lists any of them, in any order.
FEATURES = ("bias", "logit_conf", "logit_top1", "ent")
# Features computed from the draft logits (not the confidence head).
STATS_FEATURES = ("logit_top1", "ent")
# Probabilities are clamped to [PROB_EPS, 1 - PROB_EPS] before the log-odds.
PROB_EPS = 1e-6


@dataclass
class DepthPolicy:
    """A loaded policy: CPU tensors sized for the state."""

    # [max_draft, len(FEATURES)] float32, unlisted features 0.
    weights: torch.Tensor
    # [max_num_reqs + 1, max_draft + 1] float32, T_b(k); NaN where not given.
    cost: torch.Tensor
    # [max_num_reqs + 1] float32: the threshold of a b-request step; 0: the
    # cost rule (always at b = 1, and everywhere without "tau").
    tau: torch.Tensor
    uses_stats: bool
    # Positions of p_i the kernel returns per request for the host width
    # (0: no host width).
    p_hat_width: int = 0


def _costs_by_batch(path, raw) -> dict[int, list[float]]:
    rows: dict[int, list[float]] = {}
    costs = raw.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise ValueError(f"{path}: costs must be a non-empty object")
    for key, by_k in costs.items():
        b = int(key)
        if b < 1:
            raise ValueError(f"{path}: batch size {key!r} is not positive")
        if not isinstance(by_k, dict) or not by_k:
            raise ValueError(f"{path}: batch {b} needs an object of k -> ms")
        ks = sorted(int(k) for k in by_k)
        if ks != list(range(len(ks))):
            raise ValueError(f"{path}: batch {b} lists k {ks}, not 0 .. m")
        row = [float(by_k[str(k)]) for k in ks]
        if any(not v > 0 for v in row) or any(a > c for a, c in zip(row, row[1:])):
            raise ValueError(f"{path}: batch {b} costs must be positive and nondecreasing: {row}")
        rows[b] = row
    return rows


def load_policy(
    path: str | Path,
    max_num_reqs: int,
    max_draft: int,
    verify_drafts: Sequence[int],
) -> DepthPolicy:
    """Read and check a ``dspark-depth-policy/1`` file for a drafter of
    ``max_draft`` drafts and ``max_num_reqs`` requests; ``verify_drafts[b]``
    is the draft count a b-request step verifies (index 0 unused).
    ValueError on a malformed policy or a cost table that does not cover
    ``verify_drafts``."""
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    depth = raw.get("max_depth")
    if not isinstance(depth, int) or depth < max_draft:
        raise ValueError(
            f"{path}: max_depth {depth!r} does not cover the drafter's {max_draft} drafts"
        )
    features = raw.get("features")
    if (
        not isinstance(features, list)
        or len(set(features)) != len(features)
        or any(f not in FEATURES for f in features)
    ):
        raise ValueError(f"{path}: features must be distinct names among {FEATURES}, got {features!r}")
    weights = raw.get("weights")
    if not isinstance(weights, list) or len(weights) != depth:
        raise ValueError(f"{path}: weights needs {depth} rows (one per draft position)")
    table = torch.zeros(max_draft, len(FEATURES), dtype=torch.float32)
    for i, row in enumerate(weights):
        if not isinstance(row, list) or len(row) != len(features):
            raise ValueError(f"{path}: weights row {i} needs {len(features)} values")
        values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}: weights row {i} is not finite: {values}")
        if i < max_draft:
            for name, v in zip(features, values):
                table[i, FEATURES.index(name)] = v

    rows = _costs_by_batch(path, raw)
    known = sorted(rows)
    cost = torch.full((max_num_reqs + 1, max_draft + 1), float("nan"), dtype=torch.float32)
    for b in range(1, max_num_reqs + 1):
        if b <= known[0]:
            row = rows[known[0]]
        elif b >= known[-1]:
            row = rows[known[-1]]
        else:
            lo = max(k for k in known if k <= b)
            hi = min(k for k in known if k >= b)
            w = 0.0 if hi == lo else (b - lo) / (hi - lo)
            row = [x + (y - x) * w for x, y in zip(rows[lo], rows[hi])]
        n = min(len(row), max_draft + 1)
        cost[b, :n] = torch.tensor(row[:n], dtype=torch.float32)
        k_b = verify_drafts[b] if b < len(verify_drafts) else verify_drafts[-1]
        if k_b > max_draft or (k_b > 0 and k_b >= n):
            raise ValueError(
                f"{path}: costs at batch {b} give k = 0 .. {n - 1}, but a "
                f"{b}-request step verifies {k_b} drafts"
            )
    tau = torch.zeros(max_num_reqs + 1, dtype=torch.float32)
    raw_tau = raw.get("tau")
    if raw_tau is not None:
        if not isinstance(raw_tau, dict) or not raw_tau:
            raise ValueError(f"{path}: tau must be a non-empty object of batch -> threshold")
        given = {int(b): float(t) for b, t in raw_tau.items()}
        if any(b < 1 or not 0.0 < t <= 1.0 for b, t in given.items()):
            raise ValueError(f"{path}: tau needs positive batch sizes and thresholds in (0, 1]")
        known_t = sorted(given)
        for b in range(2, max_num_reqs + 1):
            if b <= known_t[0]:
                t = given[known_t[0]]
            elif b >= known_t[-1]:
                t = given[known_t[-1]]
            else:
                lo = max(k for k in known_t if k <= b)
                hi = min(k for k in known_t if k >= b)
                w = 0.0 if hi == lo else (b - lo) / (hi - lo)
                t = given[lo] + (given[hi] - given[lo]) * w
            tau[b] = t
    p_hat_width = 0
    if raw.get("host_width") is not None:
        from vllm_sm80.spec_decode.dspark_host_width import parse_host_width

        p_hat_width = parse_host_width(path, raw).width
        if p_hat_width > max_draft:
            raise ValueError(
                f"{path}: host_width chooses K up to {p_hat_width}, past the "
                f"drafter's {max_draft} drafts"
            )
    return DepthPolicy(
        weights=table,
        cost=cost,
        tau=tau,
        uses_stats=any(f in STATS_FEATURES for f in features),
        p_hat_width=p_hat_width,
    )


@dataclass
class RowMaskState:
    """Persistent device buffers shared by the depth pieces (fixed
    addresses: CUDA graphs read and write them in place)."""

    max_num_reqs: int
    max_num_tokens: int
    max_draft: int
    device: torch.device
    # Per request-state slot, the confidence head's output for each drafted
    # position of the slot's latest draft step (1.0: no confidence yet).
    conf_by_slot: torch.Tensor = field(init=False)
    # Per slot and drafted position, the draft logits' top-1 probability and
    # entropy ([max_num_reqs, 2, max_draft]; written only when the policy
    # uses them).
    stats_by_slot: torch.Tensor = field(init=False)
    # Policy weights [max_draft, len(FEATURES)].
    weights: torch.Tensor = field(init=False)
    # T_b(k), [max_num_reqs + 1, max_draft + 1].
    cost: torch.Tensor = field(init=False)
    # Threshold per batch size (0: the cost rule), [max_num_reqs + 1].
    tau: torch.Tensor = field(init=False)
    # Per request of the current verify step (batch order): drafts kept.
    live_len: torch.Tensor = field(init=False)
    # Per token row of the current target forward: 1 live, 0 masked.
    row_live: torch.Tensor = field(init=False)
    # Host flag: row_live may hold a 0 (set by the mask kernel's caller,
    # cleared when the rows are reset to live).
    active: bool = False
    uses_stats: bool = False
    # Host width: positions of p_i returned per request (0: off), and this
    # step's [num_reqs, p_hat_width] output (None on a step without the
    # kernel); the step's async output takes it (take_p_hat).
    p_hat_width: int = 0
    p_hat_step: torch.Tensor | None = None

    def __post_init__(self) -> None:
        d = self.device
        self.conf_by_slot = torch.ones(
            self.max_num_reqs, self.max_draft, dtype=torch.float32, device=d
        )
        self.stats_by_slot = torch.zeros(
            self.max_num_reqs, 2, self.max_draft, dtype=torch.float32, device=d
        )
        self.weights = torch.zeros(
            self.max_draft, len(FEATURES), dtype=torch.float32, device=d
        )
        self.cost = torch.zeros(
            self.max_num_reqs + 1, self.max_draft + 1, dtype=torch.float32, device=d
        )
        self.tau = torch.zeros(self.max_num_reqs + 1, dtype=torch.float32, device=d)
        self.live_len = torch.full(
            (self.max_num_reqs,), self.max_draft, dtype=torch.int32, device=d
        )
        self.row_live = torch.ones(self.max_num_tokens, dtype=torch.int8, device=d)

    def set_policy(self, policy: DepthPolicy) -> None:
        self.weights.copy_(policy.weights)
        self.cost.copy_(policy.cost)
        self.tau.copy_(policy.tau)
        self.uses_stats = policy.uses_stats
        self.p_hat_width = policy.p_hat_width

    def take_p_hat(self) -> torch.Tensor | None:
        """This step's p_hat, once."""
        p_hat, self.p_hat_step = self.p_hat_step, None
        return p_hat

    def reset_live(self) -> None:
        """Every row live again (no-op unless a mask was applied)."""
        if self.active:
            self.row_live.fill_(1)
            self.live_len.fill_(self.max_draft)
            self.active = False


_STATE: RowMaskState | None = None


def enabled() -> bool:
    return bool(envs.VLLM_DSPARK_ADAPTIVE_DEPTH)


def init_row_mask_state(
    *,
    max_num_reqs: int,
    max_num_tokens: int,
    max_draft: int,
    device: torch.device,
    verify_drafts: Sequence[int],
) -> RowMaskState | None:
    """Create the state once (later calls must agree and return it); None
    when the switch is off. ``verify_drafts[b]``: the drafts a b-request
    step verifies."""
    global _STATE
    if not enabled():
        return None
    if _STATE is not None:
        same = (
            _STATE.max_num_reqs == max_num_reqs
            and _STATE.max_num_tokens == max_num_tokens
            and _STATE.max_draft == max_draft
            and _STATE.device == torch.device(device)
        )
        if not same:
            raise ValueError("DSpark depth state was created with other sizes")
        return _STATE
    policy = load_policy(
        envs.VLLM_DSPARK_ADAPTIVE_DEPTH, max_num_reqs, max_draft, verify_drafts
    )
    state = RowMaskState(max_num_reqs, max_num_tokens, max_draft, torch.device(device))
    state.set_policy(policy)
    if policy.p_hat_width:
        from vllm_sm80.spec_decode.dspark_host_width import install_output_copy

        install_output_copy()
    _STATE = state
    logger.info(
        "DSpark adaptive depth on (%s=%s): %d drafts, %d request slots%s",
        SWITCH,
        envs.VLLM_DSPARK_ADAPTIVE_DEPTH,
        max_draft,
        max_num_reqs,
        (", draft-logit features" if policy.uses_stats else "")
        + (f", p_i 1..{policy.p_hat_width} to the host width" if policy.p_hat_width else ""),
    )
    return state


def get_row_mask_state() -> RowMaskState | None:
    """The state, or None when the switch is off or nothing created it yet."""
    return _STATE


def _reset_for_tests() -> None:
    global _STATE
    _STATE = None
