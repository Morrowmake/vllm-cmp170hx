# SPDX-License-Identifier: Apache-2.0
"""Frozen DFlash2 confidence and optional row records; no online fitting."""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class FrozenCoefficients:
    slope: float
    intercepts: tuple[float, ...]
    threshold: float

    @classmethod
    def load(cls, path, width):
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1 or not data.get("calibrated"):
            raise ValueError("Draft skip requires calibrated version-1 coefficients")
        result = cls(
            float(data["slope"]),
            tuple(map(float, data["intercepts"])),
            float(data["threshold"]),
        )
        values = (result.slope, result.threshold, *result.intercepts)
        if (
            not all(math.isfinite(x) for x in values)
            or result.slope < 0
            or len(result.intercepts) != width
            or not 0 < result.threshold < 1
        ):
            raise ValueError("Invalid frozen draft confidence coefficients")
        return result


def log_odds_max_q(scores, temperature):
    temp = torch.where(temperature > 0, temperature, 1).float()
    logits = scores.float() / temp[:, None, None]
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    complement = shifted.exp().sum(dim=-1) - 1
    return (-complement.clamp_min(math.exp(-40)).log()).clamp(-40, 40)


def survival_cap(features, slope, intercepts, threshold):
    survival = torch.sigmoid(features * slope + intercepts).cumprod(dim=-1)
    # First low survival ends the prefix, even if a later feature is invalid.
    live = (survival >= threshold).to(torch.int32).cumprod(dim=-1)
    return live.sum(dim=-1).to(torch.int32)


def _apply_mask(
    boundaries,
    idx_mapping,
    positions,
    is_padding,
    logits_indices,
    predicted_positions,
    caps,
    is_prefilling,
    sentinel,
):
    # Map every input row to its request without reading device values.
    # Capture padding can repeat the final boundary; exclude it below.
    rows = torch.arange(
        is_padding.numel(), device=is_padding.device, dtype=boundaries.dtype
    )
    requests = torch.searchsorted(boundaries[1:], rows, right=True)
    safe_requests = requests.clamp_max(idx_mapping.shape[0] - 1)
    starts = boundaries[safe_requests].long()
    slots = idx_mapping[safe_requests].long()
    safe_slots = torch.where(slots >= 0, slots, sentinel)
    first_positions = positions[starts.clamp_max(rows.numel() - 1)]
    valid = (
        (requests < idx_mapping.shape[0])
        & (slots >= 0)
        & ~is_prefilling[safe_requests]
        & (predicted_positions[safe_slots, 0] == first_positions + 1)
    )
    mask = valid & (rows - starts > caps[safe_slots])
    is_padding.logical_or_(mask)
    return mask[logits_indices]


# Fuse indexing and mask construction after the boundary search. Compilation
# happens during warmup, before capture; the decode path reads no host values.
_apply_mask_compiled = torch.compile(_apply_mask, dynamic=True, fullgraph=True)


class DraftConfidence:
    def __init__(self, max_reqs, width, device, coefficients=None, log_dir=""):
        self.width = width
        self.coefficients = coefficients
        self.features = torch.full((max_reqs + 1, width), float("nan"), device=device)
        self.positions = torch.full(
            (max_reqs + 1, width), -1, device=device, dtype=torch.int64
        )
        self.caps = torch.full((max_reqs + 1,), width, device=device, dtype=torch.int32)
        self.sentinel = max_reqs
        self.bias = torch.tensor(
            coefficients.intercepts if coefficients else [-0.2] * width, device=device
        )
        self.slope = coefficients.slope if coefficients else 0.5
        self.log_path = None
        self.log_step = 0
        if log_dir:
            root = Path(log_dir)
            root.mkdir(parents=True, exist_ok=True)
            self.log_path = root / f"rows-{os.getpid()}.jsonl"
        if coefficients is not None and self.features.is_cuda:
            self._warm_mask()
        logger.info(
            "GLM-5 draft confidence active: skip=%d log=%d frozen=%d "
            "(VLLM_GLM5_DFLASH_SKIP; GLM-5 MoE row mask)",
            coefficients is not None,
            self.log_path is not None,
            coefficients is not None,
        )

    def _warm_mask(self):
        # Prime singleton and variable batch shapes before model graph capture.
        device = self.features.device
        for num_reqs, query_len in (
            (1, 1),
            (1, self.width + 1),
            (min(self.sentinel, 2), self.width + 1),
        ):
            num_tokens = num_reqs * query_len
            _apply_mask_compiled(
                torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
                * query_len,
                torch.arange(num_reqs, device=device, dtype=torch.int32),
                torch.arange(num_tokens, device=device),
                torch.zeros(num_tokens, device=device, dtype=torch.bool),
                torch.arange(num_tokens, device=device),
                self.positions,
                self.caps,
                torch.zeros(num_reqs, device=device, dtype=torch.bool),
                self.sentinel,
            )

    def predict(self, scores, idx_mapping, sample_pos, temperature, width):
        rows = scores.shape[0]
        slots = idx_mapping.view(rows, width)[:, 0].long()
        safe = torch.where(slots >= 0, slots, self.sentinel)
        values = log_odds_max_q(scores, temperature[safe.clamp_max(self.sentinel - 1)])
        self.features[safe, :width] = values
        self.positions[safe, :width] = sample_pos.view(rows, width)
        if self.coefficients is not None:
            self.caps[safe] = survival_cap(
                values, self.slope, self.bias[:width], self.coefficients.threshold
            )

    def apply_mask(self, batch, is_prefilling=None):
        if self.coefficients is None:
            return
        assert is_prefilling is not None
        apply = _apply_mask_compiled if batch.is_padding.is_cuda else _apply_mask
        batch.draft_skip_mask = apply(
            batch.query_start_loc[: batch.num_reqs + 1],
            batch.idx_mapping,
            batch.positions,
            batch.is_padding,
            batch.logits_indices,
            self.positions,
            self.caps,
            is_prefilling,
            self.sentinel,
        )

    def observe(self, batch, num_sampled, num_rejected):
        if self.log_path is None:
            return
        self.log_step += 1
        slots = batch.idx_mapping_np
        features = self.features[slots].cpu().tolist()
        positions = self.positions[slots].cpu().tolist()
        sampled = num_sampled.cpu().tolist()
        rejected = num_rejected.cpu().tolist()
        input_positions = batch.positions.cpu().tolist()
        input_tokens = batch.input_ids.cpu().tolist()
        records = []
        for r, req_id in enumerate(batch.req_ids):
            if batch.is_prefilling_np[r]:
                continue
            accepted = max(int(sampled[r]) - 1, 0)
            count = accepted + int(rejected[r])
            start = int(batch.query_start_loc_np[r])
            if not count or positions[r][0] != input_positions[start] + 1:
                continue
            request = hashlib.sha256(req_id.encode()).hexdigest()[:16]
            for j in range(min(count, self.width)):
                feature = features[r][j]
                if (
                    not math.isfinite(feature)
                    or start + j + 1 >= len(input_tokens)
                    or input_tokens[start + j + 1] < 0
                ):
                    continue
                records.append(
                    dict(
                        request=request,
                        step=positions[r][0],
                        batch_step=self.log_step,
                        worker=os.getpid(),
                        position=j,
                        width=count,
                        feature=feature,
                        accepted=j < accepted,
                        observed=j <= accepted,
                        accepted_count=accepted,
                        concurrency=batch.num_reqs,
                    )
                )
        with self.log_path.open("a") as fh:
            for record in records:
                fh.write(json.dumps(record, allow_nan=False) + "\n")


def create_draft_confidence(config, max_reqs, width, device):
    from vllm import envs

    enabled = envs.VLLM_GLM5_DFLASH_SKIP
    log_dir = envs.VLLM_GLM5_DFLASH_CONFIDENCE_LOG
    if not enabled and not log_dir:
        return None
    parallel = config.parallel_config
    spec = config.speculative_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or spec.enable_adaptive_verification
        or spec.draft_sample_method != "greedy"
    ):
        raise ValueError(
            "Draft confidence currently requires TP-only greedy DFlash2 "
            "without adaptive verification or context parallelism"
        )
    if enabled and not envs.VLLM_GLM5_MOE_MASK_PADDING:
        raise ValueError("Draft skip requires VLLM_GLM5_MOE_MASK_PADDING=1")
    coefficients = (
        FrozenCoefficients.load(envs.VLLM_GLM5_DFLASH_SKIP_COEFFICIENTS, width)
        if enabled
        else None
    )
    return DraftConfidence(max_reqs, width, device, coefficients, log_dir)
