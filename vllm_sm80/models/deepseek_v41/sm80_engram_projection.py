# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram projection shared by ordinary and early-gather execution."""

import torch


def project_rows(engram, hash_ids: torch.Tensor) -> torch.Tensor:
    if engram._batch is not None:
        return engram.wkv(engram._batch.wkv_input(engram, hash_ids.shape[0]))
    return engram.wkv(engram.embed(hash_ids).flatten(-2))


def project_kv(engram, hash_ids: torch.Tensor) -> torch.Tensor:
    project = getattr(engram, "_project_kv", None)
    if project is not None:
        return project(hash_ids)
    return project_rows(engram, hash_ids)
