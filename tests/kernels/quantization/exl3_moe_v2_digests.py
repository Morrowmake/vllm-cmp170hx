# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output digests of the EXL3 MoE operators on fixed synthetic layers, for a
byte comparison between two trees' extensions (each run in its own process).

    python exl3_moe_v2_digests.py OUT.json    (vllm imported from PYTHONPATH)

Per configuration of moe_out_v2 that predates the drafter's kernels (hidden
5120 at bits_x2 6 and 4 with intermediate 2304, 640 and 512, hidden 2560 at
bits_x2 10) the SHA-256 of the output bytes of moe_out_v2 and moe_out_v2_fold,
and of moe_out at the drafter's configurations, over row counts 1..64 and
uniform, all-local and hot routings.
Imports nothing from the tests package, so a tree without this file's
companions can run it.
"""

import hashlib
import json
import sys
from functools import partial

import numpy as np
import torch

# (name, hidden, intermediate, top_k, local, global, bits_x2, operators)
CONFIGS = (
    ("h5120-b6", 5120, 2304, 6, 96, 384, 6, ("moe_out_v2", "moe_out_v2_fold")),
    ("h5120-b4", 5120, 2304, 6, 96, 384, 4, ("moe_out_v2", "moe_out_v2_fold")),
    ("h5120-tp640-b6", 5120, 640, 6, 384, 384, 6, ("moe_out_v2", "moe_out_v2_fold")),
    ("h5120-tp512-b4", 5120, 512, 6, 384, 384, 4, ("moe_out_v2", "moe_out_v2_fold")),
    ("h2560-b10", 2560, 640, 10, 256, 512, 10, ("moe_out_v2", "moe_out_v2_fold")),
    ("drafter-b8", 5120, 2304, 3, 32, 128, 8, ("moe_out",)),
    ("drafter-tp640-b8", 5120, 640, 3, 128, 128, 8, ("moe_out",)),
    ("drafter-tp512-b8", 5120, 512, 3, 128, 128, 8, ("moe_out",)),
)
ROWS = (1, 4, 5, 6, 8, 13, 16, 17, 40, 64)
SCENARIOS = ("uniform", "local", "hot")
SETS = 4
SWIGLU_LIMIT = 10.0


def _layer(hidden, intermediate, local, bits_x2, seed, device):
    rng = np.random.default_rng(seed)
    owners, sets = [], []
    for _ in range(SETS):
        parts = []
        for k, n in (
            (hidden, intermediate),
            (hidden, intermediate),
            (intermediate, hidden),
        ):
            trellis = rng.integers(-(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2))
            suh = rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)
            svh = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64
            tensors = (
                torch.from_numpy(trellis.astype(np.int16)).to(device),
                torch.from_numpy(suh.astype(np.float16)).to(device),
                torch.from_numpy(svh.astype(np.float16)).to(device),
            )
            owners.append(tensors)
            parts.append([t.data_ptr() for t in tensors])
        sets.append(parts)
    pointers = torch.tensor(
        [sets[e % SETS] for e in range(local)], dtype=torch.int64, device=device
    )
    meta = torch.tensor(
        [
            [
                [hidden, intermediate, bits_x2, 2],
                [hidden, intermediate, bits_x2, 2],
                [intermediate, hidden, bits_x2, 2],
            ]
        ]
        * local,
        dtype=torch.int64,
        device=device,
    )
    return owners, pointers, meta


def _case(scenario, rows, top_k, local, global_experts, hidden, seed):
    generator = torch.Generator().manual_seed(seed)
    pool = local if scenario == "local" else global_experts
    ids = torch.stack(
        [torch.randperm(pool, generator=generator)[:top_k] for _ in range(rows)]
    )
    if scenario == "hot":  # two local experts in every row: several row tiles each
        ids[:, 0] = 5
        ids[:, 1] = SETS * 3 + 1
    weights = torch.rand(rows, top_k, generator=generator) + 0.01
    weights = weights / weights.sum(-1, keepdim=True)
    x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16)
    return x.cuda(), ids.to(torch.int32).cuda(), weights.cuda()


def main(path):
    from vllm_sm80.kernels.exl3 import ops

    ops.ensure_extension()
    device = torch.device("cuda")
    digests = {}
    for (
        name,
        hidden,
        inter,
        top_k,
        local,
        global_experts,
        bits_x2,
        operators,
    ) in CONFIGS:
        owners, pointers, meta = _layer(
            hidden, inter, local, bits_x2, 1022 + bits_x2, device
        )
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
        expert_map[:local] = torch.arange(local, dtype=torch.int32, device=device)
        for operator in operators:
            if operator == "moe_out":
                fn, size = ops.moe_out, ops.moe_workspace_size
            else:
                fn = partial(getattr(ops, operator), bits_x2=bits_x2)
                size = ops.moe_workspace_size_v2
                if operator == "moe_out_v2_fold":
                    ops.require_moe_v2_fold()
            for scenario in SCENARIOS:
                for rows in ROWS:
                    x, ids, weights = _case(
                        scenario,
                        rows,
                        top_k,
                        local,
                        global_experts,
                        hidden,
                        rows * 31 + len(scenario),
                    )
                    out = torch.full_like(x, float("nan"))
                    workspace = torch.full(
                        (size(rows, hidden, inter, top_k, local),),
                        float("nan"),
                        device=device,
                    )
                    fn(
                        x,
                        ids,
                        weights,
                        pointers,
                        meta,
                        expert_map,
                        inter,
                        out,
                        workspace,
                        swiglu_limit=SWIGLU_LIMIT,
                    )
                    torch.cuda.synchronize()
                    data = out.view(torch.int16).cpu().numpy().tobytes()
                    digests[f"{name}/{operator}/{scenario}/{rows}"] = hashlib.sha256(
                        data
                    ).hexdigest()
        del owners
    with open(path, "w") as handle:
        json.dump(digests, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
