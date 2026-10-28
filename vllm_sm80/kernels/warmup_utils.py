# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the pre-capture warmups of the sm_80 decode kernels.

WHY THE WARMUPS MUST RUN BEFORE CUDA GRAPH CAPTURE. Two of the families keep
an allocate-once module-level buffer -- ``routing._scratch(device, E)`` and
``kda_decode._counter(device)``. Both are left zeroed by every launch, which
is what makes a graph replay correct, and neither allocates on the timed path.
But if the first call happened *during* capture the allocation would land in
that graph's private memory pool and a second capture would reuse freed
memory. Triton also compiles on first launch, which cannot happen inside a
capture at all. So every family warmup runs from
``vllm/model_executor/warmup/kernel_warmup.py``, which
``Worker.compile_or_warm_up_model`` calls immediately before
``capture_model()``.

The warmups cover the batch sizes the decode graphs are captured for,
intersected with each family's token bound (a size above the bound can never
dispatch).
"""

# Small M values that a non-captured (eager) decode step can hit. Compiling
# them here costs a few seconds of startup and avoids a first-launch stall.
EXTRA_MS = (1, 2, 4, 8)


def token_sizes(capture_sizes, bound: int) -> list[int]:
    sizes = {int(s) for s in capture_sizes if 1 <= int(s) <= bound}
    sizes.update(m for m in EXTRA_MS if m <= bound)
    return sorted(sizes)


def find_module(model, attr: str):
    for module in model.modules():
        if hasattr(module, attr):
            return module
    return None
