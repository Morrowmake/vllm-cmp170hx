# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two DSpark adaptive-depth Triton kernels on one sm_80 card, and a CPU
proof that these checks catch a broken kernel.

- ``_row_mask_kernel`` (``compute_row_mask``) writes ``reference_row_mask``'s
  ``live_len`` wherever the float32 numpy reference's best score leads by more
  than 1e-5 (the device's exp and log may round differently near a tie), and
  ``row_live`` exactly as its own ``live_len`` lays out, eagerly and replayed
  from a CUDA graph whose inputs (slots, features, weights, costs, earlier
  row flags) change in place between replays; raw-confidence, random and
  draft-logit-statistics policies, K = 1, 3, 5, 7.
- ``_publish_kernel``, through the row-mask drafter's ``_sample_sequential``
  inside a captured draft step: replays with new slots, padding requests and
  confidences publish each request's row at its slot (zeros past a narrower
  width) and nothing for padding; a zero-draft ``propose`` between replays
  resets its batch's slots to 1.0 and the next replay overwrites them.
- CPU, Triton interpreter: the same checks pass on the real kernels and fail on
  a copy of each kernel with one deliberate fault.

    pytest -q tests/v1/spec_decode/test_dspark_row_mask_gpu.py
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_sm80.spec_decode import dspark_row_mask_conf as conf_module
from vllm_sm80.spec_decode import dspark_row_mask_kernel as kernel_module
from vllm_sm80.spec_decode.dspark_row_mask import RowMaskState

ROOT = Path(__file__).resolve().parents[3]
MAX_REQS, MAX_TOKENS, MAX_DRAFT = 8, 96, 7

needs_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80 card",
)

# One deliberate fault per kernel: (module file, exact source text, replacement).
FAULTS = {
    "row_mask": ("vllm_sm80/spec_decode/dspark_row_mask_kernel.py",
                 "live = (j <= best_k)", "live = (j < best_k)"),
    "publish": ("vllm_sm80/spec_decode/dspark_row_mask_conf.py",
                "conf = tl.load(conf_ptr + req * conf_stride + cols, mask=cols < width, other=0.0)",
                "conf = tl.load(conf_ptr + req * conf_stride + cols, mask=cols < max_draft, other=0.0)"),
}


def _capture(run):
    """A CUDA graph of ``run`` after one warm call on a side stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def _features(gen, trial):
    """Random per-slot features and weights; trial 0: raw-confidence policy,
    1: random weights over the confidence, 2: with draft-logit statistics."""
    conf = torch.rand(MAX_REQS, MAX_DRAFT, generator=gen)
    if trial == 1:
        conf = conf.sqrt().sqrt()  # near-one: keeps more drafts
    stats = torch.stack((torch.rand(MAX_REQS, MAX_DRAFT, generator=gen),
                         torch.rand(MAX_REQS, MAX_DRAFT, generator=gen) * 5.0), dim=1)
    weights = torch.zeros(MAX_DRAFT, 4)
    if trial == 0:
        weights[:, 1] = 1.0
    else:
        weights = torch.randn(MAX_DRAFT, 4, generator=gen)
        weights[:, 0] += 1.5
        weights[:, 1] = weights[:, 1].abs() + 0.5
        if trial == 1:
            weights[:, 2:4] = 0.0
    return conf, stats, weights, trial == 2


def check_row_mask(kmod, device: torch.device, graph: bool) -> None:
    """``kmod.compute_row_mask`` against ``reference_row_mask`` on CPU copies
    over batch sizes, depths, policies and refreshed inputs."""
    from tests.v1.spec_decode.test_dspark_row_mask_kernel import np_decide

    gen = torch.Generator().manual_seed(1)
    compared = 0
    for b in (1, 3, MAX_REQS):
        for k in (1, 3, 5, MAX_DRAFT):
            for uses_stats in (False, True):
                state = RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, device)
                ref = RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, torch.device("cpu"))
                state.uses_stats = ref.uses_stats = uses_stats
                idx = torch.zeros(b, dtype=torch.int64, device=device)

                def refresh(trial: int):
                    conf, stats, weights, _ = _features(
                        gen, 2 if uses_stats else trial % 2)
                    slots = torch.randperm(MAX_REQS, generator=gen)[:b]
                    conf[slots[0], 0] = 1e-4  # this request keeps no draft
                    weights[0, 1] = max(float(weights[0, 1]), 1.0)
                    steps = torch.rand(MAX_REQS + 1, MAX_DRAFT + 1, generator=gen) + 0.05
                    cost = 20.0 + steps.cumsum(1)
                    cost[0] = 0.0
                    rows = torch.randint(0, 2, (MAX_TOKENS,), generator=gen, dtype=torch.int8)
                    tau = torch.zeros(MAX_REQS + 1)
                    if trial == 1:
                        tau[2:] = torch.rand(MAX_REQS - 1, generator=gen) * 0.8 + 0.05
                    for s in (state, ref):
                        s.tau.copy_(tau)
                        s.conf_by_slot.copy_(conf)
                        s.stats_by_slot.copy_(stats)
                        s.weights.copy_(weights)
                        s.cost.copy_(cost)
                        s.row_live.copy_(rows)
                        s.live_len.fill_(-3)
                    idx.copy_(slots)
                    return slots, conf, stats, weights, cost, rows, tau

                run = lambda: kmod.compute_row_mask(state, idx, b, k)  # noqa: E731
                g = None
                if graph:
                    refresh(1)
                    g = _capture(run)
                for trial in range(3):
                    slots, conf, stats, weights, cost, rows, tau = refresh(trial)
                    if g is not None:
                        g.replay()
                        torch.cuda.synchronize()
                    else:
                        run()
                    kernel_module.reference_row_mask(ref, slots, b, k)
                    case = (b, k, uses_stats, trial, graph)
                    got = state.live_len.cpu()
                    assert got[b:].tolist() == [-3] * (MAX_REQS - b), case
                    for r, slot in enumerate(slots.tolist()):
                        _, lead = np_decide(conf[slot].tolist(), stats[slot].tolist(),
                                            weights.numpy(), cost[b].tolist(), b, k, uses_stats,
                                            float(tau[b]))
                        if lead > 1e-5:
                            compared += 1
                            assert int(got[r]) == int(ref.live_len[r]), (case, r)
                    want = rows.clone()
                    span = min(MAX_REQS * (MAX_DRAFT + 1), MAX_TOKENS)
                    want[:span] = 1
                    for r, d in enumerate(got[:b].tolist()):
                        assert 0 <= d <= k, case
                        want[r * (k + 1) + d + 1 : (r + 1) * (k + 1)] = 0
                    assert torch.equal(state.row_live.cpu(), want), case
                    assert state.active, case
    assert compared > 0


class _DraftStep:
    """Stands for the DSpark drafter under the row-mask mixin: the head writes
    a (requests, width) block of ``next_conf`` into the full confidence
    buffer, and ``propose`` drafts nothing itself."""

    def _sample_sequential(self, num_reqs, head_hidden):
        width = self.num_speculative_steps
        self.draft_token_confidence_probs[:num_reqs, :width] = self.next_conf[:num_reqs, :width]

    def propose(self, input_batch, *args, **kwargs):
        return None


def check_publish(cmod, device: torch.device, graph: bool) -> None:
    """The publish kernel through the mixin's ``_sample_sequential`` (inside a
    captured draft step when ``graph``) and the zero-draft reset of
    ``propose``."""
    gen = torch.Generator().manual_seed(2)
    d = cmod.row_mask_class(_DraftStep).__new__(cmod.row_mask_class(_DraftStep))
    d._row_mask = RowMaskState(MAX_REQS, MAX_TOKENS, MAX_DRAFT, device)
    d.dp_size = 1
    d.draft_token_confidence_probs = torch.full((MAX_REQS, MAX_DRAFT), -1.0, device=device)
    d.sample_idx_mapping = torch.full((MAX_REQS * MAX_DRAFT,), -1, dtype=torch.int64, device=device)
    d.next_conf = torch.zeros(MAX_REQS, MAX_DRAFT, device=device)
    table = d._row_mask.conf_by_slot
    table.fill_(0.25)
    want = table.cpu().clone()
    padded = 4

    def layout(slots, width):
        # prepare_dflash_inputs' layout: `width` sample rows per request
        # holding its slot, -1 for graph padding requests.
        d.sample_idx_mapping.fill_(-1)
        for r, s in enumerate(slots):
            d.sample_idx_mapping[r * width:(r + 1) * width] = s
        conf = torch.rand(MAX_REQS, MAX_DRAFT, generator=gen) + 0.5
        conf[len(slots):] = 9.0  # what padding rows of the graph compute
        d.next_conf.copy_(conf)
        for r, s in enumerate(slots):
            want[s] = 0.0
            want[s, :width] = conf[r, :width]

    for width in (MAX_DRAFT, 3):
        d.num_speculative_steps = width
        run = lambda: d._sample_sequential(padded, None)  # noqa: E731
        g = None
        if graph:
            layout([7], width)
            g = _capture(run)
            want.copy_(table.cpu())
        for slots in ([6, 1, 3], [1, 5, 0, 2], [4]):
            layout(slots, width)
            if g is not None:
                g.replay()
                torch.cuda.synchronize()
            else:
                run()
            assert torch.equal(table.cpu(), want), (width, slots, graph)
            assert not bool((table == 9.0).any()), (width, slots, graph)
        # A zero-draft step of slots 5 and 0 (no draft step runs): reset to 1.0.
        batch = type("Batch", (), {})()
        batch.num_reqs = 2
        batch.idx_mapping = torch.tensor([5, 0, 3], dtype=torch.int64, device=device)
        d.propose(batch, num_speculative_tokens=0)
        want[5] = 1.0
        want[0] = 1.0
        assert torch.equal(table.cpu(), want), (width, "zero-draft", graph)


def interpreted_results() -> dict:
    """In the Triton interpreter: each check on the real kernels and on a copy
    with one fault; the outcome as text."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    cpu = torch.device("cpu")
    out_dir = Path(os.environ["ROW_MASK_FAULT_DIR"])

    def faulty(name):
        path, old, new = FAULTS[name]
        source = (ROOT / path).read_text()
        assert source.count(old) == 1, (path, old)
        target = out_dir / f"faulty_{name}.py"
        target.write_text(source.replace(old, new))
        spec = importlib.util.spec_from_file_location(f"faulty_{name}", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def outcome(check, module):
        try:
            check(module, cpu, graph=False)
        except AssertionError as exc:
            return "AssertionError " + str(exc)[:200]
        return "pass"

    return {
        "row_mask/real": outcome(check_row_mask, kernel_module),
        "row_mask/faulty": outcome(check_row_mask, faulty("row_mask")),
        "publish/real": outcome(check_publish, conf_module),
        "publish/faulty": outcome(check_publish, faulty("publish")),
    }


@pytest.fixture(scope="module")
def interpreted(tmp_path_factory):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               ROW_MASK_FAULT_DIR=str(tmp_path_factory.mktemp("faulty")))
    code = ("import json; from tests.v1.spec_decode.test_dspark_row_mask_gpu import "
            "interpreted_results as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.cpu_test
@pytest.mark.parametrize("kernel", ("row_mask", "publish"))
def test_checks_pass_on_the_kernel_and_fail_on_a_fault(interpreted, kernel):
    assert interpreted[f"{kernel}/real"] == "pass", interpreted
    assert interpreted[f"{kernel}/faulty"].startswith("AssertionError"), interpreted


@needs_sm80
@pytest.mark.parametrize("graph", (False, True))
def test_cuda_row_mask_kernel_matches_the_reference(graph):
    check_row_mask(kernel_module, torch.device("cuda"), graph)


@needs_sm80
@pytest.mark.parametrize("graph", (False, True))
def test_cuda_publish_in_a_captured_draft_step(graph):
    check_publish(conf_module, torch.device("cuda"), graph)

