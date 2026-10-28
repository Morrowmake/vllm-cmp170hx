# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Development-only ordered admission and actual per-step execution pins.

No hook runs when the A/B switch is unset. Tagged completion arrays are held
until complete, then admitted in declared row order. Admission accounts for both
scheduled prompt tokens and input capacity including per-request draft slots.
Worker tracing is installed
only between begin/end RPCs, after capture, and restored afterwards. This is not
a timing path: reading actual input tensors deliberately synchronizes the device.
"""
import hashlib
import json
import re
from dataclasses import asdict

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)
PROTOCOL = "dev-ab-batch/1"
SCHEDULERS = ("vllm_sm80.dev_ab_batch.OrderedScheduler",
              "vllm_sm80.dev_ab_batch.OrderedAsyncScheduler")


def require_enabled():
    if not envs.VLLM_DEV_AB_SWITCH or not envs.VLLM_SERVER_DEV_MODE:
        raise ValueError("ordered A/B batches require VLLM_DEV_AB_SWITCH and VLLM_SERVER_DEV_MODE")


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def manifest(value):
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, dict) or not isinstance(value.get("id"), str)
            or not value["id"] or not isinstance(value.get("rows"), list) or not value["rows"]):
        raise ValueError("invalid ordered batch manifest")
    for row in value["rows"]:
        if (not isinstance(row, dict) or type(row.get("tokens")) is not int or row["tokens"] <= 0
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("ordered batch rows need token lengths and SHA256 identities")
    return value


def row_index(req_id, batch):
    # Completion serving appends the prompt index; InputProcessor optionally
    # appends its random uniqueness suffix. Neither changes the declared row.
    match = re.fullmatch(r"cmpl-" + re.escape(batch["id"]) + r"-(\d+)(?:-[0-9a-f]{8})?", req_id)
    if match is None or int(match[1]) >= len(batch["rows"]):
        raise ValueError("ordered batch contains an undeclared request: " + req_id)
    return int(match[1])


class OrderedAdmission:
    """Mixin over the original sync/async scheduler, not a replacement scheduler."""

    def __init__(self, *args, **kwargs):
        require_enabled()
        self._ordered_manifest = None
        self._ordered_rows = {}
        self._ordered_held = {}
        super().__init__(*args, **kwargs)

    def _clear_finished_batch(self):
        if self._ordered_manifest and not any(r in self.requests for r in self._ordered_rows):
            self._ordered_manifest = None
            self._ordered_rows.clear()
            self._ordered_held.clear()

    def add_request(self, request):
        self._clear_finished_batch()
        params = request.sampling_params
        tag = (getattr(params, "extra_args", None) or {}).get("dev_ab_batch")
        if tag is None:
            if self._ordered_manifest:
                raise ValueError("ordered batch requires exclusive admission")
            return super().add_request(request)
        require_enabled()
        batch = manifest(tag)
        row = row_index(request.request_id, batch)
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        prompt_tokens = sum(r["tokens"] for r in batch["rows"])
        input_tokens = prompt_tokens + len(batch["rows"]) * draft_slots
        if (self.policy != SchedulingPolicy.FCFS
                or len(batch["rows"]) > min(self.max_num_running_reqs, self.max_num_active_reqs)
                or prompt_tokens > self.max_num_scheduled_tokens
                or input_tokens > self.scheduler_config.max_num_batched_tokens):
            raise ValueError("ordered batch must fit one FCFS admission and prefill step, including draft reservations")
        tokens = request.prompt_token_ids
        expected = batch["rows"][row]
        if (tokens is None or len(tokens) != expected["tokens"]
                or token_digest(tokens) != expected["sha256"]):
            raise ValueError("ordered batch prompt does not match its declared row")
        if self._ordered_manifest is None:
            if self.requests:
                raise ValueError("drain before admitting an ordered batch")
            self._ordered_manifest = batch
        if batch != self._ordered_manifest or row in self._ordered_rows.values():
            raise ValueError("ordered batch is mixed or repeats a row")
        super().add_request(request)
        self.waiting.remove_request(request)
        self._ordered_rows[request.request_id] = row
        self._ordered_held[row] = request
        if len(self._ordered_held) == len(batch["rows"]):
            for i in range(len(batch["rows"])):
                self.waiting.add_request(self._ordered_held[i])
            self._ordered_held.clear()

    def get_request_counts(self):
        running, waiting = super().get_request_counts()
        return running, waiting + len(self._ordered_held)

    def has_requests(self):
        if self._ordered_held:
            # Held rows remain registered for cancellation and metrics, but
            # never start a partial GPU batch or spin the idle engine loop.
            return self.has_finished_requests()
        return super().has_requests()

    def schedule(self, *args, **kwargs):
        if self._ordered_manifest:
            self.running.sort(key=lambda r: self._ordered_rows[r.request_id])
        return super().schedule(*args, **kwargs)

    def finish_requests(self, *args, **kwargs):
        finished = super().finish_requests(*args, **kwargs)
        self._ordered_held = {i: r for i, r in self._ordered_held.items()
                              if r.request_id in self.requests}
        self._clear_finished_batch()
        return finished


class OrderedScheduler(OrderedAdmission, Scheduler):
    pass


class OrderedAsyncScheduler(OrderedAdmission, AsyncScheduler):
    pass


def configure(config):
    require_enabled()
    scheduler = config.scheduler_config
    chosen = SCHEDULERS[bool(scheduler.async_scheduling)]
    if scheduler.scheduler_cls not in (None, *SCHEDULERS):
        raise ValueError("ordered A/B batches cannot replace another scheduler extension")
    scheduler.scheduler_cls = chosen


def capability(runner):
    if not envs.VLLM_DEV_AB_SWITCH or not envs.VLLM_SERVER_DEV_MODE:
        return None
    scheduler = runner.vllm_config.scheduler_config
    return PROTOCOL if scheduler.scheduler_cls in SCHEDULERS else None


def descriptor(desc):
    value = asdict(desc)
    value["cg_mode"] = desc.cg_mode.name
    return value


class BatchTrace:
    def __init__(self, runner, batch, reference):
        self.runner, self.batch, self.reference = runner, batch, reference
        self.steps, self.restores = [], []
        self.current = None

    def replace(self, owner, name, value):
        existed = name in vars(owner)
        prior = vars(owner).get(name)
        self.restores.append((owner, name, existed, prior))
        setattr(owner, name, value)

    def restore(self):
        for owner, name, existed, prior in reversed(self.restores):
            if existed:
                setattr(owner, name, prior)
            else:
                delattr(owner, name)
        self.restores.clear()

    def mismatch(self, why):
        logger.error("Dev A/B batch pin mismatch: %s; batch=%s step=%s",
                     why, self.batch["id"], json.dumps(self.current, sort_keys=True))
        raise RuntimeError("ordered batch pin mismatch: " + why)

    def check(self, fields):
        if self.reference is None:
            return
        index = len(self.steps) - 1
        if index >= len(self.reference):
            self.mismatch("extra execution step")
        expected = self.reference[index]
        for key in fields:
            if self.current["pin"][key] != expected[key]:
                self.mismatch("step %d %s" % (index, key))

    def install(self):
        from vllm_sm80 import dev_ab as ab
        from vllm.v1.worker.gpu.cudagraph_utils import _is_compatible

        runner = self.runner
        gather = runner.gather_batch_req_state
        prepare = runner.prepare_inputs

        def gather_batch(output, dummy_run):
            state, uniform = gather(output, dummy_run)
            if dummy_run or state is None:
                self.mismatch("dummy execution inside an ordered batch")
            rows = [row_index(req, self.batch) for req in state.req_ids]
            pin = dict(rows=rows, scheduled_tokens=state.num_scheduled_tokens.tolist(),
                       verify_k=[len(output.scheduled_spec_decode_tokens.get(req, ())) for req in state.req_ids],
                       draft_k=output.num_spec_tokens_to_schedule, has_prefill=bool(state.has_prefill),
                       drafts=[])
            self.current = dict(step=len(self.steps), request_ids=list(state.req_ids),
                                state_slots=state.idx_mapping_np.tolist(), pin=pin, dispatches=[])
            self.steps.append(self.current)
            if rows != sorted(set(rows)):
                self.mismatch("execution row order differs from declared order")
            if len(self.steps) == 1 and rows != list(range(len(self.batch["rows"]))):
                self.mismatch("partial initial batch")
            self.check(("rows", "scheduled_tokens", "verify_k", "draft_k", "has_prefill"))
            return state, uniform

        def prepare_inputs(*args, **kwargs):
            batch = prepare(*args, **kwargs)
            pin = self.current["pin"]
            pin["input_ids"] = batch.input_ids[:batch.num_tokens].tolist()
            pin["positions"] = batch.positions[:batch.num_tokens].tolist()
            self.current["target_descriptor"] = descriptor(args[2])
            self.check(("input_ids", "positions"))
            return batch

        self.replace(runner, "gather_batch_req_state", gather_batch)
        self.replace(runner, "prepare_inputs", prepare_inputs)
        managers = [("target", runner.cudagraph_manager)]
        managers += [(f"draft:{w}", m) for w, m in ab._speculator(runner)._cg_managers.items()]
        for role, manager in managers:
            original = manager.dispatch

            def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                         max_query_len=None, num_ubatches=1, *, _role=role, _manager=manager,
                         _original=original):
                desc = _original(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                                 max_query_len, num_ubatches)
                effective = _manager._resolve_effective_loras(num_active_loras)
                candidate = next((d for d in _manager._candidates.get((num_tokens, effective), ())
                                  if _is_compatible(d, num_reqs, num_tokens, uniform_token_count,
                                                    effective, max_query_len, num_ubatches)), None)
                graph_key = repr(candidate) if candidate is not None else None
                logical = dict(role=_role, num_reqs=num_reqs, num_tokens=num_tokens,
                               uniform_token_count=uniform_token_count, graph_key=graph_key,
                               graph_descriptor=descriptor(candidate) if candidate is not None else None)
                if desc.cg_mode.name == "FULL" and desc != candidate:
                    self.mismatch("actual FULL descriptor differs from its dispatch pin")
                self.current["dispatches"].append(dict(logical, actual=descriptor(desc), actual_key=repr(desc)))
                if _role == "target":
                    self.current["pin"]["target"] = logical
                    self.check(("target",))
                else:
                    drafts = self.current["pin"]["drafts"]
                    drafts.append(logical)
                    if self.reference is not None:
                        expected = self.reference[len(self.steps) - 1]["drafts"]
                        if drafts != expected[:len(drafts)]:
                            self.mismatch("draft dispatch differs")
                return desc
            self.replace(manager, "dispatch", dispatch)

    def pins(self):
        if not self.steps:
            raise RuntimeError("ordered batch has no actual execution trace")
        pins = [step["pin"] for step in self.steps]
        if self.reference is not None and pins != self.reference:
            raise RuntimeError("ordered batch pin mismatch: incomplete execution or draft steps")
        return pins


def begin(worker, value, reference_key, establish):
    from vllm_sm80 import dev_ab as ab

    runner = worker.model_runner
    trace, reason = None, None
    try:
        require_enabled()
        batch = manifest(value)
        if capability(runner) != PROTOCOL:
            raise ValueError("ordered batch scheduler is not configured")
        if ab._resident_requests(runner) or "batch_trace" in ab._STATE.saved:
            raise ValueError("drain and end the previous ordered batch first")
        if (getattr(runner, "dp_size", 1) != 1 or getattr(runner, "use_pp", False)
                or getattr(runner, "pcp_manager", None) is not None
                or getattr(runner, "ubatch_runner", None) is not None):
            raise ValueError("ordered batch pins require one DP/PP/PCP group without microbatching")
        if type(establish) is not bool or not isinstance(reference_key, str) or not reference_key:
            raise ValueError("invalid ordered batch reference")
        refs = ab._STATE.saved.setdefault("batch_references", {})
        key = (ab._STATE.arm, reference_key)
        if establish == (key in refs):
            raise ValueError("ordered batch reference already exists or is missing")
        reference = None if establish else refs[key]
        trace = BatchTrace(runner, batch, reference)
    except Exception as error:
        reason = str(error)
    if not ab._transaction_vote(reason is None):
        return dict(ok=False, rank=worker.rank, reason=reason or "another rank refused batch")
    try:
        trace.install()
    except Exception as error:
        reason = str(error)
    if not ab._transaction_vote(reason is None):
        restored = True
        try:
            trace.restore()
        except Exception:
            restored = False
        if not ab._transaction_vote(restored):
            ab._fail_stop("ordered batch instrumentation rollback failed")
        return dict(ok=False, rank=worker.rank, reason=reason or "another rank could not begin batch")
    ab._STATE.saved["batch_trace"] = (trace, key, establish)
    return dict(ok=True, rank=worker.rank, protocol=PROTOCOL, batch_id=batch["id"],
                generation=ab._STATE.generation, arm=ab._arm_name(ab._STATE.arm))


def end(worker):
    from vllm_sm80 import dev_ab as ab

    entry = ab._STATE.saved.get("batch_trace")
    reason, pins = None, None
    try:
        require_enabled()
        if entry is None:
            raise ValueError("no ordered batch is active")
        if ab._resident_requests(worker.model_runner):
            raise ValueError("drain before ending the ordered batch")
        ab._synchronize()
        pins = entry[0].pins()
    except Exception as error:
        reason = str(error)
    agreed = ab._transaction_vote(reason is None)
    restored = True
    try:
        if entry is not None:
            entry[0].restore()
    except Exception:
        restored = False
    if not ab._transaction_vote(restored):
        ab._fail_stop("ordered batch instrumentation could not be removed")
    ab._STATE.saved.pop("batch_trace", None)
    if not agreed:
        return dict(ok=False, rank=worker.rank, reason=reason or "another rank has a batch pin mismatch",
                    steps=entry[0].steps if entry is not None else [])
    trace, key, establish = entry
    if establish:
        ab._STATE.saved["batch_references"][key] = pins
    return dict(ok=True, rank=worker.rank, protocol=PROTOCOL, batch_id=trace.batch["id"],
                generation=ab._STATE.generation, arm=ab._arm_name(ab._STATE.arm),
                replay=ab._STATE.replay, manifest=trace.batch, steps=trace.steps)
