# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device failures observed at an output's existing host-copy boundary.

Each producing CUDA stream owns a sticky error word, avoiding cross-stream
initialization/reset races. The output-copy stream already waits for model
work, including joined auxiliary streams. Error words join those copies and
are read after the same existing event, before any result is published.
A failure is fatal: words never reset and can invalidate pending outputs.
There is no device assertion, host synchronization, or collective here beyond
the event synchronization the output consumer already performs.
"""
import functools
import inspect

import torch

_buffers: dict[tuple[int, int], torch.Tensor] = {}
# Bits of the sticky word: each producer ORs its own (ce_ring_all_reduce: 2;
# bit 1 is free).
CE_RING_TIMEOUT = 2
MESSAGES = (
    (CE_RING_TIMEOUT, "copy-engine ring all-reduce: a peer flag did not arrive within the wait bound; "
                      "the ranks' collectives are out of step"),
)


def error_buffer(device: torch.device) -> torch.Tensor:
    stream = torch.cuda.current_stream(device)
    index = stream.device.index
    key = (index, stream.cuda_stream)
    error = _buffers.get(key)
    if error is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("initialize the step error state on this stream before graph capture")
        error = torch.zeros((), dtype=torch.int32, device=device)
        _buffers[key] = error
    return error


class ErrorCheck:
    def __init__(self, host_flags):
        self.host_flags = host_flags

    def raise_if_failed(self):
        assert all(flag.device.type == "cpu" for flag in self.host_flags)
        word = 0
        for flag in self.host_flags:
            word |= int(flag.item())
        if word:
            raise ValueError(
                "; ".join(text for bit, text in MESSAGES if word & bit)
                or f"device step error word {word:#x}"
            )


def copy_error_flags(
    device: torch.device | None = None, *, producer_streams: bool = False
) -> ErrorCheck | None:
    """Enqueue only; the caller's existing stream/event orders and joins D2H."""
    if not _buffers:
        return None
    index = torch.cuda.current_device() if device is None else device.index
    flags = [(stream, flag) for (rank_device, stream), flag in _buffers.items() if rank_device == index]
    if not flags:
        return None
    copies = []
    for producer, flag in flags:
        if producer_streams:
            # For a caller that already synchronizes the whole device: enqueue
            # behind each producer without adding cross-stream waits/events.
            with torch.cuda.stream(torch.cuda.ExternalStream(producer, device=index)):
                copies.append(flag.to("cpu", non_blocking=True, copy=True))
        else:
            copies.append(flag.to("cpu", non_blocking=True, copy=True))
    return ErrorCheck(copies)


class _CheckedEvent:
    def __init__(self, event, errors):
        self.event, self.errors = event, errors

    def __getattr__(self, name):
        return getattr(self.event, name)

    def synchronize(self):
        self.event.synchronize()
        self.errors.raise_if_failed()


def install_output_checks(*classes, sync_runner=None):
    """Attach to existing output classes; no buffers means the original path.

    The same event is re-recorded after the error copies, not replaced with a
    new CUDA event. A proxy checks the CPU words at its existing synchronize.
    """
    for cls in classes:
        init = cls.__init__
        if getattr(init, "_step_errors", False):
            continue
        names = tuple(inspect.signature(init).parameters)
        stream_name = "copy_stream" if "copy_stream" in names else "async_output_copy_stream"
        stream_index = names.index(stream_name) - 1
        event_name = "copy_event" if stream_name == "copy_stream" else "async_copy_ready_event"

        def wrap(original, stream_name, stream_index, event_name):
            @functools.wraps(original)
            def checked_init(self, *args, **kwargs):
                original(self, *args, **kwargs)
                if not _buffers:
                    return
                copy_stream = kwargs[stream_name] if stream_name in kwargs else args[stream_index]
                with torch.cuda.stream(copy_stream):
                    errors = copy_error_flags(copy_stream.device)
                    if errors is None:
                        return
                    event = getattr(self, event_name)
                    event.record(copy_stream)
                self._sm80_step_errors = errors
                setattr(self, event_name, _CheckedEvent(event, errors))
            checked_init._step_errors = True
            return checked_init
        cls.__init__ = wrap(init, stream_name, stream_index, event_name)

    if sync_runner is not None and not getattr(sync_runner._to_list, "_step_errors", False):
        original = sync_runner._to_list
        @functools.wraps(original)
        def checked_to_list(self, sampled_token_ids):
            errors = copy_error_flags(sampled_token_ids.device)
            if errors is None:
                return original(self, sampled_token_ids)
            event = self.transfer_event
            self.transfer_event = _CheckedEvent(event, errors)
            try:
                return original(self, sampled_token_ids)
            finally:
                self.transfer_event = event
        checked_to_list._step_errors = True
        sync_runner._to_list = checked_to_list


class _DiscardedOutput:
    def __init__(self, output):
        self.output = output


def dispatch_worker_output(worker, output, output_rank):
    """Retain checked non-reply outputs without adding replies to the RPC.

    Async workers consume these on their existing output thread, never on
    the RPC thread. Other outputs keep the executor's original dispatch.
    """
    if output_rank is None or worker.rank == output_rank:
        worker.handle_output(output)
    elif getattr(output, "_sm80_step_errors", None) is not None:
        if not getattr(worker, "_step_error_dispatch", False):
            enqueue = worker.enqueue_output
            def checked_enqueue(value):
                if isinstance(value, _DiscardedOutput):
                    # Raises a normal Python error on this rank, but never
                    # enqueues an unexpected response for a later RPC.
                    value.output.get_output()
                else:
                    enqueue(value)
            worker.enqueue_output = checked_enqueue
            worker._step_error_dispatch = True
        worker.handle_output(_DiscardedOutput(output))
