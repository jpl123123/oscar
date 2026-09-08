"""Opt-in, bounded synchronized timings and integer-sort attribution.

These are diagnostic wall times (including host/JIT), not a throughput test.
No wrappers or dispatch mode are installed when PROFILE_STEPS is unset/zero.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import marshal
import os
import time
import traceback
from collections import defaultdict
from contextvars import ContextVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_active = ContextVar("oscar_perf_record", default=None)


def _sync(device):
    if device.type == "npu":
        torch.npu.synchronize(device)


def _rank():
    dist = torch.distributed
    return dist.get_rank() if dist.is_initialized() else 0


def function_identity(fn):
    fn = inspect.unwrap(fn)
    return {
        "file": fn.__code__.co_filename,
        "function": fn.__qualname__,
        "loaded_code_sha256": hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest(),
    }


class Record:
    def __init__(self, device):
        self.device = device
        self.stages = defaultdict(
            lambda: {"calls": 0, "inclusive_ms": 0.0, "wait_before_ms": 0.0}
        )
        self.sorts = defaultdict(int)
        self.integer_sort_stacks = {}
        self.first_forward = None

    def call(self, label, fn, args, kwargs):
        before = time.perf_counter()
        _sync(self.device)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _sync(self.device)
            stage = self.stages[label]
            elapsed = (time.perf_counter() - start) * 1000
            if stage["calls"] == 0:
                stage["first_ms"] = elapsed
            stage["max_ms"] = max(stage.get("max_ms", 0.0), elapsed)
            stage["calls"] += 1
            stage["inclusive_ms"] += elapsed
            stage["wait_before_ms"] += (start - before) * 1000


class SortTrace(TorchDispatchMode):
    def __init__(self, record):
        super().__init__()
        self.record = record

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.name in ("aten::sort", "aten::argsort"):
            tensor = args[0] if args else kwargs["self"]
            key = f"{func._schema.name}:{tensor.dtype}:{tensor.device.type}"
            self.record.sorts[key] += 1
            if (
                tensor.dtype in (torch.bool, torch.int32, torch.int64)
                and key not in self.record.integer_sort_stacks
            ):
                # File/function/line only; do not record tensor contents.
                self.record.integer_sort_stacks[key] = [
                    f"{f.filename}:{f.lineno}:{f.name}"
                    for f in traceback.extract_stack(limit=14)[:-1]
                ]
        return func(*args, **kwargs)


def timed(label, fn):
    if getattr(fn, "_oscar_timed", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        record = _active.get()
        if record is None:
            return fn(*args, **kwargs)
        if label == "forward" and record.first_forward is None:
            impl = args[0]
            layer = args[1] if len(args) > 1 else kwargs["layer"]
            query = args[2] if len(args) > 2 else kwargs["query"]
            record.first_forward = {
                "implementation": function_identity(fn),
                "layer": getattr(layer, "layer_name", "?"),
                "query_shape": list(query.shape),
                "query_dtype": str(query.dtype),
                "use_triton": impl._oscar_use_triton,
                "use_paged": impl._oscar.use_paged,
                "use_fused_prep": impl._oscar.use_fused_prep,
                "attention_mode": impl._oscar.attention_mode,
                "streaming_impl": "native_slabs"
                if impl._oscar.attention_mode == "streaming"
                else None,
                "cache_shape": list(impl.key_cache.shape)
                if impl.key_cache is not None
                else None,
                "stage_capacity": getattr(layer, "_oscar_stage_rows", 0)
                * getattr(impl, "stage_block", 0),
            }
        return record.call(label, fn, args, kwargs)

    wrapped._oscar_timed = True
    return wrapped


def scheduled_shape(scheduler_output):
    counts = [
        int(n)
        for n in getattr(scheduler_output, "num_scheduled_tokens", {}).values()
        if n > 0
    ]
    return {
        "requests": len(counts),
        "tokens": sum(counts),
        "max_tokens_per_request": max(counts, default=0),
    }


def capture(phase, step, device, fn, args, kwargs, *, batch=None):
    record = Record(device)
    token = _active.set(record)
    success = False
    start = time.perf_counter()
    try:
        with SortTrace(record):
            result = record.call(phase, fn, args, kwargs)
        success = True
        return result
    finally:
        _active.reset(token)
        print(
            "[oscar-ascend] PERF "
            + json.dumps(
                {
                    "phase": phase,
                    "step": step,
                    "rank": _rank(),
                    "success": success,
                    "wall_ms": (time.perf_counter() - start) * 1000,
                    "stages": dict(record.stages),
                    "sorts": dict(record.sorts),
                    "integer_sort_stacks": record.integer_sort_stacks,
                    "first_forward": record.first_forward,
                    "batch": batch,
                    "scope": "synchronized diagnostic; inclusive stages overlap; cold calls include JIT; not throughput",
                },
                sort_keys=True,
            ),
            flush=True,
        )


def install_diagnostics(runner_class):
    limit = int(os.environ.get("OSCAR_ASCEND_PROFILE_STEPS", "0"))
    if limit <= 0 or getattr(runner_class, "_oscar_perf_installed", False):
        return
    min_requests = int(os.environ.get("OSCAR_ASCEND_PROFILE_MIN_REQUESTS", "0"))
    max_per_request = int(
        os.environ.get("OSCAR_ASCEND_PROFILE_MAX_TOKENS_PER_REQUEST", "0")
    )
    if min_requests < 0 or max_per_request < 0:
        raise ValueError("OSCAR diagnostic request filters must be nonnegative")
    from . import backend
    from .kernels import paged_attention, prefill, slab_attention, streaming_attention

    cls = backend.AscendOscarAttentionBackendImpl
    for name in (
        "forward",
        "do_kv_cache_update",
        "_rotate_clip",
        "_staging_write",
        "_prefill_attention",
        "_decode_attention",
    ):
        setattr(cls, name, timed(name.removeprefix("_"), getattr(cls, name)))
    backend.staging_order = timed("staging_sort", backend.staging_order)
    backend.prepare_native_kv = timed("prepare_native_kv", backend.prepare_native_kv)
    paged_attention.oscar_paged_attention_triton = timed(
        "paged_attention", paged_attention.oscar_paged_attention_triton
    )
    prefill.npu_prefill_prepared = timed(
        "native_attention", prefill.npu_prefill_prepared
    )
    streaming_attention.streaming_attention_triton = timed(
        "direct_streaming_attention", streaming_attention.streaming_attention_triton
    )
    for name, label in (
        ("slab_attention", "streaming_attention"),
        ("dequant_slab", "slab_dequant"),
        ("native_slab_attention", "slab_native_attention"),
        ("merge_slab", "slab_merge"),
    ):
        setattr(slab_attention, name, timed(label, getattr(slab_attention, name)))
    original_execute = runner_class.execute_model
    original_sample = runner_class.sample_tokens

    @functools.wraps(original_execute)
    def execute(runner, scheduler_output, *args, **kwargs):
        total = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        count = getattr(runner, "_oscar_perf_count", 0)
        runner._oscar_perf_sample = None
        if _rank() != 0 or total <= 0 or count >= limit:
            return original_execute(runner, scheduler_output, *args, **kwargs)
        batch = scheduled_shape(scheduler_output)
        if batch["requests"] < min_requests or (
            max_per_request and batch["max_tokens_per_request"] > max_per_request
        ):
            if not getattr(runner, "_oscar_perf_waiting_printed", False):
                print(
                    f"[oscar-ascend] PERF waiting for requests>={min_requests}, "
                    f"max tokens/request<={max_per_request or 'unlimited'}; "
                    "nonmatching steps do not consume the capture budget",
                    flush=True,
                )
                runner._oscar_perf_waiting_printed = True
            return original_execute(runner, scheduler_output, *args, **kwargs)
        runner._oscar_perf_count = count + 1
        runner._oscar_perf_sample = count + 1
        runner._oscar_perf_batch = batch
        return capture(
            "execute_model",
            count + 1,
            runner.device,
            original_execute,
            (runner, scheduler_output, *args),
            kwargs,
            batch=batch,
        )

    @functools.wraps(original_sample)
    def sample(runner, *args, **kwargs):
        step = getattr(runner, "_oscar_perf_sample", None)
        runner._oscar_perf_sample = None
        if step is None:
            return original_sample(runner, *args, **kwargs)
        return capture(
            "sample_tokens",
            step,
            runner.device,
            original_sample,
            (runner, *args),
            kwargs,
            batch=runner._oscar_perf_batch,
        )

    runner_class.execute_model = execute
    runner_class.sample_tokens = sample
    runner_class._oscar_perf_installed = True
    print(
        f"[oscar-ascend] PERF enabled: first {limit} matching steps on rank 0; "
        f"min_requests={min_requests}, max_tokens_per_request={max_per_request or 'unlimited'}; "
        "synchronized timings alter throughput",
        flush=True,
    )
