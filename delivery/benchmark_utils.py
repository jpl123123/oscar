"""Interleaved A/B timings with first-call cost separated from warm samples."""

import statistics
import time

import torch


def compare_calls(
    calls, sync, repeats=6, *, clock=time.perf_counter, label="PREP BENCH"
):
    if repeats < 2:
        raise ValueError("A/B timing requires at least two rounds")
    names = list(calls)
    if len(names) < 2:
        raise ValueError("A/B timing requires at least two variants")
    first, outputs, samples = {}, {}, {name: [] for name in names}

    def measure(fn):
        sync()
        start = clock()
        value = fn()
        sync()
        return value, (clock() - start) * 1000

    for name, fn in calls.items():
        print(f"{label} first call: {name}", flush=True)
        value, first[name] = measure(fn)
        # Outputs may share the same destination buffer across variants.
        outputs[name] = value.detach().float().cpu().clone()
    for name in names[1:]:
        torch.testing.assert_close(
            outputs[name], outputs[names[0]], atol=1e-2, rtol=1e-2
        )
    for round_id in range(repeats):
        for name in names if round_id % 2 == 0 else names[::-1]:
            _, elapsed = measure(calls[name])
            samples[name].append(elapsed)
    return {
        name: {
            "first_call_ms": first[name],
            "median_ms": statistics.median(samples[name]),
            "samples_ms": samples[name],
            "numerical_match": True,
        }
        for name in names
    }
