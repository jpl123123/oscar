"""Same-input full-forward comparison: dense native baseline vs streaming."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from delivery.benchmark_utils import compare_calls
from delivery.probe_streaming import bootstrap, make_case, make_impl
from oscar_ascend.kernels.slab_attention import plan_slabs

CASES = (
    ("mtp", 1, 24579, 4),
    ("concurrent", 8, 24579, 4),
    ("prefill", 1, 15360, 9256),
)


def benchmark_case(name, batch, prefix, count, device):
    print(
        f"STREAM BENCH case={name} batch={batch} prefix={prefix} query={count}",
        flush=True,
    )
    case = make_case([prefix] * batch, [count] * batch)
    impl, layer, inputs = make_impl(case, device, "native")
    output = torch.empty_like(inputs[0])
    plan = plan_slabs(
        batch, 1, 256, inputs[0].element_size(), impl._oscar.stream_workspace_bytes
    )
    print(
        f"STREAM BENCH plan: impl=native_slabs chunk_tokens={plan.chunk_tokens} "
        f"scratch_bytes={plan.scratch_bytes}; first calls then 6 warm rounds; "
        "each completion waits for device synchronization",
        flush=True,
    )

    def call(mode):
        impl._oscar.attention_mode = mode
        return impl.forward(layer, *inputs, output=output)

    def sync():
        if device == "npu":
            torch.npu.synchronize()

    timing = compare_calls(
        {
            "baseline": lambda: call("native"),
            "streaming": lambda: call("streaming"),
        },
        sync,
        label="STREAM BENCH",
        progress=True,
    )
    baseline = timing["baseline"]["median_ms"]
    streaming = timing["streaming"]["median_ms"]
    if not all(math.isfinite(t) and t > 0 for t in (baseline, streaming)):
        raise RuntimeError(f"Invalid benchmark timing for {name}: {timing}")
    return {
        "case": name,
        "device": device,
        "batch": batch,
        "prefix": prefix,
        "q_len": count,
        "timings": timing,
        "baseline_over_streaming": baseline / streaming,
        "streaming_impl": "native_slabs",
        "streaming_kv_scratch_bytes": plan.scratch_bytes,
        "streaming_chunk_tokens": plan.chunk_tokens,
        "dense_history_pair_bytes": batch * prefix * 256 * 2 * 2,
        "memory_note": "calculated tensor sizes; not measured device peak",
        "scope": "same-input single-layer full forward; includes write/window/rotation; excludes model/communication",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument(
        "--cases", choices=["mtp", "concurrent", "prefill", "all"], default="all"
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Stop at the first target shape over 10%% slower than baseline",
    )
    args = ap.parse_args(argv)
    if args.gate and (args.device != "npu" or args.cases != "all"):
        ap.error("Performance gate requires every target case on NPU")
    torch.set_num_threads(2)
    bootstrap(args.device)
    for index, (name, batch, prefix, count) in enumerate(CASES):
        if args.cases not in ("all", name):
            continue
        record = benchmark_case(name, batch, prefix, count, args.device)
        print("STREAM BENCH " + json.dumps(record), flush=True)
        speedup = record["baseline_over_streaming"]
        if args.gate and (not math.isfinite(speedup) or speedup < 1 / 1.10):
            skipped = ", ".join(case[0] for case in CASES[index + 1 :]) or "none"
            raise SystemExit(
                f"STREAM performance gate failed for {name}: "
                f"baseline/streaming={speedup:.4f}, required >= {1 / 1.10:.4f}; "
                f"remaining cases skipped: {skipped}; serving was not enabled"
            )
    if args.gate:
        print("STREAM performance gate PASS: all target cases on NPU", flush=True)


if __name__ == "__main__":
    main()
