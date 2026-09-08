"""Same-input full-forward comparison: dense native baseline vs streaming."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from delivery.benchmark_utils import compare_calls
from delivery.probe_streaming import bootstrap, make_case, make_impl
from oscar_ascend.kernels.streaming_attention import plan_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument(
        "--cases", choices=["mtp", "concurrent", "prefill", "all"], default="all"
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Fail if any target shape is over 10%% slower than baseline",
    )
    args = ap.parse_args()
    if args.gate and (args.device != "npu" or args.cases != "all"):
        ap.error("Performance gate requires every target case on NPU")
    torch.set_num_threads(2)
    bootstrap(args.device)
    failed = []
    for name, batch, prefix, count in (
        ("mtp", 1, 24579, 4),
        ("concurrent", 8, 24579, 4),
        ("prefill", 1, 15360, 9256),
    ):
        if args.cases not in ("all", name):
            continue
        print(
            f"STREAM BENCH case={name} batch={batch} prefix={prefix} query={count}",
            flush=True,
        )
        case = make_case([prefix] * batch, [count] * batch)
        impl, layer, inputs = make_impl(case, args.device, "native")
        output = torch.empty_like(inputs[0])

        def call(mode, impl=impl, layer=layer, inputs=inputs, output=output):
            impl._oscar.attention_mode = mode
            return impl.forward(layer, *inputs, output=output)

        def sync():
            if args.device == "npu":
                torch.npu.synchronize()

        timing = compare_calls(
            {
                "baseline": lambda call=call: call("native"),
                "streaming": lambda call=call: call("streaming"),
            },
            sync,
            label="STREAM BENCH",
        )
        speedup = timing["baseline"]["median_ms"] / timing["streaming"]["median_ms"]
        plan = plan_stream(case[6], case[7], 6, 1, 256)
        print(
            "STREAM BENCH "
            + json.dumps(
                {
                    "case": name,
                    "device": args.device,
                    "batch": batch,
                    "prefix": prefix,
                    "q_len": count,
                    "timings": timing,
                    "baseline_over_streaming": speedup,
                    "streaming_split_scratch_bytes": plan.scratch_bytes,
                    "dense_history_pair_bytes": batch * prefix * 256 * 2 * 2,
                    "memory_note": "calculated tensor sizes; not measured device peak",
                    "scope": "same-input single-layer full forward; includes write/window/rotation; excludes model/communication",
                }
            ),
            flush=True,
        )
        if speedup < 1 / 1.10:
            failed.append(name)
    if args.gate and failed:
        raise SystemExit(
            "STREAM performance gate failed for "
            + ", ".join(failed)
            + "; serving was not enabled"
        )


if __name__ == "__main__":
    main()
