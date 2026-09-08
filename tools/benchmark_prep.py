"""Compare KV preparation paths on the shapes observed in target serving."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from delivery.probe_prefill import check_native_mtp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--cases", choices=["mtp", "prefill", "all"], default="all")
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(2)
    cases = {"mtp": (24579, 4), "prefill": (15360, 9256)}
    for name, (prefix, nq) in cases.items():
        if args.cases in (name, "all"):
            print(f"PREP BENCH case={name} prefix={prefix} q_len={nq}", flush=True)
            check_native_mtp(
                args.device,
                compare=True,
                prefix=prefix,
                nq=nq,
                block_sizes=(128,),
                windows=(True,),
            )


if __name__ == "__main__":
    main()
