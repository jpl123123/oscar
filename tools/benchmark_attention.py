"""Microbenchmark read paths; does not measure end-to-end serving throughput.

Use --device npu --triton after probe_paged passes. CPU mode compares the two
dense implementations and is useful for correctness and complexity checks.
"""

# ruff: noqa: B023 -- closures execute synchronously within each benchmark case
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from oscar_ascend.kernels.decode_kernel import oscar_prefill_ref
from oscar_ascend.kernels.dequant_kernel import oscar_full_dequant
from oscar_ascend.kernels.paged_attention import oscar_paged_attention_triton
from oscar_ascend.kernels.store_kernel import oscar_store_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 8192, 32768])
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(2)
    torch.manual_seed(123)
    d, hk, hq, bs, nq = 256, 1, 8, 128, 4
    rk, rv = [torch.linalg.qr(torch.randn(d, d)).Q.to(args.device) for _ in range(2)]
    for length in args.lengths:
        blocks = math.ceil(length / bs)
        kc = torch.zeros(blocks, bs, hk, d, device=args.device, dtype=torch.int8)
        vc = torch.zeros_like(kc)
        oldk, oldv = [torch.randn(length, hk, d, device=args.device) for _ in range(2)]
        oscar_store_ref(
            oldk @ rk, oldv @ rv, kc, vc, torch.arange(length, device=args.device)
        )
        bt = torch.arange(blocks, device=args.device, dtype=torch.int32).unsqueeze(0)
        q = torch.randn(nq, hq, d, device=args.device)
        k, v = [torch.randn(nq, hk, d, device=args.device) for _ in range(2)]

        def dense(inverse_history):
            kr, vr = oscar_full_dequant(
                kc, vc, bt[0], length, hk, d, use_triton=args.triton
            )
            if inverse_history:
                return oscar_prefill_ref(
                    q, k, v, kr.float() @ rk.t(), vr.float() @ rv.t(), d**-0.5, hk, d
                )
            return (
                oscar_prefill_ref(
                    q @ rk, k @ rk, v @ rv, kr.float(), vr.float(), d**-0.5, hk, d
                )
                @ rv.t()
            )

        paths = {
            "legacy_inverse_history": lambda: dense(True),
            "rotated_dense": lambda: dense(False),
        }
        if args.triton:
            paths["paged_triton"] = lambda: (
                oscar_paged_attention_triton(
                    q @ rk, k @ rk, v @ rv, kc, vc, bt, [0, nq], [length + nq], d**-0.5
                )
                @ rv.t()
            )

        def sync():
            if args.device == "npu":
                torch.npu.synchronize()

        expected = paths["legacy_inverse_history"]()
        for name, fn in paths.items():
            actual = fn()
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            for _ in range(3):
                fn()
            sync()
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn()
            sync()
            print(
                json.dumps(
                    {
                        "device": args.device,
                        "path": name,
                        "prefix": length,
                        "q_len": nq,
                        "ms": (time.perf_counter() - start) * 1000 / args.iterations,
                        "scope": "single-layer read-only microbenchmark; no staging",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
