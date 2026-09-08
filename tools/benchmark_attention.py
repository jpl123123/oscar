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
from oscar_ascend.kernels.prefill import oscar_prefill
from oscar_ascend.kernels.store_kernel import oscar_store_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 8192, 32768])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode")
    ap.add_argument(
        "--kv-tiles",
        nargs="+",
        type=int,
        choices=[4, 16, 32, 64, 128],
        default=[4],
        help="Default: validated tile 4. Larger tiles are experimental; 32 overflowed UB on Ascend910B4.",
    )
    ap.add_argument("--block-size", type=int, default=1536)
    ap.add_argument("--prefill-tokens", type=int, default=15360)
    ap.add_argument("--prefill-prefix", type=int, default=0)
    args = ap.parse_args()
    if (
        args.iterations <= 0
        or args.block_size <= 0
        or args.prefill_tokens <= 0
        or args.prefill_prefix < 0
    ):
        ap.error(
            "iterations, block size and prefill tokens must be positive; prefix must be nonnegative"
        )
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(2)
    torch.manual_seed(123)
    if args.mode in ("prefill", "all"):
        benchmark_prefill(args)
    if args.mode == "prefill":
        return
    d, hk, hq, bs, nq = 256, 1, 8, args.block_size, 4
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

        def reconstructed_native():
            kr, vr = oscar_full_dequant(
                kc, vc, bt[0], length, hk, d, use_triton=args.triton
            )
            return (
                oscar_prefill(
                    (q @ rk).to(torch.bfloat16),
                    (k @ rk).to(torch.bfloat16),
                    (v @ rv).to(torch.bfloat16),
                    kr.to(torch.bfloat16),
                    vr.to(torch.bfloat16),
                    d**-0.5,
                    hk,
                    d,
                ).float()
                @ rv.t()
            )

        paths["reconstructed_native_bf16"] = reconstructed_native

        def paged(tile):
            return (
                oscar_paged_attention_triton(
                    q @ rk,
                    k @ rk,
                    v @ rv,
                    kc,
                    vc,
                    bt,
                    [0, nq],
                    [length + nq],
                    d**-0.5,
                    block_kv=tile,
                )
                @ rv.t()
            )

        if args.triton:
            for tile in args.kv_tiles:
                paths[f"paged_triton_kv{tile}"] = lambda tile=tile: paged(tile)

        def sync():
            if args.device == "npu":
                torch.npu.synchronize()

        expected = paths["legacy_inverse_history"]()
        for name, fn in paths.items():
            actual = fn()
            assert torch.isfinite(actual).all().item()
            tolerance = 1e-2 if name == "reconstructed_native_bf16" else 2e-3
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
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
                        "block_size": bs,
                        "ms": (time.perf_counter() - start) * 1000 / args.iterations,
                        "scope": "single-layer read-only microbenchmark; no staging",
                    }
                ),
                flush=True,
            )


def benchmark_prefill(args):
    n, prefix, hq, hk, d = args.prefill_tokens, args.prefill_prefix, 8, 1, 256
    q = torch.randn(n, hq, d, device=args.device, dtype=torch.bfloat16)
    k, v = [
        torch.randn(prefix + n, hk, d, device=args.device, dtype=q.dtype)
        for _ in range(2)
    ]
    inputs = (q, k[prefix:], v[prefix:], k[:prefix], v[:prefix], d**-0.5, hk, d)
    expected = oscar_prefill_ref(*inputs)
    for name, fn in (
        ("prefill_sdpa_old", oscar_prefill_ref),
        ("prefill_native", oscar_prefill),
    ):
        actual = fn(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        for _ in range(2):
            fn(*inputs)
        if args.device == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            fn(*inputs)
        if args.device == "npu":
            torch.npu.synchronize()
        print(
            json.dumps(
                {
                    "device": args.device,
                    "path": name,
                    "prefix": prefix,
                    "q_len": n,
                    "ms": (time.perf_counter() - start) * 1000 / args.iterations,
                    "scope": "single-layer dense prefill; excludes rotation/dequant/write/staging; CPU uses reference",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
