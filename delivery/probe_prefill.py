"""Verify native prefill causality/GQA across the 2048 compressed-mask boundary."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from oscar_ascend.kernels.decode_kernel import oscar_prefill_ref
from oscar_ascend.kernels.prefill import oscar_prefill


def run(device):
    torch.manual_seed(73)
    torch.set_num_threads(2)
    for dtype in (torch.bfloat16, torch.float16):
        for prefix, n in ((0, 17), (257, 1), (2049, 513), (0, 2051)):
            d, hk, hq = 256, 1, 8
            q = torch.randn(n, hq, d, dtype=dtype)
            k, v = [torch.randn(prefix + n, hk, d, dtype=dtype) for _ in range(2)]
            # Independent fp32 CPU oracle, avoiding vendor SDPA dispatch.
            expected = oscar_prefill_ref(
                q.float(),
                k[prefix:].float(),
                v[prefix:].float(),
                k[:prefix].float(),
                v[:prefix].float(),
                d**-0.5,
                hk,
                d,
            )
            q, k, v = [x.to(device) for x in (q, k, v)]
            actual = (
                oscar_prefill(
                    q,
                    k[prefix:],
                    v[prefix:],
                    k[:prefix],
                    v[:prefix],
                    d**-0.5,
                    hk,
                    d,
                )
                .float()
                .cpu()
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            print(
                f"PREFILL PASS device={device} dtype={dtype} prefix={prefix} q_len={n}",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    args = ap.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    run(args.device)
