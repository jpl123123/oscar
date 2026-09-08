"""Verify native prefill causality/GQA across the 2048 compressed-mask boundary."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    if os.environ.get("OSCAR_ASCEND_FUSED_PREP", "0") == "1":
        check_prepared_buffers(device)
    check_native_mtp(device)


def check_prepared_buffers(device):
    from oscar_ascend.kernels.prepare_kv import prepare_native_kv
    from oscar_ascend.kernels.store_kernel import oscar_store_ref

    torch.manual_seed(95)
    prefix, bs, d = 257, 128, 256
    bt = torch.tensor([3, 0, 2], dtype=torch.int32)
    pos = torch.arange(prefix)
    slots = bt[pos // bs].long() * bs + pos % bs
    for hk in (1, 2):
        for dtype in (torch.bfloat16, torch.int8):
            kc = torch.zeros(4, bs, hk, d, dtype=dtype)
            vc = torch.zeros_like(kc)
            oldk, oldv = [torch.randn(prefix, hk, d) for _ in range(2)]
            oscar_store_ref(oldk, oldv, kc, vc, slots)
            sk, sv = [torch.randn(2, bs, hk, d) for _ in range(2)]
            owner = torch.full((2, bs), -1, dtype=torch.int64)
            for p in (0, 127, 128, 256):
                block, off = int(bt[p // bs]), p % bs
                owner[block % 2, off] = block
            for output_dtype in (torch.bfloat16, torch.float16):
                new = [torch.randn(4, hk, d, dtype=output_dtype) for _ in range(2)]
                for stage in (None, (sk, sv, owner)):
                    expected = prepare_native_kv(
                        kc, vc, bt, prefix, *new, stage, use_triton=False
                    )
                    dev_stage = (
                        None if stage is None else tuple(t.to(device) for t in stage)
                    )
                    actual = prepare_native_kv(
                        kc.to(device),
                        vc.to(device),
                        bt.to(device),
                        prefix,
                        *(t.to(device) for t in new),
                        dev_stage,
                        use_triton=device == "npu",
                    )
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)
            print(
                f"PREP BUFFERS PASS device={device} hk={hk} cache={dtype}", flush=True
            )


def check_native_mtp(
    device,
    *,
    compare=False,
    prefix=24579,
    nq=4,
    block_sizes=(128, 1536),
    windows=(False, True),
):
    """Real OSCAR write/stage/forward, actual 6:1 GQA and 24K MTP history."""
    if device == "npu":
        from vllm.platforms import current_platform

        if current_platform.device_type != "npu":
            raise RuntimeError("Native MTP probe requires the Ascend platform")
        current_platform.pre_register_and_update()
    from oscar_ascend.backend import (
        AscendAttentionState,
        AscendOscarAttentionBackendImpl,
    )
    from oscar_ascend.kernels.paged_attention import oscar_paged_attention_ref

    torch.manual_seed(93)
    d, hk, hq = 256, 1, 6
    rk, rv = [torch.linalg.qr(torch.randn(d, d)).Q for _ in range(2)]
    oldk, oldv = [torch.randn(prefix, hk, d, dtype=torch.bfloat16) for _ in range(2)]
    q, k, v = [torch.randn(nq, h, d, dtype=torch.bfloat16) for h in (hq, hk, hk)]
    for bs in block_sizes:
        for window in windows:
            blocks = (prefix + nq + bs - 1) // bs
            bt = (
                torch.arange(blocks - 1, -1, -1, dtype=torch.int32)
                .unsqueeze(0)
                .to(device)
            )
            pos = torch.arange(prefix + nq, device=device)
            slots = bt[0, pos // bs].long() * bs + pos % bs
            cache = [
                torch.zeros(blocks, bs, hk, d, dtype=torch.int8, device=device)
                for _ in range(2)
            ]
            impl = AscendOscarAttentionBackendImpl.__new__(
                AscendOscarAttentionBackendImpl
            )
            impl.head_size, impl.num_heads, impl.num_kv_heads = d, hq, hk
            impl.scale = d**-0.5
            impl.key_cache = impl.value_cache = None
            impl._oscar_setup()
            impl._oscar.attention_mode = "native"  # this probe also provides the baseline for A/B
            impl._oscar.use_paged = False
            impl._oscar.use_fused_prep = (
                os.environ.get("OSCAR_ASCEND_FUSED_PREP", "0") == "1"
            )
            impl._oscar_use_triton = (
                device == "npu"
                and os.environ.get("OSCAR_ASCEND_USE_TRITON", "1") == "1"
            )
            impl._oscar.window_enabled = window
            impl._oscar.sink_tokens = 128
            impl._oscar.recent_tokens = 256
            impl._oscar.staging_tokens = 8192
            if compare:
                # Match serving defaults for the work shared by both variants.
                impl._oscar.k_clip_ratio = float(
                    os.environ.get("OSCAR_ASCEND_K_CLIP_RATIO", "0.96")
                )
                impl._oscar.v_clip_ratio = float(
                    os.environ.get("OSCAR_ASCEND_V_CLIP_RATIO", "0.92")
                )
            layer = SimpleNamespace(
                layer_name="probe.layers.0.self_attn.attn",
                _oscar_rots=(rk.to(device), rv.to(device)),
            )
            impl._set_caches(cache)
            old = [t.to(device) for t in (oldk, oldv)]
            impl.do_kv_cache_update(layer, *old, cache, slots[:prefix])
            if window:
                impl._ensure_staging(layer, cache)
                impl._staging_write(
                    layer,
                    *old,
                    SimpleNamespace(
                        num_actual_tokens=prefix,
                        slot_mapping=slots[:prefix],
                        actual_seq_lengths_q=[prefix],
                        seq_lens_list=[prefix],
                    ),
                )
            md = SimpleNamespace(
                attn_state=AscendAttentionState.SpecDecoding,
                num_actual_tokens=nq,
                slot_mapping=slots[prefix:],
                actual_seq_lengths_q=[nq],
                seq_lens_list=[prefix + nq],
                block_tables=bt,
            )
            fresh = [t.to(device) for t in (q, k, v)]
            output = torch.empty(nq, hq, d, dtype=q.dtype, device=device)
            if compare:
                from delivery.benchmark_utils import compare_calls

                def call(
                    fused,
                    impl=impl,
                    layer=layer,
                    fresh=fresh,
                    cache=cache,
                    md=md,
                    output=output,
                ):
                    impl._oscar.use_fused_prep = fused
                    return impl.forward(layer, *fresh, cache, md, output=output)

                def sync():
                    if device == "npu":
                        torch.npu.synchronize()

                times = compare_calls(
                    {
                        "baseline": lambda call=call: call(False),
                        "fused": lambda call=call: call(True),
                    },
                    sync,
                )
                print(
                    "PREP BENCH "
                    + json.dumps(
                        {
                            "device": device,
                            "prefix": prefix,
                            "q_len": nq,
                            "hq": hq,
                            "block_size": bs,
                            "window": window,
                            "clip_ratios": [
                                impl._oscar.k_clip_ratio,
                                impl._oscar.v_clip_ratio,
                            ],
                            "timings": times,
                            "baseline_over_fused": times["baseline"]["median_ms"]
                            / times["fused"]["median_ms"],
                            "scope": "same inputs; interleaved single-layer full forward; excludes model/communication; does not change serving defaults",
                        }
                    ),
                    flush=True,
                )
                continue
            actual = impl.forward(layer, *fresh, cache, md, output=output).cpu().float()
            stage = (
                None
                if not window
                else tuple(
                    t.cpu()
                    for t in (
                        layer._oscar_stage_k,
                        layer._oscar_stage_v,
                        layer._oscar_slot_owner,
                    )
                )
            )
            expected = (
                oscar_paged_attention_ref(
                    q.float() @ rk,
                    k.float() @ rk,
                    v.float() @ rv,
                    cache[0].cpu(),
                    cache[1].cpu(),
                    bt.cpu(),
                    [0, nq],
                    [prefix + nq],
                    impl.scale,
                    stage,
                )
                @ rv.t()
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            if device == "npu":
                torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                impl.forward(layer, *fresh, cache, md, output=output)
            if device == "npu":
                torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / 3
            print(
                f"NATIVE MTP PASS device={device} prefix={prefix} q_len={nq} hq={hq} "
                f"block_size={bs} window={window} full_forward_ms={elapsed:.3f} "
                f"fused_prep={impl._oscar.use_fused_prep} "
                "(single layer; excludes model/communication)",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    args = ap.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    run(args.device)
