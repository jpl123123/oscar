"""Bounded KV slabs + native attention: numerics, lifecycle and real backend."""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from oscar_ascend.kernels.slab_attention import plan_slabs, slab_attention
from oscar_ascend.kernels.store_kernel import oscar_store_ref
from oscar_ascend.kernels.streaming_attention import (
    streaming_attention_ref,
    streaming_attention_triton,
)


def bootstrap(device):
    if device == "npu":
        import torch_npu  # noqa: F401
        from vllm.platforms import current_platform

        if current_platform.device_type != "npu":
            raise RuntimeError("Streaming gate requires the Ascend platform")
        current_platform.pre_register_and_update()


def make_case(
    prefixes,
    lengths,
    *,
    hk=1,
    dtype=torch.bfloat16,
    cache_dtype=torch.int8,
    bs=128,
    window=True,
    fresh=True,
):
    torch.manual_seed(117)
    d, hq = 256, hk * 6
    seqs = [p + n if fresh else p for p, n in zip(prefixes, lengths)]
    width = max(16, math.ceil(max(seqs, default=0) / bs / 16) * 16)
    blocks = len(seqs) * width
    bt = torch.randperm(blocks).reshape(len(seqs), width).int()
    caches = [torch.zeros(blocks, bs, hk, d, dtype=cache_dtype) for _ in range(2)]
    qsl = [0]
    slots = []
    for req, (prefix, n) in enumerate(zip(prefixes, lengths)):
        pos = torch.arange(prefix)
        slots.append(bt[req, pos // bs].long() * bs + pos % bs)
        qsl.append(qsl[-1] + n)
    old = [torch.randn(sum(prefixes), hk, d).to(dtype) for _ in range(2)]
    oscar_store_ref(*old, *caches, torch.cat(slots))
    q = torch.randn(qsl[-1], hq, d).to(dtype)
    k, v = (
        [torch.randn(qsl[-1], hk, d).to(dtype) for _ in range(2)]
        if fresh
        else (None, None)
    )
    stage = None
    if window:
        rows = max(math.ceil(8192 / bs), 4)
        sk = torch.zeros(rows, bs, hk, d)
        sv = torch.zeros_like(sk)
        owner = torch.full((rows, bs), -1, dtype=torch.int64)
        offset = 0
        for req, prefix in enumerate(prefixes):
            retained = list(range(min(128, prefix))) + list(
                range(max(128, prefix - 256), prefix)
            )
            for pos in retained:
                block = int(bt[req, pos // bs])
                row, col = block % rows, pos % bs
                owner[row, col] = block
                sk[row, col], sv[row, col] = old[0][offset + pos], old[1][offset + pos]
            offset += prefix
        stage = (sk, sv, owner)
    return q, k, v, *caches, bt, qsl, seqs, d**-0.5, stage


def to_device(case, device):
    q, k, v, kc, vc, bt, qsl, seqs, scale, stage = case
    return (
        q.to(device),
        None if k is None else k.to(device),
        None if v is None else v.to(device),
        kc.to(device),
        vc.to(device),
        bt.to(device),
        qsl,
        seqs,
        scale,
        None if stage is None else tuple(t.to(device) for t in stage),
    )


def kernel_cases(compatibility=None):
    if compatibility is not None:
        options = {
            "fp16": {"dtype": torch.float16},
            "large-page": {"bs": 1536},
            "fp16-large-page": {"bs": 1536, "dtype": torch.float16},
        }
        return [(compatibility, [1537], [4], options[compatibility])]
    return [
        ("mixed", [0, 129, 1025, 0], [1, 4, 3, 0], {}),
        ("batch32", [128 + i % 3 for i in range(32)], [4] * 32, {}),
        ("batch16", [129] * 16, [4] * 16, {}),
        ("batch64", [129] * 64, [4] * 64, {}),
        ("cached_prefill", [129], [1024], {}),
        ("long_24k", [24579], [4], {}),
        ("cache_only_empty", [0, 257], [1, 1], {"fresh": False}),
        ("cache_only_multi", [3, 263], [7, 9], {"fresh": False}),
        ("slab_boundary_prefill", [8193], [257], {}),
        ("multi_kv_legacy", [129], [7], {"hk": 2, "cache_dtype": torch.bfloat16}),
    ]


def check_kernel(device, compatibility=None):
    cases = kernel_cases(compatibility)
    for name, prefixes, lengths, options in cases:
        for window in (False, True):
            print(
                f"STREAM RUN device={device} case={name} window={window} "
                f"dtype={options.get('dtype', torch.bfloat16)} block_size={options.get('bs', 128)} "
                f"compatibility={compatibility is not None}",
                flush=True,
            )
            case = make_case(prefixes, lengths, window=window, **options)
            expected, expected_lse = streaming_attention_ref(*case)
            args = to_device(case, device)
            if compatibility is not None and device == "npu":
                actual, lse = streaming_attention_triton(
                    *args, experimental_profile=True
                )
            else:
                actual, lse = slab_attention(*args)
            torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(lse.cpu(), expected_lse, atol=3e-3, rtol=3e-3)
            plan = plan_slabs(
                sum(b > a for a, b in zip(case[6], case[6][1:])),
                case[3].shape[2],
                256,
                case[0].element_size(),
            )
            print(
                f"STREAM {'COMPAT PASS' if compatibility is not None else 'PASS'} device={device} case={name} window={window} "
                f"impl={'experimental_direct' if compatibility is not None else 'native_slabs'} "
                f"chunk_tokens={plan.chunk_tokens} scratch_bytes={plan.scratch_bytes}",
                flush=True,
            )


def make_impl(case, device, mode):
    from oscar_ascend.backend import AscendOscarAttentionBackendImpl

    args = to_device(case, device)
    q, k, v, kc, vc, bt, qsl, seqs, scale, stage = args
    impl = AscendOscarAttentionBackendImpl.__new__(AscendOscarAttentionBackendImpl)
    impl.head_size, impl.num_heads, impl.num_kv_heads = (
        q.shape[2],
        q.shape[1],
        kc.shape[2],
    )
    impl.scale = scale
    impl.key_cache = impl.value_cache = None
    impl._oscar_setup()
    impl._oscar.attention_mode = mode
    impl._oscar.use_paged = impl._oscar.use_fused_prep = False
    impl._oscar_use_triton = device == "npu"
    impl._oscar.k_clip_ratio, impl._oscar.v_clip_ratio = 0.96, 0.92
    impl._oscar.window_enabled = stage is not None
    impl._set_caches((kc, vc))
    eye = torch.eye(impl.head_size, device=device)
    layer = NS(layer_name="probe.layers.0.self_attn.attn", _oscar_rots=(eye, eye))
    if stage is not None:
        layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner = stage
        layer._oscar_stage_rows = stage[2].shape[0]
        layer._oscar_stage_ready = impl._oscar_stage_ready = True
        impl.stage_block = kc.shape[1]
        impl.sink_eff = 128 // impl.stage_block * impl.stage_block
    slots = []
    for req, (a, b, seq) in enumerate(zip(qsl, qsl[1:], seqs)):
        pos = torch.arange(seq - (b - a), seq, device=device)
        slots.append(
            bt[req, pos // kc.shape[1]].long() * kc.shape[1] + pos % kc.shape[1]
        )
    metadata = NS(
        num_actual_tokens=q.shape[0],
        slot_mapping=torch.cat(slots),
        actual_seq_lengths_q=qsl[1:],
        seq_lens_list=seqs,
        block_tables=bt,
    )
    return impl, layer, (q, k, v, (kc, vc), metadata)


def check_backend(device):
    from oscar_ascend import backend

    case = make_case([129, 257], [4, 3])
    impl, layer, args = make_impl(case, device, "native")
    q, _k, _v, cache, metadata = args
    expected = impl.forward(layer, *args, output=torch.empty_like(q)).cpu().float()
    impl._oscar.attention_mode = "streaming"
    with (
        patch.object(
            backend,
            "oscar_full_dequant",
            side_effect=AssertionError("dense history used"),
        ),
        patch.object(
            backend,
            "prepare_native_kv",
            side_effect=AssertionError("dense preparation used"),
        ),
        patch.object(
            impl,
            "_stage_splice",
            side_effect=AssertionError("dense window splice used"),
        ),
    ):
        actual = impl.forward(layer, *args, output=torch.empty_like(q)).cpu().float()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    # Rewrite an already-staged historical token through the standalone update
    # hook, then issue cache-only queries. The overwritten owner must be invalid.
    replacement = torch.full(
        (1, impl.num_kv_heads, impl.head_size), 2.0, device=device, dtype=q.dtype
    )
    slot = metadata.block_tables[0, 0].long().reshape(1) * impl.stage_block
    impl.do_kv_cache_update(layer, replacement, replacement, cache, slot)
    assert (
        layer._oscar_slot_owner[
            int(slot.item()) // impl.stage_block % layer._oscar_stage_rows, 0
        ].item()
        == -1
    )
    qsl, seqs = [0] + list(metadata.actual_seq_lengths_q), metadata.seq_lens_list
    stage = tuple(
        t.cpu()
        for t in (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
    )
    expected, _ = streaming_attention_ref(
        q.cpu(),
        None,
        None,
        cache[0].cpu(),
        cache[1].cpu(),
        metadata.block_tables.cpu(),
        qsl,
        seqs,
        impl.scale,
        stage,
    )
    actual = (
        impl.forward(layer, q, None, None, cache, metadata, output=torch.empty_like(q))
        .cpu()
        .float()
    )
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    print(
        f"STREAM BACKEND PASS device={device} no_dense=True cache_only_rewrite=True",
        flush=True,
    )
    # Prefix-free native fast path is still used by the streaming backend.
    case = make_case([0, 0], [17, 3])
    impl, layer, inputs = make_impl(case, device, "streaming")
    expected, _ = streaming_attention_ref(*case)
    actual = (
        impl.forward(layer, *inputs, output=torch.empty_like(inputs[0])).cpu().float()
    )
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    print(f"STREAM PREFIX-FREE PASS device={device}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument(
        "--compat",
        choices=["fp16", "large-page", "fp16-large-page"],
        help="Run one experimental profile independently; does not enable serving support",
    )
    args = ap.parse_args()
    torch.set_num_threads(2)
    bootstrap(args.device)
    check_kernel(args.device, args.compat)
    if args.compat is None:
        check_backend(args.device)
        print(
            f"STREAM DEPLOYMENT PASS device={args.device} impl=native_slabs query_dtype=bf16 kernel_block_size=128",
            flush=True,
        )


if __name__ == "__main__":
    main()
