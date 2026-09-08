"""Multi-request, cross-page MTP attention gate. CPU mode tests the oracle.

NPU usage: python3 delivery/probe_paged.py --device npu --triton
No serving or external requests are generated.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from oscar_ascend.kernels.paged_attention import (
    oscar_paged_attention_ref,
    oscar_paged_attention_triton,
)
from oscar_ascend.kernels.store_kernel import oscar_store_ref, oscar_store_triton


def run(device, use_triton):
    torch.manual_seed(42)
    d, hk, hq, bs = 256, 1, 8, 128
    qsl, seqs = [0, 1, 5, 7], [1, 133, 259]
    prefixes = [0, 129, 257]
    # Non-monotonic pages, multiple requests, both empty and long prefixes.
    bt = torch.tensor(
        [[7, 0, 0], [6, 2, 0], [4, 1, 5]], device=device, dtype=torch.int32
    )
    slots = torch.cat(
        [
            bt[i, torch.arange(c, device=device) // bs].long() * bs
            + torch.arange(c, device=device) % bs
            for i, c in enumerate(prefixes)
        ]
    )
    kold = torch.randn(len(slots), hk, d, device=device)
    vold = torch.randn_like(kold)
    q, k, v = (torch.randn(7, h, d, device=device) for h in (hq, hk, hk))
    for dtype in (torch.bfloat16, torch.int8):
        kc = torch.zeros(8, bs, hk, d, device=device, dtype=dtype)
        vc = torch.zeros_like(kc)
        store = oscar_store_triton if use_triton else oscar_store_ref
        # Invalid slot must not write the last page; initial cache bytes are zero.
        store(
            torch.cat([kold, kold[:1]]),
            torch.cat([vold, vold[:1]]),
            kc,
            vc,
            torch.cat([slots, slots.new_tensor([-1])]),
        )
        assert not kc[-1].view(torch.uint8).any().item()
        assert not vc[-1].view(torch.uint8).any().item()
        owner = torch.full((8, bs), -1, device=device, dtype=torch.int64)
        sk = torch.zeros(8, bs, hk, d, device=device)
        sv = torch.zeros_like(sk)
        chosen = torch.arange(0, len(slots), 5, device=device)
        block, off = slots[chosen] // bs, slots[chosen] % bs
        owner[block, off] = block
        sk[block, off], sv[block, off] = kold[chosen], vold[chosen]
        for stage in (None, (sk, sv, owner)):
            expected = oscar_paged_attention_ref(
                q, k, v, kc, vc, bt, qsl, seqs, d**-0.5, stage
            )
            actual = (
                oscar_paged_attention_triton(
                    q, k, v, kc, vc, bt, qsl, seqs, d**-0.5, stage
                )
                if use_triton
                else expected
            )
            assert (
                torch.isfinite(expected).all().item()
                and torch.isfinite(actual).all().item()
            )
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        if use_triton:
            # Reducer empty splits/empty sequences must produce zero/-inf, not NaN.
            from oscar_ascend.kernels.decode_kernel import (
                oscar_decode_ref,
                oscar_decode_triton,
            )

            lengths = torch.tensor([0, 1, 129], device="cpu", dtype=torch.int32)
            out, lse = oscar_decode_triton(q[:3], kc, vc, bt, lengths, d**-0.5, hk, d)
            ref, reflse = oscar_decode_ref(q[:3], kc, vc, bt, lengths, d**-0.5, hk, d)
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(lse[1:], reflse[1:], atol=1e-3, rtol=1e-3)
            assert torch.isneginf(lse[0]).all().item()
            # Exercise the real backend staging/metadata/forward seams on NPU,
            # not just a manually assembled arena passed to the new kernel.
            from types import SimpleNamespace

            from oscar_ascend.backend import AscendOscarAttentionBackendImpl

            impl = AscendOscarAttentionBackendImpl.__new__(
                AscendOscarAttentionBackendImpl
            )
            impl.head_size, impl.num_heads, impl.num_kv_heads = d, hq, hk
            impl.scale = d**-0.5
            impl.key_cache = impl.value_cache = None
            impl._oscar_setup()
            impl._oscar.use_paged = True
            impl._oscar_use_triton = True
            impl._oscar.window_enabled = True
            impl._oscar.k_clip_ratio = impl._oscar.v_clip_ratio = 0
            layer = SimpleNamespace(
                layer_name="probe.layers.0.self_attn.attn",
                _oscar_rots=(torch.eye(d, device=device), torch.eye(d, device=device)),
            )
            impl._set_caches([kc, vc])
            impl._ensure_staging(layer, [kc, vc])
            old_ends = [0, 129, 386]
            old_meta = SimpleNamespace(
                num_actual_tokens=len(slots),
                slot_mapping=slots,
                actual_seq_lengths_q=old_ends,
                seq_lens_list=prefixes,
            )
            impl._staging_write(layer, kold, vold, old_meta)
            address = layer._oscar_stage_k.data_ptr()
            fresh_slots = torch.cat(
                [
                    bt[i, torch.arange(c, s, device=device) // bs].long() * bs
                    + torch.arange(c, s, device=device) % bs
                    for i, (c, s) in enumerate(zip(prefixes, seqs))
                ]
            )
            md = SimpleNamespace(
                num_actual_tokens=7,
                slot_mapping=fresh_slots,
                actual_seq_lengths_q=qsl[1:],
                seq_lens_list=seqs,
                seq_lens=torch.tensor(seqs, device="cpu"),
                block_tables=bt,
            )
            result = impl.forward(
                layer, q, k, v, [kc, vc], md, output=torch.empty_like(q)
            )
            assert layer._oscar_stage_k.data_ptr() == address
            stage = (
                layer._oscar_stage_k,
                layer._oscar_stage_v,
                layer._oscar_slot_owner,
            )
            expected = oscar_paged_attention_ref(
                q, k, v, kc, vc, bt, qsl, seqs, d**-0.5, stage
            )
            torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
        print(
            f"PAGED PASS device={device} triton={use_triton} dtype={dtype} q_len=1/4/2 prefix=0/129/257"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument("--triton", action="store_true")
    args = ap.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    run(args.device, args.triton)
