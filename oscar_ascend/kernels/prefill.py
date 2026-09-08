"""Native Ascend dense prefill after OSCAR history has been dequantized.

All inputs must be in the same rotation domain. TND sparse mode 3 applies
right-aligned causality: query i sees keys through prefix_length + i.
"""

from functools import lru_cache

import torch

from .decode_kernel import oscar_prefill_ref


@lru_cache(maxsize=8)
def _causal_mask(device):
    # Same compressed mask as Ascend's AttentionMaskBuilder. Shared by layers;
    # it does not grow with prompt length or materialize attention scores.
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), 1)


def npu_prefill(q, k, v, k_cached, v_cached, scale, hk, d):
    prefix = k_cached.shape[0]
    # Avoid copying a full prompt when there is no cached prefix.
    k_full = torch.cat((k_cached.to(q.dtype), k.to(q.dtype))) if prefix else k
    v_full = torch.cat((v_cached.to(q.dtype), v.to(q.dtype))) if prefix else v
    return npu_prefill_prepared(q, k_full, v_full, scale, hk, d)


def npu_prefill_prepared(q, k_full, v_full, scale, hk, d):
    import torch_npu

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Native OSCAR prefill requires bf16/fp16 queries")
    n, hq, _ = q.shape
    if n == 0:
        return torch.empty_like(q)
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=q.contiguous(),
        key=k_full.contiguous(),
        value=v_full.contiguous(),
        input_layout="TND",
        num_heads=hq,
        num_key_value_heads=hk,
        actual_seq_lengths=[n],
        actual_seq_lengths_kv=[k_full.shape[0]],
        atten_mask=_causal_mask(q.device),
        sparse_mode=3,
        scale=scale,
    )
    return out.view(n, hq, d)


def oscar_prefill_prepared(q, k_full, v_full, scale, hk, d):
    if q.device.type == "npu":
        return npu_prefill_prepared(q, k_full, v_full, scale, hk, d)
    prefix = k_full.shape[0] - q.shape[0]
    return oscar_prefill_ref(
        q,
        k_full[prefix:],
        v_full[prefix:],
        k_full[:prefix],
        v_full[:prefix],
        scale,
        hk,
        d,
    )


def oscar_prefill(q, k, v, k_cached, v_cached, scale, hk, d):
    if q.device.type == "npu":
        return npu_prefill(q, k, v, k_cached, v_cached, scale, hk, d)
    return oscar_prefill_ref(q, k, v, k_cached, v_cached, scale, hk, d)
