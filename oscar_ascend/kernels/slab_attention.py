"""Bounded KV slabs shared by all queries, using native TND attention + LSE.

Historical KV is decoded once per slab (not once per query tile). The two
reusable floating KV buffers depend on a fixed budget and request count, never
sequence length. Current unquantized KV is a separate causal native call.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, pairwise

import torch

from ..format import K_IDX_OFF, check_d
from .decode_kernel import tl, triton
from .prefill import _causal_mask
from .streaming_attention import _read_tile, _validate_tensors, validate_npu_profile

DEFAULT_CHUNK_TOKENS = 8192


@dataclass(frozen=True)
class SlabPlan:
    capacity_tokens: int
    max_requests: int
    chunk_tokens: int
    scratch_bytes: int


def plan_slabs(
    requests,
    hk,
    d,
    element_size=2,
    workspace_bytes=32 * 1024 * 1024,
    chunk_tokens=DEFAULT_CHUNK_TOKENS,
):
    check_d(d)
    if requests < 0 or hk < 1 or element_size < 1 or chunk_tokens < 1:
        raise ValueError("Invalid slab geometry")
    per_token = 2 * hk * d * element_size
    limit = workspace_bytes // per_token
    if limit < 1:
        raise ValueError("Streaming workspace must hold at least one K/V token pair")
    capacity = min(requests * chunk_tokens, limit)
    # Limit metadata/native batch size as well as the floating KV workspace.
    max_requests = max(1, min(requests, 64, limit))
    chunk = min(chunk_tokens, max(1, capacity // max_requests))
    return SlabPlan(capacity, max_requests, chunk, capacity * per_token)


@dataclass(frozen=True)
class SlabRead:
    request: int
    start: int
    end: int
    q_start: int
    q_end: int


def slab_steps(qsl, seqs, plan, *, has_new):
    """Yield bounded host plans; no list grows with historical sequence length.

    Cache-only multi-query chunks need both a causal query slice and a later
    all-visible slice. Right-aligned causality for the former is exactly the native
    TND sparse-mode-3 convention. Queries before key zero remain empty.
    """
    active = [i for i, (a, b) in enumerate(pairwise(qsl)) if b > a]
    for batch_start in range(0, len(active), plan.max_requests):
        batch = active[batch_start : batch_start + plan.max_requests]
        lengths = [
            seqs[i] - (qsl[i + 1] - qsl[i]) if has_new else seqs[i] for i in batch
        ]
        for start in range(0, max(lengths, default=0), plan.chunk_tokens):
            plain, causal = [], []
            for req, length in zip(batch, lengths):
                end = min(start + plan.chunk_tokens, length)
                if start >= end:
                    continue
                a, b = qsl[req : req + 2]
                base = seqs[req] - (b - a)
                cut = max(a, min(b, a + end - base))
                masked_start = max(a, min(b, a + start - base))
                if masked_start < cut:
                    causal.append(SlabRead(req, start, end, masked_start, cut))
                if cut < b:
                    plain.append(SlabRead(req, start, end, cut, b))
            if plain:
                yield False, plain
            if causal:
                yield True, causal


if triton is not None:
    from .prepare_kv import _prepare_side

    @triton.jit
    def _slab_side(
        Cache,
        Meta,
        Table,
        Stage,
        Owner,
        Out,
        Entries,
        sbt,
        scb,
        scp,
        sch,
        smb,
        smp,
        smh,
        BS: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        INDEX: tl.constexpr,
        META: tl.constexpr,
        HAS_STAGE: tl.constexpr,
        ROWS: tl.constexpr,
        BT: tl.constexpr,
    ):
        entry = tl.program_id(2) * 4
        req = tl.load(Entries + entry)
        start = tl.load(Entries + entry + 1)
        length = tl.load(Entries + entry + 2)
        target = tl.load(Entries + entry + 3)
        _prepare_side(
            Cache,
            Meta,
            Table + req * sbt,
            Stage,
            Owner,
            Out + target * HK * D,
            length,
            start,
            scb,
            scp,
            sch,
            smb,
            smp,
            smh,
            BS,
            HK,
            D,
            BD,
            INDEX,
            META,
            HAS_STAGE,
            ROWS,
            BT,
        )

    @triton.jit
    def _merge_slab(
        Part,
        PartLse,
        Out,
        Lse,
        QueryIds,
        query_start,
        n,
        HQ: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        INDEXED: tl.constexpr,
        BT: tl.constexpr,
    ):
        row = tl.program_id(0) * BT + tl.arange(0, BT)
        local_q, head = row // HQ, row % HQ
        valid = local_q < n
        target_q = local_q + query_start
        if INDEXED:
            target_q = tl.load(QueryIds + local_q, valid, 0)
        target = target_q * HQ + head
        old_lse = tl.load(Lse + target, valid, -float("inf"))
        new_lse = tl.load(PartLse + row, valid, -float("inf"))
        maximum = tl.maximum(old_lse, new_lse)
        safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
        alpha = tl.exp(old_lse - safe_max)
        beta = tl.exp(new_lse - safe_max)
        denom = alpha + beta
        denom = tl.where(denom == 0.0, 1.0, denom)
        dims = tl.arange(0, BD)
        mask = valid[:, None] & (dims[None, :] < D)
        old = tl.load(Out + target[:, None] * D + dims[None, :], mask, 0.0)
        new = tl.load(Part + row[:, None] * D + dims[None, :], mask, 0.0).to(tl.float32)
        # Empty partials may contain unspecified native output; never use it.
        old = tl.where((old_lse == -float("inf"))[:, None], 0.0, old)
        new = tl.where((new_lse == -float("inf"))[:, None], 0.0, new)
        value = (old * alpha[:, None] + new * beta[:, None]) / denom[:, None]
        tl.store(Out + target[:, None] * D + dims[None, :], value, mask)
        tl.store(Lse + target, maximum + tl.log(denom), valid)


def dequant_slab(kc, vc, table, reads, k_buffer, v_buffer, stage=None):
    """Fill the same bounded buffers for this wave, including owner overrides."""
    counts = [r.end - r.start for r in reads]
    ends = list(accumulate(counts))
    if not ends or ends[-1] > k_buffer.shape[0]:
        raise ValueError("Slab read exceeds its fixed workspace")
    if kc.device.type == "cpu":
        target = 0
        for read, count in zip(reads, counts):
            k, v = _read_tile(
                kc,
                vc,
                table[read.request],
                read.start,
                read.end,
                read.end,
                None,
                None,
                stage,
                k_buffer.dtype,
            )
            k_buffer[target : target + count].copy_(k)
            v_buffer[target : target + count].copy_(v)
            target += count
    else:
        if triton is None:
            raise RuntimeError("Triton is required for bounded OSCAR slab decoding")
        entries = torch.tensor(
            [
                [r.request, r.start, count, end - count]
                for r, count, end in zip(reads, counts, ends)
            ],
            dtype=torch.int32,
            device=kc.device,
        )
        k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
        sk, sv, owner = (k_buffer, v_buffer, table) if stage is None else stage
        rows = 1 if stage is None else owner.shape[0]
        hk, d = kc.shape[2:]
        for source, staged, out, index, meta in (
            (k8, sk, k_buffer, K_IDX_OFF, 0),
            (v8, sv, v_buffer, 0, 4),
        ):
            _slab_side[(triton.cdiv(max(counts), 4), hk, len(reads))](
                source,
                k8,
                table,
                staged,
                owner,
                out,
                entries,
                table.stride(0),
                *source.stride()[:3],
                *k8.stride()[:3],
                BS=kc.shape[1],
                HK=hk,
                D=d,
                BD=triton.next_power_of_2(d),
                INDEX=index,
                META=meta,
                HAS_STAGE=stage is not None,
                ROWS=rows,
                BT=4,
                num_warps=1,
                num_stages=1,
            )
    return k_buffer[: ends[-1]], v_buffer[: ends[-1]], ends


def native_slab_attention(q, k, v, q_ends, kv_ends, scale, *, causal):
    """TND output and natural-log LSE, matching Ascend context-parallel calls."""
    if q.device.type == "cpu":
        # CPU oracle only: cap score tiles too; never a device performance path.
        output = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        qa = ka = 0
        for qb, kb in zip(q_ends, kv_ends):
            kh = (
                k[ka:kb]
                .float()
                .transpose(0, 1)
                .repeat_interleave(q.shape[1] // k.shape[1], 0)
            )
            vh = (
                v[ka:kb]
                .float()
                .transpose(0, 1)
                .repeat_interleave(q.shape[1] // k.shape[1], 0)
            )
            for a in range(qa, qb, 32):
                b = min(a + 32, qb)
                scores = q[a:b].float().transpose(0, 1) @ kh.transpose(-1, -2) * scale
                if causal:
                    ends = kb - ka - (qb - qa) + torch.arange(a - qa, b - qa) + 1
                    scores.masked_fill_(
                        torch.arange(kb - ka)[None, :] >= ends[:, None], -torch.inf
                    )
                logs = torch.logsumexp(scores, -1)
                probs = torch.exp(scores - logs.unsqueeze(-1))
                output[a:b] = (probs @ vh).transpose(0, 1)
                lse[a:b] = logs.transpose(0, 1)
            qa, ka = qb, kb
        return output, lse
    return npu_slab_attention(q, k, v, q_ends, kv_ends, scale, causal=causal)


def npu_slab_attention(q, k, v, q_ends, kv_ends, scale, *, causal):
    import torch_npu

    out, lse = torch_npu.npu_fused_infer_attention_score(
        query=q.contiguous(),
        key=k,
        value=v,
        input_layout="TND",
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=scale,
        actual_seq_lengths=q_ends,
        actual_seq_lengths_kv=kv_ends,
        atten_mask=_causal_mask(q.device) if causal else None,
        sparse_mode=3 if causal else 0,
        softmax_lse_flag=True,
        antiquant_mode=0,
        antiquant_scale=None,
    )
    expected = (*q.shape[:2], 1)
    if tuple(lse.shape) != expected:
        raise RuntimeError(
            f"Unexpected native TND LSE layout {tuple(lse.shape)}; expected {expected}"
        )
    return out.view_as(q), lse.squeeze(-1).contiguous()


def _pack_queries(q, ranges):
    ends = list(accumulate(b - a for a, b in ranges))
    if all(b == c for (_, b), (c, _) in pairwise(ranges)):
        start, end = ranges[0][0], ranges[-1][1]
        return q[start:end], ends, start, None
    ids = torch.tensor(
        [i for a, b in ranges for i in range(a, b)], dtype=torch.int32, device=q.device
    )
    return q[ids.long()], ends, 0, ids


def merge_slab(output, lse, part, part_lse, start, ids):
    if output.device.type == "cpu":
        target = slice(start, start + part.shape[0]) if ids is None else ids.long()
        old_lse = lse[target]
        maximum = torch.maximum(old_lse, part_lse)
        safe = torch.where(maximum == -torch.inf, 0.0, maximum)
        alpha, beta = (old_lse - safe).exp(), (part_lse - safe).exp()
        denom = alpha + beta
        denom = torch.where(denom == 0.0, 1.0, denom)
        old = torch.where((old_lse == -torch.inf)[..., None], 0.0, output[target])
        new = torch.where((part_lse == -torch.inf)[..., None], 0.0, part.float())
        output[target] = (old * alpha[..., None] + new * beta[..., None]) / denom[
            ..., None
        ]
        lse[target] = maximum + denom.log()
    else:
        n, hq, d = part.shape
        _merge_slab[(triton.cdiv(n * hq, 8),)](
            part.contiguous(),
            part_lse.contiguous(),
            output,
            lse,
            lse if ids is None else ids,
            start,
            n,
            HQ=hq,
            D=d,
            BD=triton.next_power_of_2(d),
            INDEXED=ids is not None,
            BT=8,
            num_warps=1,
            num_stages=1,
        )


def slab_attention(
    q,
    k_new,
    v_new,
    kc,
    vc,
    table,
    qsl,
    seqs,
    scale,
    stage=None,
    *,
    workspace_bytes=32 * 1024 * 1024,
    chunk_tokens=DEFAULT_CHUNK_TOKENS,
):
    validate_npu_profile(q.dtype, kc.shape[1], device_type=q.device.type)
    if q.device.type != "cpu" and triton is None:
        raise RuntimeError("Triton is required for bounded OSCAR slab attention")
    if (
        q.ndim != 3
        or kc.ndim != 4
        or q.shape[1] < 1
        or kc.shape[2] < 1
        or len(qsl) != len(seqs) + 1
        or not qsl
        or qsl[0] != 0
        or qsl[-1] != q.shape[0]
        or q.shape[1] % kc.shape[2]
    ):
        raise ValueError("Invalid slab query layout")
    if any(
        a > b or seq < 0 or (k_new is not None and seq < b - a)
        for a, b, seq in zip(qsl, qsl[1:], seqs)
    ):
        raise ValueError("Sequence length must cover its query chunk")
    _validate_tensors(q, k_new, v_new, kc, vc, table, qsl, seqs, stage)
    q = q.contiguous()
    table = table.to(device=q.device, dtype=torch.int32).contiguous()
    active = [i for i, (a, b) in enumerate(pairwise(qsl)) if b > a]
    plan = plan_slabs(
        len(active),
        kc.shape[2],
        q.shape[2],
        q.element_size(),
        workspace_bytes,
        chunk_tokens,
    )
    output = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if k_new is None:
        output.zero_()
        lse.fill_(-torch.inf)
    if not active:
        return output, lse
    if k_new is not None:
        # Current raw KV needs no historical workspace, even for a 9K prompt.
        # Bound native batch metadata; query ranges cover all nonempty requests.
        for first in range(0, len(active), plan.max_requests):
            batch = active[first : first + plan.max_requests]
            ranges = [(qsl[i], qsl[i + 1]) for i in batch]
            part_q, ends, start, ids = _pack_queries(q, ranges)
            part_k, part_v = (
                k_new[start : start + part_q.shape[0]],
                v_new[start : start + part_q.shape[0]],
            )
            if ids is not None:
                part_k, part_v = k_new[ids.long()], v_new[ids.long()]
            part, logs = native_slab_attention(
                part_q,
                part_k.contiguous(),
                part_v.contiguous(),
                ends,
                ends,
                scale,
                causal=True,
            )
            if ids is None:
                output[start : start + part.shape[0]].copy_(part)
                lse[start : start + part.shape[0]].copy_(logs)
            else:
                output[ids.long()], lse[ids.long()] = part.float(), logs
    k_buffer = v_buffer = None
    for causal, reads in slab_steps(qsl, seqs, plan, has_new=k_new is not None):
        if k_buffer is None:
            k_buffer = torch.empty(
                plan.capacity_tokens,
                kc.shape[2],
                q.shape[2],
                dtype=q.dtype,
                device=q.device,
            )
            v_buffer = torch.empty_like(k_buffer)
        k, v, kv_ends = dequant_slab(kc, vc, table, reads, k_buffer, v_buffer, stage)
        part_q, q_ends, start, ids = _pack_queries(
            q, [(r.q_start, r.q_end) for r in reads]
        )
        part, logs = native_slab_attention(
            part_q, k, v, q_ends, kv_ends, scale, causal=causal
        )
        merge_slab(output, lse, part, logs, start, ids)
    return output, lse
