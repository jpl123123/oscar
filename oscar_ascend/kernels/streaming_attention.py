"""Bounded-workspace attention over packed OSCAR pages.

The CPU implementation is an online-softmax oracle, not a fast NPU fallback.
The Triton implementation groups query/head rows and uses matrix products;
packed K/V exist only as small tiles inside the kernel. Floating KV scratch
depends on current query tokens and split count, not historical sequence length;
the integer page table remains a metadata input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..format import K_IDX_OFF, check_d
from .decode_kernel import tl, triton
from .store_kernel import dequant_split_ref

BM, BN, BK = 32, 32, 64


@dataclass(frozen=True)
class StreamPlan:
    tiles: tuple[tuple[int, int], ...]
    splits: int
    scratch_bytes: int
    query_tokens: int


def _validate_tensors(q, k, v, kc, vc, bt, qsl, seqs, stage):
    if q.ndim != 3 or kc.ndim != 4 or kc.shape != vc.shape or kc.shape[3] != q.shape[2]:
        raise ValueError(
            "Streaming attention requires [tokens,heads,D] queries and native 4-D caches"
        )
    if not kc.is_contiguous() or not vc.is_contiguous():
        raise ValueError("Streaming caches must be contiguous")
    if (k is None) != (v is None):
        raise ValueError("Both new K and V must be supplied together")
    if k is not None and (
        k.shape != v.shape
        or k.shape != (q.shape[0], kc.shape[2], q.shape[2])
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Current K/V must match query tokens, dtype and KV geometry")
    if bt.ndim != 2 or bt.shape[0] < len(seqs):
        raise ValueError("Missing request block-table rows")
    for a, b, seq in zip(qsl, qsl[1:], seqs):
        prefix = seq - (b - a) if k is not None else seq
        if b > a and math.ceil(prefix / kc.shape[1]) > bt.shape[1]:
            raise ValueError("Block table cannot cover the history")
    if stage is not None:
        sk, sv, owner = stage
        shape = (owner.shape[0], kc.shape[1], kc.shape[2], q.shape[2])
        if (
            owner.shape != shape[:2]
            or sk.shape != shape
            or sv.shape != shape
            or shape[0] < 1
        ):
            raise ValueError("Invalid streaming staging geometry")
        if not all(t.is_contiguous() for t in stage):
            raise ValueError("Streaming staging tensors must be contiguous")


def plan_stream(
    qsl, seqs, hq, hk, d, workspace_bytes=32 * 1024 * 1024, *, has_new=True
):
    check_d(d)
    if (
        d <= 0
        or hq <= 0
        or hk <= 0
        or hq % hk
        or len(qsl) != len(seqs) + 1
        or not qsl
        or qsl[0] != 0
    ):
        raise ValueError("Invalid streaming attention layout")
    tiles = []
    group = hq // hk
    for req, (a, b, seq) in enumerate(zip(qsl, qsl[1:], seqs)):
        if a > b or seq < 0 or (has_new and seq < b - a):
            raise ValueError("Sequence length must cover its query chunk")
        tiles.extend((req, row) for row in range(0, (b - a) * group, BM))
    if workspace_bytes < 0:
        raise ValueError("Streaming workspace budget must be nonnegative")
    # Only query work controls split parallelism, not growing history lengths.
    wanted = max(
        1, min(32, 2 ** math.ceil(math.log2(max(1, 128 / max(1, len(tiles) * hk)))))
    )
    per_split = qsl[-1] * hq * (d + 1) * 4
    splits = wanted
    while splits > 1 and per_split * splits > workspace_bytes:
        splits //= 2
    return StreamPlan(
        tuple(tiles), splits, per_split * splits if splits > 1 else 0, qsl[-1]
    )


def _read_tile(kc, vc, bt, start, end, prefix, k_new, v_new, stage, dtype):
    hk, d = kc.shape[2], kc.shape[3]
    # Cache tensors have D elements even when their byte views have 2*D.
    k = torch.empty(end - start, hk, d, device=kc.device, dtype=dtype)
    v = torch.empty_like(k)
    old_end = min(end, prefix)
    if start < old_end:
        pos = torch.arange(start, old_end, device=kc.device)
        bs = kc.shape[1]
        blocks = bt[pos // bs].long()
        oldk, oldv = dequant_split_ref(
            kc.view(torch.uint8), vc.view(torch.uint8), blocks, pos % bs, hk, d
        )
        if stage is not None:
            sk, sv, owner = stage
            row, off = blocks % owner.shape[0], pos % bs
            hit = (owner[row, off] == blocks).view(-1, 1, 1)
            oldk = torch.where(hit, sk[row, off], oldk)
            oldv = torch.where(hit, sv[row, off], oldv)
        # Match the native reconstruction path's intermediate rounding.
        count = old_end - start
        k[:count], v[:count] = oldk.half().to(dtype), oldv.half().to(dtype)
    fresh_start = max(start, prefix)
    if fresh_start < end:
        if k_new is None or v_new is None:
            raise ValueError("Missing current KV")
        k[fresh_start - start :] = k_new[fresh_start - prefix : end - prefix]
        v[fresh_start - start :] = v_new[fresh_start - prefix : end - prefix]
    return k, v


def streaming_attention_ref(
    q,
    k_new,
    v_new,
    kc,
    vc,
    bt,
    qsl,
    seqs,
    scale,
    stage=None,
    *,
    kv_tile=128,
    query_tile=32,
):
    """Exact online softmax in fp32 with bounded KV tiles; includes cache-only decode."""
    if (k_new is None) != (v_new is None) or kv_tile < 1 or query_tile < 1:
        raise ValueError("Invalid streaming inputs")
    n, hq, d = q.shape
    hk = kc.shape[2]
    plan = plan_stream(qsl, seqs, hq, hk, d, has_new=k_new is not None)
    _validate_tensors(q, k_new, v_new, kc, vc, bt, qsl, seqs, stage)
    if plan.query_tokens != n:
        raise ValueError("Query layout does not cover the input")
    output = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((n, hq), -torch.inf, dtype=torch.float32, device=q.device)
    for req, (a, b, seq) in enumerate(zip(qsl, qsl[1:], seqs)):
        prefix = seq - (b - a) if k_new is not None else seq
        newk = None if k_new is None else k_new[a:b]
        newv = None if v_new is None else v_new[a:b]
        for qa in range(a, b, query_tile):
            qb = min(qa + query_tile, b)
            qpart = q[qa:qb].transpose(0, 1).float()
            ends = seq - (b - a) + torch.arange(qa - a, qb - a, device=q.device) + 1
            maximum = torch.full(
                (hq, qb - qa), -torch.inf, dtype=torch.float32, device=q.device
            )
            denom = torch.zeros_like(maximum)
            acc = torch.zeros(hq, qb - qa, d, dtype=torch.float32, device=q.device)
            visible_end = seq - (b - a) + qb - a
            for start in range(0, visible_end, kv_tile):
                end = min(start + kv_tile, visible_end)
                k, v = _read_tile(
                    kc, vc, bt[req], start, end, prefix, newk, newv, stage, q.dtype
                )
                kh = k.transpose(0, 1).repeat_interleave(hq // hk, 0).float()
                vh = v.transpose(0, 1).repeat_interleave(hq // hk, 0).float()
                scores = (qpart @ kh.transpose(-1, -2)) * scale
                valid = (
                    torch.arange(start, end, device=q.device)[None, :] < ends[:, None]
                )
                scores = scores.masked_fill(~valid[None, :, :], -torch.inf)
                new_max = torch.maximum(maximum, scores.amax(-1))
                safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
                old_weight = torch.exp(maximum - safe_max)
                probs = torch.exp(scores - safe_max.unsqueeze(-1))
                acc = acc * old_weight.unsqueeze(-1) + probs @ vh
                denom = denom * old_weight + probs.sum(-1)
                maximum = new_max
            safe_denom = torch.where(denom > 0, denom, 1.0)
            output[qa:qb] = (acc / safe_denom.unsqueeze(-1)).transpose(0, 1)
            lse[qa:qb] = (maximum + torch.log(safe_denom)).transpose(0, 1)
    return output, lse


if triton is not None:

    @triton.jit
    def _kv_tile(
        Cache,
        Meta,
        New,
        Stage,
        Owner,
        blocks,
        pos,
        dims,
        prefix,
        qstart,
        head,
        valid,
        scb,
        scp,
        sch,
        smb,
        smp,
        smh,
        D: tl.constexpr,
        HK: tl.constexpr,
        BS: tl.constexpr,
        INDEX: tl.constexpr,
        META: tl.constexpr,
        HAS_STAGE: tl.constexpr,
        ROWS: tl.constexpr,
        HAS_NEW: tl.constexpr,
    ):
        old = valid & (pos < prefix)
        slot = blocks * scb + (pos % BS).to(tl.int64) * scp + head * sch
        mo = blocks * smb + (pos % BS).to(tl.int64) * smp + head * smh + META
        mask = old[:, None] & (dims[None, :] < D)
        packed = tl.load(
            Cache + slot[:, None] + INDEX + dims[None, :] // 4, mask, 0
        ).to(tl.int32)
        code = ((packed >> ((dims[None, :] % 4) * 2)) & 3).to(tl.float32)
        lo = tl.load(Meta + mo, old, 0).to(tl.uint16)
        hi = tl.load(Meta + mo + 1, old, 0).to(tl.uint16)
        scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        lo = tl.load(Meta + mo + 2, old, 0).to(tl.uint16)
        hi = tl.load(Meta + mo + 3, old, 0).to(tl.uint16)
        zero = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        value = code * scale[:, None] + zero[:, None]
        if HAS_STAGE:
            seat = (blocks % ROWS) * BS + pos % BS
            tag = tl.load(Owner + seat, old, -1)
            hit = old & (tag == blocks)
            staged = tl.load(
                Stage + (seat[:, None] * HK + head) * D + dims[None, :],
                hit[:, None] & (dims[None, :] < D),
                0.0,
            )
            value = tl.where(hit[:, None], staged, value)
        value = value.to(tl.float16).to(tl.float32)
        if HAS_NEW:
            fresh = valid & (pos >= prefix)
            fresh_value = tl.load(
                New
                + ((qstart + pos[:, None] - prefix) * HK + head) * D
                + dims[None, :],
                fresh[:, None] & (dims[None, :] < D),
                0.0,
            ).to(tl.float32)
            value = tl.where(fresh[:, None], fresh_value, value)
        return value

    @triton.jit
    def _stream_stage1(
        Q,
        KN,
        VN,
        KC,
        VC,
        BTABLE,
        SK,
        SV,
        OWNER,
        TILE_REQ,
        TILE_ROW,
        QSTART,
        SEQS,
        MID,
        OUT,
        LSE,
        sbt,
        skb,
        skp,
        skh,
        svb,
        svp,
        svh,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        GROUP: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_STAGE: tl.constexpr,
        ROWS: tl.constexpr,
        HAS_NEW: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SCALE: tl.constexpr,
        K_OFFSET: tl.constexpr,
    ):
        tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        req = tl.load(TILE_REQ + tile)
        row_start = tl.load(TILE_ROW + tile)
        qs = tl.load(QSTART + req)
        qe = tl.load(QSTART + req + 1)
        seq = tl.load(SEQS + req)
        rows = row_start + tl.arange(0, M)
        query = qs + rows // GROUP
        qhead = head * GROUP + rows % GROUP
        qvalid = rows < (qe - qs) * GROUP
        query_end = seq - (qe - qs) + rows // GROUP + 1
        prefix = seq
        if HAS_NEW:
            prefix = seq - (qe - qs)
        split_size = tl.cdiv(tl.cdiv(seq, SPLITS), N) * N
        first = split * split_size
        last = tl.minimum(first + split_size, seq)
        last = tl.minimum(
            last,
            seq - (qe - qs) + tl.minimum((row_start + M - 1) // GROUP + 1, qe - qs),
        )
        dim = tl.arange(0, K)
        maximum = tl.full([M], -float("inf"), tl.float32)
        denom = tl.zeros([M], tl.float32)
        acc0 = tl.zeros([M, K], tl.float32)
        acc1 = tl.zeros([M, K], tl.float32)
        acc2 = tl.zeros([M, K], tl.float32)
        acc3 = tl.zeros([M, K], tl.float32)
        for start in range(first, last, N):
            pos = start + tl.arange(0, N)
            valid = pos < last
            blocks = tl.load(
                BTABLE + req * sbt + pos // BS, valid & (pos < prefix), 0
            ).to(tl.int64)
            scores = tl.zeros([M, N], tl.float32)
            for ds in range(tl.cdiv(D, K)):
                dims = ds * K + dim
                qt = tl.load(
                    Q + (query[:, None] * HQ + qhead[:, None]) * D + dims[None, :],
                    qvalid[:, None] & (dims[None, :] < D),
                    0.0,
                )
                kt = _kv_tile(
                    KC,
                    KC,
                    KN,
                    SK,
                    OWNER,
                    blocks,
                    pos,
                    dims,
                    prefix,
                    qs,
                    head,
                    valid,
                    skb,
                    skp,
                    skh,
                    skb,
                    skp,
                    skh,
                    D,
                    HK,
                    BS,
                    K_OFFSET,
                    0,
                    HAS_STAGE,
                    ROWS,
                    HAS_NEW,
                )
                scores += tl.dot(qt, tl.trans(kt.to(Q.dtype.element_ty)))
            scores *= SCALE
            visible = (
                qvalid[:, None] & valid[None, :] & (pos[None, :] < query_end[:, None])
            )
            scores = tl.where(visible, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max > -float("inf"), new_max, 0.0)
            alpha = tl.exp(maximum - safe_max)
            probs = tl.exp(scores - safe_max[:, None])
            denom = denom * alpha + tl.sum(probs, 1)
            maximum = new_max
            p = probs.to(Q.dtype.element_ty)
            vt = _kv_tile(
                VC,
                KC,
                VN,
                SV,
                OWNER,
                blocks,
                pos,
                dim,
                prefix,
                qs,
                head,
                valid,
                svb,
                svp,
                svh,
                skb,
                skp,
                skh,
                D,
                HK,
                BS,
                0,
                4,
                HAS_STAGE,
                ROWS,
                HAS_NEW,
            )
            acc0 = acc0 * alpha[:, None] + tl.dot(p, vt.to(Q.dtype.element_ty))
            if D > K:
                vt = _kv_tile(
                    VC,
                    KC,
                    VN,
                    SV,
                    OWNER,
                    blocks,
                    pos,
                    dim + K,
                    prefix,
                    qs,
                    head,
                    valid,
                    svb,
                    svp,
                    svh,
                    skb,
                    skp,
                    skh,
                    D,
                    HK,
                    BS,
                    0,
                    4,
                    HAS_STAGE,
                    ROWS,
                    HAS_NEW,
                )
                acc1 = acc1 * alpha[:, None] + tl.dot(p, vt.to(Q.dtype.element_ty))
            if D > 2 * K:
                vt = _kv_tile(
                    VC,
                    KC,
                    VN,
                    SV,
                    OWNER,
                    blocks,
                    pos,
                    dim + 2 * K,
                    prefix,
                    qs,
                    head,
                    valid,
                    svb,
                    svp,
                    svh,
                    skb,
                    skp,
                    skh,
                    D,
                    HK,
                    BS,
                    0,
                    4,
                    HAS_STAGE,
                    ROWS,
                    HAS_NEW,
                )
                acc2 = acc2 * alpha[:, None] + tl.dot(p, vt.to(Q.dtype.element_ty))
            if D > 3 * K:
                vt = _kv_tile(
                    VC,
                    KC,
                    VN,
                    SV,
                    OWNER,
                    blocks,
                    pos,
                    dim + 3 * K,
                    prefix,
                    qs,
                    head,
                    valid,
                    svb,
                    svp,
                    svh,
                    skb,
                    skp,
                    skh,
                    D,
                    HK,
                    BS,
                    0,
                    4,
                    HAS_STAGE,
                    ROWS,
                    HAS_NEW,
                )
                acc3 = acc3 * alpha[:, None] + tl.dot(p, vt.to(Q.dtype.element_ty))
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        logs = maximum + tl.log(safe_denom)
        base = (query * HQ + qhead) * D
        if SPLITS > 1:
            base = ((query * HQ + qhead) * SPLITS + split) * (D + 1)
            tl.store(MID + base + D, logs, qvalid)
        else:
            tl.store(LSE + query * HQ + qhead, logs, qvalid)
        target = OUT
        if SPLITS > 1:
            target = MID
        tl.store(
            target + base[:, None] + dim[None, :],
            acc0 / safe_denom[:, None],
            qvalid[:, None] & (dim[None, :] < D),
        )
        if D > K:
            tl.store(
                target + base[:, None] + K + dim[None, :],
                acc1 / safe_denom[:, None],
                qvalid[:, None] & (K + dim[None, :] < D),
            )
        if D > 2 * K:
            tl.store(
                target + base[:, None] + 2 * K + dim[None, :],
                acc2 / safe_denom[:, None],
                qvalid[:, None] & (2 * K + dim[None, :] < D),
            )
        if D > 3 * K:
            tl.store(
                target + base[:, None] + 3 * K + dim[None, :],
                acc3 / safe_denom[:, None],
                qvalid[:, None] & (3 * K + dim[None, :] < D),
            )

    @triton.jit
    def _stream_merge(
        MID,
        OUT,
        LSE,
        HQ: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        query, head = tl.program_id(0), tl.program_id(1)
        dim = tl.arange(0, BD)
        maximum = -float("inf")
        denom = 0.0
        acc = tl.zeros([BD], tl.float32)
        for split in range(SPLITS):
            base = ((query * HQ + head) * SPLITS + split) * (D + 1)
            log = tl.load(MID + base + D)
            if log != -float("inf"):
                value = tl.load(MID + base + dim, dim < D, 0.0)
                new_max = tl.maximum(maximum, log)
                alpha, beta = tl.exp(maximum - new_max), tl.exp(log - new_max)
                acc = acc * alpha + value * beta
                denom = denom * alpha + beta
                maximum = new_max
        safe = tl.where(denom > 0.0, denom, 1.0)
        tl.store(OUT + (query * HQ + head) * D + dim, acc / safe, dim < D)
        tl.store(LSE + query * HQ + head, maximum + tl.log(safe))


def validate_npu_profile(dtype, block_size, *, device_type="npu"):
    """Packed-kernel probes passed on NPU for the deployed bf16/128 profile.

    FP16 with 1536-token pages failed static buffer planning on the deployed
    compiler; the two axes are not yet isolated. Do not silently claim support
    for either unverified configuration or downcast/reformat the model.
    """
    if device_type == "npu" and (dtype != torch.bfloat16 or block_size != 128):
        raise ValueError(
            "Streaming NPU deployment currently requires bf16 queries and "
            f"128-token cache blocks; got dtype={dtype}, block_size={block_size}. "
            "Other profiles require the standalone compatibility probe."
        )


def streaming_attention_triton(
    q,
    k_new,
    v_new,
    kc,
    vc,
    bt,
    qsl,
    seqs,
    scale,
    stage=None,
    *,
    workspace_bytes=32 * 1024 * 1024,
    experimental_profile=False,
):
    if triton is None:
        raise RuntimeError("Triton is required for streaming NPU attention")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Streaming matrix attention requires bf16/fp16 queries")
    if not experimental_profile:
        validate_npu_profile(q.dtype, kc.shape[1], device_type=q.device.type)
    if (k_new is None) != (v_new is None):
        raise ValueError("Both new K and V must be supplied together")
    n, hq, d = q.shape
    hk = kc.shape[2]
    plan = plan_stream(qsl, seqs, hq, hk, d, workspace_bytes, has_new=k_new is not None)
    _validate_tensors(q, k_new, v_new, kc, vc, bt, qsl, seqs, stage)
    if plan.query_tokens != n:
        raise ValueError("Query layout does not cover the input")
    q = q.contiguous()
    output = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(n, hq, dtype=torch.float32, device=q.device)
    if n == 0:
        return output, lse
    tile_req = torch.tensor(
        [t[0] for t in plan.tiles], dtype=torch.int32, device=q.device
    )
    tile_row = torch.tensor(
        [t[1] for t in plan.tiles], dtype=torch.int32, device=q.device
    )
    starts = torch.tensor(qsl, dtype=torch.int32, device=q.device)
    seq = torch.tensor(seqs, dtype=torch.int32, device=q.device)
    table = bt.to(device=q.device, dtype=torch.int32).contiguous()
    mid = (
        torch.empty(n, hq, plan.splits, d + 1, dtype=torch.float32, device=q.device)
        if plan.splits > 1
        else output
    )
    sk, sv, owner = (q, q, starts) if stage is None else stage
    rows = 1 if stage is None else owner.shape[0]
    kn, vn = (q, q) if k_new is None else (k_new.contiguous(), v_new.contiguous())
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    _stream_stage1[(len(plan.tiles), hk, plan.splits)](
        q,
        kn,
        vn,
        k8,
        v8,
        table,
        sk,
        sv,
        owner,
        tile_req,
        tile_row,
        starts,
        seq,
        mid,
        output,
        lse,
        table.stride(0),
        k8.stride(0),
        k8.stride(1),
        k8.stride(2),
        v8.stride(0),
        v8.stride(1),
        v8.stride(2),
        HQ=hq,
        HK=hk,
        D=d,
        BS=kc.shape[1],
        GROUP=hq // hk,
        SPLITS=plan.splits,
        HAS_STAGE=stage is not None,
        ROWS=rows,
        HAS_NEW=k_new is not None,
        M=BM,
        N=BN,
        K=BK,
        SCALE=scale,
        K_OFFSET=K_IDX_OFF,
        num_warps=4,
        num_stages=1,
    )
    if plan.splits > 1:
        _stream_merge[(n, hq)](
            mid,
            output,
            lse,
            HQ=hq,
            D=d,
            BD=triton.next_power_of_2(d),
            SPLITS=plan.splits,
            num_warps=4,
            num_stages=1,
        )
    return output, lse
