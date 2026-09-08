"""oscar_ascend.backend — AscendOSCAR 注意力实现（继承原生 impl，类外科手术后生效）。

零侵入接入点（见 plugin.py / plan §5.5）：插件把 FULL 层 impl 的 __class__ 换成
`AscendOscarAttentionBackendImpl`；本类只重写 forward / do_kv_cache_update / 存储与
读取路径，**不动** backend 类、metadata builder、allocator、页表、GDN 路径。
"""
from __future__ import annotations

import torch

from .config import OscarAscendConfig
from .format import K_IDX_OFF, VALUES_PER_BYTE, check_d
from .kernels.store_kernel import oscar_store_ref
from .kernels.decode_kernel import oscar_decode_ref, oscar_prefill_ref
from .kernels.dequant_kernel import oscar_full_dequant, triton as _k_triton
from .rotation import get_layer_rotation


def metadata_batch_lists(attn_metadata) -> tuple[list, list]:
    """提取 (query_start_loc, seq_lens) 的 host 列表——**禁止 Tensor 作布尔值**。

    真机 2026-09-04 05:21 教训：`(seq_lens_cpu or seq_lens)` 在 seq_lens_cpu 为
    **多元素 Tensor** 时抛 `Boolean value of Tensor with more than one value is
    ambiguous`（MTP 目标层 decode 走 __prefill_attention__ 时元数据为 chunked-prefill 形）。
    兼容：Tensor(任意设备，tolist 自动同步)、None 回退、_seq_lens_cpu、裸 list。
    """

    def _as_list(x, name: str) -> list:
        if x is None:
            raise RuntimeError(f"attn_metadata.{name} 缺失")
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    q_sl_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    ends = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if q_sl_cpu is not None:
        qsl = _as_list(q_sl_cpu, "query_start_loc_cpu")
    elif ends is not None:
        qsl = [0] + _as_list(ends, "actual_seq_lengths_q")
    else:
        qsl = _as_list(attn_metadata.query_start_loc, "query_start_loc")
    seq_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if seq_cpu is None:
        seq_cpu = getattr(attn_metadata, "_seq_lens_cpu", None)
    seq_list = getattr(attn_metadata, "seq_lens_list", None)
    seqs = _as_list(seq_list if seq_list is not None else (
        seq_cpu if seq_cpu is not None else attn_metadata.seq_lens), "seq_lens")
    return qsl, seqs

try:
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl,
        AscendAttentionState,
    )
except Exception:  # pragma: no cover — 平台缺失时仅作占位，plugin 不会启用
    AscendAttentionBackendImpl = object  # type: ignore
    AscendAttentionState = object  # type: ignore


class AscendOscarAttentionBackendImpl(AscendAttentionBackendImpl):  # type: ignore[misc]
    """OSCAR INT2 FULL 层注意力实现（参考 OSCAR PR oscar_attn.py 的 impl 部分）。"""

    # ------------------------------------------------------------------ setup
    def _oscar_setup(self) -> None:
        if getattr(self, "_oscar_cfg", None) is not None:
            return
        cfg = OscarAscendConfig.from_env(head_dim=self.head_size)
        check_d(cfg.head_dim)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("OSCAR requires Hq divisible by Hk")
        self._oscar_cfg = cfg
        self._oscar_rot_ready = False
        self._oscar_stage_ready = False
        self._oscar_warned_quantile = False
        self._oscar_use_triton = (
            self._oscar_cfg.use_triton and _k_triton is not None
        )
        # ★ 自证点 2：配置生效摘要（每层首次 setup 打一次）
        print(
            f"[oscar-ascend] ★ OSCAR 配置生效: D={self.head_size}, 逻辑槽=160B "
            f"(K 96B+V 64B), K旋转={'已加载' if self._oscar_cfg.k_rotation_path else '单位阵(未加载)'}, "
            f"V旋转={'已加载' if self._oscar_cfg.v_rotation_path else '单位阵(未加载)'}, "
            f"路径={self._oscar_cfg.k_rotation_path or '-'}, "
            f"triton={'启用' if self._oscar_use_triton else 'torch参考路径'}, "
            f"窗口(sink={self._oscar_cfg.sink_tokens}, recent={self._oscar_cfg.recent_tokens})"
        )
        self._oscar_stats = {"writes": 0, "kv_bytes_written": 0, "reads": 0}

    @property
    def _oscar(self) -> OscarAscendConfig:
        return self._oscar_cfg

    # ------------------------------------------------------------------ 旋转/裁剪
    def _rotate_clip(self, x: torch.Tensor, R: torch.Tensor, clip_ratio: float) -> torch.Tensor:
        x_rot = torch.matmul(x.float(), R)
        if clip_ratio > 0.0:
            if not self._oscar_warned_quantile:
                self._oscar_warned_quantile = True
                print(
                    "[oscar-ascend] 裁剪实现 = 排序分位数（sort-based，对齐论文内核"
                    " test_oscar_rotation_clip_int2.py；不依赖 torch.quantile）"
                )
            try:
                # 论文内核同款：idx = int(ratio*D)，阈值 = 第 idx 大 |x|（per-vector）
                D = x_rot.shape[-1]
                idx = min(int(clip_ratio * D), D - 1)
                sorted_abs, _ = x_rot.abs().sort(dim=-1)
                thr = sorted_abs[..., idx : idx + 1]
                x_rot = torch.clamp(x_rot, -thr, thr)
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] sort 裁剪不可用，跳过: {e}")
        return x_rot

    # ------------------------------------------------------------------ 写路径
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        slot_mapping: torch.Tensor,
    ) -> None:
        slot_mapping = slot_mapping.to(device=key.device, dtype=torch.int64) if key is not None else slot_mapping
        N = slot_mapping.shape[0]
        if N <= 0 or key is None or value is None:
            return
        D = self.head_size
        Hk = self.num_kv_heads
        self._set_caches(kv_cache)
        k_cache, v_cache = self.key_cache, self.value_cache
        k = key[:N].view(N, Hk, D)
        v = value[:N].view(N, Hk, D)
        rk, rv = self._layer_rots(layer, k.device)
        k_rot = self._rotate_clip(k, rk, self._oscar.k_clip_ratio)
        v_rot = self._rotate_clip(v, rv, self._oscar.v_clip_ratio)
        if not getattr(layer, "_oscar_wrote_once", False):
            layer._oscar_wrote_once = True
            # ★ 自证点 3：INT2 写路径真实执行（每层首写一次日志 + 字节统计）
            print(
                f"[oscar-ascend] ★ INT2 写路径首次执行: {layer.layer_name} "
                f"tokens={N} heads={Hk} — 每 token·head 逻辑槽 {160}B (原生KV {4 * D}B, "
                f"写入开销 -{(1 - 160 / (4 * D)) * 100:.1f}%)"
            )
        self._oscar_stats["writes"] += 1
        self._oscar_stats["kv_bytes_written"] += N * Hk * 160
        if self._oscar_use_triton:
            try:
                from .kernels.store_kernel import oscar_store_triton

                oscar_store_triton(k_rot, v_rot, k_cache, v_cache, slot_mapping)
                return
            except Exception as e:  # pragma: no cover — triton 编译/执行失败则回退
                print(f"[oscar-ascend] triton store 失败，回退 torch 参考路径: {e}")
        oscar_store_ref(k_rot, v_rot, k_cache, v_cache, slot_mapping)

    def _set_caches(self, kv_cache) -> None:
        if isinstance(kv_cache, (tuple, list)) and len(kv_cache) >= 2:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        elif isinstance(kv_cache, torch.Tensor) and kv_cache.dim() > 0:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        assert self.key_cache is not None and self.value_cache is not None
        # 几何对账（DESIGN-20260904-E，每 impl 一次）：槽宽只允许
        #   head_size（packed×2：int8 几何，512B/token·head，block_size=1536）或
        #   2×head_size（legacy：bf16 几何，1024B/token·head，block_size=768）；
        # 且槽内落位（K 96B / V 64B）不得越界。fork 漂移/参数漏传在此拦截。
        if not getattr(self, "_oscar_geo_ok", False):
            k8 = self.key_cache.view(torch.uint8)
            v8 = self.value_cache.view(torch.uint8)
            if k8.ndim != 4 or v8.ndim != 4 or not k8.is_contiguous() or not v8.is_contiguous():
                raise ValueError("OSCAR requires contiguous [blocks, tokens, heads, bytes] caches")
            slot_k, slot_v = int(k8.stride(2)), int(v8.stride(2))
            need_k = K_IDX_OFF + self.head_size // VALUES_PER_BYTE  # 32+64=96
            need_v = self.head_size // VALUES_PER_BYTE  # 64
            assert slot_k in (self.head_size, 2 * self.head_size), (
                f"[oscar-ascend] 几何对账失败: k 槽宽 {slot_k}B（预期 "
                f"{self.head_size}=packed×2 或 {2 * self.head_size}=legacy）"
            )
            assert need_k <= slot_k and need_v <= slot_v, (
                f"[oscar-ascend] 槽容量不足: K 需 {need_k}B/V 需 {need_v}B "
                f"vs 槽 {slot_k}/{slot_v}B（布局契约 N-01 越界）"
            )
            self._oscar_geo_ok = True
            mode = (
                "packed×2（int8 几何：512B/token·head，block_size=1536，FULL 密度×2）"
                if slot_k == self.head_size
                else "legacy（bf16 几何：1024B/token·head，block_size=768，无显存收益）"
            )
            print(
                f"[oscar-ascend] ★ 几何对账: K 槽 {slot_k}B / V 槽 {slot_v}B → {mode}；"
                f"槽内落位 K{need_k}B+V{need_v}B 无越界"
            )

    def _layer_rots(self, layer: torch.nn.Module, device: torch.device):
        if not getattr(layer, "_oscar_rots", None):
            cfg = self._oscar
            rk = get_layer_rotation(cfg.k_rotation_path, layer.layer_name, self.head_size, device, mode="k", strict=True)
            rv = get_layer_rotation(cfg.v_rotation_path, layer.layer_name, self.head_size, device, mode="v", strict=True)
            layer._oscar_rots = (rk, rv)
            layer._oscar_rkT = rk.t().contiguous()
            layer._oscar_rvT = rv.t().contiguous()
        return layer._oscar_rots

    # ------------------------------------------------------------------ 主入口
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("OSCAR fused output quantization is unsupported")
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._set_caches(kv_cache)
        state = getattr(attn_metadata, "attn_state", AscendAttentionState.ChunkedPrefill)

        # 1) 写路径：本步新 token K/V → INT2（decode/prefill 均写；前缀命中时旧前缀已在缓存）
        if key is not None and value is not None:
            self.do_kv_cache_update(
                layer, key, value, kv_cache,
                attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens],
            )
            if self._oscar.window_enabled:
                self._ensure_staging(layer, kv_cache)
                self._staging_write(layer, key, value, attn_metadata)

        # MTP and mixed batches: retain each request's actual q_len. The fused
        # kernel supports causal multi-query attention, unlike DecodeOnly.
        if self._oscar.use_paged and self._oscar_use_triton and key is not None and value is not None:
            qsl, seqs = metadata_batch_lists(attn_metadata)
            actual = attn_metadata.num_actual_tokens
            pairs = [(i, a, min(b, actual), seqs[i]) for i, (a, b) in enumerate(zip(qsl, qsl[1:])) if a < actual]
            short = [(i, a, b, seq) for i, a, b, seq in pairs if 0 < b - a <= 16]
            if short:
                from .kernels.paged_attention import oscar_paged_attention_triton
                rk, rv = self._layer_rots(layer, query.device)
                stage = None
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    stage = (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
                token_ids = torch.tensor([t for _, a, b, _ in short for t in range(a, b)], device=query.device)
                request_ids = torch.tensor([i for i, _, _, _ in short], device=attn_metadata.block_tables.device)
                starts = [0]
                for _, a, b, _ in short:
                    starts.append(starts[-1] + b - a)
                # Pack only short-query requests. A simultaneous long prefill
                # must not force the entire MTP/decode batch onto the dense path.
                out_short = oscar_paged_attention_triton(
                    query[token_ids].float() @ rk, key[token_ids].float() @ rk, value[token_ids].float() @ rv,
                    self.key_cache, self.value_cache, attn_metadata.block_tables[request_ids],
                    starts, [seq for _, _, _, seq in short], self.scale, stage,
                ) @ rv.t()
                large = [i for i, a, b, _ in pairs if b - a > 16]
                if large:
                    attn_out = self._prefill_attention(query, key, value, kv_cache, attn_metadata, layer, request_indices=large)
                else:
                    attn_out = torch.zeros_like(query)
                attn_out[token_ids] = out_short.to(attn_out.dtype)
                output.copy_(attn_out.reshape(output.shape).to(output.dtype))
                return output

        # 2) 读路径
        if state == getattr(AscendAttentionState, "DecodeOnly", None):
            attn_out = self._decode_attention(query, kv_cache, attn_metadata, layer)
        else:
            attn_out = self._prefill_attention(query, key, value, kv_cache, attn_metadata, layer)

        if output.ndim == 3:
            output[:num_tokens] = attn_out[:num_tokens].to(output.dtype)
        else:
            output[:num_tokens] = attn_out.reshape(num_tokens, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ decode
    def _decode_attention(self, query, kv_cache, attn_metadata, layer) -> torch.Tensor:
        if not getattr(layer, "_oscar_read_once", False):
            layer._oscar_read_once = True
            self._oscar_stats["reads"] += 1
            print(
                f"[oscar-ascend] ★ INT2 读路径(decode) 首次执行: {layer.layer_name} "
                f"seq_len={int(attn_metadata.seq_lens.max()) if attn_metadata.seq_lens.numel() else 0}, "
                f"窗口={'开' if self._oscar.window_enabled and self._oscar_stage_ready else '关(纯INT2)'}"
            )
        if self._oscar.window_enabled and self._oscar_stage_ready:
            return self._decode_attention_windowed(query, kv_cache, attn_metadata, layer)
        q = query.float()
        rk, _ = self._layer_rots(layer, q.device)
        q_rot = torch.matmul(q, rk)
        bt = attn_metadata.block_tables
        seq = attn_metadata.seq_lens.to(device=q.device, dtype=torch.int32)
        if self._oscar_use_triton:
            try:
                from .kernels.decode_kernel import oscar_decode_triton

                out_rot, _ = oscar_decode_triton(
                    q_rot, self.key_cache, self.value_cache, bt, seq,
                    self.scale, self.num_kv_heads, self.head_size,
                )
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] triton decode 失败，回退 torch: {e}")
                out_rot, _ = oscar_decode_ref(
                    q_rot, self.key_cache, self.value_cache, bt, seq,
                    self.scale, self.num_kv_heads, self.head_size,
                )
        else:
            out_rot, _ = oscar_decode_ref(
                q_rot, self.key_cache, self.value_cache, bt, seq,
                self.scale, self.num_kv_heads, self.head_size,
            )
        _, rv = self._layer_rots(layer, q.device)
        return torch.matmul(out_rot, rv.t().contiguous())

    # ------------------------------------------------------------------ prefill
    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer, request_indices=None) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = self.num_kv_heads
        qsl_list, seq_lens_list = metadata_batch_lists(attn_metadata)
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        num_reqs = len(qsl_list) - 1
        for i in (range(num_reqs) if request_indices is None else request_indices):
            q_start, q_end = qsl_list[i], qsl_list[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            cached_len = seq_len - q_len
            if cached_len <= 0:
                out = oscar_prefill_ref(
                    q_seq, k_seq, v_seq,
                    torch.zeros(0, Hk, D, device=query.device),
                    torch.zeros(0, Hk, D, device=query.device),
                    self.scale, Hk, D,
                )
            else:
                bt_row = attn_metadata.block_tables[i]
                k_cached, v_cached = oscar_full_dequant(
                    self.key_cache, self.value_cache, bt_row, cached_len, Hk, D,
                    use_triton=self._oscar_use_triton,
                )
                rk, rv = self._layer_rots(layer, query.device)
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    k_cached, v_cached = self._stage_splice(
                        layer, bt_row, cached_len, k_cached, v_cached
                    )
                out = oscar_prefill_ref(
                    (q_seq.float() @ rk).to(query.dtype), (k_seq.float() @ rk).to(query.dtype),
                    (v_seq.float() @ rv).to(query.dtype), k_cached.to(query.dtype), v_cached.to(query.dtype),
                    self.scale, Hk, D,
                )
                out = out.float() @ rv.t()
            output[q_start:q_end] = out.to(query.dtype)
        return output

    # ------------------------------------------------------------------ 窗口（BF16 sink/recent staging，port PR oscar_attn.py:245-336/618-750）
    def _ensure_staging(self, layer: torch.nn.Module, kv_cache) -> None:
        if getattr(layer, "_oscar_stage_ready", False):
            return
        bs = kv_cache[0].shape[1]
        cfg = self._oscar
        self.stage_block = bs
        self.sink_eff = (cfg.sink_tokens // bs) * bs
        self.sink_pages = self.sink_eff // bs
        self.tail_pages = (cfg.recent_tokens + bs - 1) // bs + 1
        rows = max(
            (cfg.staging_tokens + bs - 1) // bs,
            self.sink_pages + self.tail_pages + 2,
        )
        dev = kv_cache[0].device
        layer._oscar_stage_k = torch.zeros(rows, bs, self.num_kv_heads, self.head_size, dtype=torch.float32, device=dev)
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full((rows, bs), -1, dtype=torch.int64, device=dev)
        layer._oscar_stage_rows = rows
        layer._oscar_stage_ready = True
        self._oscar_stage_ready = True

    def _staging_write(self, layer, key, value, attn_metadata) -> None:
        N = attn_metadata.num_actual_tokens
        if N == 0:
            return
        dev = key.device
        slot = attn_metadata.slot_mapping[:N].to(device=dev, dtype=torch.int64)
        # Ascend already provides cumulative query ends on the host. Clip out
        # dummy padding requests before repeat_interleave, whose size is fixed.
        qsl, seqs = metadata_batch_lists(attn_metadata)
        lengths = [max(0, min(end, N) - min(start, N)) for start, end in zip(qsl, qsl[1:])]
        req = torch.repeat_interleave(
            torch.arange(len(lengths), device=dev),
            torch.tensor(lengths, device=dev), output_size=N,
        )
        seq = torch.tensor(seqs[:len(lengths)], device=dev)
        ends = torch.tensor(qsl[1:], device=dev)
        pos = seq[req] - ends[req] + torch.arange(N, device=dev)
        keep = (pos >= seq[req] - self._oscar.recent_tokens) | (pos < self.sink_eff)
        bs = self.stage_block
        valid = torch.nonzero(slot >= 0, as_tuple=True)[0]
        seats = (slot[valid] // bs % layer._oscar_stage_rows) * bs + slot[valid] % bs
        # Hash collisions: choose one deterministic writer per seat. Owners and
        # values must come from the SAME token, including invalidating non-window
        # writes after a physical block is recycled or a draft token is rejected.
        order = torch.argsort(seats, stable=True)
        sorted_seats = seats[order]
        last = torch.cat([sorted_seats[1:] != sorted_seats[:-1],
                          torch.ones(min(1, sorted_seats.numel()), device=dev, dtype=torch.bool)])
        selected = valid[order[last]]
        seats = sorted_seats[last]
        rows, offsets = seats // bs, seats % bs
        owner = torch.where(keep[selected], slot[selected] // bs, -1)
        layer._oscar_slot_owner[rows, offsets] = owner
        retained = keep[selected]
        selected, rows, offsets = selected[retained], rows[retained], offsets[retained]
        if selected.numel() == 0:
            return
        rk, rv = self._layer_rots(layer, dev)
        # Preserve the model's unquantized K/V in fp32 rotated space. No clipping
        # or INT2 rounding is applied to staging, and historical tokens never
        # need to be rotated again on the read path.
        layer._oscar_stage_k[rows, offsets] = key[:N].reshape(N, self.num_kv_heads, self.head_size)[selected].float() @ rk
        layer._oscar_stage_v[rows, offsets] = value[:N].reshape(N, self.num_kv_heads, self.head_size)[selected].float() @ rv

    def _stage_splice(self, layer, bt_row, cached_len, k_cached, v_cached):
        bs = self.stage_block
        rows_total = layer._oscar_stage_rows
        dev = k_cached.device
        npg = (cached_len + bs - 1) // bs
        blk = bt_row[:npg].to(torch.int64)
        rows = blk % rows_total
        staged = layer._oscar_slot_owner[rows] == blk.unsqueeze(-1)
        pos = (torch.arange(npg, device=dev) * bs).unsqueeze(-1) + torch.arange(bs, device=dev)
        staged = (staged & (pos < cached_len)).reshape(-1)[:cached_len]
        ks = layer._oscar_stage_k[rows].reshape(npg * bs, self.num_kv_heads, -1)
        vs = layer._oscar_stage_v[rows].reshape(npg * bs, self.num_kv_heads, -1)
        m = staged.view(-1, 1, 1)
        k_out = torch.where(m, ks[:cached_len].to(k_cached.dtype), k_cached)
        v_out = torch.where(m, vs[:cached_len].to(v_cached.dtype), v_cached)
        return k_out, v_out

    def _decode_attention_windowed(self, query, kv_cache, attn_metadata, layer) -> torch.Tensor:
        B = query.shape[0]
        Hq, Hk, D = self.num_heads, self.num_kv_heads, self.head_size
        g = Hq // Hk
        bs = self.stage_block
        R = layer._oscar_stage_rows
        dev = query.device
        owner = layer._oscar_slot_owner
        bt = attn_metadata.block_tables
        seq = attn_metadata.seq_lens.to(device=dev, dtype=torch.int64)
        maxpg = bt.shape[1]
        S_nb, TP, S_eff = self.sink_pages, self.tail_pages, self.sink_eff
        W = self._oscar.recent_tokens
        offs = torch.arange(bs, device=dev)

        if S_nb > 0:
            sblk = bt[:, :S_nb].to(torch.int64)
            sown = owner[(sblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
            s_staged = sown == sblk.unsqueeze(-1)
            sink_active = (seq > S_eff) & s_staged.reshape(B, -1).all(dim=1)
            spos = (torch.arange(S_nb, device=dev) * bs).view(1, S_nb, 1) + offs.view(1, 1, bs)
            s_valid = sink_active.view(B, 1, 1) & (spos < S_eff)
            s_valid = s_valid.expand(B, S_nb, bs)
        else:
            sblk = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            s_valid = torch.zeros(B, 0, bs, dtype=torch.bool, device=dev)
            sink_active = torch.zeros(B, dtype=torch.bool, device=dev)
        si = torch.where(sink_active, torch.full_like(seq, S_nb), torch.zeros_like(seq))

        last_page = (seq - 1) // bs
        pg = (last_page - (TP - 1)).unsqueeze(1) + torch.arange(TP, device=dev).unsqueeze(0)
        pg_ok = pg >= 0
        tblk = torch.gather(bt.to(torch.int64), 1, pg.clamp(0, maxpg - 1))
        town = owner[(tblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
        t_staged = (town == tblk.unsqueeze(-1)) & pg_ok.unsqueeze(-1)
        pos = (pg * bs).unsqueeze(-1) + offs.view(1, 1, bs)
        t0 = torch.maximum(seq - W, si * bs)
        inrange = (pos >= t0.view(B, 1, 1)) & (pos < seq.view(B, 1, 1))
        ok = torch.where(inrange, t_staged, torch.ones_like(t_staged))
        sv = (torch.flip(torch.cumprod(torch.flip(ok.reshape(B, -1).long(), [1]), 1), [1]) > 0)
        posf = pos.reshape(B, -1)
        cand = sv & inrange.reshape(B, -1)
        big = torch.iinfo(torch.int64).max
        cut = torch.where(cand, posf, torch.full_like(posf, big)).amin(1)
        cut = torch.minimum(cut, seq)
        t_valid = t_staged & inrange & (pos >= cut.view(B, 1, 1))

        # INT2 中段 [sink, cut)：块表按 sink 页平移（块表为 kernel 粒度，bs 即 kernel 块）
        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (torch.arange(maxpg, device=dev).unsqueeze(0) + si.unsqueeze(1)).clamp(max=maxpg - 1)
        bt_eff = torch.gather(bt, 1, gidx)
        q_rot = torch.matmul(query.float(), self._layer_rots(layer, dev)[0])
        out1, lse1 = self._oscar_int2_decode(q_rot, bt_eff, seq_eff.clamp(min=0))
        o1 = out1
        empty1 = (seq_eff <= 0).view(B, 1)
        lse1 = torch.where(
            empty1 | ~torch.isfinite(lse1),
            torch.full_like(lse1, float("-inf")), lse1,
        )
        o1 = torch.nan_to_num(o1)

        all_blk = torch.cat([sblk, tblk], dim=1)
        valid = torch.cat([s_valid, t_valid], dim=1)
        P = all_blk.shape[1]
        L = P * bs
        rowsP = all_blk % R
        kseg = layer._oscar_stage_k[rowsP].reshape(B, L, Hk, D).float()
        vseg = layer._oscar_stage_v[rowsP].reshape(B, L, Hk, D).float()
        vmask = valid.reshape(B, L)
        qh = q_rot.view(B, Hk, g, D)
        sc = torch.einsum("bkgd,blkd->bkgl", qh, kseg) * self.scale
        sc = sc.masked_fill(~vmask.view(B, 1, 1, L), float("-inf"))
        m2 = sc.amax(dim=-1)
        m2s = torch.where(torch.isfinite(m2), m2, torch.zeros_like(m2))
        p2 = torch.exp(sc - m2s.unsqueeze(-1))
        p2 = torch.where(vmask.view(B, 1, 1, L), p2, torch.zeros_like(p2))
        s2 = p2.sum(-1)
        o2 = torch.einsum("bkgl,blkd->bkgd", p2, vseg) / s2.clamp_min(1e-38).unsqueeze(-1)
        lse2 = torch.where(
            s2 > 0, m2s + torch.log(s2.clamp_min(1e-38)),
            torch.full_like(s2, float("-inf")),
        )
        o2 = o2.reshape(B, Hq, D)
        lse2 = lse2.reshape(B, Hq)

        new_lse = torch.logaddexp(lse1, lse2)
        safe_lse = torch.where(torch.isfinite(new_lse), new_lse, torch.zeros_like(new_lse))
        w1 = torch.exp(lse1 - safe_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - safe_lse).unsqueeze(-1)
        return ((o1 * w1 + torch.nan_to_num(o2) * w2) @ self._layer_rots(layer, dev)[1].t()).to(query.dtype)

    def _oscar_int2_decode(self, q_rot, bt_eff, seq_eff):
        """INT2 段 decode → (out_rot[B,Hq,D], lse[B,Hq])；seq_eff<=0 时返回空。"""
        if seq_eff.numel() == 0:
            B, Hq, D = q_rot.shape
            o = torch.zeros(B, Hq, D, device=q_rot.device)
            l = torch.full((B, Hq), float("-inf"), device=q_rot.device)
            return o, l
        if self._oscar_use_triton:
            try:
                from .kernels.decode_kernel import oscar_decode_triton

                return oscar_decode_triton(
                    q_rot, self.key_cache, self.value_cache, bt_eff, seq_eff,
                    self.scale, self.num_kv_heads, self.head_size,
                )
            except Exception:  # pragma: no cover
                pass
        return oscar_decode_ref(
            q_rot, self.key_cache, self.value_cache, bt_eff, seq_eff,
            self.scale, self.num_kv_heads, self.head_size,
        )
