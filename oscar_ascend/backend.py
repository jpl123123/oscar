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
        self._oscar_cfg: OscarAscendConfig = OscarAscendConfig.from_env(
            head_dim=self.head_size
        )
        check_d(self._oscar_cfg.head_dim)
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
                    "[oscar-ascend] 使用 torch.quantile 裁剪（NPU 支持未正式验证；"
                    "失败将跳过裁剪并继续）"
                )
            try:
                thr = torch.quantile(x_rot.abs(), clip_ratio, dim=-1, keepdim=True)
                x_rot = torch.clamp(x_rot, -thr, thr)
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] quantile 裁剪不可用，跳过: {e}")
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
                f"tokens={N} heads={Hk} — 每 token·head IO {160}B (原生 {2 * D}B, "
                f"写入开销 -{(1 - 160 / (2 * D)) * 100:.1f}%)"
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

    def _layer_rots(self, layer: torch.nn.Module, device: torch.device):
        if not getattr(layer, "_oscar_rots", None):
            cfg = self._oscar
            rk = get_layer_rotation(cfg.k_rotation_path, layer.layer_name, self.head_size, device, mode="k")
            rv = get_layer_rotation(cfg.v_rotation_path, layer.layer_name, self.head_size, device, mode="v")
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
                try:
                    self._ensure_staging(layer, kv_cache)
                    self._staging_write(layer, key, value, attn_metadata)
                except Exception as e:  # pragma: no cover — 窗口是精度增益，失败降级纯 INT2
                    print(f"[oscar-ascend] 窗口 staging 跳过（降级纯 INT2）: {e}")

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
        seq = attn_metadata.seq_lens
        if self._oscar_use_triton:
            try:
                from .kernels.decode_kernel import oscar_decode_triton

                out_rot, _ = oscar_decode_triton(
                    q_rot, self.key_cache, self.value_cache, bt, seq,
                    self.scale, self.num_kv_heads, self.head_size,
                )
            except Exception as e:  # pragma: no cover
                print(f"[oscar-ascend] triton decode 失败，回退 torch: {e}")
                out_rot = oscar_decode_ref(
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
    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = self.num_kv_heads
        q_sl = attn_metadata.query_start_loc
        qsl_list = (
            attn_metadata.query_start_loc_cpu.tolist()
            if getattr(attn_metadata, "query_start_loc_cpu", None) is not None
            else q_sl.tolist()
        )
        seq_lens_list = (
            (getattr(attn_metadata, "seq_lens_cpu", None) or attn_metadata.seq_lens).tolist()
        )
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        num_reqs = len(qsl_list) - 1
        for i in range(num_reqs):
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
                k_cached = torch.matmul(k_cached.float(), layer._oscar_rkT)
                v_cached = torch.matmul(v_cached.float(), layer._oscar_rvT)
                if self._oscar.window_enabled and self._oscar_stage_ready:
                    k_cached, v_cached = self._stage_splice(
                        layer, bt_row, cached_len, k_cached, v_cached
                    )
                out = oscar_prefill_ref(
                    q_seq, k_seq, v_seq, k_cached.to(query.dtype), v_cached.to(query.dtype),
                    self.scale, Hk, D,
                )
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
        layer._oscar_stage_k = torch.zeros(rows, bs, self.num_kv_heads, self.head_size, dtype=torch.bfloat16, device=dev)
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full((rows, bs), -1, dtype=torch.int64, device=dev)
        layer._oscar_stage_rows = rows
        self._oscar_stage_ready = True

    def _staging_write(self, layer, key, value, attn_metadata) -> None:
        N = attn_metadata.num_actual_tokens
        slot = attn_metadata.slot_mapping[:N].to(torch.int64)
        seq = attn_metadata.seq_lens.to(torch.int64)
        qsl = attn_metadata.query_start_loc.to(torch.int64)
        q_lens = qsl[1:] - qsl[:-1]
        req = torch.repeat_interleave(torch.arange(seq.shape[0], device=slot.device), q_lens)
        pos = seq[req] - qsl[req + 1] + torch.arange(N, device=slot.device)
        keep = (slot >= 0) & (
            (pos >= seq[req] - self._oscar.recent_tokens) | (pos < self.sink_eff)
        )
        bs = self.stage_block
        if keep.numel() == 0:
            return
        rows = (slot // bs) % layer._oscar_stage_rows
        kb, kr, ko = (slot // bs)[keep], rows[keep], (slot % bs)[keep]
        if kb.numel() == 0:
            return
        layer._oscar_slot_owner.index_put_((kr, ko), kb)
        win = layer._oscar_slot_owner[kr, ko] == kb
        sel = keep.nonzero(as_tuple=True)[0][win]
        key_sel = key[:N].view(N, self.num_kv_heads, self.head_size)[sel].to(torch.bfloat16)
        val_sel = value[:N].view(N, self.num_kv_heads, self.head_size)[sel].to(torch.bfloat16)
        layer._oscar_stage_k.index_put_((kr[win], ko[win]), key_sel)
        layer._oscar_stage_v.index_put_((kr[win], ko[win]), val_sel)

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
        seq = attn_metadata.seq_lens.to(torch.int64)
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
        o1 = torch.matmul(out1, self._layer_rots(layer, dev)[1].t().contiguous())
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
        qh = query.float().view(B, Hk, g, D)
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
        w1 = torch.exp(lse1 - new_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - new_lse).unsqueeze(-1)
        return (o1 * w1 + torch.nan_to_num(o2) * w2).to(query.dtype)

    def _oscar_int2_decode(self, q_rot, bt_eff, seq_eff):
        """INT2 段 decode → (out_rot[B,Hq,D], lse[B,Hq])；seq_eff<=0 时返回空。"""
        if int(seq_eff.max().item()) <= 0:
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
