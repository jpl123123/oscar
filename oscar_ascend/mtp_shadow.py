"""oscar_ascend.mtp_shadow — MTP 草稿层 BF16 影子池（DESIGN-20260904-E 方案 A）。

背景：packed ×2 几何（--kv-cache-dtype int8_per_token_head，见 delivery/serve_oscar.sh）
把所有 attention 层的池视图变成 256B/槽 int8。MTP 草稿层必须保持 BF16 原生路径
（R-20260904：量化草稿 KV 击穿接受率），而 vllm-ascend 没有 per_token_head 的原生
量化算子（全树 grep 零命中）→ 原生路径在 int8 池上不可用。

解法：把 MTP 层 impl 换成本类；forward 时用**插件自建的 BF16 影子池**替换 kv_cache
入参。影子池与原生逻辑视图同形状 `(nb×block_chunk, 128, Hk, D)`，逻辑块号空间与
主池一一对应（slot_mapping / block_tables 原样复用），原生 forward 内部
`self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]`
（attention_v1.py:1527）会自行重绑定 → 写读其余 100% 走原生算子。
代价 ≈ 2 × nb × block_chunk × 128 × Hk × D × 2B ≈ 2.15 GiB/rank
（nb=1369, block_chunk=12, Hk=1, D=256）。

失败协议：影子池建立/替换失败**不回退原生**（int8 池上原生路径会写错数据，静默
错数据比崩溃更糟）→ 打印后向上抛出。
"""
from __future__ import annotations

import torch

try:
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
except Exception:  # pragma: no cover — 平台缺失时 plugin 不会走到本分支
    AscendAttentionBackendImpl = object  # type: ignore


class MTPShadowAttentionImpl(AscendAttentionBackendImpl):  # type: ignore[misc]
    """MTP 草稿层 impl：kv_cache 入参 → BF16 影子池，其余 100% 原生。"""

    def _shadow_kv(self, kv_cache):
        sh = getattr(self, "_oscar_shadow_kv", None)
        if sh is not None:
            return sh
        k_native, v_native = kv_cache[0], kv_cache[1]
        # bf16 固定：本部署模型 bf16（w8a8 只作用权重）；形状取自原生逻辑视图，
        # 与主池几何（packed 256B 或 legacy 512B）自动同构，逻辑块号一一对应。
        self._oscar_shadow_kv = (
            torch.zeros(k_native.shape, dtype=torch.bfloat16, device=k_native.device),
            torch.zeros(v_native.shape, dtype=torch.bfloat16, device=v_native.device),
        )
        gib = (k_native.numel() + v_native.numel()) * 2 / 2**30
        print(
            f"[oscar-ascend] ★ MTP 影子池已建立: shape={tuple(k_native.shape)} bf16 ×2 "
            f"（{gib:.2f} GiB；逻辑块号空间与主池一一对应，原生算子直用）"
        )
        return self._oscar_shadow_kv

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        if (
            isinstance(kv_cache, (tuple, list))
            and len(kv_cache) >= 2
            and kv_cache[0] is not None
            and kv_cache[0].dtype != torch.bfloat16
        ):
            kv_cache = self._shadow_kv(kv_cache)
        # super() = 原生 AscendAttentionBackendImpl：绑定影子池 → 写读全原生
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
