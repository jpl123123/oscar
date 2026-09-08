"""oscar_ascend.rotation — per-layer 正交旋转加载/缓存。

Port of vllm OSCAR PR `oscar/rotation.py`（快照 57286d5d）：
检查点格式 {layers: {layer_id: {rotation: [D,D] fp32, ...}}} 或 {layer_id: matrix}
或堆叠 [num_layers, D, D]；缺层/缺路径 → 单位阵（退化 clipped INT2，保持可跑）。
加载一次 lru_cache（CPU tensor），调用方按需 .to(device, dtype)。
"""
from __future__ import annotations

import re
from functools import lru_cache

import torch

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index_from_name(layer_name: str) -> int | None:
    """从 vllm 层名提取全局 decoder 层索引；失败返回 None（调用方回退单位阵）。"""
    m = _LAYER_IDX_RE.search(layer_name)
    if m is not None:
        return int(m.group(1))
    ints = re.findall(r"\d+", layer_name)
    return int(ints[-1]) if ints else None


@lru_cache(maxsize=8)
def _resolve_lid(key) -> int | None:
    """检查点键容错：int / '11' / 'language_model.model.layers.11.self_attn.attn' 均可。"""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            m = _LAYER_IDX_RE.search(key)
            if m:
                return int(m.group(1))
            ints = re.findall(r"\d+", key)
            return int(ints[-1]) if ints else None
    return None


@lru_cache(maxsize=8)
def _load_checkpoint(path: str) -> dict[int, dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and obj.get("format_version", 0) >= 2
            and "r_h_pbr" in str(obj.get("objective", ""))):
        print(
            f"[oscar-ascend] ⚠️ 旋转检查点 {path!r} 非 v2 配方（缺 U@H@P_br 组合；"
            "format_version/objective 检查未通过）——per-vector INT2 精度将显著劣化；"
            "请重新运行校准（delivery/install_and_launch.sh 会自动触发）。"
        )
    out: dict[int, dict[str, torch.Tensor]] = {}
    if isinstance(obj, dict) and "layers" in obj:
        for k, entry in obj["layers"].items():
            lid = _resolve_lid(k)
            if lid is None:
                continue
            if isinstance(entry, dict) and "rotation" in entry:
                # gen_rotations / PR 格式：entry 含 rotation(+rotation_v/eigenvalues)
                out[lid] = {
                    key: value.float().contiguous()
                    for key, value in entry.items()
                    if isinstance(value, torch.Tensor)
                }
            else:
                out[lid] = {"rotation": entry.float().contiguous()}
    elif isinstance(obj, dict):
        for k, rot in obj.items():
            lid = _resolve_lid(k)
            if lid is None:
                continue
            out[lid] = (
                {"rotation": rot.float().contiguous()}
                if not (isinstance(rot, dict) and "rotation" in rot)
                else {
                    key: value.float().contiguous()
                    for key, value in rot.items()
                    if isinstance(value, torch.Tensor)
                }
            )
    elif torch.is_tensor(obj) and obj.dim() == 3:
        for lid in range(obj.shape[0]):
            out[lid] = {"rotation": obj[lid].float().contiguous()}
    else:
        raise ValueError(
            f"Unrecognized OSCAR rotation checkpoint structure at {path!r}: {type(obj)}"
        )
    return out


def get_layer_rotation(
    path: str,
    layer_name: str,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mode: str = "k",
    strict: bool = False,
) -> torch.Tensor:
    """返回本层 [D,D] 旋转；空路径/缺层 → 单位阵（fp32 contiguous）。

    mode="k" 取 entry["rotation"]（PR 兼容）；mode="v" 优先 entry["rotation_v"]
    （gen_rotations.py 产物），缺失时回退 entry["rotation"]。
    """
    if not path:
        return torch.eye(head_dim, device=device, dtype=dtype)
    table = _load_checkpoint(path)
    lid = layer_index_from_name(layer_name)
    entry = table.get(lid) if lid is not None else None
    if entry is None:
        if strict:
            raise ValueError(f"Rotation checkpoint {path!r} is missing layer {layer_name}")
        return torch.eye(head_dim, device=device, dtype=dtype)
    rot = entry.get("rotation_v") if mode == "v" else entry.get("rotation")
    if rot is None:
        rot = entry.get("rotation")
    if rot is None:
        return torch.eye(head_dim, device=device, dtype=dtype)
    if rot.shape != (head_dim, head_dim):
        raise ValueError(
            f"OSCAR rotation for layer {lid} has shape {tuple(rot.shape)}, "
            f"expected ({head_dim}, {head_dim})."
        )
    if strict:
        if not torch.isfinite(rot).all() or not torch.allclose(rot @ rot.t(), torch.eye(head_dim), atol=1e-3, rtol=1e-3):
            raise ValueError(f"Rotation checkpoint {path!r} layer {lid} is not finite/orthogonal")
    return rot.to(device=device, dtype=dtype).contiguous()
