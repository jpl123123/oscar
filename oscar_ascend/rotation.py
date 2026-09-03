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
def _load_checkpoint(path: str) -> dict[int, dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[int, dict[str, torch.Tensor]] = {}
    if isinstance(obj, dict) and "layers" in obj:
        for k, entry in obj["layers"].items():
            lid = int(k)
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
            out[int(k)] = (
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
    return rot.to(device=device, dtype=dtype).contiguous()
