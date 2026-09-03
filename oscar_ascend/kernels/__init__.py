"""oscar_ascend.kernels — Triton/torch 双路径算子包。"""
from .store_kernel import oscar_store_ref
from .decode_kernel import oscar_decode_ref, oscar_prefill_ref, oscar_full_dequant_ref
from .dequant_kernel import oscar_full_dequant

try:
    from .store_kernel import triton as _triton
    HAS_TRITON = _triton is not None
except Exception:  # pragma: no cover
    HAS_TRITON = False

__all__ = [
    "oscar_store_ref", "oscar_decode_ref", "oscar_prefill_ref",
    "oscar_full_dequant_ref", "oscar_full_dequant", "HAS_TRITON",
]
