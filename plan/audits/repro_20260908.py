"""Review reproductions without PyTorch; not numerical/NPU integration tests.

Run from any directory: python3 /absolute/path/to/this/file.py
The first two cases execute methods extracted from the actual backend AST,
with allocation/math doubles. Remaining cases check Python indexing and
IEEE float semantics used by the implementation.
"""
import ast
import struct
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT / "oscar_ascend/backend.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))


def method(name, namespace):
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "backend.py (extracted)", "exec"), namespace)
    return namespace[name]


fake_torch = types.SimpleNamespace(
    zeros=lambda *a, **k: object(), zeros_like=lambda *a, **k: object(),
    full=lambda *a, **k: object(), bfloat16="bf16", int64="int64",
)
ensure = method("_ensure_staging", {"torch": fake_torch})
impl = types.SimpleNamespace(
    _oscar=types.SimpleNamespace(sink_tokens=128, recent_tokens=256, staging_tokens=8192),
    num_kv_heads=1, head_size=256,
)
layer = types.SimpleNamespace()
cache = [types.SimpleNamespace(shape=(4, 128, 1, 256), device="cpu")]
ensure(impl, layer, cache)
first = layer._oscar_stage_k
ensure(impl, layer, cache)
assert first is not layer._oscar_stage_k
assert not getattr(layer, "_oscar_stage_ready", False)
print("REPRODUCED: staging reallocates on second call; layer ready flag absent")


class FakeTensor:
    device = "cpu"

    def float(self):
        return self

    def t(self):
        return self

    def contiguous(self):
        return self


def matmul(a, b):
    if isinstance(a, tuple):
        raise TypeError("matmul input is tuple, not Tensor")
    return FakeTensor()


def fail_triton(*args):
    raise RuntimeError("injected Triton failure")


module_name = "oscar_ascend.kernels.decode_kernel"
stub = types.ModuleType(module_name)
stub.oscar_decode_triton = fail_triton
sys.modules[module_name] = stub
decode = method("_decode_attention", {
    "__package__": "oscar_ascend",
    "torch": types.SimpleNamespace(matmul=matmul),
    "oscar_decode_ref": lambda *a: (FakeTensor(), FakeTensor()),
})
impl = types.SimpleNamespace(
    _oscar=types.SimpleNamespace(window_enabled=False), _oscar_use_triton=True,
    _layer_rots=lambda *a: (FakeTensor(), FakeTensor()), key_cache=None,
    value_cache=None, scale=0.1, num_kv_heads=1, head_size=256,
)
try:
    decode(impl, FakeTensor(), None,
           types.SimpleNamespace(block_tables=None, seq_lens=None),
           types.SimpleNamespace(_oscar_read_once=True))
except TypeError as exc:
    print(f"REPRODUCED: decode fallback crashes: {exc}")
else:
    raise AssertionError("Expected tuple matmul error")

bs, slot_width, block_count = 128, 512, 4
slot = -1
offset = (slot // bs) * (bs * slot_width) + (slot % bs) * slot_width
buf = bytearray(block_count * bs * slot_width)
buf[offset] = 1
assert buf[(block_count * bs - 1) * slot_width] == 1
print(f"REPRODUCED: slot=-1 yields byte offset {offset}, indexing the last slot")

assert not (float("nan") > 1e-4)
print("REPRODUCED: NaN error passes the probe's `error > tolerance` rejection check")
floor_half = struct.unpack("e", struct.pack("e", 1e-8))[0]
assert floor_half == 0.0
print("REPRODUCED: SCALE_FLOOR=1e-8 becomes zero when stored as IEEE fp16")
