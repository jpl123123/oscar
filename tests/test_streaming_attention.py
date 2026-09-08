"""Numerics, bounds and actual JIT source arithmetic without an NPU compiler."""

import ast
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oscar_ascend.kernels import streaming_attention as stream
from oscar_ascend.kernels.store_kernel import oscar_store_ref


class Tensor(np.ndarray):
    def to(self, dtype, bitcast=False):
        array = np.asarray(self)
        return tensor(array.view(dtype) if bitcast else array.astype(dtype))


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset
        self.dtype = SimpleNamespace(element_ty=data.dtype.type)

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class TL:
    constexpr = int
    float16, float32, int32, int64, uint16 = (
        np.float16,
        np.float32,
        np.int32,
        np.int64,
        np.uint16,
    )
    pid = (0, 0, 0)

    def program_id(self, axis):
        return self.pid[axis]

    @staticmethod
    def load(ptr, mask=None, other=0):
        offsets, valid = np.broadcast_arrays(ptr.offset, True if mask is None else mask)
        indexes = offsets[valid].astype(np.int64)
        assert (indexes >= 0).all() and (indexes < ptr.data.size).all(), (
            "out-of-bounds load"
        )
        result = np.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[valid] = ptr.data[indexes]
        return tensor(result)

    @staticmethod
    def store(ptr, value, mask=None):
        offsets, values, valid = np.broadcast_arrays(
            ptr.offset, value, True if mask is None else mask
        )
        indexes = offsets[valid].astype(np.int64)
        assert (indexes >= 0).all() and (indexes < ptr.data.size).all(), (
            "out-of-bounds store"
        )
        ptr.data[indexes] = values[valid]

    arange = staticmethod(lambda a, b: tensor(np.arange(a, b, dtype=np.int32)))
    cdiv = staticmethod(lambda a, b: (a + b - 1) // b)
    minimum = staticmethod(lambda a, b: tensor(np.minimum(a, b)))
    maximum = staticmethod(lambda a, b: tensor(np.maximum(a, b)))
    where = staticmethod(lambda c, a, b: tensor(np.where(c, a, b)))
    zeros = staticmethod(lambda s, d: tensor(np.zeros(s, dtype=d)))
    full = staticmethod(lambda s, v, d: tensor(np.full(s, v, dtype=d)))
    max = staticmethod(lambda x, a: tensor(np.max(x, axis=a)))
    sum = staticmethod(lambda x, a: tensor(np.sum(x, axis=a)))
    exp = staticmethod(lambda x: tensor(np.exp(x)))
    log = staticmethod(lambda x: tensor(np.log(x)))
    trans = staticmethod(lambda x: tensor(np.asarray(x).T))
    dot = staticmethod(
        lambda a, b: tensor(
            np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        )
    )


def jit_source_functions(tl):
    tree = ast.parse(Path(stream.__file__).read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_kv_tile", "_stream_stage1", "_stream_merge")
    ]
    for fn in functions:
        fn.decorator_list = []
    namespace = {"tl": tl}
    # Execute only trusted repository JIT definitions with bounds-checking
    # NumPy operations. This does not claim Ascend compilation or device timing.
    exec(  # noqa: S102
        compile(ast.Module(body=functions, type_ignores=[]), stream.__file__, "exec"),
        namespace,
    )
    return namespace


def emulate(q, k, v, kc, vc, bt, qsl, seqs, scale, stage, splits):
    tl = TL()
    fns = jit_source_functions(tl)
    n, hq, d = q.shape
    hk = kc.shape[2]
    plan = stream.plan_stream(qsl, seqs, hq, hk, d, has_new=k is not None)
    out = np.full((n, hq, d), np.nan, np.float32)
    lse = np.full((n, hq), np.nan, np.float32)
    mid = np.full((n, hq, splits, d + 1), np.nan, np.float32) if splits > 1 else out
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    sk, sv, owner = (q, q, bt) if stage is None else stage
    inputs = [
        q,
        q if k is None else k,
        q if v is None else v,
        k8,
        v8,
        bt,
        sk,
        sv,
        owner,
    ]
    pointers = [Pointer(t.numpy()) for t in inputs]
    pointers += [
        Pointer(np.array([t[i] for t in plan.tiles], np.int32)) for i in (0, 1)
    ]
    pointers += [
        Pointer(np.array(qsl, np.int32)),
        Pointer(np.array(seqs, np.int32)),
        Pointer(mid),
        Pointer(out),
        Pointer(lse),
    ]
    for pid in product(range(len(plan.tiles)), range(hk), range(splits)):
        tl.pid = pid
        fns["_stream_stage1"](
            *pointers,
            bt.stride(0),
            *k8.stride()[:3],
            *v8.stride()[:3],
            HQ=hq,
            HK=hk,
            D=d,
            BS=kc.shape[1],
            GROUP=hq // hk,
            SPLITS=splits,
            HAS_STAGE=stage is not None,
            ROWS=1 if stage is None else owner.shape[0],
            HAS_NEW=k is not None,
            M=stream.BM,
            N=stream.BN,
            K=stream.BK,
            SCALE=scale,
            K_OFFSET=32,
        )
    if splits > 1:
        for query, head in product(range(n), range(hq)):
            tl.pid = (query, head, 0)
            fns["_stream_merge"](
                Pointer(mid),
                Pointer(out),
                Pointer(lse),
                HQ=hq,
                D=d,
                BD=2 ** math_ceil_log2(d),
                SPLITS=splits,
            )
    return torch.from_numpy(out), torch.from_numpy(lse)


def math_ceil_log2(n):
    return (n - 1).bit_length()


def case(d=64, hk=1, cache_dtype=torch.int8, fresh=True, window=True):
    torch.manual_seed(47)
    hq, bs, nb = hk * 6, 8, 27
    qsl = [0, 1, 5, 5]
    seqs = [1, 71, 15] if fresh else [0, 67, 0]
    bt = torch.randperm(nb).reshape(3, 9).int()
    kc = torch.zeros(nb, bs, hk, d, dtype=cache_dtype)
    vc = torch.zeros_like(kc)
    old = [torch.randn(nb * bs, hk, d) for _ in range(2)]
    oscar_store_ref(*old, kc, vc, torch.arange(nb * bs))
    q = torch.randn(5, hq, d).half()
    k, v = [torch.randn(5, hk, d).half() for _ in range(2)] if fresh else (None, None)
    sk, sv = [torch.randn(3, bs, hk, d) for _ in range(2)]
    owner = torch.full((3, bs), -1, dtype=torch.int64)
    for req in range(3):
        block = int(bt[req, 0])
        owner[block % 3] = block
    return q, k, v, kc, vc, bt, qsl, seqs, d**-0.5, (sk, sv, owner) if window else None


@pytest.mark.parametrize("splits", [1, 3, 16])
@pytest.mark.parametrize("fresh,window", [(True, False), (True, True), (False, True)])
def test_jit_source_matches_online_oracle_and_never_accesses_invalid_rows(
    splits, fresh, window
):
    args = case(fresh=fresh, window=window)
    actual, lse = emulate(*args, splits)
    expected, expected_lse = stream.streaming_attention_ref(
        *args, kv_tile=17, query_tile=2
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
def test_jit_source_multikv_d256(dtype):
    args = case(d=256, hk=2, cache_dtype=dtype)
    actual, lse = emulate(*args, splits=2)
    expected, expected_lse = stream.streaming_attention_ref(*args)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=1e-4)


def test_workspace_does_not_grow_with_history_and_respects_budget():
    p = stream.plan_stream([0, 4, 8], [24_000, 24_000], 6, 1, 256, 100_000)
    longer = stream.plan_stream([0, 4, 8], [262_144, 262_144], 6, 1, 256, 100_000)
    assert p == longer and p.scratch_bytes <= 100_000
    assert stream.plan_stream([0, 15360], [20000], 6, 1, 256).splits == 1
    assert stream.plan_stream([0, 4], [100000], 6, 1, 256, 0).scratch_bytes == 0


def test_cpu_path_only_dequantizes_bounded_tiles(monkeypatch):
    args = case()
    calls = []
    original = stream.dequant_split_ref

    def checked(k8, v8, blocks, pos, hk, d):
        calls.append(blocks.numel())
        assert blocks.numel() <= 11
        return original(k8, v8, blocks, pos, hk, d)

    monkeypatch.setattr(stream, "dequant_split_ref", checked)
    stream.streaming_attention_ref(*args, kv_tile=11, query_tile=2)
    assert calls and max(calls) <= 11


def test_invalid_scores_are_not_silently_ignored_by_merge():
    args = case()
    args[0][0, 0, 0] = torch.nan
    with np.errstate(invalid="ignore"):
        actual, lse = emulate(*args, splits=16)
    assert torch.isnan(actual[0, 0]).all()
    assert torch.isnan(lse[0, 0])
