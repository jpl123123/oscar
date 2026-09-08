"""Bounded work, causal partitioning, native API contract, and device-source math."""

import ast
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_streaming_attention import TL, Pointer
from test_streaming_attention import case as small_case

from delivery.probe_streaming import make_case
from oscar_ascend.kernels import prepare_kv
from oscar_ascend.kernels import slab_attention as slab
from oscar_ascend.kernels.streaming_attention import streaming_attention_ref


@pytest.mark.parametrize("fresh,window,legacy", list(product([False, True], repeat=3)))
def test_small_slabs_match_independent_online_oracle(fresh, window, legacy):
    args = small_case(
        d=64,
        hk=2,
        fresh=fresh,
        window=window,
        cache_dtype=torch.bfloat16 if legacy else torch.int8,
    )
    expected = streaming_attention_ref(*args, kv_tile=11, query_tile=3)
    actual = slab.slab_attention(
        *args, chunk_tokens=7, workspace_bytes=14 * 2 * 2 * 64 * 2
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


def test_cache_only_causal_tail_across_slabs_and_leading_empty_queries():
    args = make_case([3, 35, 0], [7, 17, 2], fresh=False)
    actual = slab.slab_attention(*args, chunk_tokens=5, workspace_bytes=15 * 1024)
    expected = streaming_attention_ref(*args, kv_tile=11, query_tile=3)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(actual[0][:4]) == 0
    assert torch.isneginf(actual[1][:4]).all()
    assert torch.count_nonzero(actual[0][-2:]) == 0


def test_each_history_token_decoded_once_for_all_queries_with_reused_buffers(
    monkeypatch,
):
    args = make_case([33, 53], [35, 37])
    decoded, buffers, calls = [], [], []
    real_dequant, real_native = slab.dequant_slab, slab.native_slab_attention

    def observe_dequant(kc, vc, table, reads, kb, vb, stage):
        decoded.extend((r.request, p) for r in reads for p in range(r.start, r.end))
        buffers.append((kb.data_ptr(), vb.data_ptr(), kb.shape[0]))
        return real_dequant(kc, vc, table, reads, kb, vb, stage)

    def observe_native(q, k, v, qe, ke, scale, *, causal):
        calls.append((q.shape[0], k.shape[0], causal))
        return real_native(q, k, v, qe, ke, scale, causal=causal)

    monkeypatch.setattr(slab, "dequant_slab", observe_dequant)
    monkeypatch.setattr(slab, "native_slab_attention", observe_native)
    actual = slab.slab_attention(*args, chunk_tokens=5, workspace_bytes=10 * 1024)
    expected = streaming_attention_ref(*args)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    assert len(decoded) == len(set(decoded)) == 86
    assert set(decoded) == {
        (req, p) for req, length in enumerate([33, 53]) for p in range(length)
    }
    assert len(set(buffers)) == 1 and buffers[0][2] == 10
    assert calls[0] == (72, 72, True)  # raw current KV, no history concatenation
    assert len(calls) == 12  # one current call plus 11 shared history waves
    assert all(q in (72, 37) and k <= 10 and not c for q, k, c in calls[1:])


def test_workspace_and_batch_plan_are_independent_of_history():
    plan = slab.plan_slabs(8, 1, 256)
    assert plan.scratch_bytes == 32 * 1024 * 1024
    assert plan.chunk_tokens == 4096
    for length in [24579, 262144]:
        steps = slab.slab_steps(
            list(range(0, 33, 4)), [length + 4] * 8, plan, has_new=True
        )
        causal, reads = next(steps)
        assert not causal
        assert sum(r.end - r.start for r in reads) == plan.capacity_tokens
    with pytest.raises(ValueError, match="one K/V token"):
        slab.plan_slabs(1, 1, 256, workspace_bytes=0)


def test_more_requests_than_slab_capacity_are_scheduled_in_bounded_groups():
    args = make_case([5] * 7, [2] * 7, window=False)
    actual = slab.slab_attention(*args, workspace_bytes=3 * 1024, chunk_tokens=2)
    expected = streaming_attention_ref(*args)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_native_tnd_api_requests_lse_and_correct_masks(monkeypatch, causal):
    q = torch.randn(8, 6, 256, dtype=torch.bfloat16)
    k = torch.randn(11, 1, 256, dtype=q.dtype)
    observed = {}

    def native(**kwargs):
        observed.update(kwargs)
        return torch.zeros_like(q), torch.zeros(8, 6, 1)

    monkeypatch.setitem(
        sys.modules, "torch_npu", NS(npu_fused_infer_attention_score=native)
    )
    out, lse = slab.npu_slab_attention(q, k, k, [3, 8], [5, 11], 0.0625, causal=causal)
    assert out.shape == q.shape and lse.shape == (8, 6)
    assert observed["input_layout"] == "TND"
    assert observed["softmax_lse_flag"] is True
    assert observed["actual_seq_lengths"] == [3, 8]
    assert observed["actual_seq_lengths_kv"] == [5, 11]
    assert observed["sparse_mode"] == (3 if causal else 0)
    assert (observed["atten_mask"] is not None) == causal


def test_native_lse_layout_must_not_be_silently_transposed(monkeypatch):
    q = torch.ones(2, 6, 64, dtype=torch.bfloat16)
    k = torch.ones(3, 1, 64, dtype=q.dtype)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        NS(npu_fused_infer_attention_score=lambda **kwargs: (q, torch.zeros(6, 2, 1))),
    )
    with pytest.raises(RuntimeError, match="Unexpected native TND LSE"):
        slab.npu_slab_attention(q, k, k, [2], [3], 0.1, causal=False)


def jit_functions(tl):
    functions = []
    for module, names in (
        (prepare_kv, {"_prepare_side"}),
        (slab, {"_slab_side", "_merge_slab"}),
    ):
        tree = ast.parse(Path(module.__file__).read_text())
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name in names:
                fn.decorator_list = []
                functions.append(fn)
    namespace = {"tl": tl}
    # Execute only these trusted local kernel definitions with checked pointers.
    exec(  # noqa: S102
        compile(ast.Module(body=functions, type_ignores=[]), slab.__file__, "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("legacy,window", list(product([False, True], repeat=2)))
def test_actual_dequant_launch_and_jit_handle_offsets_and_packed_requests(
    monkeypatch, legacy, window
):
    args = small_case(
        d=256, hk=2, cache_dtype=torch.bfloat16 if legacy else torch.int8, window=window
    )
    _, _, _, kc, vc, table, _, _, _, stage = args
    reads = [slab.SlabRead(1, 6, 29, 0, 1), slab.SlabRead(1, 31, 66, 1, 2)]
    counts = [r.end - r.start for r in reads]
    expected = [torch.empty(sum(counts), 2, 256, dtype=torch.float16) for _ in range(2)]
    slab.dequant_slab(kc, vc, table, reads, *expected, stage)
    tl = TL()
    functions = jit_functions(tl)

    class CacheProxy:
        device = NS(type="npu")

        def __init__(self, data):
            self.data, self.shape = data, data.shape

        def view(self, dtype):
            return self.data.view(dtype)

    class Launch:
        def __getitem__(self, grid):
            def execute(*args, **kwargs):
                kwargs.pop("num_warps")
                kwargs.pop("num_stages")
                args = [
                    Pointer(a.numpy()) if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                for pid in product(*(range(n) for n in grid)):
                    tl.pid = pid
                    functions["_slab_side"](*args, **kwargs)

            return execute

    tensor = torch.tensor

    def host_tensor(data, **kwargs):
        kwargs["device"] = "cpu"
        return tensor(data, **kwargs)

    monkeypatch.setattr(slab.torch, "tensor", host_tensor)
    monkeypatch.setattr(
        slab,
        "triton",
        NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
        ),
    )
    monkeypatch.setattr(slab, "_slab_side", Launch(), raising=False)
    actual = [torch.full_like(t, torch.nan) for t in expected]
    slab.dequant_slab(CacheProxy(kc), CacheProxy(vc), table, reads, *actual, stage)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("indexed", [False, True])
def test_actual_merge_jit_source_handles_empty_parts_large_lse_and_nan(indexed):
    torch.manual_seed(91)
    out = torch.randn(5, 2, 64)
    lse = torch.tensor(
        [[-torch.inf, 1000.0], [2.0, -torch.inf], [1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]
    )
    part = torch.randn(3, 2, 64).half()
    logs = torch.tensor([[-torch.inf, -1000.0], [float("nan"), 4.0], [3.0, -torch.inf]])
    ids = torch.tensor([0, 3, 1], dtype=torch.int32) if indexed else None
    out[0, 0] = part[0, 0] = torch.nan  # empty values are unspecified
    wanted, wanted_lse = out.clone(), lse.clone()
    slab.merge_slab(wanted, wanted_lse, part, logs, 0, ids)
    tl = TL()
    fn = jit_functions(tl)["_merge_slab"]
    actual, actual_lse = out.numpy().copy(), lse.numpy().copy()
    for block in range(2):
        tl.pid = (block, 0, 0)
        with np.errstate(invalid="ignore", over="ignore"):
            fn(
                Pointer(part.numpy()),
                Pointer(logs.numpy()),
                Pointer(actual),
                Pointer(actual_lse),
                Pointer(np.zeros(1, np.int32) if ids is None else ids.numpy()),
                0,
                3,
                HQ=2,
                D=64,
                BD=64,
                INDEXED=indexed,
                BT=4,
            )
    torch.testing.assert_close(
        torch.from_numpy(actual), wanted, atol=2e-6, rtol=2e-6, equal_nan=True
    )
    torch.testing.assert_close(
        torch.from_numpy(actual_lse), wanted_lse, atol=2e-6, rtol=2e-6, equal_nan=True
    )
    assert torch.count_nonzero(wanted[0, 0]) == 0
