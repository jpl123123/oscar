"""Lifecycle tests for the no-dense-history default backend path."""

from types import SimpleNamespace as NS

import pytest
import torch
from test_backend import fixture, meta

from oscar_ascend import backend
from oscar_ascend.config import OscarAscendConfig
from oscar_ascend.integration import streaming_reserve
from oscar_ascend.kernels.paged_attention import oscar_paged_attention_ref


def forbid_dense(monkeypatch, impl):
    def fail(*args, **kwargs):
        pytest.fail("full-history materialization was invoked")

    monkeypatch.setattr(backend, "oscar_full_dequant", fail)
    monkeypatch.setattr(backend, "prepare_native_kv", fail)
    monkeypatch.setattr(impl, "_stage_splice", fail)


@pytest.mark.parametrize("state_name", ["DecodeOnly", "SpecDecoding", "ChunkedPrefill"])
def test_default_streaming_avoids_dense_history_for_mixed_requests_and_padding(
    monkeypatch, state_name
):
    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    impl._oscar_use_triton = False
    rk, rv = [torch.linalg.qr(torch.randn(64, 64)).Q for _ in range(2)]
    layer._oscar_rots = (rk, rv)
    oldk, oldv = [torch.randn(9, 1, 64) for _ in range(2)]
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 0, 1, 2])
    impl.do_kv_cache_update(layer, oldk, oldv, cache, slots)
    impl._ensure_staging(layer, cache)
    impl._staging_write(layer, oldk, oldv, meta(slots.tolist(), [6, 9], [6, 3]))
    q, k, v = [torch.randn(7, h, 64).half() for h in (2, 1, 1)]
    md = meta(
        [6, 7, 24, 25, 3], [4, 5, 7], [10, 4, 0], bt=[[4, 1, 6], [0, 0, 0], [0, 0, 0]]
    )
    md.attn_state = getattr(backend.AscendAttentionState, state_name)
    forbid_dense(monkeypatch, impl)
    actual = impl.forward(
        layer, q, k, v, cache, md, output=torch.empty(7, 128, dtype=q.dtype)
    )
    stage = (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
    expected = (
        oscar_paged_attention_ref(
            (q[:5].float() @ rk).half(),
            (k[:5].float() @ rk).half(),
            (v[:5].float() @ rv).half(),
            *cache,
            md.block_tables[:2],
            [0, 4, 5],
            [10, 4],
            impl.scale,
            stage,
        ).float()
        @ rv.t()
    )
    torch.testing.assert_close(
        actual[:5].float().reshape(5, 2, 64), expected, atol=3e-3, rtol=3e-3
    )
    assert torch.count_nonzero(actual[5:]) == 0


def test_cache_only_update_invalidates_old_window_and_uses_streaming(monkeypatch):
    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    old = torch.ones(4, 1, 64)
    impl.do_kv_cache_update(layer, old, old, cache, torch.arange(4))
    impl._ensure_staging(layer, cache)
    impl._staging_write(layer, old, old, meta([0, 1, 2, 3], [4], [4]))
    assert layer._oscar_slot_owner[0, 1] == 0
    replacement = torch.full((1, 1, 64), 9.0)
    impl.do_kv_cache_update(layer, replacement, replacement, cache, torch.tensor([1]))
    assert layer._oscar_slot_owner[0, 1] == -1
    assert layer._oscar_slot_owner[0, 0] == 0
    q = torch.zeros(1, 2, 64)
    md = meta([], [1], [4])
    md.num_actual_tokens = 1
    md.attn_state = backend.AscendAttentionState.DecodeOnly
    forbid_dense(monkeypatch, impl)
    actual = impl.forward(layer, q, None, None, cache, md, output=torch.empty_like(q))
    torch.testing.assert_close(actual, torch.full_like(actual, 3.0))  # mean(1,9,1,1)


def test_rejected_mtp_tail_and_recycled_blocks_cannot_leak(monkeypatch):
    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    oldv = torch.arange(6).float()[:, None, None].expand(6, 1, 64).contiguous()
    oldk = torch.zeros_like(oldv)
    impl.do_kv_cache_update(layer, oldk, oldv, cache, torch.arange(6))
    impl._ensure_staging(layer, cache)
    impl._staging_write(layer, oldk, oldv, meta(list(range(6)), [6], [6]))
    forbid_dense(monkeypatch, impl)
    firstv = torch.tensor([10.0, 100.0, 1000.0, 10000.0])[:, None, None].expand(
        4, 1, 64
    )
    impl.forward(
        layer,
        torch.zeros(4, 2, 64),
        torch.zeros_like(firstv),
        firstv,
        cache,
        meta([6, 7, 8, 9], [4], [10]),
        output=torch.empty(4, 2, 64),
    )
    # Only token 6 was accepted; positions 7/8 are replaced and 9 is invisible.
    secondv = torch.tensor([20.0, 30.0])[:, None, None].expand(2, 1, 64)
    out = impl.forward(
        layer,
        torch.zeros(2, 2, 64),
        torch.zeros_like(secondv),
        secondv,
        cache,
        meta([7, 8], [2], [9]),
        output=torch.empty(2, 2, 64),
    )
    torch.testing.assert_close(out[0], torch.full_like(out[0], 45 / 8))
    torch.testing.assert_close(out[1], torch.full_like(out[1], 75 / 9))
    # A new request recycles the same pages. Fresh prefill must ignore old data.
    newv = torch.full((4, 1, 64), -3.0)
    out = impl.forward(
        layer,
        torch.zeros(4, 2, 64),
        torch.zeros_like(newv),
        newv,
        cache,
        meta([0, 1, 2, 3], [4], [4]),
        output=torch.empty(4, 2, 64),
    )
    torch.testing.assert_close(out, torch.full_like(out, -3.0))
    tail = torch.full((1, 1, 64), -8.0)
    out = impl.forward(
        layer,
        torch.zeros(1, 2, 64),
        torch.zeros_like(tail),
        tail,
        cache,
        meta([4], [1], [5]),
        output=torch.empty(1, 2, 64),
    )
    torch.testing.assert_close(out, torch.full_like(out, -4.0))


def test_cached_long_prefill_is_also_streaming(monkeypatch):
    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    impl._oscar.window_enabled = False
    old = torch.ones(4, 1, 64)
    impl.do_kv_cache_update(layer, old, old, cache, torch.arange(4))
    forbid_dense(monkeypatch, impl)
    q = torch.zeros(20, 2, 64)
    new = torch.ones(20, 1, 64)
    md = meta(list(range(4, 24)), [20], [24], bt=[[0, 1, 2, 3, 4, 5]])
    out = impl.forward(layer, q, new, new, cache, md, output=torch.empty_like(q))
    torch.testing.assert_close(out, torch.ones_like(out))


def test_streaming_is_default_and_transient_reserve_is_per_worker(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_ATTENTION_MODE", raising=False)
    cfg = OscarAscendConfig.from_env()
    assert cfg.attention_mode == "streaming"
    impl = NS(_oscar=cfg, head_size=256, num_heads=6, num_kv_heads=1)
    reserve = streaming_reserve([impl], 15360)
    assert reserve == streaming_reserve([impl] * 16, 15360)
    assert reserve > cfg.stream_workspace_bytes
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        streaming_reserve([impl], 0)


def test_mixed_prefix_free_prefill_does_not_force_it_into_streaming(monkeypatch):
    from oscar_ascend.kernels import slab_attention as stream

    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    impl._oscar.window_enabled = False
    old = torch.ones(6, 1, 64)
    impl.do_kv_cache_update(
        layer, old, old, cache, torch.tensor([24, 25, 26, 27, 28, 29])
    )
    q = torch.zeros(21, 2, 64)
    current = torch.ones(21, 1, 64)
    md = meta(
        list(range(17)) + [30, 31, 20, 21],
        [17, 21],
        [17, 10],
        bt=[[0, 1, 2, 3, 4], [6, 7, 5, 0, 0]],
    )
    forbid_dense(monkeypatch, impl)
    calls = []
    original = stream.slab_attention

    def checked(q, k, v, *args, **kwargs):
        calls.append(q.shape[0])
        return original(q, k, v, *args, **kwargs)

    monkeypatch.setattr(stream, "slab_attention", checked)
    result = impl.forward(
        layer, q, current, current, cache, md, output=torch.empty_like(q)
    )
    assert calls == [4]
    torch.testing.assert_close(result, torch.ones_like(result))


def test_streaming_failure_has_no_dense_fallback(monkeypatch):
    from oscar_ascend.kernels import slab_attention as stream

    impl, layer, cache = fixture()
    impl._oscar.attention_mode = "streaming"
    impl._oscar.window_enabled = False
    ones = torch.ones(2, 1, 64)
    impl.do_kv_cache_update(layer, ones, ones, cache, torch.tensor([0, 1]))
    forbid_dense(monkeypatch, impl)

    def failed(*args, **kwargs):
        raise RuntimeError("stream failure")

    monkeypatch.setattr(stream, "slab_attention", failed)
    with pytest.raises(RuntimeError, match="stream failure"):
        impl.forward(
            layer,
            torch.zeros(1, 2, 64),
            ones[:1],
            ones[:1],
            cache,
            meta([2], [1], [3]),
            output=torch.empty(1, 2, 64),
        )
