"""Backend regression tests with real CPU tensors and Ascend-shaped metadata.

The vLLM import is optional; these tests do not pretend to execute NPU kernels.
"""

import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from delivery.check_ready import validate
from oscar_ascend import format as fmt
from oscar_ascend.backend import AscendOscarAttentionBackendImpl as Impl
from oscar_ascend.backend import metadata_batch_lists
from oscar_ascend.integration import memory_budget, verify_layers
from oscar_ascend.kernels.decode_kernel import oscar_full_dequant_ref, oscar_prefill_ref
from oscar_ascend.kernels.paged_attention import oscar_paged_attention_ref, query_layout
from oscar_ascend.kernels.store_kernel import oscar_store_ref
from oscar_ascend.plugin import install_impl
from oscar_ascend.rotation import get_layer_rotation

torch.set_num_threads(2)


def fixture(hk=1, dtype=torch.bfloat16, d=64):
    impl = Impl.__new__(Impl)
    impl.head_size, impl.num_kv_heads, impl.num_heads = d, hk, hk * 2
    impl.scale = d**-0.5
    impl.key_cache = impl.value_cache = None
    impl._oscar_setup()
    impl._oscar.sink_tokens = 4
    impl._oscar.recent_tokens = 4
    impl._oscar.staging_tokens = 8
    layer = NS(layer_name="model.layers.0.self_attn.attn")
    cache = [torch.zeros(8, 4, hk, d, dtype=dtype) for _ in range(2)]
    impl._set_caches(cache)
    return impl, layer, cache


def meta(slots, ends, seqs, bt=None):
    return NS(
        num_actual_tokens=len(slots),
        slot_mapping=torch.tensor(slots),
        query_start_loc=torch.tensor([0] + ends),
        actual_seq_lengths_q=ends,
        seq_lens=torch.tensor(seqs),
        seq_lens_cpu=torch.tensor(seqs),
        seq_lens_list=seqs,
        block_tables=torch.tensor(bt or [[0, 1, 2, 3]], dtype=torch.int32),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("hk", [1, 2, 4])
def test_geometry_and_negative_slots(dtype, hk):
    _impl, _, cache = fixture(hk, dtype)
    k, v = torch.randn(3, hk, 64), torch.randn(3, hk, 64)
    before = [t.clone() for t in cache]
    oscar_store_ref(k, v, *cache, torch.tensor([-1, 2, -1]))
    expected = [t.clone() for t in before]
    oscar_store_ref(k[1:2], v[1:2], *expected, torch.tensor([2]))
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(cache, expected)
    )
    before = [t.clone() for t in cache]
    oscar_store_ref(k, v, *cache, torch.full((3,), -1))
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(cache, before)
    )


def test_staging_persists_and_invalidates_recycled_slots():
    impl, layer, cache = fixture()
    impl._ensure_staging(layer, cache)
    address = layer._oscar_stage_k.data_ptr()
    k, v = torch.randn(4, 1, 64), torch.randn(4, 1, 64)
    impl._staging_write(layer, k, v, meta([0, 1, 2, 3], [4], [4]))
    impl._ensure_staging(layer, cache)
    assert layer._oscar_stage_k.data_ptr() == address
    impl._staging_write(layer, k[:1] + 5, v[:1], meta([4], [1], [5]))
    torch.testing.assert_close(layer._oscar_stage_k[0], k)
    # Reuse old physical slots for middle tokens outside both windows.
    impl._staging_write(
        layer,
        torch.randn(12, 1, 64),
        torch.randn(12, 1, 64),
        meta(list(range(12)), [12], [100]),
    )
    assert (layer._oscar_slot_owner[0] == -1).all()


def test_staging_collision_owners_match_values_and_padding_is_ignored():
    impl, layer, cache = fixture()
    impl._ensure_staging(layer, cache)
    # Physical blocks 0 and stage_rows have the same hash seat.
    slot = layer._oscar_stage_rows * impl.stage_block
    k = torch.stack([torch.ones(1, 64), torch.full((1, 64), 7.0)])
    md = meta([0, slot], [1, 2, 4], [1, 1, 1])  # dummy padding request
    impl._staging_write(layer, k, k, md)
    assert layer._oscar_slot_owner[0, 0].item() == slot // 4
    torch.testing.assert_close(layer._oscar_stage_k[0, 0], k[1])


def test_host_metadata_does_not_read_device_query_starts():
    class NoRead:
        def tolist(self):
            raise AssertionError("unexpected device sync")

    md = meta([0, 1], [1, 2], [9, 12])
    md.query_start_loc = NoRead()
    assert metadata_batch_lists(md) == ([0, 1, 2], [9, 12])


def test_decode_failure_falls_back_to_tensor_result():
    impl, layer, cache = fixture()
    impl._oscar.window_enabled = False
    k, v = torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    impl.do_kv_cache_update(layer, k, v, cache, torch.arange(3))
    query = torch.randn(1, 2, 64)
    md = meta([2], [1], [3])
    expected = impl._decode_attention(query, cache, md, layer)
    impl._oscar_use_triton = True
    with patch(
        "oscar_ascend.kernels.decode_kernel.oscar_decode_triton",
        create=True,
        side_effect=RuntimeError("injected"),
    ):
        actual = impl._decode_attention(query, cache, md, layer)
    torch.testing.assert_close(actual, expected)


def test_failed_surgery_restores_class_and_state():
    class Original:
        pass

    class Broken(Original):
        def _oscar_setup(self):
            self.partial = True
            raise ValueError("unsupported")

    impl = Original()
    impl.keep = 123
    with pytest.raises(ValueError):
        install_impl(impl, Broken)
    assert type(impl) is Original and vars(impl) == {"keep": 123}


@pytest.mark.parametrize("value", [0.0, 1.0, -1.25, 1e-7])
def test_constant_vector_quantization(value):
    x = torch.full((2, 1, 64), value)
    packed, scale, zero = fmt.quantize(x)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    out = fmt.dequant(packed, scale, zero, 64)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("window", [False, True])
def test_rotated_prefill_matches_original_domain_reference(window):
    torch.manual_seed(7)
    impl, layer, cache = fixture()
    impl._oscar.window_enabled = window
    rk = torch.linalg.qr(torch.randn(64, 64)).Q
    rv = torch.linalg.qr(torch.randn(64, 64)).Q
    layer._oscar_rots = (rk, rv)
    layer._oscar_rkT, layer._oscar_rvT = rk.t(), rv.t()
    oldk, oldv = torch.randn(6, 1, 64), torch.randn(6, 1, 64)
    impl.do_kv_cache_update(layer, oldk, oldv, cache, torch.arange(6))
    if window:
        impl._ensure_staging(layer, cache)
        impl._staging_write(layer, oldk, oldv, meta(list(range(6)), [6], [6]))
    q, k, v = torch.randn(3, 2, 64), torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    md = meta([6, 7, 8], [3], [9])
    kr, vr = oscar_full_dequant_ref(*cache, md.block_tables[0], 6, 1, 64)
    if window:
        kr, vr = impl._stage_splice(layer, md.block_tables[0], 6, kr, vr)
    expected = oscar_prefill_ref(q, k, v, kr @ rk.t(), vr @ rv.t(), impl.scale, 1, 64)
    actual = impl._prefill_attention(q, k, v, cache, md, layer)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_paged_oracle_mixed_queries_noncontiguous_pages_and_staging():
    torch.manual_seed(8)
    impl, layer, cache = fixture()
    oldk, oldv = torch.randn(9, 1, 64), torch.randn(9, 1, 64)
    # req0: pages 3,1 prefix5; req1: page4 prefix4
    slots = torch.tensor([12, 13, 14, 15, 4, 16, 17, 18, 19])
    oscar_store_ref(oldk, oldv, *cache, slots)
    impl._ensure_staging(layer, cache)
    impl._staging_write(layer, oldk, oldv, meta(slots.tolist(), [5, 9], [5, 4]))
    stage = (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
    q, k, v = torch.randn(5, 2, 64), torch.randn(5, 1, 64), torch.randn(5, 1, 64)
    bt = torch.tensor([[3, 1, 2], [4, 7, 0]])
    out = oscar_paged_attention_ref(
        q, k, v, *cache, bt, [0, 2, 5], [7, 7], impl.scale, stage
    )
    # Independently assemble original K/V from the staged physical cache.
    expected = torch.cat(
        [
            oscar_prefill_ref(
                q[:2], k[:2], v[:2], oldk[:5], oldv[:5], impl.scale, 1, 64
            ),
            oscar_prefill_ref(
                q[2:], k[2:], v[2:], oldk[5:], oldv[5:], impl.scale, 1, 64
            ),
        ]
    )
    # A hash collision may legitimately evict a stage row; compare only a
    # non-colliding configuration by making all pages' hashes distinct above.
    torch.testing.assert_close(out, expected)
    _, prefix, _, ends = query_layout([0, 2, 5], [7, 7], "cpu")
    assert prefix.tolist() == [5, 4] and ends.tolist() == [6, 7, 5, 6, 7]


def test_budget_accounts_shadow_and_fixed_buffers():
    budget = memory_budget(10000, 100, 25, 1000)
    assert budget == 7200 and budget + (budget // 100) * 25 + 1000 <= 10000
    with pytest.raises(ValueError):
        memory_budget(100, 100, 25, 100)


def test_incomplete_layer_and_rank_manifests_fail():
    verify_layers(["linear_attention", "full_attention"], [1])
    with pytest.raises(RuntimeError):
        verify_layers(["full_attention"] * 2, [0])
    record = {
        "rank": 0,
        "world_size": 2,
        "layers": ["layer.1"],
        "verified": True,
        "sha256": "a",
    }
    line = lambda r: "[oscar-ascend] READY " + json.dumps(r)
    with pytest.raises(ValueError):
        validate(line(record))
    logs = line(record) + "\n" + line(dict(record, rank=1))
    assert len(validate(logs, "a")) == 2
    with pytest.raises(ValueError):
        validate(logs, "stale")


def test_missing_rotation_layer_rejected_when_strict(tmp_path):
    path = tmp_path / "rotation.pt"
    torch.save({"layers": {"1": {"rotation": torch.eye(64)}}}, path)
    with pytest.raises(ValueError, match="missing layer"):
        get_layer_rotation(
            str(path), "model.layers.0.self_attn.attn", 64, "cpu", strict=True
        )


def test_worker_hooks_budget_then_initialize_and_report(monkeypatch, capsys):
    """Exercise the actual hook bodies with the v0.23 worker API replaced."""
    import sys
    from types import ModuleType

    from oscar_ascend.integration import install_runner_hooks
    from oscar_ascend.mtp_shadow import MTPShadowAttentionImpl

    impl, layer, cache = fixture()
    layer.impl = impl
    shadow = MTPShadowAttentionImpl.__new__(MTPShadowAttentionImpl)
    draft = NS(impl=shadow)
    draft_name = "model.mtp.layers.0.self_attn.attn"
    ctx = {layer.layer_name: layer, draft_name: draft}
    specs = {
        name: NS(block_size=4, num_kv_heads=1, head_size=64, head_size_v=64)
        for name in ctx
    }
    config = NS(kv_cache_tensors=[NS(size=8192)], num_blocks=8)

    class Runner:
        compilation_config = NS(static_forward_context=ctx)
        vllm_config = NS(
            cache_config=NS(num_gpu_blocks_override=None),
            parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1),
        )
        model_config = NS(hf_text_config=NS(layer_types=["full_attention"]))

        def get_kv_cache_spec(self):
            return specs

        def initialize_kv_cache_tensors(self, config):
            return {
                layer.layer_name: cache,
                draft_name: [t.to(torch.int8) for t in cache],
            }

    class Worker:
        model_runner = Runner()

        def determine_available_memory(self):
            return 200000

    for name, attrs in {
        "vllm_ascend.worker.worker": {"NPUWorker": Worker},
        "vllm_ascend.worker.model_runner_v1": {"NPUModelRunner": Runner},
        "vllm.v1.core.kv_cache_utils": {
            "get_kv_cache_groups": lambda *a: [],
            "get_kv_cache_config_from_groups": lambda *a: config,
        },
    }.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    install_runner_hooks()
    worker = Worker()
    assert worker.determine_available_memory() < 200000
    worker.model_runner.initialize_kv_cache_tensors(config)
    assert layer._oscar_stage_ready
    assert shadow.key_cache.dtype == torch.bfloat16
    records = validate(capsys.readouterr().out)
    assert records[0]["layer_ids"] == [0]
    assert records[0]["shadow_layers"] == [draft_name]


def test_forward_routes_mtp_to_paged_without_full_dequant(monkeypatch):
    from oscar_ascend import backend

    impl, layer, cache = fixture()
    impl._oscar.use_paged = True
    impl._oscar_use_triton = True
    monkeypatch.setattr(
        "oscar_ascend.kernels.store_kernel.oscar_store_triton", oscar_store_ref
    )
    calls = []

    def paged(*args):
        calls.append(args[6:8])
        return oscar_paged_attention_ref(*args)

    monkeypatch.setattr(
        "oscar_ascend.kernels.paged_attention.oscar_paged_attention_triton", paged
    )
    with patch.object(
        backend,
        "oscar_full_dequant",
        side_effect=AssertionError("dense history path used"),
    ):
        for step, length in enumerate([4, 2]):
            start = 0 if step == 0 else 4
            q, k, v = (
                torch.randn(length, 2, 64),
                torch.randn(length, 1, 64),
                torch.randn(length, 1, 64),
            )
            md = meta(list(range(start, start + length)), [length], [start + length])
            md.attn_state = "SpecDecoding"
            out = impl.forward(
                layer, q, k, v, cache, md, output=torch.empty(length, 128)
            )
            assert torch.isfinite(out).all()
            if step == 0:
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1),
                    k.transpose(0, 1).repeat_interleave(2, 0),
                    v.transpose(0, 1).repeat_interleave(2, 0),
                    is_causal=True,
                    scale=impl.scale,
                )
                torch.testing.assert_close(
                    out.reshape(length, 2, 64), expected.transpose(0, 1)
                )
    assert calls == [([0, 4], [4]), ([0, 2], [6])]

    # A long prefill plus a short MTP query still routes the short request to
    # paged attention, while preserving each request's independent causal mask.
    q, k, v = torch.randn(19, 2, 64), torch.randn(19, 1, 64), torch.randn(19, 1, 64)
    md = meta(list(range(19)), [17, 19], [17, 2], bt=[[0, 1, 2, 3, 4], [4, 5, 0, 0, 0]])
    md.attn_state = "ChunkedPrefill"
    out = impl.forward(layer, q, k, v, cache, md, output=torch.empty(19, 128))
    empty = k[:0]
    expected = torch.cat(
        [
            oscar_prefill_ref(q[:17], k[:17], v[:17], empty, empty, impl.scale, 1, 64),
            oscar_prefill_ref(q[17:], k[17:], v[17:], empty, empty, impl.scale, 1, 64),
        ]
    )
    torch.testing.assert_close(out.reshape(19, 2, 64), expected)
    assert calls[-1] == ([0, 2], [2])


def test_forward_without_attention_state_uses_prefill():
    impl, layer, cache = fixture()
    impl._oscar.use_paged = False
    impl._oscar_use_triton = False
    impl._oscar.window_enabled = False
    md = meta([0, 1], [2], [2])
    q, k, v = torch.randn(2, 2, 64), torch.randn(2, 1, 64), torch.randn(2, 1, 64)
    out = impl.forward(layer, q, k, v, cache, md, output=torch.empty_like(q))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1).repeat_interleave(2, 0),
        v.transpose(0, 1).repeat_interleave(2, 0), is_causal=True,
    ).transpose(0, 1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_native_import_resolves_platform_before_attention(monkeypatch):
    from oscar_ascend import backend

    monkeypatch.setattr(backend.util, "find_spec", lambda name: object())
    calls = []
    native = NS(AscendAttentionBackendImpl=type("NativeImpl", (), {}),
                AscendAttentionState=backend._CPUAttentionState)

    class PlatformModule:
        @property
        def current_platform(self):
            calls.append("resolve_platform")
            return NS(device_type="npu")

    def load(name):
        calls.append(name)
        return PlatformModule() if name == "vllm.platforms" else native

    monkeypatch.setattr(backend, "import_module", load)
    assert backend._load_attention_types() == (native.AscendAttentionBackendImpl, native.AscendAttentionState)
    assert calls == ["vllm.platforms", "resolve_platform", "vllm_ascend.attention.attention_v1"]


@pytest.mark.parametrize("error", [ImportError("vendor symbol missing"), RuntimeError("vendor init failed")])
def test_installed_native_import_failure_is_not_hidden(monkeypatch, error):
    from oscar_ascend import backend

    monkeypatch.setattr(backend.util, "find_spec", lambda name: object())

    def load(name):
        if name == "vllm.platforms":
            return NS(current_platform=NS(device_type="npu"))
        raise error

    monkeypatch.setattr(backend, "import_module", load)
    with pytest.raises(RuntimeError, match="could not import") as caught:
        backend._load_attention_types()
    assert caught.value.__cause__ is error


def test_cpu_attention_states_only_when_both_packages_absent(monkeypatch):
    from oscar_ascend import backend

    monkeypatch.setattr(backend.util, "find_spec", lambda name: None)
    base, state = backend._load_attention_types()
    assert base is object
    assert state.ChunkedPrefill != state.DecodeOnly


@pytest.mark.parametrize("capacity", [8192, 2**24, 2**24 + 2])
def test_staging_sort_exact_and_stable_at_float32_boundary(monkeypatch, capacity):
    from oscar_ascend.backend import staging_order

    seats = torch.tensor([capacity - 1, 0, capacity - 2, capacity - 1, 0])
    original = torch.argsort
    expected = original(seats, stable=True)
    seen = []

    def checked(keys, **kwargs):
        seen.append(keys.dtype)
        return original(keys, **kwargs)

    monkeypatch.setattr(torch, "argsort", checked)
    assert torch.equal(staging_order(seats, capacity), expected)
    assert seen == [torch.float32 if capacity <= 2**24 else torch.int64]


@pytest.mark.parametrize("state_name", ["DecodeOnly", "SpecDecoding"])
@pytest.mark.parametrize("window", [False, True])
def test_native_mtp_default_preserves_history_window_and_padding(monkeypatch, state_name, window):
    from oscar_ascend import backend

    impl, layer, cache = fixture()
    impl.num_heads = 6
    impl._oscar.use_paged = False
    impl._oscar_use_triton = False
    impl._oscar.window_enabled = window
    rk, rv = [torch.linalg.qr(torch.randn(64, 64)).Q for _ in range(2)]
    layer._oscar_rots = (rk, rv)
    oldk, oldv = [torch.randn(9, 1, 64) for _ in range(2)]
    oldslots = [16, 17, 18, 19, 4, 5, 0, 1, 2]
    impl.do_kv_cache_update(layer, oldk, oldv, cache, torch.tensor(oldslots))
    if window:
        impl._ensure_staging(layer, cache)
        impl._staging_write(layer, oldk, oldv, meta(oldslots, [6, 9], [6, 3]))
    q, k, v = torch.randn(6, 6, 64), torch.randn(6, 1, 64), torch.randn(6, 1, 64)
    md = meta([6, 7, 24, 25, 3], [4, 5, 6], [10, 4, 0], bt=[[4, 1, 6], [0, 0, 0], [0, 0, 0]])
    md.attn_state = getattr(backend.AscendAttentionState, state_name)
    seen = []
    original = backend.oscar_full_dequant

    def dequant(*args, **kwargs):
        seen.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "oscar_full_dequant", dequant)
    monkeypatch.setattr("oscar_ascend.kernels.paged_attention.oscar_paged_attention_triton",
                        lambda *a, **k: pytest.fail("slow vector kernel used"))
    monkeypatch.setattr(impl, "_decode_attention", lambda *a: pytest.fail("legacy decode used"))
    actual = impl.forward(layer, q, k, v, cache, md, output=torch.empty_like(q))
    stage = None if not window else (layer._oscar_stage_k, layer._oscar_stage_v, layer._oscar_slot_owner)
    expected = oscar_paged_attention_ref(
        q[:5] @ rk, k[:5] @ rk, v[:5] @ rv, *cache, md.block_tables[:2],
        [0, 4, 5], [10, 4], impl.scale, stage,
    ) @ rv.t()
    assert seen == [6, 3]  # One reconstruction per request, not per query.
    torch.testing.assert_close(actual[:5], expected, atol=1e-5, rtol=1e-4)
    assert torch.count_nonzero(actual[5:]) == 0
