"""v0.23 worker seams: account plugin memory and verify every target layer.

Installed after Attention construction, when Ascend platform imports are safe.
No upstream files are edited. Persistent arenas are allocated at cache setup,
not lazily on the first request. Transient operator workspaces still need NPU
peak-memory validation, just like native attention.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


def memory_budget(available, native_per_block, shadow_per_block, fixed_bytes):
    if native_per_block <= 0 or available <= fixed_bytes:
        raise ValueError("Insufficient memory for OSCAR persistent buffers")
    blocks = (available - fixed_bytes) // (native_per_block + shadow_per_block)
    if blocks < 1:
        raise ValueError("OSCAR memory budget cannot hold one cache block")
    return blocks * native_per_block


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def streaming_reserve(impls, max_tokens):
    """Reserve one worker's peak query/split work, not a BF16 history cache."""
    active = [impl for impl in impls if impl._oscar.attention_mode == "streaming"]
    if not active:
        return 0
    if max_tokens <= 0:
        raise ValueError("Streaming reserve requires max_num_batched_tokens")
    # Conservative allowance for rotated current tensors, output/inverse
    # rotation, and current-KV write temporaries. These scale with query work.
    return max(
        impl._oscar.stream_workspace_bytes
        + max_tokens * impl.head_size * 4 * (3 * impl.num_heads + 4 * impl.num_kv_heads)
        for impl in active
    )


def verify_layers(layer_types, actual_ids):
    expected = {
        i
        for i, kind in enumerate(layer_types)
        if kind in ("full_attention", "attention")
    }
    if not expected or set(actual_ids) != expected or len(actual_ids) != len(expected):
        raise RuntimeError(
            f"Incomplete OSCAR layers: expected={sorted(expected)}, actual={sorted(actual_ids)}"
        )


def install_runner_hooks():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    if getattr(NPUWorker, "_oscar_memory_hook", False):
        return
    original_memory = NPUWorker.determine_available_memory
    original_caches = NPUModelRunner.initialize_kv_cache_tensors

    def determine_available_memory(worker):
        available = original_memory(worker)
        runner = worker.model_runner
        ctx = runner.compilation_config.static_forward_context
        targets = [
            m for m in ctx.values() if hasattr(getattr(m, "impl", None), "_oscar_cfg")
        ]
        if not targets:
            return available
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        from .mtp_shadow import MTPShadowAttentionImpl

        cfg = runner.vllm_config
        if getattr(cfg.cache_config, "num_gpu_blocks_override", None) is not None:
            raise ValueError("num_gpu_blocks_override bypasses OSCAR memory budgeting")
        if getattr(cfg.parallel_config, "data_parallel_size", 1) != 1:
            raise ValueError("OSCAR manifest validation currently supports DP=1")
        if cfg.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("OSCAR memory accounting currently supports PP=1")
        specs = deepcopy(runner.get_kv_cache_spec())
        groups = get_kv_cache_groups(cfg, deepcopy(specs))
        provisional = get_kv_cache_config_from_groups(cfg, groups, available)
        if provisional.num_blocks < 1:
            raise ValueError("No native cache blocks fit")
        native = (
            sum(t.size for t in provisional.kv_cache_tensors) // provisional.num_blocks
        )
        shadow = 0
        fixed = 0
        for name, module in ctx.items():
            impl = getattr(module, "impl", None)
            if isinstance(impl, MTPShadowAttentionImpl):
                spec = specs[name]
                shadow += (
                    spec.block_size
                    * spec.num_kv_heads
                    * (spec.head_size + (spec.head_size_v or spec.head_size))
                    * 2
                )
            elif hasattr(impl, "_oscar_cfg"):
                c = impl._oscar
                # Rotations and contiguous transposes are loaded after profiling.
                fixed += 4 * c.head_dim * c.head_dim * 4
                if c.window_enabled:
                    # Reserve using the engine page size, which upper-bounds
                    # the arena rounded at the smaller kernel block size.
                    bs = specs[name].block_size
                    rows = max(
                        (c.staging_tokens + bs - 1) // bs,
                        c.sink_tokens // bs + (c.recent_tokens + bs - 1) // bs + 3,
                    )
                    fixed += (
                        rows * bs * (2 * impl.num_kv_heads * impl.head_size * 4 + 8)
                    )
        workspace = streaming_reserve(
            [m.impl for m in targets],
            getattr(
                getattr(cfg, "scheduler_config", None), "max_num_batched_tokens", 0
            ),
        )
        fixed += workspace
        budget = memory_budget(available, native, shadow, fixed)
        runner._oscar_memory_plan = {
            "available": available,
            "native": native,
            "shadow": shadow,
            "fixed": fixed,
            "budget": budget,
            "streaming_transient_bytes": workspace,
        }
        worker.available_kv_cache_memory_bytes = budget
        print(
            "[oscar-ascend] MEMORY "
            + json.dumps(runner._oscar_memory_plan, sort_keys=True)
        )
        return budget

    def initialize_kv_cache_tensors(runner, config):
        caches = original_caches(runner, config)
        import torch.distributed as dist

        from .mtp_shadow import MTPShadowAttentionImpl
        from .rotation import layer_index_from_name

        ctx = runner.compilation_config.static_forward_context
        names, ids, shadows = [], [], []
        persistent = 0
        for name, module in ctx.items():
            impl = getattr(module, "impl", None)
            if hasattr(impl, "_oscar_cfg"):
                cache = caches[name]
                if impl._oscar.attention_mode == "streaming":
                    from .kernels.streaming_attention import validate_npu_profile

                    validate_npu_profile(
                        getattr(runner.model_config, "dtype", None),
                        cache[0].shape[1],
                        device_type=cache[0].device.type,
                    )
                impl._set_caches(cache)
                impl._layer_rots(module, cache[0].device)
                persistent += 4 * impl.head_size**2 * 4
                if impl._oscar.window_enabled:
                    impl._ensure_staging(module, cache)
                    persistent += sum(
                        t.numel() * t.element_size()
                        for t in (
                            module._oscar_stage_k,
                            module._oscar_stage_v,
                            module._oscar_slot_owner,
                        )
                    )
                names.append(name)
                ids.append(layer_index_from_name(name))
            elif isinstance(impl, MTPShadowAttentionImpl):
                shadow_kv = impl._shadow_kv(caches[name])
                impl.key_cache, impl.value_cache = shadow_kv
                persistent += sum(t.numel() * t.element_size() for t in shadow_kv)
                shadows.append(name)
        if names:
            plan = runner._oscar_memory_plan
            native_bytes = sum(t.size for t in config.kv_cache_tensors)
            if (
                native_bytes + persistent + plan["streaming_transient_bytes"]
                > plan["available"]
            ):
                raise RuntimeError(
                    "Actual OSCAR persistent allocation exceeds profiled budget"
                )
            hf = runner.model_config.hf_text_config
            verify_layers(getattr(hf, "layer_types", []), ids)
            manifest = {
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "world_size": runner.vllm_config.parallel_config.tensor_parallel_size,
                "layers": sorted(names),
                "layer_ids": sorted(ids),
                "shadow_layers": shadows,
                "native_bytes": native_bytes,
                "persistent_bytes": persistent,
                "streaming_transient_bytes": plan["streaming_transient_bytes"],
                "source": str(Path(__file__).parent.resolve()),
                "sha256": source_fingerprint(),
                "verified": True,
            }
            print(
                "[oscar-ascend] READY " + json.dumps(manifest, sort_keys=True),
                flush=True,
            )
        return caches

    NPUWorker.determine_available_memory = determine_available_memory
    NPUModelRunner.initialize_kv_cache_tensors = initialize_kv_cache_tensors
    from .diagnostics import install_diagnostics

    install_diagnostics(NPUModelRunner)
    NPUWorker._oscar_memory_hook = True
