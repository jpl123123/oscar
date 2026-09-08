"""Default deployment gates must match the profile actually verified on NPU."""

from types import SimpleNamespace as NS

import pytest
import torch

from delivery.probe_streaming import kernel_cases
from oscar_ascend.kernels import streaming_attention as stream


def test_deployment_cases_match_verified_dtype_and_block_size():
    cases = kernel_cases()
    assert {name for name, *_ in cases} >= {
        "mixed",
        "batch32",
        "long_24k",
        "cache_only_empty",
        "multi_kv_legacy",
    }
    for _, _, _, options in cases:
        stream.validate_npu_profile(
            options.get("dtype", torch.bfloat16), options.get("bs", 128)
        )


@pytest.mark.parametrize(
    "dtype,bs", [(torch.float16, 128), (torch.bfloat16, 1536), (torch.float16, 1536)]
)
def test_unverified_npu_profiles_fail_before_launch(dtype, bs, monkeypatch):
    with pytest.raises(ValueError, match="requires bf16 queries"):
        stream.validate_npu_profile(dtype, bs)
    # The public launcher must reject the profile before tensor planning or allocation.
    monkeypatch.setattr(stream, "triton", object())
    query = NS(dtype=dtype, device=NS(type="npu"))
    cache = NS(shape=(1, bs, 1, 256))
    with pytest.raises(ValueError, match="requires bf16 queries"):
        stream.streaming_attention_triton(
            query, None, None, cache, cache, None, None, None, 0.0625
        )


def test_cpu_oracle_keeps_compatibility_coverage():
    stream.validate_npu_profile(torch.float16, 1536, device_type="cpu")


def test_compatibility_cases_isolate_the_axes():
    fp16 = kernel_cases("fp16")[0][3]
    large = kernel_cases("large-page")[0][3]
    combined = kernel_cases("fp16-large-page")[0][3]
    assert fp16 == {"dtype": torch.float16}
    assert large == {"bs": 1536}
    assert combined == {"bs": 1536, "dtype": torch.float16}
