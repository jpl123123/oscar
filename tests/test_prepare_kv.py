import pytest
import torch

from oscar_ascend.kernels.decode_kernel import oscar_full_dequant_ref
from oscar_ascend.kernels.prepare_kv import prepare_native_kv
from oscar_ascend.kernels.store_kernel import oscar_store_ref


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [False, True])
def test_prepared_buffers_preserve_rounding_and_owner_collisions(
    cache_dtype, output_dtype, window
):
    torch.manual_seed(33)
    prefix, bs, hk, d = 9, 4, 2, 64
    bt = torch.tensor([3, 0, 2], dtype=torch.int32)
    cache = [torch.zeros(4, bs, hk, d, dtype=cache_dtype) for _ in range(2)]
    pos = torch.arange(prefix)
    blocks = bt[pos // bs].long()
    old = [torch.randn(prefix, hk, d) for _ in range(2)]
    oscar_store_ref(*old, *cache, blocks * bs + pos % bs)
    fresh = [torch.randn(3, hk, d, dtype=output_dtype) for _ in range(2)]
    sk, sv = [torch.randn(2, bs, hk, d) for _ in range(2)]
    owner = torch.tensor([[2, -1, 0, -1], [3, -1, -1, 3]])
    stage = (sk, sv, owner) if window else None
    actual = prepare_native_kv(*cache, bt, prefix, *fresh, stage, use_triton=False)
    expected = list(oscar_full_dequant_ref(*cache, bt, prefix, hk, d))
    for i, values in enumerate(expected):
        if window:
            for p in range(prefix):
                block = int(blocks[p])
                if owner[block % 2, p % bs] == block:
                    values[p] = (sk, sv)[i][block % 2, p % bs]
        expected[i] = torch.cat([values.half().to(output_dtype), fresh[i]])
    for a, e in zip(actual, expected):
        assert a.is_contiguous()
        torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_no_prefix_needs_no_kernel_or_extra_buffer():
    k, v = [torch.randn(4, 1, 64) for _ in range(2)]
    actual = prepare_native_kv(None, None, None, 0, k, v)
    assert actual[0].data_ptr() == k.data_ptr()
    assert actual[1].data_ptr() == v.data_ptr()
