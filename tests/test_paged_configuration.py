"""Keep unvalidated Ascend tile experiments out of the default serving path."""

import pytest

from oscar_ascend.kernels.paged_attention import paged_block_kv


def test_default_uses_npu_validated_tile(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_PAGED_BLOCK_KV", raising=False)
    assert paged_block_kv() == 4


def test_larger_tile_requires_explicit_configuration(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_PAGED_BLOCK_KV", "32")
    assert paged_block_kv() == 32


def test_unsupported_tile_is_rejected(monkeypatch):
    monkeypatch.setenv("OSCAR_ASCEND_PAGED_BLOCK_KV", "12")
    with pytest.raises(ValueError, match="must be"):
        paged_block_kv()
