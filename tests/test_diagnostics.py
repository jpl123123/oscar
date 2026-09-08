import json
from types import SimpleNamespace as NS

import pytest
import torch

from oscar_ascend import diagnostics as diag


def records(capsys):
    return [
        json.loads(line.split(" PERF ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if " PERF {" in line
    ]


def test_disabled_diagnostics_do_not_wrap_runner(monkeypatch):
    monkeypatch.delenv("OSCAR_ASCEND_PROFILE_STEPS", raising=False)
    runner = type("Runner", (), {})
    diag.install_diagnostics(runner)
    assert not hasattr(runner, "_oscar_perf_installed")


def test_capture_attributes_integer_sort_and_preserves_results(capsys):
    def operation():
        x = torch.tensor([3, 1, 2])
        a = torch.argsort(x, stable=True)
        b = torch.argsort(x.float(), stable=True)
        return a + b

    measured = diag.timed("sort_test", operation)
    result = diag.capture("execute_model", 1, torch.device("cpu"), measured, (), {})
    assert torch.equal(result, torch.tensor([2, 4, 0]))
    assert diag._active.get() is None
    (report,) = records(capsys)
    assert report["success"]
    assert report["stages"]["sort_test"]["calls"] == 1
    assert any("torch.int64" in k for k in report["integer_sort_stacks"])
    assert any("torch.float32" in k for k in report["sorts"])
    assert any(
        "test_diagnostics.py" in frame
        for frames in report["integer_sort_stacks"].values()
        for frame in frames
    )


def test_capture_preserves_exceptions_and_resets_context(capsys):
    def fail():
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        diag.capture("execute_model", 1, torch.device("cpu"), fail, (), {})
    assert diag._active.get() is None
    assert records(capsys)[0]["success"] is False


def test_runner_capture_is_bounded_and_skips_empty_steps(monkeypatch, capsys):
    from oscar_ascend import backend
    from oscar_ascend.kernels import paged_attention, prefill

    class Impl:
        forward = do_kv_cache_update = _rotate_clip = _staging_write = (
            _prefill_attention
        ) = _decode_attention = lambda *a: None

    # Keep instrumentation mutations local to this test.
    monkeypatch.setattr(backend, "AscendOscarAttentionBackendImpl", Impl)
    monkeypatch.setattr(backend, "staging_order", backend.staging_order)
    monkeypatch.setattr(backend, "prepare_native_kv", backend.prepare_native_kv)
    monkeypatch.setattr(
        paged_attention,
        "oscar_paged_attention_triton",
        paged_attention.oscar_paged_attention_triton,
    )
    monkeypatch.setattr(prefill, "npu_prefill_prepared", prefill.npu_prefill_prepared)
    monkeypatch.setenv("OSCAR_ASCEND_PROFILE_STEPS", "2")

    class Runner:
        device = torch.device("cpu")

        def execute_model(self, scheduler_output):
            return scheduler_output.total_num_scheduled_tokens + 1

        def sample_tokens(self, offset):
            return offset + 3

    diag.install_diagnostics(Runner)
    runner = Runner()
    assert runner.execute_model(NS(total_num_scheduled_tokens=0)) == 1
    for _ in range(4):
        assert runner.execute_model(NS(total_num_scheduled_tokens=7)) == 8
        assert runner.sample_tokens(5) == 8
    reports = records(capsys)
    assert [(r["phase"], r["step"]) for r in reports] == [
        ("execute_model", 1),
        ("sample_tokens", 1),
        ("execute_model", 2),
        ("sample_tokens", 2),
    ]
    assert diag._active.get() is None
