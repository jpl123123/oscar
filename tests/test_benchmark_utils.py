import pytest
import torch

from delivery.benchmark_utils import compare_calls


def test_first_call_is_excluded_and_warm_order_alternates():
    now = [0.0]
    order = []
    counts = {"baseline": 0, "fused": 0}

    def call(name, cold, warm):
        order.append(name)
        now[0] += cold if counts[name] == 0 else warm
        counts[name] += 1
        return torch.ones(3)

    result = compare_calls(
        {
            "baseline": lambda: call("baseline", 1, 0.001),
            "fused": lambda: call("fused", 100, 0.002),
        },
        lambda: None,
        repeats=2,
        clock=lambda: now[0],
    )
    assert order == ["baseline", "fused", "baseline", "fused", "fused", "baseline"]
    assert result["baseline"]["first_call_ms"] == 1000
    assert result["fused"]["first_call_ms"] == 100000
    assert result["baseline"]["median_ms"] == pytest.approx(1)
    assert result["fused"]["median_ms"] == pytest.approx(2)
    assert len(result["baseline"]["samples_ms"]) == 2


def test_comparison_copies_shared_output_before_next_variant():
    output = torch.empty(3)
    with pytest.raises(AssertionError):
        compare_calls(
            {
                "baseline": lambda: output.fill_(1),
                "fused": lambda: output.fill_(0),
            },
            lambda: None,
            repeats=2,
        )


def test_progress_printing_is_outside_timing(monkeypatch):
    now, lines = [0.0], []

    def slow_print(message, **kwargs):
        lines.append(message)
        now[0] += 10.0

    def call():
        now[0] += 0.002
        return torch.ones(3)

    monkeypatch.setattr("builtins.print", slow_print)
    result = compare_calls(
        {"baseline": call, "streaming": call},
        lambda: None,
        repeats=2,
        clock=lambda: now[0],
        label="STREAM BENCH",
        progress=True,
    )
    for timing in result.values():
        assert timing["first_call_ms"] == pytest.approx(2)
        assert timing["median_ms"] == pytest.approx(2)
    assert any("first call complete: streaming" in line for line in lines)
    assert any("warm sample 2/2: streaming start" in line for line in lines)
    assert any("warm sample 2/2: streaming complete" in line for line in lines)
