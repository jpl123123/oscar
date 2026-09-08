"""Keep a failed deployment gate from launching more expensive NPU work."""

import pytest

from tools import benchmark_streaming as bench


@pytest.fixture
def runs(monkeypatch):
    visited = []
    ratios = {case[0]: 1.0 for case in bench.CASES}
    monkeypatch.setattr(bench, "bootstrap", lambda device: None)
    monkeypatch.setattr(bench.torch, "set_num_threads", lambda count: None)

    def fake_case(name, batch, prefix, count, device):
        visited.append(name)
        return {"case": name, "baseline_over_streaming": ratios[name]}

    monkeypatch.setattr(bench, "benchmark_case", fake_case)
    return visited, ratios


@pytest.mark.parametrize("failed_case", ["mtp", "concurrent", "prefill"])
def test_gate_stops_at_first_regression(runs, capsys, failed_case):
    visited, ratios = runs
    ratios[failed_case] = 0.47
    with pytest.raises(SystemExit, match=f"failed for {failed_case}") as exc:
        bench.main(["--gate"])
    names = [case[0] for case in bench.CASES]
    assert visited == names[: names.index(failed_case) + 1]
    assert "remaining cases skipped:" in str(exc.value)
    assert "serving was not enabled" in str(exc.value)
    assert "gate PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("bad_ratio", [float("nan"), float("inf"), 0.0, -1.0])
def test_gate_does_not_accept_invalid_measurement(runs, bad_ratio):
    visited, ratios = runs
    ratios["mtp"] = bad_ratio
    with pytest.raises(SystemExit, match="failed for mtp"):
        bench.main(["--gate"])
    assert visited == ["mtp"]


def test_gate_only_passes_after_all_shapes(runs, capsys):
    visited, ratios = runs
    ratios["mtp"] = 1 / 1.10
    bench.main(["--gate"])
    assert visited == ["mtp", "concurrent", "prefill"]
    assert "gate PASS: all target cases on NPU" in capsys.readouterr().out


def test_standalone_benchmark_still_collects_all_shapes(runs, capsys):
    visited, ratios = runs
    ratios["mtp"] = 0.47
    bench.main([])
    assert visited == ["mtp", "concurrent", "prefill"]
    assert "gate PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--device", "cpu"], ["--cases", "mtp"]])
def test_gate_cannot_pass_partial_or_cpu_coverage(runs, args):
    with pytest.raises(SystemExit) as exc:
        bench.main(["--gate", *args])
    assert exc.value.code == 2
    assert runs[0] == []
