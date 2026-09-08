"""Deployment version guard regressions, independent of NPU availability."""

from pathlib import Path

import pytest
import tomllib

from delivery.runtime_versions import ASCEND_VERSION, check_runtime_version


def test_selected_ascend_dev_build_is_accepted():
    check_runtime_version("vllm-ascend", "0.23.1.dev0+g5cb98caaa.d20260822")
    check_runtime_version("vllm", "0.23.0")


@pytest.mark.parametrize("version", ["0.23.0", "0.23.1", "0.24.0", "0.23.1.dev0+other"])
def test_unselected_ascend_build_is_rejected(version):
    with pytest.raises(ValueError, match="expected 0.23.1.dev0"):
        check_runtime_version("vllm-ascend", version)


def test_dependency_matches_startup_guard():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert f"vllm-ascend=={ASCEND_VERSION}" in config["project"]["dependencies"]
