"""Exact Ascend build selected for the deployment; no framework imports."""

from packaging.version import Version

ASCEND_VERSION = "0.23.1.dev0+g5cb98caaa.d20260822"


def check_runtime_version(package: str, installed: str) -> None:
    version = Version(installed)
    if package == "vllm-ascend":
        expected = ASCEND_VERSION
        supported = version == Version(expected)
    elif package == "vllm":
        expected = "0.23.0"
        supported = version.base_version == expected
    else:
        raise ValueError(f"Unknown runtime package: {package}")
    if not supported:
        raise ValueError(
            f"Unsupported {package} version: {installed}; expected {expected}"
        )
