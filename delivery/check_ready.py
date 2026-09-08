"""Validate the per-rank cache initialization manifests, not log counters."""

import json
import sys
from pathlib import Path


def validate(log, expected_sha=None):
    records = {}
    for line in log.splitlines():
        marker = "[oscar-ascend] READY "
        if marker in line:
            record = json.loads(line.split(marker, 1)[1])
            if not record.get("verified") or not record.get("layers"):
                raise ValueError("Unverified or empty layer manifest")
            records[record["rank"]] = record
    if not records:
        raise ValueError("No verified cache initialization manifests")
    first = next(iter(records.values()))
    if set(records) != set(range(first["world_size"])):
        raise ValueError("Missing tensor-parallel rank manifests")
    for r in records.values():
        if r["world_size"] != first["world_size"] or r["layers"] != first["layers"]:
            raise ValueError("Ranks disagree about model coverage")
        if r["sha256"] != (expected_sha or first["sha256"]):
            raise ValueError("Worker source does not match this checkout")
    if "[oscar-ascend] 注入失败" in log or "Traceback (most recent call last)" in log:
        raise ValueError("Server log contains initialization/runtime failure")
    return records


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from oscar_ascend.integration import source_fingerprint

    try:
        records = validate(Path(sys.argv[1]).read_text(), source_fingerprint())
    except (ValueError, KeyError) as exc:
        print(f"VERDICT: NOT-ACTIVE — {exc}")
        sys.exit(1)
    print(
        f"VERDICT: OSCAR ACTIVE — {len(records)} ranks verified at cache initialization; numerical/performance acceptance is separate"
    )
