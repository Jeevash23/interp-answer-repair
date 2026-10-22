"""Verify frozen sources and deterministic Experiment 001 result artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FREEZE = PROJECT_ROOT / "data" / "experiment-001-implementation-freeze.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment-001" / "verification.json"

EXPECTED_RESULTS = {
    "exploratory": {
        "behavior-per-example.jsonl": "c654df4d5d402784a10cc00ca0ca90a939c84edd833abeace9974ad572601b1a",
        "patches-per-example.jsonl": "d7911e331a67cd1594fed4b0875489b9dc8b3085ea06f52f8029c9c3a82f21b9",
        "behavior-summary.json": "ebe5f428067c213ab38a324c197ad5c7cdcc883b38904af347bcf0a3761edb0b",
        "transport-summary.json": "d4574f1ac206eed4f4ca050a2c387ff0ffe2b13a74a17607eaabaf6d75272fa2",
    },
    "confirmatory": {
        "behavior-per-example.jsonl": "5b81754c17e176c26681ed7e55cab29af3a009c26c9fe9f41ffc8c5ae3286442",
        "patches-per-example.jsonl": "fdeafc8015b6247422dd17505e0437f6736c19b3b0b0d23a207a3d2beee79d86",
        "behavior-summary.json": "bde7523a6306896a08dd25a000295d613abc39cb9e9eec2ccf75ff189cab992c",
        "transport-summary.json": "467c59f8f76a90f3f52b4e00910ddf45c4b3615f90b5b808f702edc2093a78d5",
    },
}
EXPECTED_ROWS = {
    "exploratory": {"behavior-per-example.jsonl": 320, "patches-per-example.jsonl": 3_840},
    "confirmatory": {"behavior-per-example.jsonl": 640, "patches-per-example.jsonl": 7_680},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_jsonl(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    nonfinite = [
        (index, key)
        for index, row in enumerate(rows)
        for key, value in row.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if nonfinite:
        raise RuntimeError(f"{path} contains non-finite values: {nonfinite[:3]}")
    return {"rows": len(rows), "unique_examples": len({row["example_id"] for row in rows})}


def main() -> None:
    freeze = json.loads(IMPLEMENTATION_FREEZE.read_text())
    source_checks = {}
    for relative_path, expected in freeze["source_files_sha256"].items():
        actual = sha256(PROJECT_ROOT / relative_path)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {relative_path}: {actual} != {expected}")
        source_checks[relative_path] = actual

    result_checks: dict[str, Any] = {}
    for run, files in EXPECTED_RESULTS.items():
        result_checks[run] = {}
        for filename, expected in files.items():
            path = PROJECT_ROOT / "results" / "experiment-001" / run / filename
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Result changed: {path}: {actual} != {expected}")
            details: dict[str, Any] = {"sha256": actual}
            if filename in EXPECTED_ROWS[run]:
                details.update(validate_jsonl(path, EXPECTED_ROWS[run][filename]))
            result_checks[run][filename] = details

    output = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "frozen_sources": source_checks,
        "deterministic_results": result_checks,
        "paper_pdf_sha256": sha256(PROJECT_ROOT / "paper" / "build" / "main.pdf"),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
