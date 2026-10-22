"""Freeze Experiment 003 code and analysis before model outcomes are evaluated."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "experiment-003-implementation-freeze.json"
RESULT_ROOT = PROJECT_ROOT / "results" / "experiment-003"

SOURCE_FILES = (
    "scripts/run_experiment_003.py",
    "scripts/analyze_experiment_003.py",
    "src/circuit_transport/experiment_003.py",
    "tests/test_dataset_003.py",
    "tests/test_experiment_003.py",
    "tests/test_analysis_003.py",
    "docs/EXPERIMENT-003.md",
    "data/experiment-003-manifest.jsonl",
    "data/experiment-003-data-freeze.json",
    "results/experiment-002/localization/selected-heads.json",
    "scripts/run_experiment_002.py",
    "src/circuit_transport/experiment_002.py",
    "uv.lock",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("Experiment 003 implementation is already frozen")
    if any((RESULT_ROOT / split).exists() for split in ("development", "test")):
        raise RuntimeError(
            "Experiment 003 result directories already exist; refusing a late freeze"
        )
    data_freeze = json.loads(
        (PROJECT_ROOT / "data" / "experiment-003-data-freeze.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (
            PROJECT_ROOT / "results" / "experiment-002" / "localization" / "selected-heads.json"
        ).read_text(encoding="utf-8")
    )
    if data_freeze["status"] != "frozen before any Experiment 003 model evaluation":
        raise RuntimeError("Experiment 003 data freeze is not valid")
    if selection["status"] != "complete":
        raise RuntimeError("Experiment 002 selected heads are incomplete")
    output = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "implementation and analysis frozen before development/test model evaluation",
        "protocol": {
            "model": "gpt2-medium",
            "model_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
            "batch_size": 20,
            "corruption": "primary",
            "bootstrap_samples": 5000,
            "bootstrap_seed": 20260903,
            "template_sign_flips": 256,
            "multiple_testing": "Holm across four END controls",
        },
        "source_files_sha256": {
            relative_path: sha256(PROJECT_ROOT / relative_path) for relative_path in SOURCE_FILES
        },
        "freeze_script_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
