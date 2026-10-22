"""Freeze Experiment 004 code and hypotheses before localization outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "experiment-004-implementation-freeze.json"
MODEL_MANIFEST = PROJECT_ROOT / "data" / "pythia-model-manifest.json"
FILES = (
    "data/experiment-004-manifest.jsonl",
    "data/experiment-004-feasibility-freeze.json",
    "data/pythia-model-manifest.json",
    "docs/EXPERIMENT-004.md",
    "src/circuit_transport/pythia.py",
    "src/circuit_transport/experiment_002.py",
    "src/circuit_transport/experiment_003.py",
    "src/circuit_transport/experiment_004.py",
    "scripts/run_experiment_002.py",
    "scripts/analyze_experiment_003.py",
    "scripts/run_experiment_004_localization.py",
    "scripts/run_experiment_004.py",
    "scripts/analyze_experiment_004.py",
    "tests/test_experiment_004.py",
    "uv.lock",
)
FEASIBILITY_FILES = (
    "results/experiment-004/feasibility/behavior-per-example.jsonl",
    "results/experiment-004/feasibility/behavior-summary.json",
    "results/experiment-004/feasibility/decision.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    forbidden = (
        PROJECT_ROOT / "results" / "experiment-004" / "localization" / "selected-heads.json",
        PROJECT_ROOT / "results" / "experiment-004" / "development" / "patches-per-example.jsonl",
        PROJECT_ROOT / "results" / "experiment-004" / "test" / "patches-per-example.jsonl",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Refusing to freeze after an Experiment 004 circuit outcome exists")
    model = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen before Pythia localization and transport outcomes",
        "model_revision": model["revision"],
        "weight_sha256": next(
            item["sha256"] for item in model["weight_files"] if item["name"] == "model.safetensors"
        ),
        "selection": {
            "development_records": 160,
            "development_templates": 8,
            "sites": ["s2", "end"],
            "heads_per_condition_site": 8,
            "positive_only": True,
        },
        "inference": {
            "bootstrap_samples": 5000,
            "bootstrap_seed": 20260904,
            "control_test": "exact 2^8 template-level sign flip with Holm correction",
            "primary_replication_site": "end",
        },
        "feasibility_result_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in FEASIBILITY_FILES
        },
        "feasibility_reproduction": {
            "scientific_artifacts_byte_identical": all(
                sha256(PROJECT_ROOT / relative)
                == sha256(
                    PROJECT_ROOT / relative.replace("/feasibility/", "/feasibility-reproduction/")
                )
                for relative in FEASIBILITY_FILES
            )
        },
        "source_files_sha256": {relative: sha256(PROJECT_ROOT / relative) for relative in FILES},
        "freeze_script_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
