"""Freeze the Pythia feasibility manifest and implementation before behavior."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset_004 import retokenize_manifest, validate_manifest, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "experiment-003-manifest.jsonl"
MODEL_PATH = PROJECT_ROOT / "models" / "pythia-410m-deduped"
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-004-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-004-feasibility-freeze.json"
RESULT_PATH = PROJECT_ROOT / "results" / "experiment-004" / "feasibility"

SOURCE_FILES = (
    "src/circuit_transport/dataset_004.py",
    "src/circuit_transport/pythia.py",
    "scripts/run_experiment_004_feasibility.py",
    "tests/test_dataset_004.py",
    "docs/EXPERIMENT-004-FEASIBILITY.md",
    "data/pythia-model-manifest.json",
    "data/experiment-003-manifest.jsonl",
    "uv.lock",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST_PATH.exists() or FREEZE_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("Experiment 004 feasibility is already frozen or evaluated")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    source_records = [json.loads(line) for line in SOURCE_MANIFEST.read_text().splitlines()]
    records = retokenize_manifest(tokenizer, source_records)
    audit = validate_manifest(records)
    manifest_hash = write_manifest(records, MANIFEST_PATH)
    output = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen before Pythia IOI behavior evaluation",
        "model_revision": "c4fc8d586d62df497f1f9b69d66d3ca419992d3e",
        "weight_sha256": "e7ae132489f63d5d86009a8178a75c7d5d195410d067fca01a3160623e370fae",
        "manifest_sha256": manifest_hash,
        "audit": audit,
        "source_files_sha256": {
            relative_path: sha256(PROJECT_ROOT / relative_path) for relative_path in SOURCE_FILES
        },
        "freeze_script_sha256": sha256(Path(__file__)),
    }
    FREEZE_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
