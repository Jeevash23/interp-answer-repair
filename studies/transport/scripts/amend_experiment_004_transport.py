"""Record the pre-outcome memory-safe transport amendment for Experiment 004."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-001.json"
OUTPUT = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-002.json"
FILES = (
    "docs/EXPERIMENT-004.md",
    "scripts/run_experiment_004.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    forbidden = (
        PROJECT_ROOT / "results" / "experiment-004" / "development" / "patches-per-example.jsonl",
        PROJECT_ROOT / "results" / "experiment-004" / "test" / "patches-per-example.jsonl",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Refusing to amend after an Experiment 004 transport outcome exists")
    payload = {
        "amended_at_utc": datetime.now(UTC).isoformat(),
        "status": "structural memory amendment before any transport outcome",
        "previous_amendment_sha256": sha256(PREVIOUS),
        "failure": {
            "stage": "first development template, first patched transport forward pass",
            "scientific_results_written": False,
            "requested_allocation_gib": 1.09,
            "cause": "four cached hook_result runs plus patched hook_result materialization",
        },
        "change": {
            "batch_size_from": 20,
            "batch_size_to": 4,
            "complete_abba_baba_pairs_per_batch": 2,
            "entity_mismatch_shift_rows": 2,
            "scientific_design_changed": False,
        },
        "amended_source_files_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in FILES
        },
        "amendment_script_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
