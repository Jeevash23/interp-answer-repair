"""Record the pre-outcome memory-safe localization amendment for Experiment 004."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT_ROOT / "data" / "experiment-004-implementation-freeze.json"
OUTPUT = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-001.json"
FILES = (
    "docs/EXPERIMENT-004.md",
    "scripts/run_experiment_004_localization.py",
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
        raise RuntimeError("Refusing to amend after an Experiment 004 circuit outcome exists")
    payload = {
        "amended_at_utc": datetime.now(UTC).isoformat(),
        "status": "structural memory amendment before any localization outcome",
        "previous_freeze_sha256": sha256(ORIGINAL),
        "failure": {
            "stage": "first development template, first patched localization forward pass",
            "scientific_results_written": False,
            "requested_allocation_gib": 17.5,
            "cause": "expanded TransformerLens hook_result materialization",
        },
        "change": {
            "from": "patch per-head result after W_O",
            "to": "patch per-head z before its fixed linear W_O projection",
            "equivalence": "(z_clean-z_corrupt) @ W_O equals result_clean-result_corrupt",
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
