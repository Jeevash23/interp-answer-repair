"""Generate and freeze the pre-outcome Experiment 003 prompt manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset_003 import (
    MANIFEST_SEED,
    TEMPLATES,
    generate_manifest,
    validate_manifest,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "experiment-003-manifest.jsonl"
FREEZE = PROJECT_ROOT / "data" / "experiment-003-data-freeze.json"
AUDIT = PROJECT_ROOT / "data" / "experiment-003-audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST.exists() or FREEZE.exists():
        raise RuntimeError("Experiment 003 data is already frozen; refusing to overwrite it")
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "gpt2-small", local_files_only=True
    )
    excluded_records = [json.loads(line) for line in OLD_MANIFEST.read_text().splitlines()]
    records = generate_manifest(tokenizer, excluded_records)
    audit = validate_manifest(records, excluded_records)
    manifest_hash = write_manifest(records, MANIFEST)
    audit_output = {
        **audit,
        "status": "passed",
        "manifest_sha256": manifest_hash,
        "template_ids": {
            split: [template.template_id for template in TEMPLATES if template.split == split]
            for split in ("development", "test")
        },
    }
    AUDIT.write_text(json.dumps(audit_output, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen before any Experiment 003 model evaluation",
        "manifest_seed": MANIFEST_SEED,
        "manifest_sha256": manifest_hash,
        "prior_manifest_sha256": sha256(OLD_MANIFEST),
        "generator_sha256": sha256(PROJECT_ROOT / "src/circuit_transport/dataset_003.py"),
        "freeze_script_sha256": sha256(Path(__file__)),
        "specification_sha256": sha256(PROJECT_ROOT / "docs/EXPERIMENT-003.md"),
        "audit_sha256": sha256(AUDIT),
        "audit": audit,
    }
    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "freeze": freeze}, indent=2))


if __name__ == "__main__":
    main()
