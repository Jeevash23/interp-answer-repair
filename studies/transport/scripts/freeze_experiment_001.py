"""Generate and freeze the paired prompt manifest for Experiment 001."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset import MANIFEST_SEED, generate_manifest, write_audit, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-small"
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
AUDIT_PATH = PROJECT_ROOT / "data" / "experiment-001-audit.md"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-001-freeze.json"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    records = generate_manifest(tokenizer)
    digest = write_manifest(records, MANIFEST_PATH)
    write_audit(records, AUDIT_PATH, count=100)
    freeze = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_seed": MANIFEST_SEED,
        "manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "manifest_sha256": digest,
        "example_count": len(records),
        "audit_path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
        "audit_rows": 100,
        "independent_human_audit_signed": False,
        "model_repo_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
