"""Local provenance and deterministic controls."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from activation_transfer.paths import recorded_path, verify_frozen_file

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260908


def read(path):
    return json.loads(recorded_path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    with recorded_path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "big") % 2**63


def verify(record):
    for path, sha in record.items():
        verify_frozen_file(ROOT / path, sha)
