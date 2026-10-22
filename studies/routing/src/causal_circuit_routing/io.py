"""Small explicit provenance helpers; outputs never silently replace previous runs."""

from __future__ import annotations

from activation_transfer.paths import recorded_path, frozen_file_matches
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    h = hashlib.sha256()
    with recorded_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(recorded_path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_records():
    return [json.loads(line) for line in (ROOT / "data/pilot-001.jsonl").read_text().splitlines()]


def verify_freeze():
    frozen = read_json(ROOT / "data/pilot-001-freeze.json")
    bad = [name for name, sha in frozen["files"].items() if not frozen_file_matches(ROOT / name, sha)]
    model_path = ROOT / frozen["model"]["path"]
    bad += [
        "model/" + name
        for name, sha in frozen["model"]["files"].items()
        if digest(model_path / name) != sha
    ]
    if bad:
        raise RuntimeError(f"Frozen inputs changed: {bad}")
    return frozen
