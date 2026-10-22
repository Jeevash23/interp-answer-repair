"""Where frozen files live in this release, and whether each still matches its freeze record.

Some frozen files were changed for the release, to replace personal paths or to point the code at
folders inside the release (see RELEASE-CHANGES.md). Every such change is listed in
release-changes.json with the SHA-256 of the original. A frozen file passes when its bytes match the
freeze record, or when release-changes.json records exactly that change (the freeze's digest as the
original and the file's current digest as the release copy). Files a freeze names but the release does
not ship (logs, for example) pass only if release-changes.json lists them with the frozen digest.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("MODELS_DIR", RELEASE / "models"))
PROJECT_KEY = "<project>/"

UNCHANGED = "unchanged"
DOCUMENTED = "changed for the release (documented)"
NOT_SHIPPED = "not shipped (documented)"
OK = {UNCHANGED, DOCUMENTED, NOT_SHIPPED}


@lru_cache(maxsize=1)
def changes():
    return json.loads((RELEASE / "release-changes.json").read_text())


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_key(key: str, base: Path | None = None):
    """Path of a file named in a freeze record: '<project>/x' is x inside this release; any other
    '<...>' placeholder names a file outside the project (not shipped); plain keys are relative to base.
    Keys that pass through a folder listed in release-changes.json "path_tokens" (the names under which
    earlier projects merged into this one were recorded) are mapped to that folder in the release."""
    if key.startswith(PROJECT_KEY):
        return RELEASE / key[len(PROJECT_KEY):]
    if key.startswith("<"):
        return None
    path = Path(os.path.normpath((base or RELEASE) / key))
    tokens = changes().get("path_tokens", {})
    if tokens and not path.exists():
        parts = Path(key).parts
        for token, folder in tokens.items():
            if token in parts:
                path = Path(os.path.normpath(RELEASE / folder / Path(*parts[parts.index(token) + 1:])))
                break
    for alias, target in changes().get("path_aliases", {}).items():   # folders that were links in the project
        rel = release_relative(path)
        if rel is not None and (rel == alias or rel.startswith(alias + "/")):
            path = RELEASE / target / rel[len(alias):].lstrip("/")
    return path


def release_relative(path) -> str | None:
    try:
        return Path(os.path.abspath(path)).relative_to(RELEASE).as_posix()
    except ValueError:
        return None


def status(path, expected: str, key: str | None = None) -> str:
    """Status of one frozen file: unchanged, documented change, not shipped (documented), changed, missing."""
    ch = changes()
    rel = release_relative(path) if path is not None else None
    gone = ch.get("frozen_files_not_shipped", {})
    entry = next((gone[k] for k in (key, rel) if k is not None and k in gone and gone[k]["original_sha256"] == expected),
                 None)
    if path is None or not Path(path).exists():
        return NOT_SHIPPED if entry else "missing"
    got = sha256(path)
    if got == expected:
        return UNCHANGED
    if entry and entry.get("if_present") == "ignore":
        # A record the release does not ship and that a reader's own run recreates with other contents
        # (for example the cache record a model download writes); the freeze cannot speak for it.
        return NOT_SHIPPED
    rec = ch.get("changed_files", {}).get(rel)
    if rec and rec["original_sha256"] == expected and rec["release_sha256"] == got:
        return DOCUMENTED
    return "changed"


def matches(path, expected: str, key: str | None = None) -> bool:
    return status(path, expected, key) in OK


def check_record(record: dict, base: Path) -> dict:
    """Status of every file in a freeze record {'files': {key: sha256}} whose keys are relative to base."""
    out = {}
    for key, digest in record["files"].items():
        out[key] = status(resolve_key(key, base), digest, key)
    return out
