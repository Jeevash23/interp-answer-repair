"""Resolve recorded source locations and verify explicitly recorded relocations.

Scientific measurements and freeze records are never rewritten. A relocated source
must match both its recorded original digest and the new digest in the migration
ledger. This is an acknowledged source migration, not byte-identical source replay.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    # Release: the earlier projects' recorded names are replaced by these tokens (RELEASE-CHANGES.md).
    'transport-project': ROOT / 'studies/transport',
    'routing-project': ROOT / 'studies/routing',
    'bridge-project': ROOT,
}


def recorded_path(path):
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    for name, destination in PROJECTS.items():
        if name in parts:
            found = destination.joinpath(*parts[parts.index(name) + 1:])
            # Release: the transport study's models/ folder was a link to the shared models/ folder.
            transport_models = ROOT / 'studies/transport/models'
            if not found.exists() and found.is_relative_to(transport_models):
                found = ROOT / 'models' / found.relative_to(transport_models)
            return found
    return path


def sha256(path):
    with recorded_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=1)
def relocation_records():
    path = ROOT / 'provenance/source-relocations.json'
    return json.loads(path.read_text())['files']


def frozen_file_matches(path, expected):
    path = recorded_path(path)
    if not path.is_file():
        return False
    actual = sha256(path)
    if actual == expected:
        return True
    try:
        key = str(path.resolve().relative_to(ROOT))
    except ValueError:
        return False
    record = relocation_records().get(key)
    if record and record['original_sha256'] == expected and record['relocated_sha256'] == actual:
        return True
    return _release_matches(path, expected)


def _release_matches(path, expected):
    """Release: a frozen file changed for the release in a way release-changes.json documents."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    import release_paths
    return release_paths.matches(path, expected)


def verify_frozen_file(path, expected):
    if not frozen_file_matches(path, expected):
        raise ValueError(f'Frozen file or documented relocation does not match: {path}')
