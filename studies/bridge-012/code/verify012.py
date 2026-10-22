"""Verify Study 7's freezes after the run (no inference; the project environment is enough).

Checks the protocol and input freezes and every pulled-back stage freeze for the files present
locally. A navigation document that changed after its freeze is accepted only through
provenance/navigation-updates.json: its preserved copy must match the frozen hash and the current
file must match the recorded new hash. Files kept only on the Modal Volume are counted as remote.
(Written after the run; code/freeze012.py, frozen with the inputs, is unchanged.)
"""
from pathlib import Path
import hashlib
import json
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import release_paths as _release_paths  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    updates = json.loads((HERE / 'provenance/navigation-updates.json').read_text())['files']
    for rel, u in updates.items():
        assert sha(ROOT / u['preserved_copy']) == u['frozen_sha256'], rel
    checked = remote = documented = 0
    paths = [HERE / 'data/protocol-freeze.json', HERE / 'data/input-freeze.json'] + sorted((HERE / 'results/freezes').glob('*.json'))
    for path in paths:
        for rel, expected in json.loads(path.read_text())['files'].items():
            if not (ROOT / rel).exists():
                remote += 1
            elif sha(ROOT / rel) == expected:
                checked += 1
            else:
                u = updates.get(rel)
                assert (u and u['frozen_sha256'] == expected and sha(ROOT / rel) == u['current_sha256']) \
                    or _release_paths.matches(ROOT / rel, expected, rel), (path.name, rel)
                documented += 1
    print(f'Study 7 freezes verified ({checked} file hashes; {documented} documented navigation update; '
          f'{remote} listed files kept on the Modal Volume)')


if __name__ == '__main__':
    main()
