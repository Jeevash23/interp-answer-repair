"""Read-only release audit of the completed October research extensions.

Each study has its own protocol, terminal outcome and independent saved-array
audit. Recompute audits in temporary directories; never rewrite their retained
timestamps or observations during a release check.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('007', '008', '009')


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key != 'utc'}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def main():
    with tempfile.TemporaryDirectory(prefix='ai01-extension-audits-') as folder:
        for number in STUDIES:
            study = ROOT / f'studies/bridge-{number}'
            retained = study / 'results/verification.json'
            assert retained.is_file(), f'Bridge {number} has no completed audit'
            original = json.loads(retained.read_text())
            assert original['status'] == 'passed', retained
            output = Path(folder) / f'bridge-{number}.json'
            subprocess.run([sys.executable, str(study / f'code/verify{number}.py'),
                            '--output', str(output)], cwd=ROOT, check=True)
            assert output.is_file(), f'Bridge {number} verifier did not complete'
            repeated = json.loads(output.read_text())
            assert repeated['status'] == 'passed', output
            assert stable(original) == stable(repeated), f'Bridge {number} audit changed'
            print(f'Bridge {number}: retained audit independently replayed')


if __name__ == '__main__':
    main()
