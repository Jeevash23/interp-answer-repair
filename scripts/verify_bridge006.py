"""Verify the Bridge 006 integrity records (protocol and implementation freezes, measurement and analysis manifests).

Run from the project root: .venv/bin/python scripts/verify_bridge006.py   (also part of `make verify`)
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ['protocol-freeze.json', 'implementation-freeze.json', 'measurement-manifest.json', 'analysis-manifest.json']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def main():
    failures = []
    for name in RECORDS:
        files = json.loads((ROOT / 'studies/bridge-006' / name).read_text())['files']
        bad = [p for p, h in files.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
        failures += bad
        print(f'bridge-006 {name}: {len(files) - len(bad)}/{len(files)} files match')
    if failures:
        print('changed or missing:', *failures[:20], sep='\n  ')
        sys.exit(1)


if __name__ == '__main__':
    main()
