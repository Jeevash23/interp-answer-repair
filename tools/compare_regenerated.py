"""Compare files a script has just regenerated with the copies shipped in this release.

    python tools/compare_regenerated.py paper/sections/study2-secondary-table.tex paper/review/posthoc-baseline.json

Each named file must have the SHA-256 that SHA256SUMS records for it, so a table or analysis output that a
script rewrites from the saved outputs must come out byte for byte the same. Exit status 1 otherwise.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def main():
    sums = {}
    for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        sums[path] = digest
    bad = 0
    for name in sys.argv[1:]:
        got = hashlib.sha256((RELEASE / name).read_bytes()).hexdigest()
        want = sums.get(name)
        ok = got == want
        bad += not ok
        print(f"{'identical' if ok else 'DIFFERENT'}  {name}")
    sys.exit(1 if bad or len(sys.argv) < 2 else 0)


if __name__ == "__main__":
    main()
