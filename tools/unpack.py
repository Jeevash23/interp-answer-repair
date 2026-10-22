"""Unpack the compressed inputs and the saved outputs into the folders the study code reads.

    python tools/unpack.py            # unpack everything listed in data/PACKED-FILES.json, checking every SHA-256
    python tools/unpack.py --check    # report what is unpacked, change nothing
    python tools/unpack.py --remove   # delete the unpacked copies (the packed files stay)

Large frozen inputs are shipped as single .xz files next to where the code expects them; saved outputs
are tar.xz archives under data/saved-outputs/ whose member paths are relative to the release root.
An existing file is left alone when its digest is already right, and the run stops if it differs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import sys
import tarfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_target(rel: str) -> Path:
    p = (RELEASE / rel).resolve()
    if not str(p).startswith(str(RELEASE.resolve()) + "/"):
        raise SystemExit(f"refusing a path outside the release: {rel}")
    return p


def place(rel: str, data: bytes, want: str, mode: str) -> str:
    target = safe_target(rel)
    if target.exists():
        got = sha(target.read_bytes())
        if got == want:
            if mode == "remove":
                target.unlink()
                return "removed"
            return "present"
        raise SystemExit(f"{rel} exists with a different SHA-256; move it away first")
    if mode in ("check", "remove"):
        return "absent"
    if sha(data) != want:
        raise SystemExit(f"{rel}: packed copy does not match its recorded SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return "unpacked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    mode = "remove" if args.remove else "check" if args.check else "unpack"
    packed = json.loads((RELEASE / "data/PACKED-FILES.json").read_text())
    counts = {}
    for xz_rel, rec in sorted(packed.get("xz", {}).items()):
        data = lzma.decompress((RELEASE / xz_rel).read_bytes()) if mode == "unpack" else b""
        s = place(rec["unpacked"], data, rec["sha256"], mode)
        counts[s] = counts.get(s, 0) + 1
    for arc_rel, rec in sorted(packed.get("archives", {}).items()):
        members = rec["members"]
        if mode == "unpack":
            raw = lzma.decompress((RELEASE / arc_rel).read_bytes())
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                for m in tar.getmembers():
                    if not m.isfile() or m.name not in members:
                        raise SystemExit(f"{arc_rel}: unexpected member {m.name}")
                    s = place(m.name, tar.extractfile(m).read(), members[m.name]["sha256"], mode)
                    counts[s] = counts.get(s, 0) + 1
        else:
            for name, meta in members.items():
                s = place(name, b"", meta["sha256"], mode)
                counts[s] = counts.get(s, 0) + 1
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing packed")
    if mode == "check" and counts.get("absent"):
        sys.exit(1)


if __name__ == "__main__":
    main()
