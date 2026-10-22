"""Check every freeze record in the release against the files it names.

    python tools/check_freezes.py          # after tools/unpack.py; exit status 1 if any file fails

Each file named in a freeze must be unchanged, changed for the release in a documented way, or
documented as not shipped (tools/release_paths.py explains the rule). The records are listed in
release-changes.json under "freeze_records"; for each, "base" is the folder its paths are relative to,
"field" names the record's map of paths to SHA-256 when it is not "files", and the study's own recorded
exceptions are honoured exactly as its verifier honours them:
"amendments" (a study's AMENDMENTS.json with the amended digest of each changed file), "superseded_by"
(later amendment records whose digests replace this record's), "relocations" (a record of files whose
paths were updated when the study moved, with their digests before and after) and "copies" (a frozen file
kept as a separate frozen copy because the original is updated as the study runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_paths as rp  # noqa: E402


def read(rel):
    return json.loads((rp.RELEASE / rel).read_text())


def main():
    records = rp.changes().get("freeze_records", [])
    totals, failures = {}, []
    for entry in records:
        field = entry.get("field", "files")
        record = read(entry["path"])
        base = rp.RELEASE / entry.get("base", ".")
        amended = {}
        if entry.get("amendments"):
            amended = {a["file"]: a["amended_sha256"] for a in read(entry["amendments"])["files"]}
        later = {}
        for rel in entry.get("superseded_by", []):
            later.update(read(rel)[field])
        moved = {}                     # release path -> relocated digest, for files the freeze recorded before a move
        for rel in ([entry["relocations"]] if entry.get("relocations") else []):
            rbase = rp.RELEASE / entry.get("relocations_base", entry.get("base", "."))
            for key, rec in read(rel)["files"].items():
                moved[rp.release_relative(rp.resolve_key(key, rbase))] = (rec.get("original_sha256"), rec["relocated_sha256"])
        copies = entry.get("copies", {})
        result = {}
        for key, digest in record[field].items():
            path = rp.resolve_key(key, base)
            s = rp.status(path, digest, key)
            if s not in rp.OK and key in amended and rp.matches(path, amended[key], key):
                s = "amended (recorded in the study's amendments)"
            if s not in rp.OK and key in later and rp.matches(path, later[key], key):
                s = "superseded by a recorded amendment"
            mv = moved.get(rp.release_relative(path)) if path is not None else None
            if s not in rp.OK and mv and mv[0] == digest and rp.matches(path, mv[1], key):
                s = "relocated (recorded with digests before and after)"
            if s not in rp.OK and key in copies and rp.matches(rp.RELEASE / copies[key], digest):
                s = "frozen copy matches"
            result[key] = s
        for key, s in result.items():
            totals[s] = totals.get(s, 0) + 1
            if s in ("changed", "missing"):
                failures.append(f"{entry['path']}: {key}: {s}")
        counts = {}
        for s in result.values():
            counts[s] = counts.get(s, 0) + 1
        print(f"{entry['path']}: {len(result)} files: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print("total: " + ", ".join(f"{v} {k}" for k, v in sorted(totals.items())))
    for f in failures:
        print("FAIL", f)
    if not records:
        print("no freeze records listed")
    sys.exit(1 if failures or not records else 0)


if __name__ == "__main__":
    main()
