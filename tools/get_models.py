"""Download the pinned model checkpoints into models/ and check every file's SHA-256.

    python tools/get_models.py --list              # what each model is, its size and which studies use it
    python tools/get_models.py gpt2-small          # download one model (needs network access)
    python tools/get_models.py --all               # download every model listed in models/models.json
    python tools/get_models.py --check             # check the files already in models/ (no network)
    python tools/get_models.py --no-weights NAME   # only the configuration and tokeniser files (a few MB)
    python tools/get_models.py --link-from DIR     # use checkpoints already in DIR/<folder> (links, no copy)

Each model is fetched from the Hugging Face Hub at the exact revision the studies used. A file whose
digest differs from the recorded one is reported and the run stops. Set MODELS_DIR to keep the
checkpoints somewhere else; the study code reads the same variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("MODELS_DIR", RELEASE / "models"))
SPEC = json.loads((RELEASE / "models/models.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".msgpack", ".h5", ".gguf")


def selected(spec, no_weights=False):
    return {f: m for f, m in spec["files"].items() if not (no_weights and f.endswith(WEIGHTS))}


def check(name: str, no_weights=False) -> list[str]:
    spec = SPEC["models"][name]
    folder = MODELS / spec["folder"]
    bad = []
    for fname, meta in selected(spec, no_weights).items():
        p = folder / fname
        if not p.exists():
            bad.append(f"{spec['folder']}/{fname}: missing")
        elif meta.get("sha256") and sha256(p) != meta["sha256"]:
            bad.append(f"{spec['folder']}/{fname}: SHA-256 differs from the record")
    return bad


def download(name: str, no_weights=False) -> None:
    from huggingface_hub import hf_hub_download

    spec = SPEC["models"][name]
    folder = MODELS / spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for fname in selected(spec, no_weights):
        hf_hub_download(spec["repo"], fname, revision=spec["revision"], local_dir=str(folder))
    bad = check(name, no_weights)
    if bad:
        raise SystemExit("\n".join(bad))
    print(f"{name}: {len(spec['files'])} files at {spec['repo']}@{spec['revision'][:12]}, digests match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--link-from", metavar="DIR")
    args = ap.parse_args()
    if args.link_from:
        src = Path(args.link_from).resolve()
        for n, s in SPEC["models"].items():
            target, link = src / s["folder"], MODELS / s["folder"]
            if target.is_dir() and not link.exists():
                link.symlink_to(target, target_is_directory=True)
                print(f"{n}: models/{s['folder']} -> {target}")
        return
    names = list(SPEC["models"]) if args.all or (args.check and not args.names) else args.names
    if args.list or not names:
        for n, s in SPEC["models"].items():
            size = sum(m.get("bytes", 0) for m in s["files"].values())
            print(f"{n:16s} {s['repo']}@{s['revision'][:12]}  {size / 1e9:5.2f} GB  -> models/{s['folder']}  ({s['used_by']})")
        return
    unknown = [n for n in names if n not in SPEC["models"]]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}")
    failed = False
    for n in names:
        if args.check:
            bad = check(n, args.no_weights)
            print(f"{n}: " + ("all files present and matching" if not bad else "; ".join(bad)))
            failed |= bool(bad)
        else:
            download(n, args.no_weights)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
