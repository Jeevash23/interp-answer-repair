"""Freeze study 002 data and pre-inference protocol without touching pilot 001."""

from activation_transfer.paths import recorded_path

import os
import tarfile

from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

from causal_circuit_routing.study2_data import build


def main():
    target = ROOT / "data/study-002-freeze.json"
    if target.exists():
        raise FileExistsError(target)
    config = read_json(ROOT / "configs/study-002.json")
    tokenizer = AutoTokenizer.from_pretrained(recorded_path(ROOT / config["model_path"]), local_files_only=True)
    rows, library = build(tokenizer)
    write_json(ROOT / "data/study-002-recipients.json", rows)
    write_json(ROOT / "data/study-002-prompts.json", library)
    paths = [
        "configs/study-002.json",
        "docs/STUDY-002-PROTOCOL.md",
        "uv.lock",
        "pyproject.toml",
        "data/study-002-recipients.json",
        "data/study-002-prompts.json",
    ]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*study2*.py")]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*study2*.py")]
    frozen = {
        "frozen_utc": now(),
        "stage": "before_study_inference",
        "recipients": len(rows),
        "unique_prompts": len(library),
        "files": {p: digest(ROOT / p) for p in sorted(paths)},
        "model": read_json(ROOT / "data/pilot-001-freeze.json")["model"],
    }
    write_json(target, frozen)
    archive = ROOT / "data/study-002-frozen-source.tar.gz"
    with tarfile.open(archive, "x:gz") as tf:
        for path in paths + ["data/study-002-freeze.json"]:
            tf.add(ROOT / path, arcname=path)
    print(f"Frozen {len(rows)} recipients and {len(library)} unique prompts")


if __name__ == "__main__":
    main()
