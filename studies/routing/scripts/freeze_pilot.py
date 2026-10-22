"""Create tokenized inputs and freeze protocol, code, dependencies and model hashes."""

from activation_transfer.paths import recorded_path

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

from causal_circuit_routing.dataset import make_records
from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json


def main():
    target = ROOT / "data/pilot-001-freeze.json"
    if target.exists():
        raise FileExistsError("Freeze exists; use a new numbered pilot for protocol changes")
    config = read_json(ROOT / "configs/pilot-001.json")
    model = recorded_path(ROOT / config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = make_records(tokenizer)
    manifest = ROOT / "data/pilot-001.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    paths = [
        Path("data/pilot-001.jsonl"),
        Path("configs/pilot-001.json"),
        Path("uv.lock"),
        Path("pyproject.toml"),
        Path("docs/EXPERIMENT_PLAN.md"),
        Path("docs/FAMILY_SWITCH_METRIC.md"),
    ]
    for directory in ("src", "scripts", "tests"):
        paths.extend(p.relative_to(ROOT) for p in (ROOT / directory).rglob("*.py"))
    model_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    write_json(
        target,
        {
            "frozen_utc": now(),
            "status": "local_pre_inference_freeze_not_public_preregistration",
            "count": len(rows),
            "files": {str(p): digest(ROOT / p) for p in sorted(paths)},
            "model": {
                "path": config["model_path"],
                "repository": config["model_repository"],
                "revision": config["model_revision"],
                "files": {name: digest(model / name) for name in model_files},
            },
        },
    )
    print(f"Frozen {len(rows)} prompts, protocol, code, lock, and seven model files")


if __name__ == "__main__":
    main()
