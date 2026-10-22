"""Restore the two Study 5 model revisions without rewriting frozen manifests.

Run with --verify-only to check existing configurations and weights without network
access. Downloads use the recorded commit IDs, never a repository's latest revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt2-medium", "pythia-410m-deduped")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(model: str, *, verify_only: bool, root: Path = ROOT) -> dict:
    run = json.loads((root / "results/experiment-005" / model / "run-config.json").read_text())
    manifest_name = (
        "data/model-manifest.json" if model == "gpt2-medium" else "data/pythia-model-manifest.json"
    )
    manifest_path = root / manifest_name
    if digest(manifest_path) != run["input_sha256"][manifest_name]:
        raise ValueError("The frozen model manifest has changed")
    manifest = json.loads(manifest_path.read_text())
    record = (
        next(item for item in manifest["models"] if item["name"] == model)
        if model == "gpt2-medium"
        else manifest
    )
    files = (f"models/{model}/config.json", f"models/{model}/model.safetensors")
    for relative in files:
        path = root / relative
        if path.exists() and digest(path) != run["input_sha256"][relative]:
            raise ValueError(f"Existing model file differs from the frozen run: {relative}")
    if not verify_only:
        from huggingface_hub import hf_hub_download, snapshot_download

        snapshot_download(
            repo_id=record["repo_id"],
            revision=record["revision"],
            local_dir=root / "models" / model,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
            max_workers=4,
        )
        if model == "pythia-410m-deduped":
            # The retained TransformerLens loader also resolves this exact tiny
            # configuration in its separate offline Hugging Face cache.
            hf_hub_download(
                repo_id=record["repo_id"],
                revision=record["revision"],
                filename="config.json",
                cache_dir=root / "models/huggingface-config-cache",
            )
    for relative in files:
        if digest(root / relative) != run["input_sha256"][relative]:
            raise ValueError(f"Restored model file differs from the frozen run: {relative}")
    return {
        "model": model,
        "revision": record["revision"],
        "configuration_and_weight_hashes_match": True,
        "frozen_manifests_unchanged": digest(manifest_path) == run["input_sha256"][manifest_name],
        "network_used": not verify_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Models must not be repeated")
    print(
        json.dumps(
            [restore(model, verify_only=args.verify_only) for model in args.models], indent=2
        )
    )


if __name__ == "__main__":
    main()
