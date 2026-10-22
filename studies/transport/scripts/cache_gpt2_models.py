"""Download reproducible GPT-2 snapshots for the circuit-transport experiments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "data" / "model-manifest.json"

MODEL_SPECS = {
    "gpt2-small": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
}

# Keep PyTorch-compatible weights and tokenizer/configuration files. Deliberately skip
# duplicate TensorFlow, Flax, and legacy pytorch_model.bin weights.
ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
]


def directory_size(path: Path) -> int:
    """Return the size of regular files under path, without double-counting symlinks."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    records: list[dict[str, object]] = []

    for local_name, repo_id in MODEL_SPECS.items():
        revision = api.model_info(repo_id).sha
        if revision is None:
            raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")

        destination = MODELS_DIR / local_name
        print(f"Downloading {repo_id} at {revision} to {destination}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=4,
        )

        config = AutoConfig.from_pretrained(destination, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
        weight_files = sorted(path.name for path in destination.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No safetensors weights were downloaded for {repo_id}")

        records.append(
            {
                "name": local_name,
                "repo_id": repo_id,
                "revision": revision,
                "local_path": str(destination.relative_to(PROJECT_ROOT)),
                "architecture": config.architectures,
                "parameters_reported_by_config": {
                    "layers": getattr(config, "n_layer", None),
                    "hidden_width": getattr(config, "n_embd", None),
                    "attention_heads": getattr(config, "n_head", None),
                    "vocabulary_size": tokenizer.vocab_size,
                },
                "weight_files": weight_files,
                "bytes_on_disk": directory_size(destination),
            }
        )
        print(
            f"Validated {local_name}: {directory_size(destination) / 1024**3:.2f} GiB", flush=True
        )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "Hugging Face Hub",
        "format": "safetensors",
        "models": records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproducibility manifest to {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
