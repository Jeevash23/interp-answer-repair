"""Download and record the Pythia checkpoint used for cross-architecture replication."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "EleutherAI/pythia-410m-deduped"
DESTINATION = PROJECT_ROOT / "models" / "pythia-410m-deduped"
CONFIG_CACHE = PROJECT_ROOT / "models" / "huggingface-config-cache"
MANIFEST = PROJECT_ROOT / "data" / "pythia-model-manifest.json"
ALLOW_PATTERNS = ("*.json", "*.model", "*.safetensors", "*.txt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    api = HfApi()
    revision = api.model_info(REPO_ID).sha
    if revision is None:
        raise RuntimeError(f"No revision returned for {REPO_ID}")
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        local_dir=DESTINATION,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=4,
    )
    # TransformerLens' legacy Pythia converter still resolves configuration by
    # repository name. Cache that tiny file locally so all experiment runs are offline.
    AutoConfig.from_pretrained(REPO_ID, revision=revision, cache_dir=CONFIG_CACHE)
    config = AutoConfig.from_pretrained(DESTINATION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(DESTINATION, local_files_only=True)
    weights = sorted(DESTINATION.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("Pythia safetensors weights are missing")
    output = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": REPO_ID,
        "revision": revision,
        "local_path": str(DESTINATION.relative_to(PROJECT_ROOT)),
        "architecture": config.architectures,
        "layers": config.num_hidden_layers,
        "hidden_width": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "vocabulary_size": tokenizer.vocab_size,
        "weight_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in weights
        ],
    }
    MANIFEST.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
