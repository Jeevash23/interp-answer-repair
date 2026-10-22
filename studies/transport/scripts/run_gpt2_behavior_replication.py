"""Exploratory behavioral replication across cached GPT-2 model sizes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-001-freeze.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "gpt2-size-behavior"
MODELS = {
    "gpt2-small": torch.float32,
    "gpt2-medium": torch.bfloat16,
    "gpt2-large": torch.bfloat16,
}
CORRUPTIONS = {
    "primary": "corrupt_token_ids",
    "alternative": "alternative_corrupt_token_ids",
}
BATCH_SIZE = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    condition: str,
    token_key: str,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[float]] = defaultdict(list)
    device = torch.device("cuda:0")
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        tokens = torch.tensor([row[condition][token_key] for row in batch], device=device)
        with torch.inference_mode():
            logits = model(tokens).logits[:, -1, :].float()
        rows = torch.arange(len(batch), device=device)
        io_ids = torch.tensor([row["io_token_id"] for row in batch], device=device)
        subject_ids = torch.tensor([row["subject_token_id"] for row in batch], device=device)
        probabilities = logits.softmax(dim=-1)
        io_logits = logits[rows, io_ids]
        subject_logits = logits[rows, subject_ids]
        io_probabilities = probabilities[rows, io_ids]
        subject_probabilities = probabilities[rows, subject_ids]
        outputs["probability_difference"].extend(
            (io_probabilities - subject_probabilities).cpu().tolist()
        )
        outputs["logit_difference"].extend((io_logits - subject_logits).cpu().tolist())
        outputs["pairwise_accuracy"].extend((io_logits > subject_logits).float().cpu().tolist())
    return {key: np.asarray(value) for key, value in outputs.items()}


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    if sha256(MANIFEST_PATH) != freeze["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    records = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    configuration: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "exploratory behavioral replication; no circuit sites transferred across models",
        "manifest_sha256": freeze["manifest_sha256"],
        "models": {},
    }

    for model_name, dtype in MODELS.items():
        model_path = PROJECT_ROOT / "models" / model_name
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        first = records[0]
        for condition in ("canonical", "controlled"):
            if (
                tokenizer.encode(first[condition]["clean_text"], add_special_tokens=False)
                != first[condition]["clean_token_ids"]
            ):
                raise RuntimeError(f"Tokenizer mismatch for {model_name}")

        print(f"Loading {model_name} ({dtype})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to("cuda")
        model.eval()
        for split in ("exploratory", "development", "test"):
            split_records = [record for record in records if record["split"] == split]
            for condition in ("canonical", "controlled"):
                clean = evaluate(
                    model, split_records, condition=condition, token_key="clean_token_ids"
                )
                for corruption, token_key in CORRUPTIONS.items():
                    corrupt = evaluate(
                        model, split_records, condition=condition, token_key=token_key
                    )
                    summaries.append(
                        {
                            "model": model_name,
                            "dtype": str(dtype).removeprefix("torch."),
                            "split": split,
                            "condition": condition,
                            "corruption": corruption,
                            "n": len(split_records),
                            "clean_pairwise_accuracy": float(clean["pairwise_accuracy"].mean()),
                            "clean_probability_difference": float(
                                clean["probability_difference"].mean()
                            ),
                            "clean_logit_difference": float(clean["logit_difference"].mean()),
                            "corrupt_probability_difference": float(
                                corrupt["probability_difference"].mean()
                            ),
                            "corrupt_logit_difference": float(corrupt["logit_difference"].mean()),
                            "behavioral_floor_passed": bool(
                                clean["pairwise_accuracy"].mean() >= 0.70
                                and clean["probability_difference"].mean() > 0
                            ),
                        }
                    )
        configuration["models"][model_name] = {
            "dtype": str(dtype).removeprefix("torch."),
            "local_path": str(model_path.relative_to(PROJECT_ROOT)),
            "allocated_gpu_memory_mib_after_evaluation": round(
                torch.cuda.memory_allocated() / 1024**2, 1
            ),
        }
        del model
        torch.cuda.empty_cache()

    configuration["completed_at_utc"] = datetime.now(UTC).isoformat()
    configuration["script_sha256"] = sha256(Path(__file__))
    write_csv(OUTPUT_DIR / "behavior-summary.csv", summaries)
    (OUTPUT_DIR / "behavior-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "run-config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
