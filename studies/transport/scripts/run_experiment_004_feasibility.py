"""Evaluate frozen Pythia hook fidelity and multi-template IOI behavior."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from circuit_transport.pythia import load_pythia_for_hooks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-004-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-004-feasibility-freeze.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "experiment-004" / "feasibility"
FROZEN_LOGIT_TOLERANCE = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def evaluate_rows(
    model: Any, records: list[dict[str, Any]], condition: str, device: torch.device
) -> list[dict[str, Any]]:
    tokens = torch.tensor(
        [record[condition]["clean_token_ids"] for record in records], device=device
    )
    with torch.inference_mode():
        logits = model(tokens, return_type="logits").float()[:, -1, :]
    probabilities = logits.softmax(dim=-1)
    rows = torch.arange(len(records), device=device)
    io_ids = torch.tensor([record["io_token_id"] for record in records], device=device)
    subject_ids = torch.tensor([record["subject_token_id"] for record in records], device=device)
    io_logits = logits[rows, io_ids]
    subject_logits = logits[rows, subject_ids]
    io_probabilities = probabilities[rows, io_ids]
    subject_probabilities = probabilities[rows, subject_ids]
    output = []
    for index, record in enumerate(records):
        output.append(
            {
                "example_id": record["example_id"],
                "split": record["split"],
                "template_id": record["template_id"],
                "pattern": record["pattern"],
                "condition": condition,
                "sequence_length": len(record[condition]["clean_token_ids"]),
                "io_wins_pairwise": bool(io_logits[index] > subject_logits[index]),
                "logit_difference": float(io_logits[index] - subject_logits[index]),
                "probability_difference": float(
                    io_probabilities[index] - subject_probabilities[index]
                ),
            }
        )
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["condition"], row["template_id"])].append(row)
    templates = []
    for (split, condition, template), group in sorted(grouped.items()):
        accuracy = float(np.mean([row["io_wins_pairwise"] for row in group]))
        probability = float(np.mean([row["probability_difference"] for row in group]))
        templates.append(
            {
                "split": split,
                "condition": condition,
                "template_id": template,
                "n": len(group),
                "accuracy": accuracy,
                "probability_difference": probability,
                "logit_difference": float(np.mean([row["logit_difference"] for row in group])),
                "gate_passed": bool(accuracy >= 0.70 and probability > 0),
            }
        )
    aggregate = []
    for split in ("development", "test"):
        for condition in ("canonical", "controlled"):
            group = [row for row in rows if row["split"] == split and row["condition"] == condition]
            accuracy = float(np.mean([row["io_wins_pairwise"] for row in group]))
            probability = float(np.mean([row["probability_difference"] for row in group]))
            template_group = [
                row for row in templates if row["split"] == split and row["condition"] == condition
            ]
            aggregate.append(
                {
                    "split": split,
                    "condition": condition,
                    "n": len(group),
                    "accuracy": accuracy,
                    "probability_difference": probability,
                    "logit_difference": float(np.mean([row["logit_difference"] for row in group])),
                    "gate_passed": bool(accuracy >= 0.70 and probability > 0),
                    "templates_passed": sum(row["gate_passed"] for row in template_group),
                    "templates_total": len(template_group),
                }
            )
    return {"aggregate": aggregate, "templates": templates}


def main() -> None:
    args = parse_args()
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if sha256(MANIFEST_PATH) != freeze["manifest_sha256"]:
        raise RuntimeError("Experiment 004 feasibility manifest hash does not match its freeze")
    records = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines()]

    model, hf_model, _ = load_pythia_for_hooks()
    smoke_tokens = torch.tensor([records[0]["canonical"]["clean_token_ids"]])
    with torch.inference_mode():
        hf_logits = hf_model(smoke_tokens).logits.float()
        hooked_logits, cache = model.run_with_cache(
            smoke_tokens, names_filter=lambda name: name.endswith("attn.hook_result")
        )
    maximum_logit_error = float((hooked_logits.float() - hf_logits).abs().max())
    hook_shapes = {name: list(value.shape) for name, value in cache.items()}
    fidelity = {
        "maximum_absolute_logit_error": maximum_logit_error,
        "tolerance": FROZEN_LOGIT_TOLERANCE,
        "logit_fidelity_passed": maximum_logit_error <= FROZEN_LOGIT_TOLERANCE,
        "attention_result_hooks": len(hook_shapes),
        "hook_shapes": hook_shapes,
        "hook_shape_passed": len(hook_shapes) == 24
        and all(shape == [1, len(smoke_tokens[0]), 16, 1024] for shape in hook_shapes.values()),
    }
    del hf_model, cache, hf_logits, hooked_logits
    gc.collect()

    device = torch.device("cuda:0")
    model.to(device)
    behavior_rows = []
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_template[record["template_id"]].append(record)
    for template, template_records in sorted(by_template.items()):
        for condition in ("canonical", "controlled"):
            print(f"{template}: {condition}", flush=True)
            behavior_rows.extend(evaluate_rows(model, template_records, condition, device))

    behavior = summarize(behavior_rows)
    aggregate_passed = all(row["gate_passed"] for row in behavior["aggregate"])
    controlled_template_passed = all(
        row["templates_passed"] >= 6
        for row in behavior["aggregate"]
        if row["condition"] == "controlled"
    )
    advance = bool(
        aggregate_passed
        and controlled_template_passed
        and fidelity["logit_fidelity_passed"]
        and fidelity["hook_shape_passed"]
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "behavior-per-example.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in behavior_rows),
        encoding="utf-8",
    )
    (output_dir / "behavior-summary.json").write_text(
        json.dumps(behavior, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    configuration = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "model": "EleutherAI/pythia-410m-deduped",
        "model_revision": "c4fc8d586d62df497f1f9b69d66d3ca419992d3e",
        "dtype": "float32",
        "inference_device": "cuda:0",
        "records": len(records),
        "templates": len(by_template),
        "manifest_sha256": freeze["manifest_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = {
        "fidelity": fidelity,
        "behavior": behavior,
        "advance_to_circuit_study": advance,
        "decision": "advance" if advance else "stop at behavioral or hook boundary",
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
