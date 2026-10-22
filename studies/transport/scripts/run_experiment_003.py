"""Run frozen GPT-2 Medium transport on Experiment 003 template families."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from run_experiment_002 import (
    load_bridge,
    run_batch,
    summarize_behavior,
    summarize_patches,
    write_csv,
    write_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-003-manifest.jsonl"
MANIFEST_FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-003-data-freeze.json"
SELECTION_PATH = (
    PROJECT_ROOT / "results" / "experiment-002" / "localization" / "selected-heads.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "experiment-003"
FROZEN_BATCH_SIZE = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != FROZEN_BATCH_SIZE:
        raise ValueError(f"Experiment 003 freezes batch size {FROZEN_BATCH_SIZE}")
    manifest_freeze = json.loads(MANIFEST_FREEZE_PATH.read_text(encoding="utf-8"))
    if sha256(MANIFEST_PATH) != manifest_freeze["manifest_sha256"]:
        raise RuntimeError("Experiment 003 manifest hash does not match its freeze record")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection["status"] != "complete":
        raise RuntimeError("Experiment 002 head selection is not complete")
    records = [
        record
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))["split"] == args.split
    ]
    if args.limit_records is not None:
        records = records[: args.limit_records]
    if not records or len(records) % 2:
        raise ValueError("Selected records must preserve adjacent ABBA/BABA pairs")

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = load_bridge()
    behavior_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        print(f"primary batch {start}:{start + len(batch)} of {len(records)}", flush=True)
        run_batch(
            bridge,
            batch,
            selection=selection,
            corruption="primary",
            batch_offset=start,
            behavior_rows=behavior_rows,
            patch_rows=patch_rows,
        )

    behavior_summary = summarize_behavior(behavior_rows)
    transport_summary = summarize_patches(patch_rows)
    imported_runner = PROJECT_ROOT / "scripts" / "run_experiment_002.py"
    configuration = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if args.limit_records is None else "dry-run",
        "experiment": "003",
        "model": "gpt2-medium",
        "model_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "dtype": "float32",
        "compatibility_processing_device": "cpu",
        "inference_device": "cuda:0",
        "split": args.split,
        "records": len(records),
        "templates": len({record["template_id"] for record in records}),
        "batch_size": args.batch_size,
        "corruptions": {"primary": "corrupt"},
        "manifest_sha256": manifest_freeze["manifest_sha256"],
        "selection_sha256": sha256(SELECTION_PATH),
        "script_sha256": sha256(Path(__file__)),
        "imported_runner_sha256": sha256(imported_runner),
        "intervention_module_sha256": sha256(
            PROJECT_ROOT / "src" / "circuit_transport" / "experiment_002.py"
        ),
        "torch": torch.__version__,
        "transformer_lens": version("transformer-lens"),
        "transformers": version("transformers"),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    write_jsonl(output_dir / "behavior-per-example.jsonl", behavior_rows)
    write_jsonl(output_dir / "patches-per-example.jsonl", patch_rows)
    write_csv(output_dir / "behavior-summary.csv", behavior_summary)
    write_csv(output_dir / "transport-summary.csv", transport_summary)
    (output_dir / "behavior-summary.json").write_text(
        json.dumps(behavior_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "transport-summary.json").write_text(
        json.dumps(transport_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run-config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"behavior": behavior_summary, "config": configuration}, indent=2))
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
