"""Exhaustively localize GPT-2 Medium heads for Experiment 002."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from circuit_transport.experiment import final_outcomes, tokens_from_records
from circuit_transport.experiment_002 import (
    SITES,
    choose_random_heads,
    localization_similarity,
    make_layer_sweep_hook,
    positions_from_records,
    result_cache_filter,
    result_hook_name,
    select_top_heads,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-medium"
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
MANIFEST_FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-001-freeze.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment-002" / "localization"
SELECTION_COUNT = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--limit-layers", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def expanded_logit_difference(
    logits: torch.Tensor,
    records: Sequence[dict[str, Any]],
    *,
    copies: int,
) -> torch.Tensor:
    final_logits = logits[:, -1, :].float()
    io_ids = torch.tensor(
        [record["io_token_id"] for record in records], device=logits.device
    ).repeat_interleave(copies)
    subject_ids = torch.tensor(
        [record["subject_token_id"] for record in records], device=logits.device
    ).repeat_interleave(copies)
    rows = torch.arange(final_logits.shape[0], device=logits.device)
    return final_logits[rows, io_ids] - final_logits[rows, subject_ids]


def load_bridge() -> TransformerBridge:
    bridge = TransformerBridge.boot_transformers(
        str(MODEL_PATH), device=torch.device("cpu"), dtype=torch.float32
    )
    bridge.enable_compatibility_mode()
    bridge.set_use_attn_result(True)
    bridge.to(torch.device("cuda:0"))
    bridge.eval()
    return bridge


def serialize_heads(heads: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[layer, head] for layer, head in heads]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    manifest_freeze = json.loads(MANIFEST_FREEZE_PATH.read_text())
    if sha256(MANIFEST_PATH) != manifest_freeze["manifest_sha256"]:
        raise RuntimeError("Experiment 001 manifest hash does not match its freeze record")
    records = [
        json.loads(line)
        for line in MANIFEST_PATH.read_text().splitlines()
        if json.loads(line)["split"] == "exploratory"
    ]
    if args.limit_records is not None:
        records = records[: args.limit_records]
    if not records:
        raise ValueError("No exploratory records selected")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = load_bridge()
    n_layers = int(bridge.cfg.n_layers)
    n_heads = int(bridge.cfg.n_heads)
    if args.limit_layers is not None:
        n_layers = min(n_layers, args.limit_layers)
    is_full_run = len(records) == 80 and n_layers == int(bridge.cfg.n_layers)

    score_sums = {
        condition: {site: np.zeros((n_layers, n_heads), dtype=np.float64) for site in SITES}
        for condition in ("canonical", "controlled")
    }
    behavior_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        print(f"localization batch {start}:{start + len(batch)}", flush=True)
        for condition in ("canonical", "controlled"):
            clean_tokens = tokens_from_records(
                batch, condition, "clean", device=torch.device("cuda:0")
            )
            corrupt_tokens = tokens_from_records(
                batch, condition, "corrupt", device=torch.device("cuda:0")
            )
            with torch.inference_mode():
                clean_logits, clean_cache = bridge.run_with_cache(
                    clean_tokens, names_filter=result_cache_filter
                )
                corrupt_logits, corrupt_cache = bridge.run_with_cache(
                    corrupt_tokens, names_filter=result_cache_filter
                )
            clean_outcomes = final_outcomes(clean_logits.float(), batch)
            corrupt_outcomes = final_outcomes(corrupt_logits.float(), batch)
            for index, record in enumerate(batch):
                behavior_rows.append(
                    {
                        "example_id": record["example_id"],
                        "condition": condition,
                        "clean_probability_difference": float(
                            clean_outcomes["probability_difference"][index]
                        ),
                        "clean_logit_difference": float(clean_outcomes["logit_difference"][index]),
                        "clean_io_wins_pairwise": float(clean_outcomes["io_wins_pairwise"][index]),
                        "corrupt_probability_difference": float(
                            corrupt_outcomes["probability_difference"][index]
                        ),
                        "corrupt_logit_difference": float(
                            corrupt_outcomes["logit_difference"][index]
                        ),
                    }
                )

            corrupt_logit_difference = corrupt_outcomes["logit_difference"].to(
                torch.device("cuda:0")
            )
            for site, position_key in SITES.items():
                effects = np.empty((len(batch), n_layers, n_heads), dtype=np.float32)
                positions = positions_from_records(
                    batch, position_key, device=torch.device("cuda:0")
                )
                expanded_tokens = corrupt_tokens.repeat_interleave(n_heads, dim=0)
                for layer in range(n_layers):
                    clean_result = clean_cache[result_hook_name(layer)]
                    corrupt_result = corrupt_cache[result_hook_name(layer)]
                    rows = torch.arange(len(batch), device=clean_result.device)
                    delta = (
                        clean_result[rows, positions, :, :] - corrupt_result[rows, positions, :, :]
                    )
                    hook = make_layer_sweep_hook(delta, positions)
                    with (
                        torch.inference_mode(),
                        bridge.hooks(fwd_hooks=[(result_hook_name(layer), hook)]),
                    ):
                        patched_logits = bridge(expanded_tokens)
                    patched_logit_difference = expanded_logit_difference(
                        patched_logits, batch, copies=n_heads
                    ).reshape(len(batch), n_heads)
                    layer_effects = patched_logit_difference - corrupt_logit_difference[:, None]
                    effects[:, layer, :] = layer_effects.float().cpu().numpy()
                    score_sums[condition][site][layer, :] += effects[:, layer, :].sum(axis=0)
                    del patched_logits
                for index, record in enumerate(batch):
                    per_example_rows.append(
                        {
                            "example_id": record["example_id"],
                            "condition": condition,
                            "site": site,
                            "shape": [n_layers, n_heads],
                            "effects": effects[index].reshape(-1).tolist(),
                        }
                    )
            del clean_logits, corrupt_logits, clean_cache, corrupt_cache

    head_order = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    summary_rows: list[dict[str, Any]] = []
    score_maps: dict[str, dict[str, dict[tuple[int, int], float]]] = {
        condition: {} for condition in ("canonical", "controlled")
    }
    for condition in ("canonical", "controlled"):
        for site in SITES:
            means = score_sums[condition][site] / len(records)
            score_map = {head: float(means[head]) for head in head_order}
            score_maps[condition][site] = score_map
            for layer, head in head_order:
                summary_rows.append(
                    {
                        "condition": condition,
                        "site": site,
                        "layer": layer,
                        "head": head,
                        "mean_logit_effect": float(means[layer, head]),
                    }
                )

    selection: dict[str, Any] = {
        "status": "complete" if is_full_run else "dry-run; selection is not frozen",
        "selection_count": SELECTION_COUNT,
        "head_order": serialize_heads(head_order),
        "conditions": {},
        "similarity": {},
    }
    if is_full_run:
        for site in SITES:
            selection["similarity"][site] = localization_similarity(
                score_maps["canonical"][site], score_maps["controlled"][site]
            )
            selected_by_condition = {
                condition: select_top_heads(score_maps[condition][site], count=SELECTION_COUNT)
                for condition in ("canonical", "controlled")
            }
            excluded = tuple(
                set(selected_by_condition["canonical"]) | set(selected_by_condition["controlled"])
            )
            for condition, selected in selected_by_condition.items():
                selection["conditions"].setdefault(condition, {})[site] = {
                    "selected_heads": serialize_heads(selected),
                    "random_heads": serialize_heads(
                        choose_random_heads(selected, excluded=excluded, n_heads=n_heads)
                    ),
                }

    configuration = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": selection["status"],
        "model": "gpt2-medium",
        "model_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "dtype": "float32",
        "float_accumulation": "float32",
        "compatibility_processing_device": "cpu",
        "inference_device": "cuda:0",
        "manifest_sha256": manifest_freeze["manifest_sha256"],
        "records": len(records),
        "layers": n_layers,
        "heads_per_layer": n_heads,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "transformer_lens": version("transformer-lens"),
        "transformers": version("transformers"),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "script_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(PROJECT_ROOT / "src" / "circuit_transport" / "experiment_002.py"),
    }

    def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )

    write_jsonl(output_dir / "head-effects-per-example.jsonl", per_example_rows)
    write_jsonl(output_dir / "head-effects-summary.jsonl", summary_rows)
    write_jsonl(output_dir / "behavior-per-example.jsonl", behavior_rows)
    (output_dir / "selected-heads.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run-config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
