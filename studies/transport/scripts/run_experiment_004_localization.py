"""Exhaustively localize Pythia heads on Experiment 004 development templates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from circuit_transport.experiment import final_outcomes, tokens_from_records
from circuit_transport.experiment_002 import (
    SITES,
    choose_random_heads,
    localization_similarity,
    select_top_heads,
)
from circuit_transport.experiment_004 import condition_positions
from circuit_transport.pythia import load_pythia_for_hooks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-004-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-freeze.json"
AMENDMENT_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-001.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "experiment-004" / "localization"
SELECTION_COUNT = 8
FROZEN_BATCH_SIZE = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--limit-layers", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def expanded_logit_difference(
    logits: torch.Tensor, records: Sequence[dict[str, Any]], *, copies: int
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


def serialize_heads(heads: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[layer, head] for layer, head in heads]


def z_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


def z_cache_filter(name: str) -> bool:
    return name.endswith("attn.hook_z")


def make_layer_z_sweep_hook(delta: torch.Tensor, positions: torch.Tensor) -> Any:
    """Patch one head's z value per example-major, head-minor expanded row."""
    if delta.ndim != 3:
        raise ValueError("delta must have shape [batch, heads, d_head]")

    def patch(value: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        batch, heads, _ = delta.shape
        if value.shape[0] != batch * heads or value.shape[2] != heads:
            raise ValueError("expanded target batch does not match the localization delta")
        patched = value.clone()
        rows = torch.arange(batch * heads, device=value.device)
        example_rows = torch.div(rows, heads, rounding_mode="floor")
        head_rows = rows.remainder(heads)
        patched[rows, positions[example_rows], head_rows, :] += delta[example_rows, head_rows, :]
        return patched

    return patch


def validate_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if sha256(FREEZE_PATH) != amendment["previous_freeze_sha256"]:
        raise RuntimeError("Experiment 004 amendment does not match the original freeze")
    expected = dict(freeze["source_files_sha256"])
    expected.update(amendment["amended_source_files_sha256"])
    for relative in (
        "data/experiment-004-manifest.jsonl",
        "docs/EXPERIMENT-004.md",
        "src/circuit_transport/pythia.py",
        "src/circuit_transport/experiment_004.py",
        "scripts/run_experiment_004_localization.py",
    ):
        if sha256(PROJECT_ROOT / relative) != expected[relative]:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    return freeze


def main() -> None:
    args = parse_args()
    if args.batch_size != FROZEN_BATCH_SIZE:
        raise ValueError(f"Experiment 004 freezes batch size {FROZEN_BATCH_SIZE}")
    freeze = validate_freeze()
    records = [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == "development"
    ]
    if args.limit_records is not None:
        records = records[: args.limit_records]
    if not records:
        raise ValueError("No development records selected")

    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_template[record["template_id"]].append(record)
    if any(len(group) % 2 for group in by_template.values()):
        raise ValueError("Every selected template must preserve ABBA/BABA pairs")

    model, hf_model, _ = load_pythia_for_hooks()
    del hf_model
    gc.collect()
    model.set_use_attn_result(False)
    device = torch.device("cuda:0")
    model.to(device)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    if args.limit_layers is not None:
        n_layers = min(n_layers, args.limit_layers)
    is_full = len(records) == 160 and n_layers == int(model.cfg.n_layers)

    score_sums = {
        condition: {site: np.zeros((n_layers, n_heads), dtype=np.float64) for site in SITES}
        for condition in ("canonical", "controlled")
    }
    behavior_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()

    for template, template_records in sorted(by_template.items()):
        for start in range(0, len(template_records), args.batch_size):
            batch = template_records[start : start + args.batch_size]
            print(f"localization {template} {start}:{start + len(batch)}", flush=True)
            for condition in ("canonical", "controlled"):
                clean_tokens = tokens_from_records(batch, condition, "clean", device=device)
                corrupt_tokens = tokens_from_records(batch, condition, "corrupt", device=device)
                with torch.inference_mode():
                    clean_logits, clean_cache = model.run_with_cache(
                        clean_tokens, names_filter=z_cache_filter
                    )
                    corrupt_logits, corrupt_cache = model.run_with_cache(
                        corrupt_tokens, names_filter=z_cache_filter
                    )
                clean_outcomes = final_outcomes(clean_logits.float(), batch)
                corrupt_outcomes = final_outcomes(corrupt_logits.float(), batch)
                for index, record in enumerate(batch):
                    behavior_rows.append(
                        {
                            "example_id": record["example_id"],
                            "template_id": record["template_id"],
                            "pattern": record["pattern"],
                            "condition": condition,
                            "clean_probability_difference": float(
                                clean_outcomes["probability_difference"][index]
                            ),
                            "clean_logit_difference": float(
                                clean_outcomes["logit_difference"][index]
                            ),
                            "clean_io_wins_pairwise": float(
                                clean_outcomes["io_wins_pairwise"][index]
                            ),
                            "corrupt_probability_difference": float(
                                corrupt_outcomes["probability_difference"][index]
                            ),
                            "corrupt_logit_difference": float(
                                corrupt_outcomes["logit_difference"][index]
                            ),
                        }
                    )

                corrupt_difference = corrupt_outcomes["logit_difference"].to(device)
                expanded_tokens = corrupt_tokens.repeat_interleave(n_heads, dim=0)
                for site, position_key in SITES.items():
                    effects = np.empty((len(batch), n_layers, n_heads), dtype=np.float32)
                    positions = condition_positions(batch, condition, position_key, device=device)
                    rows = torch.arange(len(batch), device=device)
                    for layer in range(n_layers):
                        clean_result = clean_cache[z_hook_name(layer)]
                        corrupt_result = corrupt_cache[z_hook_name(layer)]
                        delta = (
                            clean_result[rows, positions, :, :]
                            - corrupt_result[rows, positions, :, :]
                        )
                        hook = make_layer_z_sweep_hook(delta, positions)
                        with (
                            torch.inference_mode(),
                            model.hooks(fwd_hooks=[(z_hook_name(layer), hook)]),
                        ):
                            patched_logits = model(expanded_tokens, return_type="logits")
                        patched_difference = expanded_logit_difference(
                            patched_logits, batch, copies=n_heads
                        ).reshape(len(batch), n_heads)
                        layer_effects = patched_difference - corrupt_difference[:, None]
                        effects[:, layer, :] = layer_effects.float().cpu().numpy()
                        score_sums[condition][site][layer, :] += effects[:, layer, :].sum(axis=0)
                        del patched_logits
                    for index, record in enumerate(batch):
                        per_example_rows.append(
                            {
                                "example_id": record["example_id"],
                                "template_id": record["template_id"],
                                "pattern": record["pattern"],
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
        "status": "complete" if is_full else "dry-run; selection is not frozen",
        "selection_count": SELECTION_COUNT,
        "positive_only": True,
        "head_order": serialize_heads(head_order),
        "conditions": {},
        "similarity": {},
    }
    if is_full:
        for site in SITES:
            selection["similarity"][site] = localization_similarity(
                score_maps["canonical"][site], score_maps["controlled"][site]
            )
            selected_by_condition = {
                condition: select_top_heads(
                    score_maps[condition][site], count=SELECTION_COUNT, positive_only=True
                )
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(output_dir / "head-effects-per-example.jsonl", per_example_rows)
    write_jsonl(output_dir / "head-effects-summary.jsonl", summary_rows)
    write_jsonl(output_dir / "behavior-per-example.jsonl", behavior_rows)
    (output_dir / "selected-heads.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    configuration = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": selection["status"],
        "model": "EleutherAI/pythia-410m-deduped",
        "model_revision": freeze["model_revision"],
        "dtype": "float32",
        "inference_device": "cuda:0",
        "records": len(records),
        "templates": len(by_template),
        "layers": n_layers,
        "heads_per_layer": n_heads,
        "batch_size": args.batch_size,
        "manifest_sha256": sha256(MANIFEST_PATH),
        "script_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(PROJECT_ROOT / "src" / "circuit_transport" / "experiment_004.py"),
        "torch": torch.__version__,
        "transformer_lens": version("transformer-lens"),
        "transformers": version("transformers"),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
