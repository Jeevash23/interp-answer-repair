"""Run the frozen GPT-2 Small causal-transport pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from circuit_transport.experiment import (
    CONTROLS,
    HEAD_GROUPS,
    RANDOM_HEAD_GROUPS,
    build_patch_hooks,
    cache_filter,
    early_local_projection,
    final_outcomes,
    scalar_local_outcome,
    tokens_from_records,
)
from circuit_transport.metrics import paired_bootstrap_transport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-small"
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-001-freeze.json"
BOOTSTRAP_SEED = 20_260_831
BOOTSTRAP_SAMPLES = 2_000
RANDOM_DIRECTION_SEED = 94_117
TRANSPORT_PAIRS = (
    ("canonical", "canonical"),
    ("controlled", "controlled"),
    ("canonical", "controlled"),
    ("controlled", "canonical"),
)
CORRUPTION_KEYS = {
    "primary": "corrupt",
    "alternative": "alternative_corrupt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("exploratory", "development", "test"),
        default=["exploratory"],
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--corruptions", nargs="+", choices=tuple(CORRUPTION_KEYS), default=list(CORRUPTION_KEYS)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: limit each split.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_records(splits: list[str], limit: int | None) -> tuple[list[dict[str, Any]], dict]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    actual_hash = file_sha256(MANIFEST_PATH)
    if actual_hash != freeze["manifest_sha256"]:
        raise RuntimeError(
            f"Manifest hash changed: expected {freeze['manifest_sha256']}, got {actual_hash}"
        )
    all_records = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines()]
    selected: list[dict[str, Any]] = []
    for split in splits:
        split_records = [record for record in all_records if record["split"] == split]
        if limit is not None:
            split_records = split_records[:limit]
        selected.extend(split_records)
    return selected, freeze


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numpy_values(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def mean_interval(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    array = numpy_values(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_SAMPLES, array.size))
    boot = array[indices].mean(axis=1)
    low, high = np.percentile(boot, (2.5, 97.5))
    return float(low), float(high)


def summarize_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["condition"], row["corruption"])].append(row)

    summaries = []
    for (split, condition, corruption), group in sorted(groups.items()):
        clean_probability = [row["clean_probability_difference"] for row in group]
        clean_logit = [row["clean_logit_difference"] for row in group]
        probability_low, probability_high = mean_interval(clean_probability)
        logit_low, logit_high = mean_interval(clean_logit)
        summaries.append(
            {
                "split": split,
                "condition": condition,
                "corruption": corruption,
                "n": len(group),
                "clean_pairwise_accuracy": float(
                    np.mean([row["clean_io_wins_pairwise"] for row in group])
                ),
                "clean_probability_difference": float(np.mean(clean_probability)),
                "clean_probability_difference_ci_low": probability_low,
                "clean_probability_difference_ci_high": probability_high,
                "clean_logit_difference": float(np.mean(clean_logit)),
                "clean_logit_difference_ci_low": logit_low,
                "clean_logit_difference_ci_high": logit_high,
                "corrupt_probability_difference": float(
                    np.mean([row["corrupt_probability_difference"] for row in group])
                ),
                "corrupt_logit_difference": float(
                    np.mean([row["corrupt_logit_difference"] for row in group])
                ),
                "behavioral_floor_passed": bool(
                    np.mean([row["clean_io_wins_pairwise"] for row in group]) >= 0.70
                    and np.mean(clean_probability) > 0
                ),
            }
        )
    return summaries


def summarize_patches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "split",
        "corruption",
        "stage",
        "source_condition",
        "target_condition",
        "control",
    )
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items()):
        identity = dict(zip(keys, group_key, strict=True))
        for metric, threshold in (("probability_difference", 0.01), ("logit_difference", 0.1)):
            summary = paired_bootstrap_transport(
                [row[f"target_clean_{metric}"] for row in group],
                [row[f"target_corrupt_{metric}"] for row in group],
                [row[f"target_patched_{metric}"] for row in group],
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
                minimum_denominator=threshold,
            )
            summaries.append({**identity, "metric": metric, **summary})
        local_summary = paired_bootstrap_transport(
            [row["local_clean"] for row in group],
            [row["local_corrupt"] for row in group],
            [row["local_patched"] for row in group],
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
            minimum_denominator=1e-8,
        )
        summaries.append({**identity, "metric": "local", **local_summary})
    return summaries


def append_behavior_rows(
    destination: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    condition: str,
    corruption: str,
    clean: dict[str, torch.Tensor],
    corrupt: dict[str, torch.Tensor],
) -> None:
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "example_id": record["example_id"],
            "split": record["split"],
            "template_id": record["template_id"],
            "pattern": record["pattern"],
            "condition": condition,
            "corruption": corruption,
        }
        for prefix, outcomes in (("clean", clean), ("corrupt", corrupt)):
            for metric, values in outcomes.items():
                row[f"{prefix}_{metric}"] = float(values[index])
        destination.append(row)


def append_patch_rows(
    destination: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    corruption: str,
    stage: str,
    source_condition: str,
    target_condition: str,
    control: str,
    target_clean: dict[str, torch.Tensor],
    target_corrupt: dict[str, torch.Tensor],
    target_patched: dict[str, torch.Tensor],
    local_clean: torch.Tensor,
    local_corrupt: torch.Tensor,
    local_patched: torch.Tensor,
) -> None:
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "example_id": record["example_id"],
            "split": record["split"],
            "template_id": record["template_id"],
            "pattern": record["pattern"],
            "corruption": corruption,
            "stage": stage,
            "source_condition": source_condition,
            "target_condition": target_condition,
            "control": control,
            "local_clean": float(local_clean[index]),
            "local_corrupt": float(local_corrupt[index]),
            "local_patched": float(local_patched[index]),
        }
        for prefix, outcomes in (
            ("target_clean", target_clean),
            ("target_corrupt", target_corrupt),
            ("target_patched", target_patched),
        ):
            for metric in ("probability_difference", "logit_difference"):
                row[f"{prefix}_{metric}"] = float(outcomes[metric][index])
        destination.append(row)


def run_batch(
    bridge: TransformerBridge,
    records: list[dict[str, Any]],
    *,
    corruption: str,
    batch_offset: int,
    behavior_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
) -> None:
    device = torch.device("cuda:0")
    corrupt_variant = CORRUPTION_KEYS[corruption]
    runs: dict[tuple[str, str], tuple[dict[str, torch.Tensor], Any]] = {}
    with torch.inference_mode():
        for condition in ("canonical", "controlled"):
            for variant in ("clean", corrupt_variant):
                tokens = tokens_from_records(records, condition, variant, device=device)
                logits, cache = bridge.run_with_cache(tokens, names_filter=cache_filter)
                runs[(condition, variant)] = (final_outcomes(logits, records), cache)

    for condition in ("canonical", "controlled"):
        append_behavior_rows(
            behavior_rows,
            records,
            condition=condition,
            corruption=corruption,
            clean=runs[(condition, "clean")][0],
            corrupt=runs[(condition, corrupt_variant)][0],
        )

    patch_specs = [
        (stage, source, target, "correct")
        for stage in HEAD_GROUPS
        for source, target in TRANSPORT_PAIRS
    ]
    patch_specs.extend(
        (stage, "canonical", "controlled", control)
        for stage in HEAD_GROUPS
        for control in CONTROLS
        if control != "correct"
    )

    for stage, source, target, control in patch_specs:
        source_clean_cache = runs[(source, "clean")][1]
        source_corrupt_cache = runs[(source, corrupt_variant)][1]
        target_clean_outcomes, target_clean_cache = runs[(target, "clean")]
        target_corrupt_outcomes, target_corrupt_cache = runs[(target, corrupt_variant)]
        hooks = build_patch_hooks(
            source_clean_cache,
            source_corrupt_cache,
            records,
            stage=stage,
            control=control,
            random_seed=RANDOM_DIRECTION_SEED + batch_offset,
        )
        target_tokens = tokens_from_records(records, target, corrupt_variant, device=device)
        with torch.inference_mode(), bridge.hooks(fwd_hooks=hooks):
            patched_logits, patched_cache = bridge.run_with_cache(
                target_tokens, names_filter=cache_filter
            )
        target_patched_outcomes = final_outcomes(patched_logits, records)

        if stage == "duplicate_induction":
            local_clean = torch.ones(len(records))
            local_corrupt = torch.zeros(len(records))
            local_patched = (
                early_local_projection(
                    target_clean_cache, target_corrupt_cache, patched_cache, records
                )
                .detach()
                .cpu()
            )
        else:
            local_clean = (
                scalar_local_outcome(stage, target_clean_cache, records, unembed=bridge.W_U)
                .detach()
                .cpu()
            )
            local_corrupt = (
                scalar_local_outcome(stage, target_corrupt_cache, records, unembed=bridge.W_U)
                .detach()
                .cpu()
            )
            local_patched = (
                scalar_local_outcome(stage, patched_cache, records, unembed=bridge.W_U)
                .detach()
                .cpu()
            )

        append_patch_rows(
            patch_rows,
            records,
            corruption=corruption,
            stage=stage,
            source_condition=source,
            target_condition=target,
            control=control,
            target_clean=target_clean_outcomes,
            target_corrupt=target_corrupt_outcomes,
            target_patched=target_patched_outcomes,
            local_clean=local_clean,
            local_corrupt=local_corrupt,
            local_patched=local_patched,
        )
        del patched_logits, patched_cache, target_patched_outcomes

    del runs
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("batch size must be an even number of at least four")
    records, freeze = load_frozen_records(args.splits, args.limit)
    if args.limit is None and any(
        sum(row["split"] == split for row in records) != 80 for split in args.splits
    ):
        raise RuntimeError("A requested frozen split does not contain exactly 80 examples")

    torch.manual_seed(RANDOM_DIRECTION_SEED)
    torch.cuda.manual_seed_all(RANDOM_DIRECTION_SEED)
    bridge = TransformerBridge.boot_transformers(
        str(MODEL_PATH), device="cuda", dtype=torch.float32
    )
    bridge.enable_compatibility_mode()
    bridge.set_use_attn_result(True)
    bridge.eval()

    behavior_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC)
    for split in args.splits:
        split_records = [record for record in records if record["split"] == split]
        print(f"Running split={split}, examples={len(split_records)}", flush=True)
        for corruption in args.corruptions:
            print(f"  corruption={corruption}", flush=True)
            for start in range(0, len(split_records), args.batch_size):
                batch = split_records[start : start + args.batch_size]
                print(f"    batch={start}:{start + len(batch)}", flush=True)
                run_batch(
                    bridge,
                    batch,
                    corruption=corruption,
                    batch_offset=start,
                    behavior_rows=behavior_rows,
                    patch_rows=patch_rows,
                )

    behavior_summary = summarize_behavior(behavior_rows)
    patch_summary = summarize_patches(patch_rows)
    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / "behavior-per-example.jsonl", behavior_rows)
    write_jsonl(output_dir / "patches-per-example.jsonl", patch_rows)
    write_csv(output_dir / "behavior-summary.csv", behavior_summary)
    write_csv(output_dir / "transport-summary.csv", patch_summary)
    (output_dir / "behavior-summary.json").write_text(
        json.dumps(behavior_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "transport-summary.json").write_text(
        json.dumps(patch_summary, indent=2) + "\n", encoding="utf-8"
    )

    configuration = {
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "splits": args.splits,
        "corruptions": args.corruptions,
        "batch_size": args.batch_size,
        "debug_limit": args.limit,
        "manifest": freeze,
        "manifest_sha256_verified": file_sha256(MANIFEST_PATH),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_revision": freeze["model_repo_revision"],
        "dtype": "float32",
        "device": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "packages": {
            "torch": torch.__version__,
            "transformer-lens": version("transformer-lens"),
            "transformers": version("transformers"),
        },
        "head_groups": HEAD_GROUPS,
        "random_head_groups": RANDOM_HEAD_GROUPS,
        "random_direction_seed": RANDOM_DIRECTION_SEED,
        "mismatched_role_row_shift": 2,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "source_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path)
            for path in (
                "scripts/run_experiment_001.py",
                "src/circuit_transport/dataset.py",
                "src/circuit_transport/experiment.py",
                "src/circuit_transport/metrics.py",
            )
        },
        "row_counts": {"behavior": len(behavior_rows), "patches": len(patch_rows)},
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
