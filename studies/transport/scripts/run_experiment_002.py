"""Run frozen GPT-2 Medium group transport for Experiment 002."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from circuit_transport.experiment import final_outcomes, tokens_from_records
from circuit_transport.experiment_002 import (
    CONTROLS,
    SITES,
    Head,
    build_group_patch_hooks,
    end_direct_contribution,
    result_cache_filter,
    s2_local_projection,
)
from circuit_transport.metrics import paired_bootstrap_transport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gpt2-medium"
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl"
MANIFEST_FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-001-freeze.json"
SELECTION_PATH = (
    PROJECT_ROOT / "results" / "experiment-002" / "localization" / "selected-heads.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment-002" / "development"
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20_260_901
RANDOM_DIRECTION_SEED = 941_170_02
TRANSPORT_PAIRS = (
    ("canonical", "canonical"),
    ("controlled", "controlled"),
    ("canonical", "controlled"),
    ("controlled", "canonical"),
)
CORRUPTIONS = {"primary": "corrupt", "alternative": "alternative_corrupt"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("exploratory", "development", "test"),
        default=["development"],
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_bridge() -> TransformerBridge:
    bridge = TransformerBridge.boot_transformers(
        str(MODEL_PATH), device=torch.device("cpu"), dtype=torch.float32
    )
    bridge.enable_compatibility_mode()
    bridge.set_use_attn_result(True)
    bridge.to(torch.device("cuda:0"))
    bridge.eval()
    return bridge


def parse_heads(values: Sequence[Sequence[int]]) -> tuple[Head, ...]:
    return tuple((int(layer), int(head)) for layer, head in values)


def selected_heads(selection: Mapping[str, Any], condition: str, site: str) -> tuple[Head, ...]:
    return parse_heads(selection["conditions"][condition][site]["selected_heads"])


def random_heads(selection: Mapping[str, Any], condition: str, site: str) -> tuple[Head, ...]:
    return parse_heads(selection["conditions"][condition][site]["random_heads"])


def append_behavior_rows(
    destination: list[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    condition: str,
    corruption: str,
    clean: Mapping[str, torch.Tensor],
    corrupt: Mapping[str, torch.Tensor],
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
    records: Sequence[dict[str, Any]],
    *,
    corruption: str,
    site: str,
    source_condition: str,
    target_condition: str,
    control: str,
    patched_heads: Sequence[Head],
    target_clean: Mapping[str, torch.Tensor],
    target_corrupt: Mapping[str, torch.Tensor],
    target_patched: Mapping[str, torch.Tensor],
    local_clean: torch.Tensor,
    local_corrupt: torch.Tensor,
    local_patched: torch.Tensor,
) -> None:
    serialized_heads = [[layer, head] for layer, head in patched_heads]
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "example_id": record["example_id"],
            "split": record["split"],
            "template_id": record["template_id"],
            "pattern": record["pattern"],
            "corruption": corruption,
            "site": site,
            "source_condition": source_condition,
            "target_condition": target_condition,
            "control": control,
            "patched_heads": serialized_heads,
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


def summarize_behavior(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["condition"], row["corruption"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (split, condition, corruption), group in sorted(groups.items()):
        accuracy = float(np.mean([row["clean_io_wins_pairwise"] for row in group]))
        probability = float(np.mean([row["clean_probability_difference"] for row in group]))
        summaries.append(
            {
                "split": split,
                "condition": condition,
                "corruption": corruption,
                "n": len(group),
                "clean_pairwise_accuracy": accuracy,
                "clean_probability_difference": probability,
                "clean_logit_difference": float(
                    np.mean([row["clean_logit_difference"] for row in group])
                ),
                "corrupt_probability_difference": float(
                    np.mean([row["corrupt_probability_difference"] for row in group])
                ),
                "corrupt_logit_difference": float(
                    np.mean([row["corrupt_logit_difference"] for row in group])
                ),
                "behavioral_floor_passed": bool(accuracy >= 0.70 and probability > 0),
            }
        )
    return summaries


def summarize_patches(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "split",
        "corruption",
        "site",
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
        local = paired_bootstrap_transport(
            [row["local_clean"] for row in group],
            [row["local_corrupt"] for row in group],
            [row["local_patched"] for row in group],
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
            minimum_denominator=1e-8,
        )
        summaries.append({**identity, "metric": "local", **local})
    return summaries


def run_batch(
    bridge: TransformerBridge,
    records: Sequence[dict[str, Any]],
    *,
    selection: Mapping[str, Any],
    corruption: str,
    batch_offset: int,
    behavior_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
) -> None:
    device = torch.device("cuda:0")
    corrupt_variant = CORRUPTIONS[corruption]
    runs: dict[tuple[str, str], tuple[dict[str, torch.Tensor], Any]] = {}
    with torch.inference_mode():
        for condition in ("canonical", "controlled"):
            for variant in ("clean", corrupt_variant):
                tokens = tokens_from_records(records, condition, variant, device=device)
                logits, cache = bridge.run_with_cache(tokens, names_filter=result_cache_filter)
                runs[(condition, variant)] = (final_outcomes(logits.float(), records), cache)

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
        (site, source, target, "correct") for site in SITES for source, target in TRANSPORT_PAIRS
    ]
    patch_specs.extend(
        (site, "canonical", "controlled", control)
        for site in SITES
        for control in CONTROLS
        if control != "correct"
    )
    for site, source, target, control in patch_specs:
        source_group = selected_heads(selection, source, site)
        random_group = random_heads(selection, source, site) if control == "random_head" else None
        hooks, active_heads = build_group_patch_hooks(
            runs[(source, "clean")][1],
            runs[(source, corrupt_variant)][1],
            records,
            heads=source_group,
            site=site,
            control=control,
            random_heads=random_group,
            random_seed=RANDOM_DIRECTION_SEED + batch_offset,
        )
        target_tokens = tokens_from_records(records, target, corrupt_variant, device=device)
        with torch.inference_mode(), bridge.hooks(fwd_hooks=hooks):
            patched_logits, patched_cache = bridge.run_with_cache(
                target_tokens, names_filter=result_cache_filter
            )
        target_patched = final_outcomes(patched_logits.float(), records)
        target_clean, target_clean_cache = runs[(target, "clean")]
        target_corrupt, target_corrupt_cache = runs[(target, corrupt_variant)]
        if site == "s2":
            local_clean = torch.ones(len(records))
            local_corrupt = torch.zeros(len(records))
            local_patched = (
                s2_local_projection(
                    target_clean_cache,
                    target_corrupt_cache,
                    patched_cache,
                    records,
                    target_end_heads=selected_heads(selection, target, "end"),
                )
                .detach()
                .cpu()
            )
        else:
            local_clean = (
                end_direct_contribution(
                    target_clean_cache,
                    records,
                    heads=active_heads,
                    unembed=bridge.W_U,
                )
                .detach()
                .cpu()
            )
            local_corrupt = (
                end_direct_contribution(
                    target_corrupt_cache,
                    records,
                    heads=active_heads,
                    unembed=bridge.W_U,
                )
                .detach()
                .cpu()
            )
            local_patched = (
                end_direct_contribution(
                    patched_cache,
                    records,
                    heads=active_heads,
                    unembed=bridge.W_U,
                )
                .detach()
                .cpu()
            )
        append_patch_rows(
            patch_rows,
            records,
            corruption=corruption,
            site=site,
            source_condition=source,
            target_condition=target,
            control=control,
            patched_heads=active_heads,
            target_clean=target_clean,
            target_corrupt=target_corrupt,
            target_patched=target_patched,
            local_clean=local_clean,
            local_corrupt=local_corrupt,
            local_patched=local_patched,
        )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size % 2:
        raise ValueError("batch size must be a positive even number so entity pairs remain aligned")
    manifest_freeze = json.loads(MANIFEST_FREEZE_PATH.read_text())
    if sha256(MANIFEST_PATH) != manifest_freeze["manifest_sha256"]:
        raise RuntimeError("Experiment 001 manifest hash does not match its freeze record")
    selection = json.loads(SELECTION_PATH.read_text())
    if selection["status"] != "complete":
        raise RuntimeError("Experiment 002 head selection is not complete")
    records = [
        record
        for line in MANIFEST_PATH.read_text().splitlines()
        if (record := json.loads(line))["split"] in set(args.splits)
    ]
    if args.limit_records is not None:
        records = records[: args.limit_records]
    if not records:
        raise ValueError("No records selected")
    if len(records) % 2:
        raise ValueError("record limit must preserve adjacent ABBA/BABA entity pairs")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = load_bridge()
    behavior_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()
    for corruption in CORRUPTIONS:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            print(f"{corruption} batch {start}:{start + len(batch)} of {len(records)}", flush=True)
            run_batch(
                bridge,
                batch,
                selection=selection,
                corruption=corruption,
                batch_offset=start,
                behavior_rows=behavior_rows,
                patch_rows=patch_rows,
            )

    behavior_summary = summarize_behavior(behavior_rows)
    transport_summary = summarize_patches(patch_rows)
    configuration = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if args.limit_records is None else "dry-run",
        "model": "gpt2-medium",
        "model_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "dtype": "float32",
        "compatibility_processing_device": "cpu",
        "inference_device": "cuda:0",
        "splits": args.splits,
        "records": len(records),
        "batch_size": args.batch_size,
        "corruptions": CORRUPTIONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "random_direction_seed": RANDOM_DIRECTION_SEED,
        "manifest_sha256": manifest_freeze["manifest_sha256"],
        "selection_sha256": sha256(SELECTION_PATH),
        "script_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(PROJECT_ROOT / "src" / "circuit_transport" / "experiment_002.py"),
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
