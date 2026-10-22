"""Run frozen Pythia cross-architecture transport for Experiment 004."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from run_experiment_002 import (
    append_behavior_rows,
    append_patch_rows,
    random_heads,
    selected_heads,
    summarize_behavior,
    summarize_patches,
    write_csv,
    write_jsonl,
)

from circuit_transport.experiment import final_outcomes, tokens_from_records
from circuit_transport.experiment_002 import CONTROLS, SITES, result_cache_filter
from circuit_transport.experiment_004 import (
    build_group_patch_hooks,
    end_direct_contribution,
    s2_local_projection,
)
from circuit_transport.pythia import load_pythia_for_hooks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "experiment-004-manifest.jsonl"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-freeze.json"
AMENDMENT_001_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-001.json"
AMENDMENT_002_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-002.json"
SELECTION_PATH = (
    PROJECT_ROOT / "results" / "experiment-004" / "localization" / "selected-heads.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "results" / "experiment-004"
FROZEN_BATCH_SIZE = 4
RANDOM_DIRECTION_SEED = 941_170_04
TRANSPORT_PAIRS = (
    ("canonical", "canonical"),
    ("controlled", "controlled"),
    ("canonical", "controlled"),
    ("controlled", "canonical"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def validate_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    amendment_001 = json.loads(AMENDMENT_001_PATH.read_text(encoding="utf-8"))
    amendment_002 = json.loads(AMENDMENT_002_PATH.read_text(encoding="utf-8"))
    if sha256(FREEZE_PATH) != amendment_001["previous_freeze_sha256"]:
        raise RuntimeError("Experiment 004 amendment 001 does not match the original freeze")
    if sha256(AMENDMENT_001_PATH) != amendment_002["previous_amendment_sha256"]:
        raise RuntimeError("Experiment 004 amendment 002 does not follow amendment 001")
    expected = dict(freeze["source_files_sha256"])
    expected.update(amendment_001["amended_source_files_sha256"])
    expected.update(amendment_002["amended_source_files_sha256"])
    for relative in (
        "data/experiment-004-manifest.jsonl",
        "docs/EXPERIMENT-004.md",
        "src/circuit_transport/pythia.py",
        "src/circuit_transport/experiment_004.py",
        "scripts/run_experiment_004.py",
    ):
        if sha256(PROJECT_ROOT / relative) != expected[relative]:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    return freeze


def run_batch(
    model: Any,
    records: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    batch_offset: int,
    behavior_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
) -> None:
    device = torch.device("cuda:0")
    runs: dict[tuple[str, str], tuple[dict[str, torch.Tensor], Any]] = {}
    with torch.inference_mode():
        for condition in ("canonical", "controlled"):
            for variant in ("clean", "corrupt"):
                tokens = tokens_from_records(records, condition, variant, device=device)
                logits, cache = model.run_with_cache(tokens, names_filter=result_cache_filter)
                runs[(condition, variant)] = (final_outcomes(logits.float(), records), cache)

    for condition in ("canonical", "controlled"):
        append_behavior_rows(
            behavior_rows,
            records,
            condition=condition,
            corruption="primary",
            clean=runs[(condition, "clean")][0],
            corrupt=runs[(condition, "corrupt")][0],
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
            runs[(source, "corrupt")][1],
            records,
            source_condition=source,
            target_condition=target,
            heads=source_group,
            site=site,
            control=control,
            random_heads=random_group,
            random_seed=RANDOM_DIRECTION_SEED + batch_offset,
        )
        target_tokens = tokens_from_records(records, target, "corrupt", device=device)
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            patched_logits, patched_cache = model.run_with_cache(
                target_tokens, names_filter=result_cache_filter
            )
        target_patched = final_outcomes(patched_logits.float(), records)
        target_clean, target_clean_cache = runs[(target, "clean")]
        target_corrupt, target_corrupt_cache = runs[(target, "corrupt")]
        if site == "s2":
            local_clean = torch.ones(len(records))
            local_corrupt = torch.zeros(len(records))
            local_patched = (
                s2_local_projection(
                    target_clean_cache,
                    target_corrupt_cache,
                    patched_cache,
                    records,
                    target_condition=target,
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
                    target_condition=target,
                    heads=active_heads,
                    unembed=model.W_U,
                )
                .detach()
                .cpu()
            )
            local_corrupt = (
                end_direct_contribution(
                    target_corrupt_cache,
                    records,
                    target_condition=target,
                    heads=active_heads,
                    unembed=model.W_U,
                )
                .detach()
                .cpu()
            )
            local_patched = (
                end_direct_contribution(
                    patched_cache,
                    records,
                    target_condition=target,
                    heads=active_heads,
                    unembed=model.W_U,
                )
                .detach()
                .cpu()
            )
        append_patch_rows(
            patch_rows,
            records,
            corruption="primary",
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


def main() -> None:
    args = parse_args()
    if args.batch_size != FROZEN_BATCH_SIZE:
        raise ValueError(f"Experiment 004 freezes batch size {FROZEN_BATCH_SIZE}")
    freeze = validate_freeze()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection["status"] != "complete":
        raise RuntimeError("Experiment 004 localization is not complete")
    records = [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == args.split
    ]
    if args.limit_records is not None:
        records = records[: args.limit_records]
    if not records or len(records) % args.batch_size:
        raise ValueError("Selected records must preserve complete four-row entity-pair batches")
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_template[record["template_id"]].append(record)

    output_dir = args.output_dir or DEFAULT_ROOT / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    model, hf_model, _ = load_pythia_for_hooks()
    del hf_model
    gc.collect()
    model.to(torch.device("cuda:0"))
    behavior_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat()
    offset = 0
    for template, template_records in sorted(by_template.items()):
        for start in range(0, len(template_records), args.batch_size):
            batch = template_records[start : start + args.batch_size]
            print(f"transport {args.split} {template} {start}:{start + len(batch)}", flush=True)
            run_batch(
                model,
                batch,
                selection=selection,
                batch_offset=offset,
                behavior_rows=behavior_rows,
                patch_rows=patch_rows,
            )
            offset += len(batch)

    behavior_summary = summarize_behavior(behavior_rows)
    transport_summary = summarize_patches(patch_rows)
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
    configuration = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if args.limit_records is None else "dry-run",
        "experiment": "004",
        "model": "EleutherAI/pythia-410m-deduped",
        "model_revision": freeze["model_revision"],
        "dtype": "float32",
        "inference_device": "cuda:0",
        "split": args.split,
        "records": len(records),
        "templates": len(by_template),
        "batch_size": args.batch_size,
        "corruption": "primary",
        "manifest_sha256": sha256(MANIFEST_PATH),
        "selection_sha256": sha256(SELECTION_PATH),
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
    print(json.dumps({"behavior": behavior_summary, "config": configuration}, indent=2))
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
