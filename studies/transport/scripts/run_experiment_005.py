"""Run the post-hoc control, scale, and node-blocking study offline.

Refuses to overwrite any output directory. CPU is supported intentionally.
The original studies and their results are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch

from circuit_transport.experiment import final_outcomes
from circuit_transport.experiment_005 import (
    SEED,
    addition_hooks,
    clamp_hooks,
    donor_map,
    positions,
    random_directions,
    sampled_heads,
    stable_seed,
    symmetric_scale,
)

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("canonical", "controlled")
DOSES = (0.5, 1, 2)
DRAWS = 5
# Historical GPU outputs and CPU contraction paths have a measured float32
# discrepancy. This is separate from the stricter same-computation hook audit.
HISTORICAL_PROBABILITY_TOLERANCE = 0.001
HISTORICAL_LOGIT_TOLERANCE = 0.01


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def specs():
    result = []
    for site in ("s2", "end"):
        for source in CONDITIONS:
            for target in CONDITIONS:
                result.append(
                    {
                        "spec": f"native_{site}_{source}_to_{target}",
                        "kind": "native",
                        "site": site,
                        "source": source,
                        "target": target,
                        "alpha": 1,
                        "draw": -1,
                    }
                )
            target = CONDITIONS[1 - CONDITIONS.index(source)]
            for dose in DOSES:
                result.append(
                    {
                        "spec": f"matched_{site}_{source}_to_{target}_alpha_{dose}",
                        "kind": "matched",
                        "site": site,
                        "source": source,
                        "target": target,
                        "alpha": dose,
                        "draw": -1,
                    }
                )
        for kind in ("mismatch", "off_target", "random_heads", "random_direction"):
            for draw in range(DRAWS) if kind.startswith("random_") else [-1]:
                suffix = f"_{draw}" if draw >= 0 else ""
                result.append(
                    {
                        "spec": f"control_{site}_{kind}{suffix}",
                        "kind": kind,
                        "site": site,
                        "source": "canonical",
                        "target": "controlled",
                        "alpha": 1,
                        "draw": draw,
                    }
                )
    for draw in [-1, *range(DRAWS)]:
        label = "selected" if draw == -1 else f"random_{draw}"
        for injection in (False, True):
            result.append(
                {
                    "spec": f"block_{label}_{'s2' if injection else 'baseline'}",
                    "kind": "block",
                    "site": "s2",
                    "source": "canonical",
                    "target": "controlled",
                    "alpha": int(injection),
                    "draw": draw,
                }
            )
    assert len(result) == 56
    return result


def load_model(name, device):
    if name == "gpt2-medium":
        from transformer_lens.model_bridge import TransformerBridge

        model = TransformerBridge.boot_transformers(
            str(ROOT / "models" / name), device=torch.device("cpu"), dtype=torch.float32
        )
        model.enable_compatibility_mode()
    else:
        from circuit_transport.pythia import load_pythia_for_hooks

        model, hf_model, _ = load_pythia_for_hooks()
        del hf_model
        gc.collect()
    model.set_use_attn_result(False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2-medium", "pythia-410m-deduped"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--smoke", action="store_true", help="First inference batch; not a study result"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the supported CPU execution path")
    study = 3 if args.model == "gpt2-medium" else 4
    selection_study = 2 if study == 3 else 4
    manifest_path = ROOT / f"data/experiment-{study:03d}-manifest.jsonl"
    selection_path = (
        ROOT / f"results/experiment-{selection_study:03d}/localization/selected-heads.json"
    )
    baseline_path = ROOT / f"results/experiment-{study:03d}/test/behavior-per-example.jsonl"
    patches_path = ROOT / f"results/experiment-{study:03d}/test/patches-per-example.jsonl"
    selection = json.loads(selection_path.read_text())
    records = [r for r in load_jsonl(manifest_path) if r["split"] == "test"]
    if len(records) != 160 or len({r["template_id"] for r in records}) != 8:
        raise ValueError("Study 005 requires all 160 test examples across eight templates")
    if len({r["example_id"] for r in records}) != len(records):
        raise ValueError("Duplicate test example identities")
    if args.smoke:
        first = min({r["template_id"] for r in records})
        records = [r for r in records if r["template_id"] == first]
    index = {r["example_id"]: i for i, r in enumerate(records)}
    donors = donor_map(records)
    groups = {
        (c, s): tuple(tuple(h) for h in selection["conditions"][c][s]["selected_heads"])
        for c in CONDITIONS
        for s in ("s2", "end")
    }
    unions = {
        s: tuple(sorted(set(groups[("canonical", s)]) | set(groups[("controlled", s)])))
        for s in ("s2", "end")
    }
    random_groups = {
        (s, d): sampled_heads(
            groups[("canonical", s)],
            unions[s],
            n_heads=16,
            seed=stable_seed(SEED, args.model, s, "heads", d),
        )
        for s in ("s2", "end")
        for d in range(DRAWS)
    }
    block_groups = {
        d: sampled_heads(
            groups[("controlled", "end")],
            unions["end"],
            n_heads=16,
            seed=stable_seed(SEED, args.model, "block", d),
        )
        for d in range(DRAWS)
    }
    output = args.output_dir or ROOT / "results/experiment-005" / (
        args.model + ("-smoke" if args.smoke else "")
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use a new explicit output path")
    output.mkdir(parents=True)
    config = {
        "status": "running",
        "study": "005",
        "design": "post-hoc robustness; reused test templates",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": str(device),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "seed": SEED,
        "dtype": "float32",
        "historical_endpoint_tolerances": {
            "probability": HISTORICAL_PROBABILITY_TOLERANCE,
            "logit": HISTORICAL_LOGIT_TOLERANCE,
        },
        "specs": specs(),
        "donors": donors,
        "groups": {f"{c}_{s}": g for (c, s), g in groups.items()},
        "unions": unions,
        "random_groups": {f"{s}_{d}": g for (s, d), g in random_groups.items()},
        "block_groups": block_groups,
        "packages": {p: version(p) for p in ("torch", "transformer-lens", "transformers")},
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                manifest_path,
                selection_path,
                baseline_path,
                patches_path,
                ROOT / "docs/EXPERIMENT-005.md",
                Path(__file__),
                ROOT / "src/circuit_transport/experiment_005.py",
                ROOT / "src/circuit_transport/experiment.py",
                ROOT / "src/circuit_transport/pythia.py",
                ROOT / "models" / args.model / "config.json",
                ROOT / "models" / args.model / "model.safetensors",
                ROOT
                / "data"
                / ("model-manifest.json" if study == 3 else "pythia-model-manifest.json"),
            )
        },
    }
    (output / "run-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    model = load_model(args.model, device)
    weights = model.W_O.detach().cpu()
    layers, heads, d_head, _ = weights.shape
    if heads != 16 or any(
        not (0 <= layer < layers and 0 <= head < heads)
        for group in groups.values()
        for layer, head in group
    ):
        raise ValueError("Selected head geometry does not match the cached model")
    n = len(records)
    z_bank = {
        (c, v): torch.empty(n, 2, layers, heads, d_head)
        for c in CONDITIONS
        for v in ("clean", "corrupt")
    }
    outcome_bank = {}
    by_template = defaultdict(list)
    for i, r in enumerate(records):
        by_template[r["template_id"]].append(i)
    batches = [
        ix[start : start + args.batch_size]
        for _, ix in sorted(by_template.items())
        for start in range(0, len(ix), args.batch_size)
    ]
    with torch.inference_mode():
        for condition in CONDITIONS:
            for variant in ("clean", "corrupt"):
                collected = {
                    k: torch.empty(n) for k in ("probability_difference", "logit_difference")
                }
                for ix in batches:
                    batch = [records[i] for i in ix]
                    tokens = torch.tensor(
                        [r[condition][f"{variant}_token_ids"] for r in batch], device=device
                    )
                    logits, cache = model.run_with_cache(
                        tokens, names_filter=lambda name: name.endswith("hook_z")
                    )
                    outcomes = final_outcomes(logits.float(), batch)
                    for k, values in collected.items():
                        values[ix] = outcomes[k]
                    for si, site in enumerate(("s2", "end")):
                        pos = positions(batch, condition, site).to(device)
                        for layer in range(layers):
                            z_bank[(condition, variant)][ix, si, layer] = cache[
                                f"blocks.{layer}.attn.hook_z"
                            ][torch.arange(len(ix), device=device), pos].cpu()
                    del logits, cache
                outcome_bank[(condition, variant)] = collected
                print(f"{args.model}: cached {condition}/{variant}", flush=True)

    old_behavior = {(r["example_id"], r["condition"]): r for r in load_jsonl(baseline_path)}
    baseline_error = {"probability": 0.0, "logit": 0.0}
    for i, r in enumerate(records):
        for c in CONDITIONS:
            for v in ("clean", "corrupt"):
                for metric, key in (
                    ("probability", "probability_difference"),
                    ("logit", "logit_difference"),
                ):
                    error = abs(
                        float(outcome_bank[(c, v)][key][i])
                        - old_behavior[(r["example_id"], c)][f"{v}_{key}"]
                    )
                    baseline_error[metric] = max(baseline_error[metric], error)
    if (
        baseline_error["probability"] > HISTORICAL_PROBABILITY_TOLERANCE
        or baseline_error["logit"] > HISTORICAL_LOGIT_TOLERANCE
    ):
        config.update(
            status="validation_failed",
            failed_at_utc=datetime.now(UTC).isoformat(),
            failure="Baseline mismatch against original saved study",
            checks={"baseline_error": baseline_error},
        )
        (output / "run-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Baseline mismatch against original saved study: {baseline_error}")
    old_patches = {
        (r["example_id"], r["site"], r["source_condition"], r["target_condition"]): r
        for r in load_jsonl(patches_path)
        if r["control"] == "correct"
    }
    checks = {
        "baseline_error": baseline_error,
        "native_probability_max_error": 0.0,
        "native_logit_max_error": 0.0,
        "baseline_clamp_max_error": 0.0,
        "baseline_clamp_logit_max_error": 0.0,
        "zero_dose_max_error": 0.0,
        "zero_dose_logit_max_error": 0.0,
        "matched_norm_relative_max_error": 0.0,
        "zero_shared_norm_rows": 0,
    }
    norm_audit = {site: [] for site in ("s2", "end")}
    seen_rows = set()
    row_count = 0
    with (output / "patches-per-example.jsonl").open("w") as sink, torch.inference_mode():
        for bi, ix in enumerate(batches[:1] if args.smoke else batches):
            batch = [records[i] for i in ix]
            ids = [r["example_id"] for r in batch]
            projected = {}

            def deltas(source, site, hs, mismatch=False, *, projected=projected, ids=ids, ix=ix):
                key = (source, site, tuple(hs), mismatch)
                if key not in projected:
                    source_ix = [index[donors[eid]] for eid in ids] if mismatch else ix
                    si = 0 if site == "s2" else 1
                    clean, corrupt = z_bank[(source, "clean")], z_bank[(source, "corrupt")]
                    projected[key] = {
                        h: (clean[source_ix, si, h[0], h[1]] - corrupt[source_ix, si, h[0], h[1]])
                        @ weights[h[0], h[1]]
                        for h in hs
                    }
                return projected[key]

            matched = {}
            for site in ("s2", "end"):
                a = deltas("canonical", site, unions[site])
                b = deltas("controlled", site, unions[site])
                sa, sb, _na, _nb, g = symmetric_scale(
                    torch.stack(list(a.values()), 1), torch.stack(list(b.values()), 1)
                )
                matched[site] = (sa, sb, g)
                norm_audit[site].extend(
                    {
                        "example_id": eid,
                        "canonical_union_norm": float(_na[j]),
                        "controlled_union_norm": float(_nb[j]),
                        "shared_norm": float(g[j]),
                    }
                    for j, eid in enumerate(ids)
                )
                checks["zero_shared_norm_rows"] += int((g == 0).sum())
            for spec in specs():
                kind, site, source, target = (spec[k] for k in ("kind", "site", "source", "target"))
                hs = (
                    unions[site]
                    if kind == "matched"
                    else (
                        random_groups[(site, spec["draw"])]
                        if kind == "random_heads"
                        else groups[(source, site)]
                    )
                )
                ds = deltas(source, site, hs, mismatch=kind == "mismatch")
                if kind == "random_direction":
                    ds = {
                        h: random_directions(
                            d, ids, model=args.model, site=site, head=h, draw=spec["draw"]
                        )
                        for h, d in ds.items()
                    }
                if kind == "matched":
                    scale = matched[site][CONDITIONS.index(source)] * spec["alpha"]
                    ds = {h: d * scale[:, None] for h, d in ds.items()}
                elif kind == "block" and spec["alpha"] == 0:
                    ds = {}
                pos = positions(batch, target, site) - (1 if kind == "off_target" else 0)
                hooks = addition_hooks(ds, pos)
                if kind == "block":
                    clamps = (
                        groups[(target, "end")]
                        if spec["draw"] == -1
                        else block_groups[spec["draw"]]
                    )
                    hooks += clamp_hooks(
                        z_bank[(target, "corrupt")][ix, 1], clamps, positions(batch, target, "end")
                    )
                tokens = torch.tensor(
                    [r[target]["corrupt_token_ids"] for r in batch], device=device
                )
                with model.hooks(fwd_hooks=hooks):
                    outcomes = final_outcomes(model(tokens).float(), batch)
                norm = (
                    torch.stack(list(ds.values()), 1).flatten(1).norm(dim=1)
                    if ds
                    else torch.zeros(len(ix))
                )
                if kind == "matched":
                    expected = matched[site][2] * spec["alpha"]
                    error = float(((norm - expected).abs() / expected.clamp_min(1e-12)).max())
                    checks["matched_norm_relative_max_error"] = max(
                        checks["matched_norm_relative_max_error"], error
                    )
                for j, (i, r) in enumerate(zip(ix, batch, strict=True)):
                    row = {k: r[k] for k in ("example_id", "template_id", "pattern")}
                    row.update(spec)
                    row["model"] = args.model
                    row["injected_norm"] = float(norm[j])
                    for prefix, variant in (("clean", "clean"), ("baseline", "corrupt")):
                        row[prefix] = float(
                            outcome_bank[(target, variant)]["probability_difference"][i]
                        )
                        row[prefix + "_logit"] = float(
                            outcome_bank[(target, variant)]["logit_difference"][i]
                        )
                    row["patched"] = float(outcomes["probability_difference"][j])
                    row["patched_logit"] = float(outcomes["logit_difference"][j])
                    if kind == "native":
                        prior = old_patches[(r["example_id"], site, source, target)]
                        for metric, key in (("probability", "patched"), ("logit", "patched_logit")):
                            error = abs(row[key] - prior[f"target_patched_{metric}_difference"])
                            checks[f"native_{metric}_max_error"] = max(
                                checks[f"native_{metric}_max_error"], error
                            )
                    if kind == "block" and spec["alpha"] == 0:
                        checks["baseline_clamp_max_error"] = max(
                            checks["baseline_clamp_max_error"],
                            abs(row["patched"] - row["baseline"]),
                        )
                        checks["baseline_clamp_logit_max_error"] = max(
                            checks["baseline_clamp_logit_max_error"],
                            abs(row["patched_logit"] - row["baseline_logit"]),
                        )
                    identity = (row["example_id"], row["spec"])
                    if identity in seen_rows:
                        raise RuntimeError(f"Duplicate intervention row: {identity}")
                    seen_rows.add(identity)
                    sink.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    row_count += 1
                del outcomes
            if bi == 0:
                for site in ("s2", "end"):
                    zero = {
                        h: torch.zeros_like(d)
                        for h, d in deltas("canonical", site, groups[("canonical", site)]).items()
                    }
                    hooks = addition_hooks(zero, positions(batch, "controlled", site))
                    tokens = torch.tensor(
                        [r["controlled"]["corrupt_token_ids"] for r in batch], device=device
                    )
                    with model.hooks(fwd_hooks=hooks):
                        outcomes = final_outcomes(model(tokens).float(), batch)
                    error = float(
                        (
                            outcomes["probability_difference"]
                            - outcome_bank[("controlled", "corrupt")]["probability_difference"][ix]
                        )
                        .abs()
                        .max()
                    )
                    checks["zero_dose_max_error"] = max(checks["zero_dose_max_error"], error)
                    logit_error = float(
                        (
                            outcomes["logit_difference"]
                            - outcome_bank[("controlled", "corrupt")]["logit_difference"][ix]
                        )
                        .abs()
                        .max()
                    )
                    checks["zero_dose_logit_max_error"] = max(
                        checks["zero_dose_logit_max_error"], logit_error
                    )
            sink.flush()
            print(
                f"{args.model}: intervention batch {bi + 1}/{1 if args.smoke else len(batches)}; elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
    used_batches = batches[:1] if args.smoke else batches
    expected_rows = {
        (records[i]["example_id"], spec["spec"])
        for batch_ix in used_batches
        for i in batch_ix
        for spec in specs()
    }
    checks["complete_row_coverage"] = seen_rows == expected_rows
    config.update(
        completed_at_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        rows=row_count,
        checks=checks,
        matched_norm_audit=norm_audit,
        status="smoke_complete" if args.smoke else "complete",
    )
    if (
        checks["native_probability_max_error"] > HISTORICAL_PROBABILITY_TOLERANCE
        or checks["native_logit_max_error"] > HISTORICAL_LOGIT_TOLERANCE
        or checks["baseline_clamp_max_error"] > 1e-5
        or checks["baseline_clamp_logit_max_error"] > 1e-4
        or checks["zero_dose_max_error"] > 1e-5
        or checks["zero_dose_logit_max_error"] > 1e-4
        or checks["matched_norm_relative_max_error"] > 1e-5
        or not checks["complete_row_coverage"]
    ):
        config["status"] = "validation_failed"
    config["output_sha256"] = sha(output / "patches-per-example.jsonl")
    (output / "run-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"status": config["status"], "checks": checks, "rows": row_count}, indent=2),
        flush=True,
    )
    if config["status"] == "validation_failed":
        raise RuntimeError("Study validation failed; output is not an accepted result")


if __name__ == "__main__":
    main()
