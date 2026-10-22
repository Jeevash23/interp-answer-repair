"""Analyze stored pilot effects using only the frozen discovery-fit procedure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from causal_circuit_routing.io import (
    ROOT,
    digest,
    load_records,
    read_json,
    verify_freeze,
    write_json,
)
from causal_circuit_routing.metrics import (
    classify,
    conditional_signflip_p,
    fit_signature,
    hierarchical_interval,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/pilot-001")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    verify_freeze()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else run / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_json(run / "run.json")
    config = read_json(ROOT / "configs/pilot-001.json")
    records = load_records()
    if metadata["record_ids"] != [r["id"] for r in records]:
        raise RuntimeError("Result row IDs differ from frozen manifest")
    if metadata["status"] != "completed":
        write_json(
            output / "summary.json", {"gate1": "not_passed", "run_status": metadata["status"]}
        )
        (output / "REPORT.md").write_text(
            f"# Pilot report\n\nRun status: {metadata['status']}. Gate 1 not passed.\n"
            "Inspect run.json and behavior.json; no claim of circuit-family separation is made.\n"
        )
        return
    for filename, sha in metadata["result_sha256"].items():
        if digest(run / filename) != sha:
            raise RuntimeError(f"Stored raw result changed: {filename}")
    with np.load(run / "signatures.npz") as stored:
        raw, prob_raw = stored["raw"], stored["probability_raw"]
    with np.load(run / "behavior.npz") as stored:
        behavior = dict(stored)
    train = np.array([r["split"] == "discovery" for r in records])
    test = ~train
    labels = np.array([int(r["family"] == "ABBA") for r in records])
    templates = sorted({r["template_id"] for r in records if r["split"] == "heldout"})
    pairs = sorted({r["name_pair_id"] for r in records if r["split"] == "heldout"})
    summary = {
        "behavior": {},
        "baselines": {},
        "n": len(records),
        "test_templates": len(templates),
        "test_unordered_name_pairs": len(pairs),
        "causal_transfer_tested": False,
        "fresh_literature_audit_complete": False,
    }
    behavior_ok = True
    corruption_ok = True
    for split in ("discovery", "heldout"):
        for family in ("ABBA", "BABA"):
            mask = np.array([r["split"] == split and r["family"] == family for r in records])
            accuracy = float(behavior["clean_pairwise_correct"][mask].mean())
            gap = float(
                (behavior["clean_logit_difference"] - behavior["corrupt_logit_difference"])[
                    mask
                ].mean()
            )
            behavior_ok &= accuracy >= config["minimum_behavior_accuracy"]
            corruption_ok &= gap >= config["require_effective_corruption_mean_gap"]
            summary["behavior"][f"{split}/{family}"] = {
                "n": int(mask.sum()),
                "pairwise_accuracy": accuracy,
                "mean_clean_logit_difference": float(
                    behavior["clean_logit_difference"][mask].mean()
                ),
                "mean_clean_minus_corrupt_logit_difference": gap,
                "open_vocab_top1_accuracy": float(
                    np.mean(
                        behavior["clean_top1_token_id"][mask]
                        == np.array([r["io_token_id"] for r in records])[mask]
                    )
                ),
            }
    fits = {}
    all_baselines_pass = True
    for bi, baseline in enumerate(config["baselines"]):
        fit = fit_signature(raw[bi, train], labels[train], config["within_family_scale_floor"])
        fits[baseline] = {key: value.tolist() for key, value in fit.items()}
        predicted = classify(raw[bi], fit)
        correct = np.where(predicted == -1, 0.5, (predicted == labels).astype(float))
        balanced = float(np.mean([correct[test & (labels == c)].mean() for c in (0, 1)]))
        template_acc = {
            t: float(correct[np.array([r["template_id"] == t for r in records])].mean())
            for t in templates
        }
        scaled = raw[bi] / fit["scale"]
        axis = fit["axis"]
        grid = np.empty((len(templates), len(pairs)))
        accuracy_grid = np.empty_like(grid)
        for ti, template in enumerate(templates):
            for pi, pair in enumerate(pairs):
                ids = [
                    i
                    for i, r in enumerate(records)
                    if r["template_id"] == template and r["name_pair_id"] == pair
                ]
                a = [i for i in ids if labels[i] == 1]
                b = [i for i in ids if labels[i] == 0]
                grid[ti, pi] = (scaled[a].mean(0) - scaled[b].mean(0)) @ axis / (axis @ axis)
                accuracy_grid[ti, pi] = correct[ids].mean()
        interval = hierarchical_interval(grid, config["bootstrap_draws"], config["seed"])
        accuracy_interval = hierarchical_interval(
            accuracy_grid, config["bootstrap_draws"], config["seed"] + 1
        )
        distances = np.linalg.norm(scaled[test, None, :] - scaled[None, test, :], axis=-1)
        equal_family = labels[test, None] == labels[None, test]
        diagonal = np.eye(int(test.sum()), dtype=bool)
        passed = (
            balanced >= config["minimum_heldout_balanced_accuracy"]
            and min(template_acc.values()) > 0.5
            and interval[0] > 0
        )
        all_baselines_pass &= passed
        # Probability effects are retained in native probability units as a sensitivity diagnostic.
        probability_mean_a = prob_raw[bi, test & (labels == 1)].mean(0)
        probability_mean_b = prob_raw[bi, test & (labels == 0)].mean(0)
        summary["baselines"][baseline] = {
            "heldout_balanced_accuracy": balanced,
            "heldout_accuracy_bootstrap_95": accuracy_interval,
            "per_template_accuracy": template_acc,
            "heldout_natural_axis_projection": float(grid.mean()),
            "axis_bootstrap_95": interval,
            "conditional_block_signflip_p_two_sided": conditional_signflip_p(grid),
            "template_name_pair_axis_grid": grid.tolist(),
            "discovery_axis_norm": float(np.linalg.norm(axis)),
            "scale_floor_coordinates": int(
                (fit["within_sd"] < config["within_family_scale_floor"]).sum()
            ),
            "mean_within_family_distance": float(distances[equal_family & ~diagonal].mean()),
            "mean_between_family_distance": float(distances[~equal_family].mean()),
            "heldout_raw_probability_centroid_gap_norm": float(
                np.linalg.norm(probability_mean_a - probability_mean_b)
            ),
            "criteria_pass": bool(passed),
        }
    summary.update(
        behavior_pass=bool(behavior_ok),
        corruption_efficacy_pass=bool(corruption_ok),
        gate1="provisional_pass"
        if behavior_ok and corruption_ok and all_baselines_pass
        else "not_passed",
    )
    summary["limits"] = [
        "Only two held-out templates and four unordered name pairs; bootstrap uncertainty is fragile.",
        "Supervised signature discrimination does not establish two discrete complete circuits.",
        "Single END-head effects omit early nodes, MLPs, interactions, and edge structure.",
        "No candidate router, donor transfer, or causal circuit switch was tested.",
        "Not an ACC++ implementation reproduction; primary-source/code audit is pending.",
    ]
    write_json(output / "discovery-fit.json", fits)
    write_json(output / "summary.json", summary)
    write_json(
        output / "analysis-provenance.json",
        {
            "script_sha256": digest(Path(__file__)),
            "freeze_sha256": metadata["freeze_sha256"],
            "inputs": metadata["result_sha256"],
            "outputs": {
                name: digest(output / name) for name in ("discovery-fit.json", "summary.json")
            },
        },
    )
    lines = [
        "# Pilot 001 — natural downstream causal signatures",
        "",
        f"**Gate 1: {summary['gate1']}.** No routing or circuit-switch intervention was tested.",
        "",
        "64 prompts; 32 discovery and 32 held out. The held-out split contains two templates",
        "and four unordered name pairs. All prompts were retained. Names and template IDs are",
        "disjoint between this pilot's splits. The model is GPT-2 Small, float32, CPU.",
        "",
        "| Split / family | Pairwise accuracy | Mean IO–S logit gap | Corruption gap |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary["behavior"].items():
        lines.append(
            f"| {key} | {value['pairwise_accuracy']:.1%} | "
            f"{value['mean_clean_logit_difference']:.3f} | "
            f"{value['mean_clean_minus_corrupt_logit_difference']:.3f} |"
        )
    lines += [
        "",
        "| Ablation | Held-out accuracy [bootstrap 95%] | Natural axis separation [95%] |",
        "|---|---:|---:|",
    ]
    for baseline, value in summary["baselines"].items():
        lo, hi = value["axis_bootstrap_95"]
        alo, ahi = value["heldout_accuracy_bootstrap_95"]
        lines.append(
            f"| {baseline} | {value['heldout_balanced_accuracy']:.1%} "
            f"[{alo:.1%}, {ahi:.1%}] | {value['heldout_natural_axis_projection']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] |"
        )
    lines += [
        "",
        "Natural axis separation uses discovery-only scales and centroids. It measures",
        "paired differences between unedited prompts; it is **not a transfer/switch score**.",
        "Intervals resample template and unordered-name-pair blocks, keeping role and family",
        "pairs together. With only two templates they do not establish broad generalization.",
        "",
        (
            f"Runtime: {metadata['duration_seconds'] / 60:.1f} minutes. Peak process RSS: "
            f"{metadata['peak_process_rss_mib']:.0f} MiB."
        ),
        "",
        "## Interpretation and next decision",
        "",
    ]
    lines += (
        [
            "The fixed pilot criteria passed provisionally. Validate family stability on more",
            "templates and verify the source method before candidate routing experiments.",
        ]
        if summary["gate1"] == "provisional_pass"
        else [
            "The frozen pilot criteria did not all pass. Do not expand to the full factorial",
            "dataset or report stable circuit families. Inspect the baseline-sensitive results",
            "and source methods, then specify a separate follow-up pilot if warranted.",
        ]
    )
    lines += ["", "## Limits", ""] + ["- " + s for s in summary["limits"]]
    lines += [
        "",
        "Raw effects, behavior, model hashes, a pre-inference local freeze and fitted",
        "discovery parameters accompany this report. The earlier research project was not",
        "used as training data and is preserved separately.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
