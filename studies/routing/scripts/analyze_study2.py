"""Prespecified block-paired analysis of completed study 002 interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_circuit_routing.io import ROOT, digest, read_json, write_json
from causal_circuit_routing.metrics import classify
from causal_circuit_routing.study2_run import verify_inputs


def block_stats(values, rows, mask, seed, draws=10000, multiplicity=16):
    templates = sorted({r["template_id"] for r, keep in zip(rows, mask, strict=True) if keep})
    pairs = sorted({r["name_pair_id"] for r, keep in zip(rows, mask, strict=True) if keep})
    grid = np.zeros((len(templates), len(pairs)))
    for ti, template in enumerate(templates):
        for pi, pair in enumerate(pairs):
            ix = mask & np.array(
                [r["template_id"] == template and r["name_pair_id"] == pair for r in rows]
            )
            if not ix.any():
                raise RuntimeError("Unbalanced block grid")
            grid[ti, pi] = np.mean(values[ix])
    rng = np.random.default_rng(seed)
    ts = rng.integers(len(templates), size=(draws, len(templates)))
    ps = rng.integers(len(pairs), size=(draws, len(pairs)))
    boot = grid[ts[:, :, None], ps[:, None, :]].mean((1, 2))
    tail = 0.05 / (2 * multiplicity)
    return {
        "mean": float(grid.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "simultaneous_ci": np.quantile(boot, [tail, 1 - tail]).tolist(),
        "multiplicity": multiplicity,
        "per_template": dict(zip(templates, grid.mean(1).tolist(), strict=True)),
        "n_prompts": int(mask.sum()),
        "templates": len(templates),
        "name_pairs": len(pairs),
    }


def signature_outcomes(raw, natural, fit, labels):
    scale, axis, centroids = (np.asarray(fit[k]) for k in ("scale", "axis", "centroids"))
    sign = 1 - 2 * labels
    vector = sign[:, None] * axis[None, :]
    delta = (raw - natural) / scale
    norm2 = axis @ axis
    if norm2 < 1e-8:
        raise RuntimeError("Degenerate family axis")
    switch = np.sum(delta * vector, axis=1) / norm2
    orth = np.linalg.norm(delta - switch[:, None] * vector, axis=1) / np.sqrt(norm2)
    target = centroids[1 - labels]
    before = np.linalg.norm(natural / scale - target, axis=1)
    after = np.linalg.norm(raw / scale - target, axis=1)
    return switch, orth, before - after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/study-002/analysis")
    args = parser.parse_args()
    verify_inputs(selection=True)
    frozen_analysis = read_json(ROOT / "data/study-002-analysis-freeze.json")
    if digest(Path(__file__)) != frozen_analysis["analyzer_sha256"]:
        raise RuntimeError("Analysis source changed after pre-test freeze")
    transfer = ROOT / "results/study-002/transfer"
    run = read_json(transfer / "run.json")
    if run["status"] != "completed":
        raise RuntimeError("Transfer run is not completed")
    for name, sha in run["hashes"].items():
        if digest(transfer / name) != sha:
            raise RuntimeError(f"Changed raw output: {name}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / "configs/study-002.json")
    selection = read_json(ROOT / "results/study-002/family/selection.json")
    rows = [r for r in read_json(ROOT / "data/study-002-recipients.json") if r["split"] == "test"]
    assert [r["id"] for r in rows] == run["record_ids"]
    n = len(rows)
    labels = np.array([int(r["family"] == "ABBA") for r in rows])
    all_rows = np.ones(n, bool)
    io_ids = np.array([r["io_token_id"] for r in rows])
    with np.load(transfer / "natural.npz") as data:
        natural = dict(data)
    fits = selection["signature_fits"]
    summary = {
        "n_test": n,
        "test_templates": 8,
        "test_name_pairs": 4,
        "literature_audit_complete": False,
        "model_replication_completed": False,
        "selection": {k: selection["selected"][k] for k in ("layer", "position", "development_r2")},
        "natural": {},
        "conditions": {},
        "primary_contrasts": {},
        "control_contrasts": {},
        "strict_gate_by_mode": {},
        "analysis_seed": config["seed"],
        "intervals": "Crossed template/name-pair bootstrap; discovery fit held fixed",
    }
    targets = []
    test_family_pass = True
    natural_top1 = float(np.mean(natural["clean_top1_token_id"] == io_ids))
    for bi, baseline in enumerate(config["baselines"]):
        fit = {k: np.array(v) for k, v in fits[baseline].items()}
        predicted = classify(natural["raw"][bi], fit)
        correct = np.where(predicted == -1, 0.5, predicted == labels).astype(float)
        accuracy = np.mean([correct[labels == c].mean() for c in (0, 1)])
        coord = (
            (natural["raw"][bi] / fit["scale"] - fit["centroids"].mean(0))
            @ fit["axis"]
            / np.sum(fit["axis"] ** 2)
        )
        targets.append(coord)
        paired = np.empty(n)
        for i, r in enumerate(rows):
            j = next(
                j
                for j, q in enumerate(rows)
                if q["template_id"] == r["template_id"]
                and q["name_pair_id"] == r["name_pair_id"]
                and q["role_assignment"] == r["role_assignment"]
                and q["family"] != r["family"]
            )
            paired[i] = (coord[i] - coord[j]) * (2 * labels[i] - 1)
        separation = block_stats(paired, rows, all_rows, config["seed"])
        summary["natural"][baseline] = {
            "family_balanced_accuracy": float(accuracy),
            "natural_axis_separation": separation,
            "per_template_accuracy": {
                t: float(correct[np.array([r["template_id"] == t for r in rows])].mean())
                for t in sorted({r["template_id"] for r in rows})
            },
        }
        test_family_pass &= (
            accuracy >= config["minimum_family_accuracy"] and separation["ci95"][0] > 0
        )
    candidate = selection["selected"]
    x = natural["features"][:, candidate["site_index"]]
    prediction = (x - np.array(candidate["mean_x"])) @ np.array(candidate["weight"]) + candidate[
        "mean_y"
    ]
    target = np.mean(targets, axis=0)
    test_r2 = 1 - np.mean((prediction - target) ** 2) / np.var(target)
    summary["selection"]["test_r2"] = float(test_r2)
    summary["natural_behavior"] = {
        "pairwise_accuracy": float(natural["clean_pairwise_correct"].mean()),
        "open_vocab_top1": natural_top1,
        "per_template": {
            t: {
                "pairwise": float(
                    natural["clean_pairwise_correct"][[r["template_id"] == t for r in rows]].mean()
                ),
                "top1": float(
                    np.mean(
                        natural["clean_top1_token_id"][[r["template_id"] == t for r in rows]]
                        == io_ids[[r["template_id"] == t for r in rows]]
                    )
                ),
            }
            for t in sorted({r["template_id"] for r in rows})
        },
    }
    measured = {}
    for label in run["completed_conditions"]:
        with np.load(transfer / f"{label}.npz") as data:
            arrays = dict(data)
        met = {"score": [], "orthogonal": [], "closer": [], "arrays": arrays}
        record = {
            "baselines": {},
            "pairwise_accuracy": float(arrays["pairwise_correct"].mean()),
            "open_vocab_top1": float(np.mean(arrays["top1_token_id"] == io_ids)),
            "mean_logit_change": float(
                np.mean(arrays["logit_difference"] - natural["clean_logit_difference"])
            ),
            "mean_delta_norm": float(np.mean(arrays["delta_norm"])),
            "alternative_answer_beats_io": float(np.mean(arrays["alternative_minus_io_logit"] > 0)),
            "alternative_vs_io_logit_change": float(
                np.mean(
                    arrays["alternative_minus_io_logit"]
                    - natural["clean_alternative_minus_io_logit"]
                )
            ),
        }
        for bi, baseline in enumerate(config["baselines"]):
            score, orth, closer = signature_outcomes(
                arrays["raw"][bi], natural["raw"][bi], fits[baseline], labels
            )
            met["score"].append(score)
            met["orthogonal"].append(orth)
            met["closer"].append(closer)
            record["baselines"][baseline] = {
                "score": block_stats(score, rows, all_rows, config["seed"]),
                "mean_orthogonal": float(np.mean(orth)),
                "mean_distance_improvement": float(np.mean(closer)),
                "recipientwise_positive_fraction": float(np.mean(score > 0)),
            }
        measured[label] = met
        summary["conditions"][label] = record
    for mode in config["modes"]:
        primary_pass, damage_pass, orth_pass, controls_pass = True, True, True, True
        for bi, baseline in enumerate(config["baselines"]):
            for family in ("ABBA", "BABA"):
                mask = np.array([r["family"] == family for r in rows])
                direction = "ABBA_to_BABA" if family == "ABBA" else "BABA_to_ABBA"
                for wording in ("same", "other"):
                    for answer in ("same", "different"):
                        cross = measured[f"{mode}__{wording}_cross_{answer}"]
                        same = measured[f"{mode}__{wording}_same_{answer}"]
                        contrast = cross["score"][bi] - same["score"][bi]
                        stats = block_stats(contrast, rows, mask, config["seed"])
                        key = f"{mode}/{baseline}/{direction}/{wording}_wording/{answer}_answer"
                        pairwise_loss = float(
                            np.mean(
                                natural["clean_pairwise_correct"][mask]
                                - cross["arrays"]["pairwise_correct"][mask]
                            )
                        )
                        mean_s = float(np.mean(cross["score"][bi][mask]))
                        mean_o = float(np.mean(cross["orthogonal"][bi][mask]))
                        stats.update(
                            pairwise_accuracy_loss=pairwise_loss,
                            mean_cross_score=mean_s,
                            mean_cross_orthogonal=mean_o,
                        )
                        summary["primary_contrasts"][key] = stats
                        primary_pass &= stats["simultaneous_ci"][0] > 0
                        damage_pass &= pairwise_loss <= config["max_pairwise_accuracy_drop"]
                        orth_pass &= mean_o <= max(mean_s, 0)
                correct = measured[f"{mode}__same_cross_same"]["score"][bi]
                random_mean = np.mean(
                    [measured[f"{mode}__random_{i}"]["score"][bi] for i in range(3)], axis=0
                )
                for control in ("random_mean", "wrong_position", "wrong_layer", "mlp_control"):
                    other = (
                        random_mean
                        if control == "random_mean"
                        else measured[f"{mode}__{control}"]["score"][bi]
                    )
                    stats = block_stats(correct - other, rows, mask, config["seed"])
                    summary["control_contrasts"][f"{mode}/{baseline}/{direction}/{control}"] = stats
                    controls_pass &= stats["simultaneous_ci"][0] > 0
        summary["strict_gate_by_mode"][mode] = {
            "all16_primary_simultaneous_lower_bounds_positive": bool(primary_pass),
            "pairwise_behavior_preserved": bool(damage_pass),
            "orthogonal_movement_bounded": bool(orth_pass),
            "all_control_contrasts_positive": bool(controls_pass),
            "passed": bool(
                primary_pass
                and damage_pass
                and orth_pass
                and controls_pass
                and test_family_pass
                and test_r2 > 0
                and natural_top1 >= config["minimum_open_vocab_accuracy"]
            ),
        }
    summary["test_family_gate_pass"] = bool(test_family_pass)
    summary["test_prediction_gate_pass"] = bool(test_r2 > 0)
    write_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "per-example-metrics.npz",
        **{
            f"{label}__{key}": np.array(met[key])
            for label, met in measured.items()
            for key in ("score", "orthogonal", "closer")
        },
    )
    write_json(
        output / "provenance.json",
        {
            "analysis_source_sha256": digest(Path(__file__)),
            "analysis_freeze_sha256": digest(ROOT / "data/study-002-analysis-freeze.json"),
            "raw_hashes": run["hashes"],
            "outputs": {
                name: digest(output / name) for name in ("summary.json", "per-example-metrics.npz")
            },
        },
    )
    print(
        json.dumps(
            {
                "natural_behavior": summary["natural_behavior"],
                "selection": summary["selection"],
                "gates": summary["strict_gate_by_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
