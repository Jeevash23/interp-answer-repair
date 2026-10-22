"""Frozen descriptive and primary analysis; never selects edits from test outcomes."""

import argparse

import numpy as np

from transport_bridge.common import ROOT, digest, now, read, verify, write
from transport_bridge.statistics import (
    axis_coordinates,
    block_interval,
    geometry,
    prediction_scores,
)
from transport_bridge.study import conditions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    model_dir = ROOT / "results" / args.model
    output = ROOT / args.output if args.output else model_dir / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    verify(read(ROOT / "data/source-freeze.json")["files"])
    verify(read(model_dir / "fit/test-freeze.json")["files"])
    run = read(model_dir / "test/run.json")
    assert run["status"] == "completed"
    assert run["completed"] == [s["name"] for s in conditions()]
    for name, sha in run["hashes"].items():
        assert digest(model_dir / "test" / name) == sha
    rows = [r for r in read(ROOT / "data/recipients.json") if r["split"] == "test"]
    index = {r["id"]: i for i, r in enumerate(rows)}
    selected = read(model_dir / "fit/selection.json")
    fit = selected["reference"]
    with np.load(model_dir / "test/natural.npz") as data:
        natural = dict(data)
    summary = {
        "model": args.model,
        "n_test": len(rows),
        "paired_templates": 8,
        "name_pairs": 4,
        "scope": "Finite individual-head dependency signatures; not a full circuit or universal router test",
        "prediction": {},
        "natural": {},
        "conditions": {},
        "primary": {},
        "control_contrasts": {},
        "baselines": ["zero", "discovery_mean"],
        "intervals": "Crossed template/name-pair bootstrap, discovery fit held fixed; primary families adjusted over 8 quantities per factor/baseline; secondary intervals pointwise",
    }
    clean, corrupt = natural["clean_behavior"], natural["corrupt_behavior"]
    for w in range(2):
        for o in range(2):
            mask = np.array([r["wording"] == w and r["order"] == o for r in rows])
            summary["natural"][f"w{w}_o{o}"] = {
                "pairwise_clean": float(clean[mask, 2].mean()),
                "top1_clean": float(clean[mask, 3].mean()),
                "pairwise_corrupt": float(corrupt[mask, 2].mean()),
                "top1_corrupt": float(corrupt[mask, 3].mean()),
                "logit_clean_minus_corrupt": block_interval(
                    clean[:, 0] - corrupt[:, 0], rows, mask
                ),
                "probability_clean_minus_corrupt": block_interval(
                    clean[:, 1] - corrupt[:, 1], rows, mask
                ),
            }
    for f in ["wording", "order"]:
        summary["prediction"][f] = prediction_scores(
            natural["clean_state"], natural["clean_raw"], rows, fit, selected["predictors"][f], f
        )
        coords = axis_coordinates(natural["clean_raw"], rows, fit, f)
        targets = np.array([r[f] for r in rows])
        summary["prediction"][f]["natural_family_accuracy_by_baseline"] = [
            float(((c > 0) == targets).mean()) for c in coords
        ]
    perexample = {}
    for spec in conditions():
        name, factor = spec["name"], spec["factor"]
        with np.load(model_dir / "test" / (name + ".npz")) as d:
            arrays = dict(d)
        b = arrays["behavior"]
        record = {
            "spec": spec,
            "pairwise_accuracy": float(b[:, 2].mean()),
            "top1_accuracy": float(b[:, 3].mean()),
            "alternative_beats_io": float((b[:, 4] > 0).mean()),
            "mean_delta_norm": float(arrays["delta_norm"].mean()),
            "raw_logit_recovery": block_interval(b[:, 0] - corrupt[:, 0], rows),
            "raw_probability_recovery": block_interval(b[:, 1] - corrupt[:, 1], rows),
            "factors": {},
        }
        perexample[name + "__behavior"] = b
        for f in ["wording", "order"] if factor == "self" else [factor]:
            donors = np.array([index[r["donors"][f]] for r in rows])
            metrics = geometry(
                arrays["raw"], natural["clean_raw"], natural["clean_raw"][:, donors], rows, fit, f
            )
            before = geometry(
                natural["corrupt_raw"],
                natural["clean_raw"],
                natural["clean_raw"][:, donors],
                rows,
                fit,
                f,
            )
            rfactor = {}
            other = "order" if f == "wording" else "wording"
            for bi, baseline in enumerate(summary["baselines"]):
                entry = {
                    "score": block_interval(metrics["score"][bi], rows),
                    "mean_orthogonal": float(metrics["orthogonal"][bi].mean()),
                    "donor_closer_margin": block_interval(
                        metrics["recipient_distance"][bi] - metrics["donor_distance"][bi], rows
                    ),
                    "score_change_from_corrupt": block_interval(
                        metrics["score"][bi] - before["score"][bi], rows
                    ),
                    "cells": {},
                }
                for direction in range(2):
                    for stratum in range(2):
                        mask = np.array([r[f] == direction and r[other] == stratum for r in rows])
                        behavior_loss = float((clean[mask, 2] - b[mask, 2]).mean())
                        rec_logit = block_interval(b[:, 0] - corrupt[:, 0], rows, mask)
                        rec_prob = block_interval(b[:, 1] - corrupt[:, 1], rows, mask)
                        denom_l = float((clean[mask, 0] - corrupt[mask, 0]).mean())
                        denom_p = float((clean[mask, 1] - corrupt[mask, 1]).mean())
                        score = block_interval(metrics["score"][bi], rows, mask)
                        orth = float(metrics["orthogonal"][bi, mask].mean())
                        margin = block_interval(
                            metrics["recipient_distance"][bi] - metrics["donor_distance"][bi],
                            rows,
                            mask,
                        )
                        coord = axis_coordinates(natural["clean_raw"], rows, fit, f)[bi]
                        family_acc = float(
                            (
                                (coord[mask] > 0)
                                == np.array([r[f] for r, k in zip(rows, mask, strict=True) if k])
                            ).mean()
                        )
                        natural_valid = bool(
                            clean[mask, 2].mean() >= 0.75
                            and clean[mask, 3].mean() >= 0.75
                            and family_acc >= 0.75
                        )
                        entry["cells"][f"{direction}_to_{1 - direction}/{other}{stratum}"] = {
                            "raw_logit_recovery": rec_logit,
                            "raw_probability_recovery": rec_prob,
                            "logit_cts": rec_logit["mean"] / denom_l if denom_l >= 0.1 else None,
                            "probability_cts": rec_prob["mean"] / denom_p
                            if denom_p >= 0.01
                            else None,
                            "score": score,
                            "mean_orthogonal": orth,
                            "donor_closer_margin": margin,
                            "pairwise_accuracy_loss": behavior_loss,
                            "natural_family_accuracy": family_acc,
                            "natural_valid": natural_valid,
                            "strict_switch_pass": bool(
                                natural_valid
                                and score["simultaneous_ci"][0] > 0.5
                                and orth <= max(score["mean"], 0)
                                and margin["mean"] > 0
                                and behavior_loss <= 0.1
                            ),
                            "axis_equivalent_to_clean_recipient": bool(
                                score["simultaneous_ci"][0] > -0.1
                                and score["simultaneous_ci"][1] < 0.1
                            ),
                        }
                rfactor[baseline] = entry
                if name == "head_" + f:
                    summary["primary"][f + "/" + baseline] = entry["cells"]
            record["factors"][f] = rfactor
            for k, v in metrics.items():
                perexample[name + "__" + f + "__" + k] = v
        summary["conditions"][name] = record
    for f in ["wording", "order"]:
        for kind, controls in [
            ("rank1_" + f + "_corrupt", [f"random_{f}_{d}" for d in range(3)]),
            *([("head_wording", [f"head_random_{d}" for d in range(3)])] if f == "wording" else []),
        ]:
            correct = perexample[kind + "__behavior"]
            random = np.mean([perexample[c + "__behavior"] for c in controls], axis=0)
            summary["control_contrasts"][kind] = {
                "logit_over_random": block_interval(correct[:, 0] - random[:, 0], rows),
                "probability_over_random": block_interval(correct[:, 1] - random[:, 1], rows),
            }
    summary["strict_switch_by_condition"] = {
        name: all(
            cell["strict_switch_pass"]
            for baselines in rec["factors"].values()
            for entry in baselines.values()
            for cell in entry["cells"].values()
        )
        for name, rec in summary["conditions"].items()
        if rec["spec"]["factor"] != "self"
    }
    write(output / "summary.json", summary)
    np.savez_compressed(output / "per-example-metrics.npz", **perexample)
    write(
        output / "provenance.json",
        {
            "utc": now(),
            "analyzer_sha256": digest(__file__),
            "summary_sha256": digest(output / "summary.json"),
            "raw_hashes": run["hashes"],
        },
    )
    print("Analysis complete:", output)
    print("Prediction:", summary["prediction"])
    print(
        "Strict switch passes:", [k for k, v in summary["strict_switch_by_condition"].items() if v]
    )


if __name__ == "__main__":
    main()
