"""Prospective statistics for opposing contributions, fixed before fresh test."""

import argparse

import numpy as np
from common004 import HERE, digest, load_arrays, read, save_arrays, verify, write, rows_for, interval
verify_freeze=verify

def block_interval(values,rows,mask=None,multiplicity=1):
    out=interval(values,rows,mask,multiplicity);out["simultaneous_ci"]=out.pop("adjusted_ci");return out


def main(output=None):
    verify_freeze("input-freeze.json")
    verify_freeze("gpt2-medium-test-freeze.json")
    directory = HERE / "results/medium-cancellation"
    record = read(directory / "run.json")
    assert record["status"] == "completed"
    for name, sha in record["hashes"].items():
        assert digest(directory / name) == sha
    rows = rows_for("gpt2-medium", "test", read(HERE/"data/gpt2-medium-selection.json")["families"])
    assert record["record_ids"] == [r["id"] for r in rows]
    out = HERE / output if output else directory / "analysis"
    out.mkdir(exist_ok=False, parents=True)
    nat = load_arrays(directory / "natural.npz")
    summary = {
        "natural": {},
        "factors": {},
        "n": len(rows),
        "multiplicity": 20,
        "scope": "Conditional on old discovery sign partitions",
    }
    masks = {
        f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
        for w in range(2)
        for o in range(2)
    }
    for k, mask in masks.items():
        b = nat["clean_behavior"][mask]
        summary["natural"][k] = {"pairwise": float(b[:, 2].mean()), "top1": float(b[:, 3].mean())}
    summary["behavior_valid"] = all(
        v["pairwise"] >= 0.95 and v["top1"] >= 0.85 for v in summary["natural"].values()
    )
    metrics = {}
    for factor in ["wording", "order"]:
        data = load_arrays(directory / (factor + ".npz"))
        effect = data["edited"][:, 0] - data["base"][:, 0]
        attenuation = effect[None] - (data["clamped"][..., 0] - data["clamped_base"][..., 0])
        oriented = attenuation * (1 if factor == "wording" else -1)
        assert abs(data["clamped_base"] - data["base"][None]).max() < 0.0005
        a, b, u = oriented[:3]
        primary = {"A": a, "B": b, "A_minus_B": a - b, "union": u}
        for group in ["A", "B"]:
            gi = record["variants"].index(group)
            for draw in range(3):
                ci = record["variants"].index(f"{group}_control{draw}")
                primary[f"{group}_over_control{draw}"] = (oriented[gi] - oriented[ci]) * (
                    1 if group == "A" else -1
                )
        primary = {k: block_interval(v, rows, multiplicity=20) for k, v in primary.items()}
        lo, hi = primary["union"]["simultaneous_ci"]
        success = primary["A"]["simultaneous_ci"][0] > 0 and primary["B"]["simultaneous_ci"][1] < 0
        success = success and all(
            v["simultaneous_ci"][0] > 0 for k, v in primary.items() if "control" in k
        )
        summary["factors"][factor] = {
            "recovery": block_interval(effect, rows, multiplicity=1),
            "primary": primary,
            "opposing_contributions_supported": bool(success),
            "union_equivalent_within_0p2": bool(lo > -0.2 and hi < 0.2),
            "nonadditivity": block_interval(u - a - b, rows, multiplicity=1),
            "four_run_decomposition": {
                k: {"NIE_noising": block_interval(attenuation[i], rows),
                    "PIE_denoising": block_interval(data["denoised"][i,:,0]-data["base"][:,0],rows),
                    "INT_difference": block_interval(attenuation[i]-(data["denoised"][i,:,0]-data["base"][:,0]),rows)}
                for i,k in enumerate(record["variants"])
            },
            "signed_components": {
                k: block_interval(attenuation[i], rows, multiplicity=1)
                for i, k in enumerate(record["variants"])
            },
            "cells": {
                k: {
                    "recovery": block_interval(effect, rows, mask, multiplicity=1),
                    "A": block_interval(a, rows, mask, multiplicity=1),
                    "B": block_interval(b, rows, mask, multiplicity=1),
                    "union": block_interval(u, rows, mask, multiplicity=1),
                }
                for k, mask in masks.items()
            },
        }
        metrics[factor + "__effect"] = effect
        metrics[factor + "__attenuation"] = attenuation
        metrics[factor + "__PIE"] = data["denoised"][...,0]-data["base"][None,:,0]
        metrics[factor + "__INT"] = attenuation-metrics[factor+"__PIE"]
    write(out / "summary.json", summary)
    save_arrays(out / "per-example.npz", **metrics)
    print(
        "MEDIUM ANALYSIS",
        summary["behavior_valid"],
        {
            f: (v["opposing_contributions_supported"], v["union_equivalent_within_0p2"])
            for f, v in summary["factors"].items()
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    a = p.parse_args()
    main(a.output)
