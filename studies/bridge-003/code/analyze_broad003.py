"""Fixed held-out analysis for broader wording across two model families."""

import argparse

import numpy as np
from broad003 import rows_for
from common003 import (
    HERE,
    MODELS,
    digest,
    donor_indices,
    load_arrays,
    read,
    save_arrays,
    verify_freeze,
    write,
)

from transport_bridge.statistics import axis_coordinates, block_interval, geometry


def main(model, output=None):
    for f in ["input-freeze.json", "discovery-freeze.json", "broad-test-freeze.json"]:
        verify_freeze(f)
    choice = read(HERE / "data/broad-selection.json")["selected"][model]
    directory = HERE / "results" / model / "test"
    record = read(directory / "run.json")
    assert record["status"] == "completed"
    for name, sha in record["hashes"].items():
        assert digest(directory / name) == sha
    rows = rows_for(model, "test", choice["families"])
    assert record["record_ids"] == [r["id"] for r in rows]
    out = HERE / output if output else HERE / "results" / model / "analysis"
    out.mkdir(exist_ok=False, parents=True)
    clean, corrupt = [load_arrays(directory / (k + ".npz")) for k in ["clean", "corrupt"]]
    natural = load_arrays(directory / "natural.npz")
    np.testing.assert_allclose(clean["behavior"], natural["clean_behavior"], atol=0.0005, rtol=0)
    np.testing.assert_allclose(
        corrupt["behavior"], natural["corrupt_behavior"], atol=0.0005, rtol=0
    )
    ref = choice["reference"]
    masks = {
        f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
        for w in range(2)
        for o in range(2)
    }
    coords = axis_coordinates(clean["raw"], rows, ref, "wording")
    targets = np.array([r["wording"] for r in rows])
    summary = {
        "model": model,
        "families": choice["families"],
        "discovery_qualified": choice["qualified"],
        "n": len(rows),
        "natural": {},
        "conditions": {},
        "primary_wording": {},
        "factorial": {},
    }
    for key, mask in masks.items():
        c = clean["behavior"][mask]
        b = corrupt["behavior"][mask]
        summary["natural"][key] = {
            "pairwise": float(c[:, 2].mean()),
            "top1": float(c[:, 3].mean()),
            "corrupt_pairwise": float(b[:, 2].mean()),
            "corrupt_top1": float(b[:, 3].mean()),
            "wording_classification": ((coords[:, mask] > 0) == targets[mask]).mean(1).tolist(),
        }
    summary["behavior_valid"] = all(
        v["pairwise"] >= 0.95 and v["top1"] >= 0.85 for v in summary["natural"].values()
    )
    summary["test_separated"] = all(
        min(v["wording_classification"]) >= 0.8 for v in summary["natural"].values()
    )
    values, metrics = {}, {}
    for factor in [
        "self",
        "wording",
        "order",
        "role",
        "order_role",
        "random0",
        "random1",
        "random2",
    ]:
        data = load_arrays(directory / (factor + ".npz"))
        values[factor] = data
        b = data["behavior"]
        gain = b[:, 0] - corrupt["behavior"][:, 0]
        v = {
            "recovery": block_interval(gain, rows, multiplicity=1),
            "probability_recovery": block_interval(
                b[:, 1] - corrupt["behavior"][:, 1], rows, multiplicity=1
            ),
            "pairwise": float(b[:, 2].mean()),
            "top1": float(b[:, 3].mean()),
            "cells": {
                k: {
                    "recovery": block_interval(
                        gain, rows, mask, multiplicity=8 if factor == "wording" else 1
                    ),
                    "pairwise": float(b[mask, 2].mean()),
                    "top1": float(b[mask, 3].mean()),
                }
                for k, mask in masks.items()
            },
        }
        metrics[factor + "__behavior"] = b
        if factor in ["wording", "order"]:
            donors = donor_indices(rows, factor)
            geo = geometry(data["raw"], clean["raw"], clean["raw"][:, donors], rows, ref, factor)
            for name, x in geo.items():
                metrics[factor + "__" + name] = x
            v["geometry"] = {}
            for bi, baseline in enumerate(["zero", "discovery_mean"]):
                cells = {}
                for key, mask in masks.items():
                    score = block_interval(
                        geo["score"][bi], rows, mask, multiplicity=8 if factor == "wording" else 1
                    )
                    margin = block_interval(
                        geo["recipient_distance"][bi] - geo["donor_distance"][bi],
                        rows,
                        mask,
                        multiplicity=1,
                    )
                    orth = float(geo["orthogonal"][bi, mask].mean())
                    loss = float((clean["behavior"][mask, 2] - b[mask, 2]).mean())
                    valid = (
                        summary["behavior_valid"]
                        and summary["test_separated"]
                        and choice["qualified"]
                    )
                    cells[key] = {
                        "score": score,
                        "margin": margin,
                        "orthogonal": orth,
                        "pairwise_loss": loss,
                        "switch": bool(
                            valid
                            and score["simultaneous_ci"][0] > 0.5
                            and orth <= max(score["mean"], 0)
                            and margin["mean"] > 0
                            and loss <= 0.1
                        ),
                    }
                v["geometry"][baseline] = cells
            if factor == "wording":
                summary["primary_wording"] = v["geometry"]
        summary["conditions"][factor] = v
    summary["wording_switch"] = all(
        v["switch"] for cells in summary["primary_wording"].values() for v in cells.values()
    )
    a, r, o, b = [values[k]["behavior"][:, 0] for k in ["self", "role", "order", "order_role"]]
    for name, x in {
        "order": (o + b - a - r) / 2,
        "role": (r + b - a - o) / 2,
        "interaction": b - o - r + a,
    }.items():
        summary["factorial"][name] = {
            k: block_interval(x, rows, mask, multiplicity=1) for k, mask in masks.items()
        }
        metrics["factorial__" + name] = x
    rand = np.mean([values["random" + str(i)]["behavior"][:, 0] for i in range(3)], axis=0)
    summary["wording_over_random"] = block_interval(
        values["wording"]["behavior"][:, 0] - rand, rows, multiplicity=1
    )
    write(out / "summary.json", summary)
    save_arrays(out / "per-example.npz", **metrics)
    print(
        "BROAD ANALYSIS",
        model,
        "behavior",
        summary["behavior_valid"],
        "separation",
        summary["test_separated"],
        "switch",
        summary["wording_switch"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output")
    a = p.parse_args()
    main(a.model, a.output)
