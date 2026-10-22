"""Post hoc paired-axis diagnostic after the original gate failed endpoint calibration."""

import argparse

import numpy as np
from broad003 import rows_for
from common003 import (
    HERE,
    MODELS,
    donor_indices,
    load_arrays,
    read,
    save_arrays,
    verify_freeze,
    write,
)

from transport_bridge.statistics import block_interval


def main(model, output=None):
    for f in ["input-freeze.json", "discovery-freeze.json", "broad-test-freeze.json"]:
        verify_freeze(f)
    choice = read(HERE / "data/broad-selection.json")["selected"][model]
    rows = rows_for(model, "test", choice["families"])
    p = HERE / "results" / model
    c = load_arrays(p / "test/clean.npz")["raw"][..., 0].astype(float)
    e = load_arrays(p / "test/wording.npz")["raw"][..., 0].astype(float)
    scale = np.array(choice["reference"]["scale"])[:, None]
    v = (c[:, donor_indices(rows, "wording")] - c) / scale
    u = (e - c) / scale
    norm2 = np.sum(v * v, axis=-1)
    assert np.all(norm2 > 1e-8)
    score = np.sum(u * v, axis=-1) / norm2
    orth = np.linalg.norm(u - score[..., None] * v, axis=-1) / np.sqrt(norm2)
    margin = (np.linalg.norm(u, axis=-1) - np.linalg.norm(u - v, axis=-1)) / np.sqrt(norm2)
    np.testing.assert_array_equal(np.sum(v * v, axis=-1) / norm2, np.ones_like(score))
    np.testing.assert_array_equal(np.linalg.norm(v - v, axis=-1), np.zeros_like(score))
    np.testing.assert_array_equal(margin > 0, score > 0.5)
    result = {
        "model": model,
        "scope": "Post hoc descriptive paired-axis diagnostic; no new switch claim",
        "exact_reference_endpoints": True,
        "cells": {},
    }
    for bi, ref in enumerate(["zero", "discovery_mean"]):
        result["cells"][ref] = {}
        for w in range(2):
            for o in range(2):
                mask = np.array([r["wording"] == w and r["order"] == o for r in rows])
                result["cells"][ref][f"w{w}_o{o}"] = {
                    "score": block_interval(score[bi], rows, mask, multiplicity=1),
                    "orthogonal": block_interval(orth[bi], rows, mask, multiplicity=1),
                    "distance_margin": block_interval(margin[bi], rows, mask, multiplicity=1),
                    "fraction_closer_to_donor": float((margin[bi, mask] > 0).mean()),
                }
    out = HERE / output if output else p / "paired-geometry"
    out.mkdir(exist_ok=False, parents=True)
    write(out / "summary.json", result)
    save_arrays(out / "per-example.npz", score=score, orthogonal=orth, distance_margin=margin)
    print(
        model,
        {
            ref: {k: (v["score"]["mean"], v["distance_margin"]["ci95"]) for k, v in cells.items()}
            for ref, cells in result["cells"].items()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output")
    a = p.parse_args()
    main(a.model, a.output)
