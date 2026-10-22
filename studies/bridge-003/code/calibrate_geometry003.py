"""Post hoc calibration of the geometry against actual clean reference endpoints."""

import argparse

import numpy as np
from broad003 import rows_for
from common003 import HERE, MODELS, donor_indices, load_arrays, read, verify_freeze, write

from transport_bridge.statistics import block_interval, geometry


def main(model, output=None):
    for f in ["input-freeze.json", "discovery-freeze.json", "broad-test-freeze.json"]:
        verify_freeze(f)
    choice = read(HERE / "data/broad-selection.json")["selected"][model]
    rows = rows_for(model, "test", choice["families"])
    clean = load_arrays(HERE / "results" / model / "test/clean.npz")["raw"]
    donor = clean[:, donor_indices(rows, "wording")]
    null = geometry(clean, clean, donor, rows, choice["reference"], "wording")
    np.testing.assert_array_equal(null["score"], np.zeros_like(null["score"]))
    np.testing.assert_array_equal(null["orthogonal"], np.zeros_like(null["orthogonal"]))
    true = geometry(donor, clean, donor, rows, choice["reference"], "wording")
    np.testing.assert_array_equal(true["donor_distance"], np.zeros_like(true["donor_distance"]))
    out = {
        "scope": "Post hoc geometry calibration; not a model intervention or a primary test",
        "model": model,
        "unchanged_recipient_S_and_orthogonal_exact_zero": True,
        "actual_donor_distance_exact_zero": True,
        "actual_donor": {},
    }
    for bi, ref in enumerate(["zero", "discovery_mean"]):
        out["actual_donor"][ref] = {}
        for w in range(2):
            for o in range(2):
                mask = np.array([r["wording"] == w and r["order"] == o for r in rows])
                s = block_interval(true["score"][bi], rows, mask, multiplicity=8)
                orth = float(true["orthogonal"][bi, mask].mean())
                margin = float(true["recipient_distance"][bi, mask].mean())
                out["actual_donor"][ref][f"w{w}_o{o}"] = {
                    "score": s,
                    "orthogonal": orth,
                    "donor_margin": margin,
                    "geometry_passes": bool(
                        s["simultaneous_ci"][0] > 0.5 and orth <= max(s["mean"], 0) and margin > 0
                    ),
                }
    path = HERE / output if output else HERE / "results" / model / "geometry-calibration.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    assert not path.exists()
    write(path, out)
    print(
        model,
        {
            ref: {
                k: (v["score"]["mean"], v["orthogonal"], v["geometry_passes"])
                for k, v in cells.items()
            }
            for ref, cells in out["actual_donor"].items()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output")
    a = p.parse_args()
    main(a.model, a.output)
