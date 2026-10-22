"""Post-freeze independent algebra, integrity and saved-statistics replay audit."""

import argparse
from datetime import datetime

import numpy as np
from assay import check_freeze, delta_for, load_arrays
from design import HERE, specs

from transport_bridge.common import ROOT, digest, now, read, write
from transport_bridge.study import verify_model


def main(model, replay):
    for freeze in ["screen-freeze.json", "screen-freeze-v2.json", "scientific-freeze.json"]:
        check_freeze(freeze)
    verify_model(model)
    directory = HERE / "results" / model
    for path, sha in read(directory / "fit/test-freeze.json")["files"].items():
        assert digest(HERE / path) == sha
    run = read(directory / "test/run.json")
    assert run["status"] == "completed"
    expected = {
        "natural.npz",
        *[s["name"] + ".npz" for s in specs()],
        *["trace_" + f + ".npz" for f in ["self", "wording", "order"]],
    }
    assert set(run["hashes"]) == expected
    for name, sha in run["hashes"].items():
        assert digest(directory / "test" / name) == sha
    source_time = datetime.fromisoformat(read(HERE / "data/scientific-freeze.json")["utc"])
    fit_time = datetime.fromisoformat(read(directory / "fit/run.json")["started"])
    test_freeze_time = datetime.fromisoformat(read(directory / "fit/test-freeze.json")["utc"])
    test_time = datetime.fromisoformat(run["started"])
    assert source_time < fit_time < test_freeze_time < test_time
    assert run["source_freeze"] == digest(HERE / "data/scientific-freeze.json")
    assert run["test_freeze"] == digest(directory / "fit/test-freeze.json")
    assert digest(directory / "analysis/summary.json") == digest(HERE / replay / "summary.json")
    metrics = load_arrays(directory / "analysis/per-example.npz")
    replayed = load_arrays(HERE / replay / "per-example.npz")
    assert set(metrics) == set(replayed)
    for name in metrics:
        np.testing.assert_array_equal(metrics[name], replayed[name])
    rows = [r for r in read(HERE / "data/recipients.json") if r["split"] == "test"]
    assert run["record_ids"] == [r["id"] for r in rows]
    natural = load_arrays(directory / "test/natural.npz")
    selected = read(directory / "fit/selection.json")
    cfg = read(ROOT / "configs/study.json")["models"][model]
    scale = np.array(selected["reference"]["scale"])
    centers = np.array(selected["reference"]["centers"])
    maximum_geometry_error = 0.0
    for spec in specs():
        d = load_arrays(directory / "test" / (spec["name"] + ".npz"))
        assert d["raw"].shape == (2, 384, len(cfg["layers"]) * cfg["heads"], 2)
        assert all(np.isfinite(a).all() for a in d.values())
        np.testing.assert_array_equal(d["behavior"][:, 2], d["behavior"][:, 0] > 0)
        delta, donors = delta_for(spec, rows, natural)
        np.testing.assert_array_equal(donors, d["donors"])
        np.testing.assert_allclose(
            delta.flatten(1).norm(dim=-1).numpy(), d["delta_norm"], atol=1e-5
        )
        f = (
            "wording"
            if "wording" in spec["donor"]
            else "order"
            if "order" in spec["donor"]
            else None
        )
        if f is None:
            continue
        displacement = (d["raw"][..., 0] - natural["clean_raw"][..., 0]) / scale[:, None]
        for i, r in enumerate(rows):
            a = 2 * r["wording"] + r["order"]
            b = (
                2 * (1 - r["wording"]) + r["order"]
                if f == "wording"
                else 2 * r["wording"] + 1 - r["order"]
            )
            axis = centers[:, b] - centers[:, a]
            norm2 = (axis**2).sum(1)
            projection = (displacement[:, i] * axis).sum(1) / norm2
            np.testing.assert_allclose(
                projection, metrics[spec["name"] + "__score"][:, i], atol=1e-10
            )
            squared = (displacement[:, i] ** 2).sum(1) / norm2 - projection**2
            saved = metrics[spec["name"] + "__orthogonal"][:, i] ** 2
            maximum_geometry_error = max(maximum_geometry_error, float(abs(squared - saved).max()))
            np.testing.assert_allclose(squared, saved, atol=1e-7)
    # Recover factorial contrasts by solving a saturated linear design independently.
    x = np.array(
        [[1, o - 0.5, r - 0.5, (o - 0.5) * (r - 0.5)] for o, r in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    )
    y = np.stack(
        [
            metrics[name + "__behavior"][:, 0]
            for name in ["head_self", "head_role", "head_order", "head_order_role"]
        ]
    )
    coefficients = np.linalg.solve(x, y)
    for i, key in enumerate(["order_effect", "role_effect", "interaction"], 1):
        np.testing.assert_allclose(coefficients[i], metrics["factorial__" + key], atol=5e-6, rtol=0)
    max_noop = 0.0
    for f in ["self", "wording", "order"]:
        t = load_arrays(directory / "test" / ("trace_" + f + ".npz"))
        assert t["clamped"].shape == (7, 384, 6)
        assert t["joint"].shape == (2, 2, 384, 6)
        assert t["attention"].shape == (2, 384, 4, 3)
        assert (t["attention"] >= 0).all() and (t["attention"] <= 1).all()
        max_noop = max(max_noop, float(abs(t["clamped_base"] - t["base"][None]).max()))
        attenuation = (t["edited"][None, :, 0] - t["clamped"][..., 0]) - (
            t["base"][None, :, 0] - t["clamped_base"][..., 0]
        )
        np.testing.assert_allclose(attenuation, metrics[f + "__attenuation"], atol=5e-6, rtol=0)
        chosen = selected["groups"][f]
        target_layers = sorted(h[0] for h in chosen["selected"])
        for controls in chosen["controls"]:
            assert sorted(h[0] for h in controls) == target_layers
            assert not {tuple(h) for h in controls} & {tuple(h) for h in chosen["selected"]}
            assert len({tuple(h) for h in controls}) == 4
    assert max_noop <= 0.0005
    previous = read(HERE / "data/prior-files.json")
    for path, sha in previous.items():
        assert digest(ROOT.parent / path) == sha, path
    result = {
        "status": "passed",
        "model": model,
        "utc": now(),
        "raw_files": len(expected),
        "test_cells": 384,
        "conditions": 13,
        "exact_replayed_arrays": len(metrics),
        "summary_byte_exact": True,
        "prior_files_unchanged": len(previous),
        "max_geometry_squared_error": maximum_geometry_error,
        "max_noop_clamp_error": max_noop,
        "factorial_verified_by_linear_solve": True,
        "freeze_timing_verified": True,
        "scope": "Saved-output replay and independent algebra; not a second full inference replication",
    }
    write(directory / "verification.json", result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--replay", required=True)
    args = p.parse_args()
    main(args.model, args.replay)
