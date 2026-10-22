"""Independent saved-output algebra and exact replay checks, added after arm A."""

import argparse

import numpy as np
from common003 import HERE, MODELS, ROOT, digest, load_arrays, now, read, verify_freeze, write


def replay(original, reproduced):
    assert digest(original / "summary.json") == digest(reproduced / "summary.json")
    a = load_arrays(original / "per-example.npz")
    b = load_arrays(reproduced / "per-example.npz")
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    return len(a)


def prior():
    files = read(HERE / "data/prior-files.json")
    for name, sha in files.items():
        assert digest(ROOT.parent / name) == sha, name
    return len(files)


def medium(reproduced):
    verify_freeze("input-freeze.json")
    verify_freeze("medium-freeze.json")
    directory = HERE / "results/medium-cancellation"
    run = read(directory / "run.json")
    assert run["status"] == "completed"
    assert set(run["hashes"]) == {"natural.npz", "wording.npz", "order.npz"}
    for n, sha in run["hashes"].items():
        assert digest(directory / n) == sha
    assert read(HERE / "data/medium-freeze.json")["utc"] < run["started"] < run["finished"]
    saved = load_arrays(directory / "analysis/per-example.npz")
    maxerror = 0
    for f in ["wording", "order"]:
        d = load_arrays(directory / (f + ".npz"))
        assert all(np.isfinite(a).all() for a in d.values())
        assert d["clamped"].shape == (19, 384, 6)
        independent = (d["edited"][None, :, 0] - d["clamped"][..., 0]) - (
            d["base"][None, :, 0] - d["clamped_base"][..., 0]
        )
        error = float(abs(independent - saved[f + "__attenuation"]).max())
        maxerror = max(maxerror, error)
        np.testing.assert_allclose(independent, saved[f + "__attenuation"], atol=5e-6, rtol=0)
        np.testing.assert_array_equal(
            d["clamped_base"], np.broadcast_to(d["base"], d["clamped_base"].shape)
        )
    record = {
        "status": "passed",
        "utc": now(),
        "raw_files": 3,
        "clamps_per_factor": 19,
        "test_cells": 384,
        "exact_replay_arrays": replay(directory / "analysis", HERE / reproduced),
        "max_attenuation_error": maxerror,
        "prior_files_unchanged": prior(),
        "summary_byte_exact": True,
        "scope": "Saved-output replay and independent algebra, not new inference",
    }
    write(directory / "verification.json", record)
    print(record)


def broad(model, reproduced):
    for f in [
        "input-freeze.json",
        "screen-freeze.json",
        "discovery-freeze.json",
        "broad-test-freeze.json",
    ]:
        verify_freeze(f)
    from broad003 import rows_for

    selection = read(HERE / "data/broad-selection.json")["selected"][model]
    directory = HERE / "results" / model
    run = read(directory / "test/run.json")
    assert run["status"] == "completed"
    assert len(run["hashes"]) == 11
    for name, sha in run["hashes"].items():
        assert digest(directory / "test" / name) == sha
        assert all(np.isfinite(a).all() for a in load_arrays(directory / "test" / name).values())
    assert read(HERE / "data/broad-test-freeze.json")["utc"] < run["started"] < run["finished"]
    rows = rows_for(model, "test", selection["families"])
    assert run["record_ids"] == [r["id"] for r in rows]
    screen = read(directory / "screen.json")
    assert digest(directory / "screen.npz") == screen["array_sha256"]
    sr = [r for r in read(HERE / "data" / (model + "-candidates.json")) if r["split"] == "screen"]
    sb = load_arrays(directory / "screen.npz")["behavior"]
    for family, entry in screen["scores"].items():
        for order, cell in entry["cells"].items():
            mask = np.array([r["family"] == int(family) and r["order"] == int(order) for r in sr])
            assert int(mask.sum()) == cell["n"]
            assert float(sb[mask, 2].mean()) == cell["pairwise"]
            assert float(sb[mask, 3].mean()) == cell["top1"]
    metrics = load_arrays(directory / "analysis/per-example.npz")
    clean = load_arrays(directory / "test/clean.npz")["raw"]
    ref = selection["reference"]
    scale = np.array(ref["scale"])
    centers = np.array(ref["centers"])
    maxerror = 0
    for factor in ["wording", "order"]:
        raw = load_arrays(directory / "test" / (factor + ".npz"))["raw"]
        displacement = (raw[..., 0] - clean[..., 0]) / scale[:, None]
        for i, r in enumerate(rows):
            a = 2 * r["wording"] + r["order"]
            b = (
                2 * (1 - r["wording"]) + r["order"]
                if factor == "wording"
                else 2 * r["wording"] + 1 - r["order"]
            )
            axis = centers[:, b] - centers[:, a]
            norm = (axis**2).sum(1)
            score = (displacement[:, i] * axis).sum(1) / norm
            orth2 = (displacement[:, i] ** 2).sum(1) / norm - score**2
            np.testing.assert_allclose(score, metrics[factor + "__score"][:, i], atol=1e-10, rtol=0)
            maxerror = max(
                maxerror, float(abs(orth2 - metrics[factor + "__orthogonal"][:, i] ** 2).max())
            )
    assert maxerror < 1e-7
    design = np.array(
        [[1, o - 0.5, r - 0.5, (o - 0.5) * (r - 0.5)] for o, r in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    )
    outcomes = np.stack(
        [metrics[k + "__behavior"][:, 0] for k in ["self", "role", "order", "order_role"]]
    )
    coefficients = np.linalg.solve(design, outcomes)
    for i, k in enumerate(["order", "role", "interaction"], 1):
        np.testing.assert_allclose(coefficients[i], metrics["factorial__" + k], atol=5e-6, rtol=0)
    record = {
        "status": "passed",
        "utc": now(),
        "model": model,
        "raw_files": 11,
        "test_cells": 384,
        "exact_replay_arrays": replay(directory / "analysis", HERE / reproduced),
        "max_geometry_error": maxerror,
        "prior_files_unchanged": prior(),
        "summary_byte_exact": True,
        "factorial_linear_solve": True,
        "screen_independently_recounted": True,
        "all_raw_arrays_finite": True,
        "scope": "Saved-output replay and independent algebra, not new inference",
    }
    write(directory / "verification.json", record)
    print(record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["medium", "broad"])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--replay", required=True)
    a = p.parse_args()
    medium(a.replay) if a.arm == "medium" else broad(a.model, a.replay)
