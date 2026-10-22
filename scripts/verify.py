"""Independent saved-output integrity, geometry and replay checks."""

import argparse
from datetime import datetime

import numpy as np
from project import verify as verify_project

from transport_bridge.common import ROOT, digest, now, read, verify, write
from transport_bridge.study import conditions, prepare_delta, verify_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2-medium")
p.add_argument("--replay", required=True)
args = p.parse_args()
directory = ROOT / "results" / args.model
source = read(ROOT / "data/source-freeze.json")
verify(source["files"])
verify_model(args.model)
frozen = read(directory / "fit/test-freeze.json")
verify(frozen["files"])
run = read(directory / "test/run.json")
assert run["status"] == "completed"
assert datetime.fromisoformat(source["utc"]) < datetime.fromisoformat(
    read(directory / "fit/run.json")["started"]
)
assert datetime.fromisoformat(frozen["utc"]) < datetime.fromisoformat(run["started"])
assert run["test_freeze_sha256"] == digest(directory / "fit/test-freeze.json")
assert run["completed"] == [c["name"] for c in conditions()]
for name, sha in run["hashes"].items():
    assert digest(directory / "test" / name) == sha
summary = read(directory / "analysis/summary.json")
assert digest(directory / "analysis/summary.json") == digest(ROOT / args.replay / "summary.json")
with (
    np.load(directory / "analysis/per-example-metrics.npz") as orig,
    np.load(ROOT / args.replay / "per-example-metrics.npz") as replay,
):
    assert orig.files == replay.files
    for key in orig.files:
        np.testing.assert_array_equal(orig[key], replay[key])
    metrics = dict(orig)
rows = [r for r in read(ROOT / "data/recipients.json") if r["split"] == "test"]
assert run["record_ids"] == [r["id"] for r in rows]
with np.load(directory / "test/natural.npz") as data:
    natural = dict(data)
selection = read(directory / "fit/selection.json")
scale = np.array(selection["reference"]["scale"])
centers = np.array(selection["reference"]["centers"])
cfg = read(ROOT / "configs/study.json")["models"][args.model]
maximum_geometry_error = 0.0
for spec in conditions():
    name = spec["name"]
    with np.load(directory / "test" / (name + ".npz")) as d:
        arrays = dict(d)
    assert arrays["raw"].shape == (2, 256, len(cfg["layers"]) * cfg["heads"], 2)
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.testing.assert_array_equal(arrays["behavior"][:, 2], arrays["behavior"][:, 0] > 0)
    assert (
        float(arrays["behavior"][:, 2].mean()) == summary["conditions"][name]["pairwise_accuracy"]
    )
    delta, donors = prepare_delta(spec, rows, natural, selection)
    np.testing.assert_array_equal(donors, arrays["donor_indices"])
    np.testing.assert_allclose(
        delta.flatten(1).norm(dim=-1).numpy(), arrays["delta_norm"], atol=1e-5
    )
    for factor in ["wording", "order"] if spec["factor"] == "self" else [spec["factor"]]:
        diff = (arrays["raw"][..., 0] - natural["clean_raw"][..., 0]) / scale[:, None]
        for i, r in enumerate(rows):
            a = 2 * r["wording"] + r["order"]
            b = (
                2 * (1 - r["wording"]) + r["order"]
                if factor == "wording"
                else 2 * r["wording"] + 1 - r["order"]
            )
            axis = centers[:, b] - centers[:, a]
            n2 = np.sum(axis**2, axis=1)
            score = np.sum(diff[:, i] * axis, axis=1) / n2
            np.testing.assert_allclose(
                score, metrics[name + "__" + factor + "__score"][:, i], atol=1e-10
            )
            perpendicular_squared = np.sum(diff[:, i] ** 2, axis=1) / n2 - score**2
            saved = metrics[name + "__" + factor + "__orthogonal"][:, i] ** 2
            maximum_geometry_error = max(
                maximum_geometry_error, float(np.max(abs(perpendicular_squared - saved)))
            )
            np.testing.assert_allclose(perpendicular_squared, saved, atol=1e-8)
for f in ["wording", "order"]:
    ref = np.load(directory / "test" / f"rank1_{f}_corrupt.npz")["delta_norm"]
    for c in [f"matched_{f}_corrupt", *[f"random_{f}_{d}" for d in range(3)]]:
        np.testing.assert_allclose(
            np.load(directory / "test" / (c + ".npz"))["delta_norm"], ref, atol=1e-5, rtol=1e-5
        )
integrity = verify_project()
record = {
    "status": "passed",
    "model": args.model,
    "verified_utc": now(),
    "conditions": 27,
    "test_recipients": 256,
    "raw_hashes": len(run["hashes"]),
    "exact_metric_arrays": len(metrics),
    "summary_exact": True,
    "retained_evidence_files_unchanged": integrity["unchanged_evidence_files"],
    "source_relocations_documented": integrity["documented_source_relocations"],
    "maximum_geometry_squared_error": maximum_geometry_error,
    "source_and_model_hashes_verified": True,
    "freeze_precedes_test": True,
    "scope": "Integrity and conditional-statistical replay; not independent inference replication",
}
write(ROOT / args.replay / "verification.json", record)
print(record)
