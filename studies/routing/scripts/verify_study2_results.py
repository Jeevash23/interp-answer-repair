"""Audit completed measurements and a statistical replay without new model inference."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from causal_circuit_routing.io import ROOT, digest, now, read_json, verify_freeze, write_json
from causal_circuit_routing.study2_run import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=ROOT / "reproduced/study-002-statistics")
    args = parser.parse_args()
    verify_freeze()
    verify_inputs(selection=True)
    transfer = ROOT / "results/study-002/transfer"
    analysis = ROOT / "results/study-002/analysis"
    run = read_json(transfer / "run.json")
    selection = read_json(ROOT / "results/study-002/family/selection.json")
    assert (
        digest(ROOT / "results/study-002/family/measurements.npz")
        == selection["source_measurements_sha256"]
    )
    summary = read_json(analysis / "summary.json")
    freeze = read_json(ROOT / "data/study-002-analysis-freeze.json")
    assert run["status"] == "completed"
    assert digest(ROOT / "scripts/analyze_study2.py") == freeze["analyzer_sha256"]
    assert datetime.fromisoformat(freeze["frozen_utc"]) < datetime.fromisoformat(run["started_utc"])
    rows = [r for r in read_json(ROOT / "data/study-002-recipients.json") if r["split"] == "test"]
    library = read_json(ROOT / "data/study-002-prompts.json")
    assert run["record_ids"] == [r["id"] for r in rows]
    assert len(rows) == 128
    expected = {
        f"{mode}__{wording}_{family}_{answer}"
        for mode in ("rank1", "full")
        for wording in ("same", "other")
        for family in ("same", "cross")
        for answer in ("same", "different")
    } | {
        f"{mode}__{control}"
        for mode in ("rank1", "full")
        for control in (
            "shuffled",
            "random_0",
            "random_1",
            "random_2",
            "wrong_position",
            "wrong_layer",
            "mlp_control",
        )
    }
    assert len(run["completed_conditions"]) == 30
    assert set(run["completed_conditions"]) == expected == set(summary["conditions"])
    for name, sha in run["hashes"].items():
        assert digest(transfer / name) == sha, name
    original_summary_hash = digest(analysis / "summary.json")
    assert original_summary_hash == digest(args.replay / "summary.json")
    with (
        np.load(analysis / "per-example-metrics.npz") as original,
        np.load(args.replay / "per-example-metrics.npz") as replay,
    ):
        assert set(original.files) == set(replay.files)
        for key in original.files:
            np.testing.assert_array_equal(original[key], replay[key], err_msg=key)
        metrics = dict(original)
    with np.load(transfer / "natural.npz") as data:
        natural = dict(data)
    io_ids = np.array([r["io_token_id"] for r in rows])
    assert summary["natural_behavior"]["open_vocab_top1"] == np.mean(
        natural["clean_top1_token_id"] == io_ids
    )
    assert summary["natural_behavior"]["pairwise_accuracy"] == np.mean(
        natural["clean_logit_difference"] > 0
    )
    norms = {}
    max_geometry_error = 0.0
    max_identity_error = 0.0
    for label in sorted(expected):
        metadata = read_json(transfer / f"{label}.json")
        assert metadata["sha256"] == digest(transfer / f"{label}.npz")
        assert len(metadata["donor_ids"]) == len(rows)
        mode, condition = label.split("__")
        assert metadata["mode"] == mode and metadata["condition"] == condition
        assert metadata["donor_io_token_ids"] == [
            library[pid]["io_token_id"] for pid in metadata["donor_ids"]
        ]
        assert metadata["donor_families"] == [
            library[pid]["family"] for pid in metadata["donor_ids"]
        ]
        if condition.startswith(("same_", "other_")):
            assert metadata["donor_ids"] == [r["donors"][condition] for r in rows]
        else:
            assert all(
                r["family"] != family
                for r, family in zip(rows, metadata["donor_families"], strict=True)
            )
            if condition != "shuffled":
                assert metadata["donor_ids"] == [r["donors"]["same_cross_same"] for r in rows]
        positions = [r["positions"][selection["selected"]["position"]] for r in rows]
        if condition == "wrong_position":
            positions = [p - 1 for p in positions]
        assert metadata["positions"] == positions
        expected_layer = selection["selected"]["layer"]
        if condition == "wrong_layer":
            expected_layer = max(0, expected_layer - 3)
        elif condition == "mlp_control":
            expected_layer = 0
        assert metadata["layer"] == expected_layer
        assert metadata["kind"] == ("mlp" if condition == "mlp_control" else "residual")
        with np.load(transfer / f"{label}.npz") as data:
            arrays = dict(data)
        assert arrays["raw"].shape == (2, 128, 72)
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.testing.assert_array_equal(arrays["pairwise_correct"], arrays["logit_difference"] > 0)
        assert summary["conditions"][label]["pairwise_accuracy"] == np.mean(
            arrays["pairwise_correct"]
        )
        norms[label] = arrays["delta_norm"]
        if label.endswith("__same_same_same"):
            for key in ("raw", "probability_raw"):
                max_identity_error = max(
                    max_identity_error, float(np.max(np.abs(arrays[key] - natural[key])))
                )
            assert np.max(arrays["delta_norm"]) < 0.0005
        for bi, baseline in enumerate(("zero", "s2_corrupt")):
            fit = selection["signature_fits"][baseline]
            axis, scale = np.array(fit["axis"]), np.array(fit["scale"])
            ds = (arrays["raw"][bi] - natural["raw"][bi]) / scale
            orient = np.array([-1 if r["family"] == "ABBA" else 1 for r in rows])
            projection = (ds @ axis) / np.dot(axis, axis)
            saved_s = metrics[label + "__score"][bi]
            saved_o = metrics[label + "__orthogonal"][bi]
            np.testing.assert_allclose(saved_s, projection * orient, atol=1e-12)
            # Independent Pythagorean check of the normalized perpendicular component.
            squared_o = np.sum(ds * ds, axis=1) / np.dot(axis, axis) - projection**2
            error = float(np.max(np.abs(saved_o**2 - squared_o)))
            max_geometry_error = max(max_geometry_error, error)
            np.testing.assert_allclose(saved_o**2, squared_o, atol=1e-10)
            np.testing.assert_allclose(
                summary["conditions"][label]["baselines"][baseline]["score"]["mean"],
                saved_s.mean(),
                atol=1e-12,
            )
    assert max_identity_error == 0
    random_norm_error = 0.0
    for mode in ("rank1", "full"):
        reference = norms[f"{mode}__same_cross_same"]
        for i in range(3):
            actual = norms[f"{mode}__random_{i}"]
            np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-6)
            random_norm_error = max(random_norm_error, float(np.max(np.abs(actual - reference))))
        for control in ("wrong_layer", "wrong_position", "mlp_control"):
            np.testing.assert_array_equal(norms[f"{mode}__{control}"], reference)
    previous = read_json(ROOT / "data/previous-project-readonly-baseline.json")
    previous_root = Path(previous["path"])
    for path, sha in previous["sha256"].items():
        assert digest(previous_root / path) == sha, f"Previous project changed: {path}"
    record = {
        "status": "passed",
        "verified_utc": now(),
        "verifier_source_sha256": digest(Path(__file__)),
        "pilot_and_study_source_freezes_intact": True,
        "analysis_frozen_before_test_started": True,
        "raw_npz_hashes_verified": len(run["hashes"]),
        "conditions_complete": len(expected),
        "donor_identity_and_location_records_verified": len(expected) * len(rows),
        "test_recipients": len(rows),
        "summary_reproduced_byte_for_byte": True,
        "summary_sha256": original_summary_hash,
        "per_example_metric_arrays_reproduced_exactly": len(metrics),
        "identity_signature_max_error": max_identity_error,
        "signature_geometry_max_squared_error": max_geometry_error,
        "random_control_norm_max_error": random_norm_error,
        "previous_project_recorded_files_unchanged": len(previous["sha256"]),
        "numerical_hook_checks": read_json(ROOT / "results/study-002/hook-verification.json"),
        "scope": "Result/provenance audit; does not validate scientific novelty or cross-model generality",
    }
    write_json(ROOT / "results/study-002/verification.json", record)
    print(
        f"Verified {len(expected)} conditions, {len(rows)} recipients and exact statistical replay."
    )
    print(f"All {len(previous['sha256'])} recorded previous-project files are unchanged.")


if __name__ == "__main__":
    main()
