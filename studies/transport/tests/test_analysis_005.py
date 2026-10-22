"""Synthetic-data tests for the new descriptive robustness analysis only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_experiment_005.py"
SPEC = importlib.util.spec_from_file_location("study5_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
sys.path.insert(0, str(SCRIPT.parent))
import verify_experiment_005 as verifier


def synthetic_rows() -> list[dict]:
    rows = []
    for name, spec in analysis.expected_specs().items():
        for template in range(8):
            for pair in range(10):
                for pattern in ("ABBA", "BABA"):
                    baseline = 0.1 if spec["target"] == "canonical" else 0.02
                    denominator = 0.4 if spec["target"] == "canonical" else 0.1
                    if name.startswith("native_"):
                        gain = 0.1 if spec["source"] == "canonical" else 0.2
                        if spec["site"] == "end":
                            gain += 0.05
                        gain *= 1 + 0.01 * template
                    elif name.startswith("matched_"):
                        gain = (0.12 if spec["source"] == "canonical" else 0.1) * spec["alpha"]
                    elif name.startswith("control_"):
                        gain = 0.002 * (spec.get("draw", 0) + 1)
                    elif name.endswith("baseline"):
                        gain = 0.0
                    elif name.startswith("block_selected"):
                        gain = 0.025
                    else:
                        gain = 0.08 + 0.001 * spec["draw"]
                    rows.append(
                        {
                            "example_id": f"test-{template:02d}-{pair:02d}-{pattern.lower()}",
                            "template_id": f"test-{template:02d}",
                            "pattern": pattern,
                            "model": "gpt2-medium",
                            "spec": name,
                            "draw": spec.get("draw"),
                            **spec,
                            "injected_norm": (
                                spec["alpha"] if spec["kind"] in {"matched", "block"} else 1.0
                            ),
                            "clean": baseline + denominator,
                            "baseline": baseline,
                            "patched": baseline + gain,
                            "clean_logit": 10 * (baseline + denominator),
                            "baseline_logit": 10 * baseline,
                            "patched_logit": 10 * (baseline + gain),
                        }
                    )
    return rows


def test_full_grid_has_56_specs_and_8960_rows() -> None:
    rows = synthetic_rows()
    assert len(analysis.expected_specs()) == 56
    assert len(rows) == 8960
    indexed = analysis.validate_rows(rows, model="gpt2-medium")
    assert len(indexed) == 56


def test_analysis_joins_ids_and_is_invariant_to_row_order() -> None:
    rows = synthetic_rows()
    original = analysis.analyze_rows(rows, model="gpt2-medium", samples=100)
    reversed_rows = analysis.analyze_rows(rows[::-1], model="gpt2-medium", samples=100)
    assert original == reversed_rows
    metric = original["directional_contrasts"]["native_end"]["probability_difference"]
    assert metric["gain_forward_minus_reverse"]["mean"] == pytest.approx(-0.1035)
    assert metric["cts_forward_minus_reverse"]["mean"] == pytest.approx(0.905625)


def test_random_draws_are_averaged_before_inference() -> None:
    result = analysis.analyze_rows(synthetic_rows(), model="gpt2-medium", samples=100)
    metric = result["controls"]["s2"]["random_heads"]["probability_difference"]
    assert metric["control_gain"]["mean"] == pytest.approx(0.006)
    assert metric["correct_minus_control_gain"]["mean"] == pytest.approx(0.0975)
    assert metric["correct_minus_control_gain"]["n_examples"] == 160
    assert metric["individual_draw_mean_contrast_range"] == pytest.approx([0.0935, 0.1015])


def test_blocking_reports_attenuation_residual_and_no_op_errors() -> None:
    result = analysis.analyze_rows(synthetic_rows(), model="gpt2-medium", samples=100)
    block = result["blocking"]["probability_difference"]
    assert block["selected_attenuation"]["mean"] == pytest.approx(0.0785)
    assert block["selected_residual_gain"]["mean"] == pytest.approx(0.025)
    assert block["mean_random_attenuation"]["mean"] == pytest.approx(0.0215)
    assert block["selected_minus_mean_random_attenuation"]["mean"] == pytest.approx(0.057)
    assert block["maximum_baseline_clamp_error"] == 0


def test_whole_template_bootstrap_does_not_resample_within_template() -> None:
    indexed = analysis.validate_rows(synthetic_rows(), model="gpt2-medium")
    estimator = analysis.TemplateAnalysis(indexed, samples=100, seed=5)
    # Within each template the values vary greatly but have the same mean, zero.
    values = np.tile(np.arange(20, dtype=float) - 9.5, 8)
    result = estimator.estimate(values)
    assert result["mean"] == 0
    assert result["ci95"] == [0, 0]


def test_ratios_are_ratios_of_means_and_suppress_unstable_intervals() -> None:
    indexed = analysis.validate_rows(synthetic_rows(), model="gpt2-medium")
    estimator = analysis.TemplateAnalysis(indexed, samples=100, seed=5)
    numerator = np.ones(160)
    denominator = np.tile(np.arange(1, 21), 8)
    result = estimator.estimate(
        np.column_stack((numerator, denominator)), ratios=((0, 1, 1.0, 0.01),)
    )
    assert result["mean"] == pytest.approx(1 / 10.5)
    denominator = np.repeat([-0.1, -0.1, -0.1, -0.1, 0.2, 0.2, 0.2, 0.2], 20)
    result = estimator.estimate(
        np.column_stack((numerator, denominator)), ratios=((0, 1, 1.0, 0.01),)
    )
    assert result["mean"] == pytest.approx(20)
    assert result["ci95"] is None
    assert result["valid_bootstrap_draws"] < 100
    assert "dropping" in result["interval_note"]


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "nonfinite", "baseline", "metadata", "norm", "draw"]
)
def test_validation_rejects_invalid_complete_grid(damage: str) -> None:
    rows = synthetic_rows()
    if damage == "duplicate":
        rows.append(dict(rows[0]))
    elif damage == "missing":
        rows.pop()
    elif damage == "nonfinite":
        rows[0]["patched"] = float("nan")
    elif damage == "baseline":
        rows[0]["clean"] += 0.01
    elif damage == "norm":
        rows[0]["injected_norm"] = float("inf")
    elif damage == "draw":
        rows[0]["draw"] = 999
    else:
        rows[0]["target"] = "controlled"
    with pytest.raises(ValueError):
        analysis.analyze_rows(rows, model="gpt2-medium", samples=20)


def test_report_explicitly_labels_posthoc_limits_and_retains_template_diagnostics() -> None:
    result = analysis.analyze_rows(synthetic_rows(), model="gpt2-medium", samples=20)
    text = analysis.markdown_report({"models": {"gpt2-medium": result}})
    assert "not a new untouched confirmatory study" in text
    assert "not an isolated edge path" in text
    assert "test-07" in text
    assert "unadjusted for multiplicity" in text
    assert "not byte-identical" in text


@pytest.fixture
def saved_full_run(tmp_path):
    """A complete, internally coherent retained run, requiring no inference or model files."""
    rows = synthetic_rows()
    model = "gpt2-medium"
    run = tmp_path / "results/experiment-005" / model
    run.mkdir(parents=True)
    manifest = []
    first = [row for row in rows if row["spec"] == "native_s2_canonical_to_canonical"]
    for row in first:
        pair = row["example_id"].rsplit("-", 1)[0]
        manifest.append(
            {key: row[key] for key in ("example_id", "template_id", "pattern")}
            | {
                "split": "test",
                "io_name": pair + "-a",
                "subject_name": pair + "-b",
                "corrupt_name": pair + "-c",
            }
        )
    selection = {"conditions": {condition: {} for condition in analysis.CONDITIONS}}
    for condition in analysis.CONDITIONS:
        for site in analysis.SITES:
            offset = 1 if condition == "controlled" else 0
            layers = range(1, 5) if site == "s2" else range(10, 14)
            selection["conditions"][condition][site] = {
                "selected_heads": [[layer, head + offset] for layer in layers for head in (0, 1)]
            }
    groups = {
        f"{condition}_{site}": selection["conditions"][condition][site]["selected_heads"]
        for condition in analysis.CONDITIONS
        for site in analysis.SITES
    }
    unions = {
        site: [
            list(head)
            for head in sorted(
                {
                    tuple(head)
                    for condition in analysis.CONDITIONS
                    for head in groups[f"{condition}_{site}"]
                }
            )
        ]
        for site in analysis.SITES
    }
    old_baselines = {}
    old_patches = []
    for row in rows:
        if not row["spec"].startswith("native_"):
            continue
        old_baselines[(row["example_id"], row["target"])] = {
            "example_id": row["example_id"],
            "condition": row["target"],
            "clean_probability_difference": row["clean"],
            "corrupt_probability_difference": row["baseline"],
            "clean_logit_difference": row["clean_logit"],
            "corrupt_logit_difference": row["baseline_logit"],
        }
        old_patches.append(
            {
                "example_id": row["example_id"],
                "site": row["site"],
                "control": "correct",
                "source_condition": row["source"],
                "target_condition": row["target"],
                "target_patched_probability_difference": row["patched"],
                "target_patched_logit_difference": row["patched_logit"],
            }
        )
    source_contents = {
        "data/experiment-003-manifest.jsonl": "\n".join(map(json.dumps, manifest)),
        "results/experiment-002/localization/selected-heads.json": json.dumps(selection),
        "results/experiment-003/test/behavior-per-example.jsonl": "\n".join(
            map(json.dumps, old_baselines.values())
        ),
        "results/experiment-003/test/patches-per-example.jsonl": "\n".join(
            map(json.dumps, old_patches)
        ),
        "docs/EXPERIMENT-005.md": "Synthetic verification fixture protocol.\n",
        "scripts/run_experiment_005.py": "# Synthetic verifier fixture only.\n",
        "src/circuit_transport/experiment_005.py": "# Synthetic verifier fixture only.\n",
        "src/circuit_transport/experiment.py": "# Synthetic verifier fixture only.\n",
        "src/circuit_transport/pythia.py": "# Synthetic verifier fixture only.\n",
        "models/gpt2-medium/config.json": "{}\n",
        "models/gpt2-medium/model.safetensors": "Synthetic placeholder; no model is loaded.\n",
        "data/model-manifest.json": "{}\n",
    }
    for relative, text in source_contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    config = {
        "status": "complete",
        "smoke": False,
        "study": "005",
        "model": model,
        "dtype": "float32",
        "seed": verifier.SEED,
        "rows": 8960,
        "started_at_utc": "2026-09-05T00:00:00+00:00",
        "completed_at_utc": "2026-09-05T00:01:00+00:00",
        "historical_endpoint_tolerances": {"probability": 0.001, "logit": 0.01},
        "input_sha256": {
            relative: verifier.sha(tmp_path / relative) for relative in source_contents
        },
        "specs": [{"spec": name, **spec} for name, spec in analysis.expected_specs().items()],
        "donors": verifier.expected_donors(manifest),
        "groups": groups,
        "unions": unions,
        "random_groups": {
            f"{site}_{draw}": verifier.sampled_heads(
                groups[f"canonical_{site}"],
                unions[site],
                verifier.stable_seed(verifier.SEED, model, site, "heads", draw),
            )
            for site in analysis.SITES
            for draw in analysis.DRAWS
        },
        "block_groups": {
            str(draw): verifier.sampled_heads(
                groups["controlled_end"],
                unions["end"],
                verifier.stable_seed(verifier.SEED, model, "block", draw),
            )
            for draw in analysis.DRAWS
        },
        "matched_norm_audit": {
            site: [
                {
                    "example_id": row["example_id"],
                    "canonical_union_norm": 1.0,
                    "controlled_union_norm": 1.0,
                    "shared_norm": 1.0,
                }
                for row in first
            ]
            for site in analysis.SITES
        },
        "checks": {
            "baseline_error": {"probability": 0.0, "logit": 0.0},
            "native_probability_max_error": 0.0,
            "native_logit_max_error": 0.0,
            "baseline_clamp_max_error": 0.0,
            "baseline_clamp_logit_max_error": 0.0,
            "zero_dose_max_error": 0.0,
            "zero_dose_logit_max_error": 0.0,
            "matched_norm_relative_max_error": 0.0,
            "zero_shared_norm_rows": 0,
            "complete_row_coverage": True,
        },
    }

    def save():
        data_path = run / "patches-per-example.jsonl"
        data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        config["output_sha256"] = verifier.sha(data_path)
        (run / "run-config.json").write_text(json.dumps(config))

    save()
    return tmp_path, run, rows, config, save


def test_full_run_verification_recomputes_norms_donors_hashes_and_endpoints(saved_full_run):
    root, run, _, _, _ = saved_full_run
    report = verifier.verify_run(run, model="gpt2-medium", project_root=root)
    assert report["status"] == "passed"
    assert report["input_hashes_checked"] == 12
    assert report["geometric_mean_union_norm_rows_checked"] == 320
    assert report["donor_ids_checked"] == 160
    assert max(report["recomputed_endpoint_errors"].values()) == 0


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "smoke",
        "output_hash",
        "source_hash",
        "missing_row",
        "wrong_manifest_ids",
        "donor_identity",
        "donor_seed",
        "head_union",
        "sampled_heads",
        "matched_norm",
        "geometric_norm",
        "random_direction_norm",
        "baseline_logit",
        "runtime_nan",
        "historical_tolerance",
        "inconsistent_runtime_error",
    ],
)
def test_saved_run_verifier_rejects_invalid_scientific_records(saved_full_run, damage):
    root, run, rows, config, save = saved_full_run
    if damage == "status":
        config["status"] = "validation_failed"
    elif damage == "smoke":
        config["smoke"] = True
    elif damage == "source_hash":
        (root / "scripts/run_experiment_005.py").write_text("# Changed after inference.\n")
    elif damage == "missing_row":
        rows.pop()
    elif damage == "wrong_manifest_ids":
        for row in rows:
            row["example_id"] = row["example_id"].replace("test-", "wrong-")
    elif damage == "donor_identity":
        config["donors"] = {key: key for key in config["donors"]}
    elif damage == "donor_seed":
        for key in config["donors"]:
            template, pair, pattern = key.rsplit("-", 2)
            config["donors"][key] = f"{template}-{(int(pair) + 1) % 10:02d}-{pattern}"
    elif damage == "head_union":
        config["unions"]["s2"].pop()
    elif damage == "sampled_heads":
        config["random_groups"]["s2_0"][0][1] = 0
    elif damage == "geometric_norm":
        config["matched_norm_audit"]["s2"][0]["canonical_union_norm"] = 4.0
    elif damage in {"matched_norm", "random_direction_norm"}:
        prefix = "matched_" if damage == "matched_norm" else "control_s2_random_direction"
        next(row for row in rows if row["spec"].startswith(prefix))["injected_norm"] += 0.5
    elif damage == "baseline_logit":
        next(row for row in rows if row["spec"] == "block_selected_baseline")["patched_logit"] += (
            0.1
        )
    elif damage == "runtime_nan":
        config["checks"]["zero_dose_max_error"] = float("nan")
    elif damage == "historical_tolerance":
        config["historical_endpoint_tolerances"]["probability"] = 0.01
    elif damage == "inconsistent_runtime_error":
        config["checks"]["native_probability_max_error"] = 0.0005
    save()
    if damage == "output_hash":
        with (run / "patches-per-example.jsonl").open("a") as sink:
            sink.write("\n")
    with pytest.raises(ValueError):
        verifier.verify_run(run, model="gpt2-medium", project_root=root)
