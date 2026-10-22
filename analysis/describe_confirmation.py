"""Describe Bridge 007's configuration-level distributions from audited inputs.

No inference, fitting, bootstrap, or new hypothesis test is performed. The
frozen protocol requested distributions as well as cellwise means; this later
reporting helper supplies empirical quantiles and descriptive exceedance counts.
It independently derives the geometry from raw signatures rather than treating
the unaudited per-example archive as its source of truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/bridge-007"
OUTPUT = ROOT / "analysis/results/confirmation-distributions.json"
REFS = ("zero", "fit_mean")
QUANTILES = ("min", "p05", "p25", "median", "p75", "p95", "max")
PROBABILITIES = (0, .05, .25, .5, .75, .95, 1)
METRICS = (("signed_axis_difference", None, None),
           ("absolute_axis_difference", "mean_absolute_axis_difference", .10),
           ("full_signature_distance", "mean_full_signature_distance", .50))
ATOL = 2e-12


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def matched_rows(candidates, families):
    rows = [dict(r, wording=families.index(r["family"])) for r in candidates
            if r["split"] == "test" and r["family"] in families]
    keys = [(r["template"], r["pair"], r["role"], r["wording"], r["order"]) for r in rows]
    blocks, triples = sorted({k[0] for k in keys}), sorted({k[1] for k in keys})
    expected = {(t, p, role, w, o) for t in blocks for p in triples
                for role in (0, 1) for w in (0, 1) for o in (0, 1)}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    lookup = {key: i for i, key in enumerate(keys)}
    donors = np.array([lookup[t, p, role, 1-w, o] for t, p, role, w, o in keys])
    assert np.array_equal(donors[donors], np.arange(len(rows)))
    return rows, donors, blocks, triples


def geometry(clean, wording, self_edit, scale, donors):
    """Directly project the wording-minus-self difference onto each clean axis."""
    clean, wording, self_edit = [np.asarray(x, np.float64) for x in (clean, wording, self_edit)]
    scale = np.asarray(scale, np.float64)[:, None, :]
    assert clean.shape == wording.shape == self_edit.shape
    assert scale.shape == (2, 1, clean.shape[-1]) and (scale > 0).all()
    assert all(np.isfinite(x).all() for x in (clean, wording, self_edit, scale))
    axis = (clean[:, donors] - clean) / scale
    difference = (wording - self_edit) / scale
    norm_squared = np.sum(axis * axis, axis=-1)
    assert np.all(norm_squared >= 1e-12), "Unresolvable endpoint; no rows may be dropped"
    signed = np.sum(difference * axis, axis=-1) / norm_squared
    full = np.sqrt(np.sum(difference * difference, axis=-1) / norm_squared)
    return dict(signed_axis_difference=signed, absolute_axis_difference=np.abs(signed),
                full_signature_distance=full)


def describe(values, threshold):
    values = np.asarray(values, np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    result = dict(n=len(values), mean=float(values.mean()),
                  quantiles=dict(zip(QUANTILES, np.quantile(values, PROBABILITIES,
                                                         method="linear").tolist())))
    if threshold is not None:
        result.update(descriptive_threshold=threshold,
                      n_strictly_above=int(np.count_nonzero(values > threshold)))
    return result


def compute():
    sources = {}

    def bind(path, expected=None):
        path = Path(path)
        actual = digest(path)
        if expected is not None:
            assert actual == expected, f"Source changed: {path}"
        sources[str(path.relative_to(ROOT))] = actual
        return actual

    audit_path = STUDY / "results/verification.json"
    bind(audit_path)
    audit = read(audit_path)
    assert audit["status"] == "passed" and audit["outcome"] == "final_test_completed"
    assert audit["stopping_branch_verified"] and audit["checks"]["primary"]["endpoints"] == 16
    bind(STUDY / "code/verify007.py", audit["helper_sha256"])
    summary_path = STUDY / "results/analysis/summary.json"
    bind(summary_path, audit["analysis_summary_sha256"])
    summary = read(summary_path)
    assert summary["n"] == 3200 and summary["blocks"] == summary["triples"] == 20
    assert summary["primary_multiplicity"] == 16 and summary["qualified"]
    for name, expected in summary["input_hashes"].items():
        bind(ROOT / name, expected)
    # Bind the candidate bank, selection/reference and numerical definitions to
    # the two canonical freezes already checked by the independent verifier.
    freezes = {}
    for name in ("input-freeze.json", "test-freeze.json"):
        path = STUDY / "data" / name
        bind(path, audit["checks"]["freezes"][name]["sha256"])
        freezes[name] = read(path)
    for name, frozen in (("data/candidates.json", "input-freeze.json"),
                         ("code/analyze007.py", "input-freeze.json"),
                         ("data/selection.json", "test-freeze.json")):
        path = STUDY / name
        bind(path, freezes[frozen]["files"][str(path.relative_to(ROOT))])
    selection = read(STUDY / "data/selection.json")
    families = selection["families"]
    assert families == summary["families"] == [10, 11]
    rows, donors, blocks, triples = matched_rows(read(STUDY / "data/candidates.json"), families)
    assert len(rows) == 3200 and len(blocks) == len(triples) == 20
    signatures = {}
    checked_records = {r["path"]: r for r in audit["checks"]["runner_records"]}
    for condition in ("clean", "wording", "self"):
        path = STUDY / "results/test" / (condition + ".npz")
        record_path = path.with_name(condition + "-record.json")
        bind(record_path)
        record = read(record_path)
        canonical = checked_records[str(record_path.relative_to(STUDY))]
        assert canonical["rows_verified"] and canonical["n"] == len(rows)
        bind(path, canonical["aggregate_sha256"])
        assert record["aggregate_sha256"] == canonical["aggregate_sha256"]
        assert record["status"] == "completed" and record["ids"] == [r["id"] for r in rows]
        assert record["inputs"]["rows_sha256"] == json_hash(rows)
        assert record["inputs"]["freeze_sha256"] == sources[str((STUDY / "data/test-freeze.json").relative_to(ROOT))]
        raw = load(path)["raw"]
        assert raw.shape == (2, len(rows), 176, 2)
        signatures[condition] = raw[..., 0]
    values = geometry(signatures["clean"], signatures["wording"], signatures["self"],
                      selection["reference"]["scale"], donors)
    saved_path = STUDY / "results/analysis/per-example.npz"
    bind(saved_path)
    saved = load(saved_path)
    assert np.array_equal(saved["donor_indices"], donors)
    errors, mean_checks = {}, {}
    result = {}
    masks = {f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
             for w in (0, 1) for o in (0, 1)}
    assert all(int(mask.sum()) == 800 for mask in masks.values())
    for metric, primary, threshold in METRICS:
        saved_key = "wording_minus_self_axis" if metric == "signed_axis_difference" else metric
        np.testing.assert_allclose(values[metric], saved[saved_key], rtol=0, atol=ATOL)
        errors[metric] = float(np.max(np.abs(values[metric] - saved[saved_key])))
        for ri, ref in enumerate(REFS):
            reference = result.setdefault(ref, {})
            reference[metric] = dict(pooled=describe(values[metric][ri], threshold),
                cells={cell: describe(values[metric][ri, mask], threshold) for cell, mask in masks.items()})
            for cell, mask in masks.items():
                expected = summary["per_cell_heterogeneity"][ref][cell][metric]["mean"]
                np.testing.assert_allclose(values[metric][ri, mask].mean(), expected, rtol=0, atol=ATOL)
            if primary:
                estimate = float(values[metric][ri].mean())
                audited = audit["checks"]["primary"]["reproduced"][primary + "/" + ref]["mean"]
                reported = summary["primary"][primary][ref]["mean"]
                np.testing.assert_allclose([estimate, reported], audited, rtol=0, atol=ATOL)
                mean_checks[primary + "/" + ref] = dict(recomputed=estimate,
                    primary_summary=reported, independent_audit=audited,
                    maximum_absolute_error=float(max(abs(estimate-audited), abs(estimate-reported))))
    bind(Path(__file__).resolve())
    return dict(schema=1, study="Bridge 007", families=families, n=len(rows),
        blocks=len(blocks), triples=len(triples), configurations_per_cell=800,
        scope="Descriptive completion of the frozen protocol's distribution reporting. "
              "No new inference, bootstrap, test, gate, or equivalence decision. Threshold exceedances "
              "describe cases relative to the primary average resolutions; they are not casewise failures.",
        units="Matched clean-wording separation units under the frozen fitting-only coordinate scale",
        quantile_method="NumPy linear empirical quantiles, probabilities 0, .05, .25, .50, .75, .95, 1",
        weighting="Every retained configuration once; the four equally sized family/order cells have equal pooled weight",
        threshold_rule="Strictly greater than .10 for absolute-axis discrepancy and .50 for full-signature discrepancy",
        distributions=result,
        validation=dict(status="passed", geometry="Direct projection of wording-minus-self onto matched clean axis",
            input_binding="Raw aggregate hashes appear in both canonical runner audit and audited primary summary; "
                          "candidate bank and reference also match canonical input/test freezes",
            per_example_archive_role="Cross-check only; independently reconstructed from audit-bound inputs",
            per_example_max_absolute_errors=errors, primary_mean_checks=mean_checks,
            all_cell_means_match_primary_summary=True, absolute_tolerance=ATOL),
        sources_sha256=dict(sorted(sources.items())))


def load_verified(path=OUTPUT, *, recompute=False):
    result = read(path)
    assert result["validation"]["status"] == "passed" and result["n"] == 3200
    assert result["sources_sha256"]["analysis/describe_confirmation.py"] == digest(Path(__file__))
    for name, expected in result["sources_sha256"].items():
        assert digest(ROOT / name) == expected, f"Descriptive source changed: {name}"
    if recompute:
        assert result == compute(), "Saved descriptive report differs from audited-input reconstruction"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Independently reconstruct and compare without writing")
    args = parser.parse_args()
    result = compute()
    if args.check or args.output.exists():
        assert read(args.output) == result, "Different existing report; use a fresh --output for review"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".partial")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps(dict(status="passed", output=str(args.output), n=result["n"],
                          distributions=30, no_resampling=True)))


if __name__ == "__main__":
    main()
