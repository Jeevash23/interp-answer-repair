"""Saved-array analysis for the correlated Bridge 008 source-coverage follow-up.

No model inference. Reuses the immutable Bridge 007 geometry/bootstrap helpers;
the new scientific estimands are four paired signed source-coverage interactions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / "studies/bridge-007"
sys.path.insert(0, str(BASE / "code"))
from analyze007 import Crossed, paired, validate_rows, cell_masks

SEED, DRAWS, MULTIPLICITY = 2026100408, 50_000, 4
REFS = ("zero", "fit_mean")
VARIANTS = ("narrow", "broad", "matched")
NORM_EPS, NORM_ATOL, NORM_RTOL = 1e-12, 1e-6, 1e-5
LAYER_IDS = list(range(6, 13))


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def describe(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return dict(n=int(values.size), finite_n=int(finite.size),
                mean=float(finite.mean()) if finite.size else None,
                quantiles=np.quantile(finite, [0, .05, .25, .5, .75, .95, 1]).tolist() if finite.size else None,
                quantile_probabilities=[0, .05, .25, .5, .75, .95, 1])


def matching_audit(data, n):
    """Audit supplied source vectors, with no omission of undefined matching rows."""
    np.testing.assert_array_equal(data["layer_ids"], LAYER_IDS)
    narrow, broad = data["narrow_layer_deltas"], data["broad_layer_deltas"]
    matched = data["layer_deltas"]
    assert narrow.shape == broad.shape == matched.shape and narrow.shape[:2] == (n, 7)
    assert narrow.dtype == broad.dtype == matched.dtype == np.float32
    assert np.isfinite(narrow).all() and np.isfinite(broad).all()
    nn = np.linalg.norm(narrow.astype(float), axis=-1)
    bn = np.linalg.norm(broad.astype(float), axis=-1)
    np.testing.assert_allclose(data["narrow_norms"], nn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(data["broad_norms"], bn, rtol=1e-12, atol=1e-12)
    valid = (bn > NORM_EPS) | (nn <= NORM_EPS)
    np.testing.assert_array_equal(data["matched_valid"], valid)
    expected_alpha = np.full(nn.shape, np.nan)
    both_nonzero = (bn > NORM_EPS) & (nn > NORM_EPS)
    expected_alpha[both_nonzero] = nn[both_nonzero] / bn[both_nonzero]
    expected_alpha[valid & ~both_nonzero] = 0
    np.testing.assert_allclose(data["alpha"], expected_alpha, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(matched[valid]).all()
    if (~valid).any():
        assert np.isnan(matched[~valid]).all(), "Undefined matched vectors must not be injectable"
    computed = (broad.astype(float) * expected_alpha[..., None]).astype(np.float32)
    np.testing.assert_allclose(matched[valid], computed[valid], rtol=1e-7, atol=1e-8)
    final_norm = np.linalg.norm(matched[valid].astype(float), axis=-1)
    np.testing.assert_allclose(final_norm, nn[valid], rtol=NORM_RTOL, atol=NORM_ATOL)
    layer_reports = {}
    for i, layer in enumerate(LAYER_IDS):
        layer_reports[str(layer)] = dict(narrow_norm=describe(nn[:, i]), broad_norm=describe(bn[:, i]),
                                         alpha=describe(expected_alpha[:, i]),
                                         undefined=int((~valid[:, i]).sum()))
    return dict(evaluable=bool(valid.all()), undefined_layers=int((~valid).sum()),
                undefined_configurations=int((~valid).any(1).sum()),
                zero_narrow_layers=int((nn <= NORM_EPS).sum()),
                zero_broad_layers=int((bn <= NORM_EPS).sum()),
                maximum_valid_norm_error=float(np.max(np.abs(final_norm-nn[valid]))) if valid.any() else None,
                epsilon=NORM_EPS, absolute_tolerance=NORM_ATOL, relative_tolerance=NORM_RTOL,
                per_layer=layer_reports)


def outcome(interval):
    if not interval["valid"]:
        return "non_evaluable"
    lo, hi = interval["adjusted_ci"]
    if lo > 0:
        return "donorward_increase"
    if hi < 0:
        return "donorward_decrease"
    return "unresolved"


def unavailable(n, reason, draws):
    return dict(valid=False, mean=None, ci95=None, adjusted_ci=None, n=n,
                multiplicity=MULTIPLICITY, seed=SEED, draws=draws, reason=reason)


def analyze_arrays(rows, reference, data, matching=None, draws=DRAWS):
    donors = validate_rows(rows)
    boot = Crossed(rows, draws=draws, seed=SEED)
    masks = cell_masks(rows)
    for required in ("clean", "corrupt", "narrow_wording", "narrow_self", "broad_wording", "broad_self"):
        assert required in data, required
    n = len(rows)
    for name, condition in data.items():
        assert condition["raw"].shape[:2] == (2, n), name
        assert condition["behavior"].shape[0] == n and condition["behavior"].shape[1] >= 4, name
        assert np.isfinite(condition["raw"]).all() and np.isfinite(condition["behavior"]).all(), name
    r = data["clean"]["raw"][..., 0].astype(float)
    d = r[:, donors]
    scale = np.asarray(reference["scale"], float)[:, None]
    assert np.isfinite(scale).all() and (scale > 0).all()
    geometry = {name: paired(condition["raw"][..., 0], r, d, scale) for name, condition in data.items()}
    assert geometry["clean"]["resolvable"].all(), "Base clean eligibility failed; do not analyse as qualified"
    differences, discrepancy = {}, {}
    metrics = dict(donor_indices=donors)
    available = {variant: all(variant + "_" + donor in data for donor in ("wording", "self")) for variant in VARIANTS}
    if matching is not None:
        for donor in ("wording", "self"):
            if not matching[donor]["evaluable"]:
                assert "matched_" + donor not in data, "Undefined matching condition was measured"
        available["matched"] = bool(available["matched"] and all(matching[d]["evaluable"] for d in ("wording", "self")))
    for variant in VARIANTS:
        if not available[variant]:
            continue
        w, s = variant + "_wording", variant + "_self"
        differences[variant] = geometry[w]["score"] - geometry[s]["score"]
        discrepancy[variant] = np.linalg.norm((data[w]["raw"][..., 0].astype(float) - data[s]["raw"][..., 0].astype(float)) / scale, axis=-1) / geometry[w]["pair_norm"]
        metrics[variant + "__signed_axis_difference"] = differences[variant]
        metrics[variant + "__full_signature_discrepancy"] = discrepancy[variant]
    primary, heterogeneity, decisions = {}, {}, {}
    for variant in ("broad", "matched"):
        primary[variant], heterogeneity[variant], decisions[variant] = {}, {}, {}
        if available[variant]:
            interaction = differences[variant] - differences["narrow"]
            metrics[variant + "__interaction"] = interaction
        for i, ref in enumerate(REFS):
            if available[variant]:
                estimate = boot.interval(interaction[i], multiplicity=MULTIPLICITY)
                heterogeneity[variant][ref] = {cell: boot.interval(interaction[i], mask) for cell, mask in masks.items()}
                heterogeneity[variant][ref]["configuration_distribution"] = describe(interaction[i])
                cell_means = [heterogeneity[variant][ref][cell]["mean"] for cell in masks]
                heterogeneity[variant][ref]["cell_mean_range"] = [min(cell_means), max(cell_means)]
            else:
                estimate = unavailable(n, "At least one matched donor condition unavailable; no examples omitted", draws)
                heterogeneity[variant][ref] = None
            primary[variant][ref] = estimate
            decisions[variant][ref] = outcome(estimate)
    conditions = {}
    baseline_y = data["corrupt"]["behavior"][:, 0]
    for name, condition in data.items():
        if name in ("clean", "corrupt"):
            continue
        b = condition["behavior"]
        recovery = b[:, 0] - baseline_y
        conditions[name] = dict(recovery=boot.interval(recovery),
                                recovery_by_cell={cell: boot.interval(recovery, mask) for cell, mask in masks.items()},
                                top1=float(b[:, 3].mean()), pairwise=float(b[:, 2].mean()), geometry={},
                                recovery_distribution=describe(recovery))
        metrics[name + "__recovery"] = recovery
        for i, ref in enumerate(REFS):
            conditions[name]["geometry"][ref] = {}
            for key in ("score", "q_recipient", "q_donor", "margin", "orthogonal"):
                value = geometry[name][key][i]
                conditions[name]["geometry"][ref][key] = dict(average=boot.interval(value),
                    cells={cell: boot.interval(value, mask) for cell, mask in masks.items()}, distribution=describe(value))
                metrics[name + "__" + ref + "__" + key] = value
    secondary = {}
    for variant in VARIANTS:
        if not available[variant]:
            secondary[variant] = None
            continue
        secondary[variant] = {}
        w, s = variant + "_wording", variant + "_self"
        behavioral_difference = data[w]["behavior"][:, 0] - data[s]["behavior"][:, 0]
        for i, ref in enumerate(REFS):
            endpoints = dict(signed_axis_difference=differences[variant][i],
                             absolute_axis_difference=np.abs(differences[variant][i]),
                             full_signature_discrepancy=discrepancy[variant][i],
                             wording_minus_self_recovery=behavioral_difference)
            secondary[variant][ref] = {key: dict(average=boot.interval(value),
                cells={cell: boot.interval(value, mask) for cell, mask in masks.items()}, distribution=describe(value))
                for key, value in endpoints.items()}
    changes = {}
    for variant in ("broad", "matched"):
        changes[variant] = {}
        for donor in ("wording", "self"):
            name, narrow = variant + "_" + donor, "narrow_" + donor
            if name not in data:
                changes[variant][donor] = None
                continue
            recovery_change = data[name]["behavior"][:, 0] - data[narrow]["behavior"][:, 0]
            changes[variant][donor] = dict(recovery=boot.interval(recovery_change),
                recovery_by_cell={cell: boot.interval(recovery_change, mask) for cell, mask in masks.items()}, geometry={})
            for i, ref in enumerate(REFS):
                changes[variant][donor]["geometry"][ref] = {}
                for key in ("q_recipient", "q_donor", "score", "margin"):
                    value = geometry[name][key][i] - geometry[narrow][key][i]
                    changes[variant][donor]["geometry"][ref][key] = dict(average=boot.interval(value),
                        cells={cell: boot.interval(value, mask) for cell, mask in masks.items()})
    supported = all(decisions[v][ref] == "donorward_increase" for v in ("broad", "matched") for ref in REFS)
    result = dict(model="gpt2-medium", families=sorted({r["family"] for r in rows}), n=n,
                  blocks=len(boot.blocks), triples=len(boot.triples), seed=SEED, draws=draws,
                  primary_multiplicity=MULTIPLICITY, source_layers=LAYER_IDS, narrow_heads=12, broad_heads=112,
                  primary=primary, primary_decisions=decisions, all_four_donorward_increases=supported,
                  matched_contrast_evaluable=available["matched"], matching=matching,
                  primary_cellwise=heterogeneity, conditions=conditions, secondary_wording_self=secondary,
                  changes_relative_to_narrow=changes,
                  clean_top1=float(data["clean"]["behavior"][:, 3].mean()), corrupt_top1=float(data["corrupt"]["behavior"][:, 3].mean()),
                  interpretation="Outcome-informed correlated follow-up on the inspected Bridge007 benchmark; primary signed contrasts do not test scope invariance or algorithm transfer",
                  uncertainty="Four-quantity adjusted primary family; all secondary intervals pointwise, conditional on benchmark, fitted references and hypotheses")
    return result, metrics


def self_test():
    rows = [dict(id=f"{t}-{p}-{role}-{w}-{o}", template=t, pair=p, role=role, wording=w, order=o, family=10+w)
            for t in range(3) for p in range(4) for role in (0, 1) for w in (0, 1) for o in (0, 1)]
    n = len(rows)
    donor = validate_rows(rows)
    raw = np.zeros((2, n, 2, 2), dtype=np.float32)
    raw[:, :, 0, 0] = [2*r["wording"] for r in rows]
    b = np.zeros((n, 6), np.float32); b[:, 0] = 2; b[:, 2:4] = 1
    axis = raw[:, donor][..., 0] - raw[..., 0]
    data = dict(clean=dict(raw=raw, behavior=b))
    for name, fraction, score in (("corrupt", .5, 0), ("narrow_self", .10, 1), ("narrow_wording", .12, 1),
                                 ("broad_self", .20, 1), ("broad_wording", .32, 1),
                                 ("matched_self", .15, 1), ("matched_wording", .23, 1)):
        edited = raw.copy(); edited[..., 0] += fraction*axis
        behavior = b.copy(); behavior[:, 0] = score
        data[name] = dict(raw=edited, behavior=behavior)
    result, _ = analyze_arrays(rows, dict(scale=np.ones((2, 2))), data, draws=100)
    for ref in REFS:
        np.testing.assert_allclose(result["primary"]["broad"][ref]["mean"], .10, atol=1e-7)
        np.testing.assert_allclose(result["primary"]["matched"][ref]["mean"], .06, atol=1e-7)
    assert result["all_four_donorward_increases"]
    missing = {k: v for k, v in data.items() if k != "matched_self"}
    result_missing, _ = analyze_arrays(rows, dict(scale=np.ones((2, 2))), missing, draws=100)
    assert not result_missing["all_four_donorward_increases"] and not result_missing["matched_contrast_evaluable"]
    assert result_missing["primary"]["broad"]["zero"]["multiplicity"] == 4
    # A zero mean interaction can mask opposite cell responses; retain all cells.
    cancellation = {k: dict(raw=v["raw"].copy(), behavior=v["behavior"].copy()) for k, v in data.items()}
    signs = np.array([1 if r["order"] == 0 else -1 for r in rows])[None, :, None]
    cancellation["broad_wording"]["raw"][..., 0] = raw[..., 0] + (.22 + .1*signs)*axis
    cancelled, _ = analyze_arrays(rows, dict(scale=np.ones((2, 2))), cancellation, draws=100)
    assert abs(cancelled["primary"]["broad"]["zero"]["mean"]) < 1e-7
    span = cancelled["primary_cellwise"]["broad"]["zero"]["cell_mean_range"]
    assert span[0] < -.09 and span[1] > .09
    # Match sums, not individual head norms; represent undefined cases explicitly.
    narrow = np.ones((2, 7, 3), np.float32)
    broad = narrow * np.float32(2)
    nn, bn = np.linalg.norm(narrow.astype(float), axis=-1), np.linalg.norm(broad.astype(float), axis=-1)
    fixture = dict(narrow_layer_deltas=narrow, broad_layer_deltas=broad, layer_deltas=narrow.copy(),
                   narrow_norms=nn, broad_norms=bn, alpha=nn/bn,
                   matched_valid=np.ones((2, 7), bool), layer_ids=np.array(LAYER_IDS))
    assert matching_audit(fixture, 2)["evaluable"]
    fixture["broad_layer_deltas"][0, 0] = 0; fixture["broad_norms"][0, 0] = 0
    fixture["alpha"][0, 0] = np.nan; fixture["layer_deltas"][0, 0] = np.nan; fixture["matched_valid"][0, 0] = False
    assert not matching_audit(fixture, 2)["evaluable"]
    return dict(status="passed", checks=["known_difference_in_differences", "all_four_adjusted_positive",
                "unavailable_matched_contrast_keeps_family_size", "signed_cancellation_exposed_by_cells",
                "valid_norm_matching", "undefined_norm_matching_retained"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results/analysis")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    source_summary_path = BASE / "results/analysis/summary.json"
    source_summary = read(source_summary_path)
    assert source_summary["qualified"], "Bridge007 clean eligibility failed: do not run this extension"
    selection_path = BASE / "data/selection.json"
    selection = read(selection_path)
    candidates_path = BASE / "data/candidates.json"
    rows = [dict(r, wording=selection["families"].index(r["family"])) for r in read(candidates_path)
            if r["split"] == "test" and r["family"] in selection["families"]]
    assert len(rows) == 3200 and len({r["template"] for r in rows}) == len({r["pair"] for r in rows}) == 20
    data, paths = {}, [source_summary_path, selection_path, candidates_path, Path(__file__), HERE / "docs/PROTOCOL.md"]
    for key, name in (("clean", "clean"), ("corrupt", "corrupt"), ("narrow_wording", "wording"), ("narrow_self", "self")):
        path = BASE / "results/test" / (name + ".npz")
        data[key] = arrays(path); paths.append(path)
        record = read(path.with_name(name + "-record.json"))
        assert record["ids"] == [r["id"] for r in rows] and record["status"] == "completed"
        assert digest(path) == record["aggregate_sha256"]
    matching = {}
    for donor in ("wording", "self"):
        path = HERE / "results/deltas" / (donor + ".npz")
        matching[donor] = matching_audit(arrays(path), len(rows)); paths.append(path)
    non_evaluable = {}
    for variant in ("broad", "matched"):
        for donor in ("wording", "self"):
            name = variant + "_" + donor
            path = HERE / "results/test" / (name + ".npz")
            if path.exists():
                if variant == "matched":
                    assert matching[donor]["evaluable"], "Undefined matching condition cannot be measured"
                data[name] = arrays(path); paths.append(path)
            else:
                marker = path.with_name(name + "-non-evaluable.json")
                assert variant == "matched" and marker.exists(), f"Missing measurement: {name}"
                assert not all(matching[k]["evaluable"] for k in ("wording", "self")), "Unexplained skipped matched condition"
                non_evaluable[name] = read(marker); paths.append(marker)
    result, metrics = analyze_arrays(rows, selection["reference"], data, matching)
    result["non_evaluable_records"] = non_evaluable
    result["base_study"] = "bridge-007"
    result["base_primary_success_did_not_gate_extension"] = True
    result["utc"] = datetime.now(timezone.utc).isoformat()
    result["input_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    write(args.output / "summary.json", result)
    np.savez_compressed(args.output / "per-example.npz", **metrics)
    print(json.dumps(dict(output=str(args.output), primary_decisions=result["primary_decisions"],
                         all_four_donorward_increases=result["all_four_donorward_increases"])))


if __name__ == "__main__":
    main()
