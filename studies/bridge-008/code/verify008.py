"""Independent saved-evidence verification of the correlated Bridge 008 follow-up.

The four contrasts are independently derived from state differences and clean
axes. Confidence intervals use explicit resampled grids, not either analyzer.
PyTorch is used only for the original float32 narrow-layer reduction order;
this script never loads a model or performs new model inference.
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
# Computational audit helpers only; no scientific analysis or runner is imported.
from verify007 import ExplicitBootstrap, Hashes, array_hash, json_hash, selected_rows, donors

SEED, DRAWS, MULTIPLICITY = 2026100408, 50_000, 4
LAYERS = list(range(6, 13))
REFS = ("zero", "fit_mean")
EPS, ATOL, RTOL = 1e-12, 1e-6, 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def freeze_audit(hasher):
    path = HERE / "data/input-freeze.json"
    value = read(path)
    assert value["files"]
    for name, expected in value["files"].items():
        target = Path(name) if Path(name).is_absolute() else ROOT / name
        assert hasher(target) == (expected["sha256"] if isinstance(expected, dict) else expected), name
    return dict(sha256=hasher(path), utc=value["utc"], files=len(value["files"]))


def compare_record(path, canonical_rows, freeze, bank, delta, hasher):
    record_path = path.with_name(path.stem + "-record.json")
    record = read(record_path)
    assert record["status"] == "completed" and record["ids"] == [r["id"] for r in canonical_rows]
    meta = record["inputs"]
    assert json_hash(meta) == record["input_key"]
    assert meta["operator_representation"] == "seven_layer_residual_sums_at_s2" and meta["layer_ids"] == LAYERS
    assert meta["freeze_sha256"] == freeze["sha256"]
    assert datetime.fromisoformat(freeze["utc"]) <= datetime.fromisoformat(record["started"])
    assert datetime.fromisoformat(record["started"]) <= datetime.fromisoformat(record["finished"])
    assert json_hash(canonical_rows) == meta["rows_sha256"], record_path
    for name, sha in meta["source_sha256"].items():
        assert hasher(ROOT / name) == sha, name
    assert meta["bank_sha256"] == (None if bank is None else array_hash(bank))
    assert meta["layer_delta_sha256"] == (None if delta is None else array_hash(delta))
    assert hasher(path) == record["aggregate_sha256"]
    aggregate = load(path)
    expected_indices = []
    for length in sorted({len(r["token_ids"]) for r in canonical_rows}):
        indices = [i for i, r in enumerate(canonical_rows) if len(r["token_ids"]) == length]
        expected_indices.extend(np.array(indices[start:start+meta["batch_size"]])
                                for start in range(0, len(indices), meta["batch_size"]))
    assert len(expected_indices) == len(record["hashes"]) == record["completed_chunks"]
    for i, indices in enumerate(expected_indices):
        chunk_path = path.with_name(f"{path.stem}-chunk{i:04d}.npz")
        assert hasher(chunk_path) == record["hashes"][chunk_path.name], chunk_path
        chunk = load(chunk_path)
        assert str(chunk.pop("_resume_input_key").item()) == record["input_key"]
        np.testing.assert_array_equal(chunk.pop("_resume_indices"), indices)
        assert set(chunk) == set(aggregate)
        for key, value in chunk.items():
            assert np.isfinite(value).all()
            np.testing.assert_array_equal(value, aggregate[key][:, indices] if key == "raw" else aggregate[key][indices])
    return aggregate, dict(path=str(path.relative_to(HERE)), n=len(canonical_rows), chunks=len(expected_indices),
                           aggregate_sha256=record["aggregate_sha256"], input_key=record["input_key"])


def actual_narrow(clean, corrupt, source_heads):
    import torch  # Float32 reduction arithmetic only; no model is imported or loaded.
    assert clean.dtype == corrupt.dtype == np.float32 and clean.shape == corrupt.shape
    assert len(source_heads) == clean.shape[1] == 12
    individual = torch.from_numpy(clean-corrupt)
    # This exact reduction order is required to reproduce the actual saved edit.
    sums = []
    for layer in LAYERS:
        indices = [i for i, head in enumerate(source_heads) if head[0] == layer]
        assert indices
        sums.append(individual[:, indices].sum(dim=1))
    return torch.stack(sums, dim=1).numpy()


def audit_delta(artifact, narrow, broad, rows):
    np.testing.assert_array_equal(artifact["layer_ids"], LAYERS)
    np.testing.assert_array_equal(artifact["narrow_layer_deltas"], narrow)
    np.testing.assert_array_equal(artifact["broad_layer_deltas"], broad)
    n2 = np.sum(narrow.astype(np.float64)**2, axis=-1)
    b2 = np.sum(broad.astype(np.float64)**2, axis=-1)
    nn, bn = np.sqrt(n2), np.sqrt(b2)
    np.testing.assert_allclose(artifact["narrow_norms"], nn, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(artifact["broad_norms"], bn, atol=1e-12, rtol=1e-12)
    valid = (bn > EPS) | (nn <= EPS)
    np.testing.assert_array_equal(artifact["matched_valid"], valid)
    alpha = np.full(nn.shape, np.nan)
    alpha[valid] = 0
    active = (bn > EPS) & (nn > EPS)
    alpha[active] = np.sqrt(n2[active]/b2[active])
    np.testing.assert_allclose(artifact["alpha"], alpha, atol=1e-12, rtol=1e-12, equal_nan=True)
    # Independent square-root ratio can differ by one float64 ulp; compare the
    # scientifically injected vector at a tighter tolerance than the dose rule.
    expected = (broad.astype(float)*alpha[..., None]).astype(np.float32)
    np.testing.assert_allclose(artifact["layer_deltas"], expected, rtol=2e-7, atol=1e-8, equal_nan=True)
    final_norm = np.sqrt(np.sum(artifact["layer_deltas"].astype(float)**2, axis=-1))
    error = np.abs(final_norm-nn)
    np.testing.assert_allclose(artifact["norm_error"], error, atol=1e-12, rtol=1e-12, equal_nan=True)
    assert np.all(error[valid] <= ATOL + RTOL*nn[valid])
    if (~valid).any():
        assert np.isnan(artifact["layer_deltas"][~valid]).all()
    bad = np.argwhere(~valid)
    return dict(evaluable=bool(valid.all()), invalid_layer_count=len(bad),
                invalid_example_count=len(set(int(i) for i, _ in bad)),
                invalid=[dict(id=rows[int(i)]["id"], row=int(i), layer=LAYERS[int(j)]) for i, j in bad],
                both_zero_count=int(((bn <= EPS) & (nn <= EPS)).sum()),
                zero_narrow_nonzero_broad_count=int(((bn > EPS) & (nn <= EPS)).sum()),
                maximum_valid_norm_error=float(error[valid].max()) if valid.any() else None)


def primary_audit(rows, reference, data, summary, draws=DRAWS):
    mapping = donors(rows)
    r = data["clean"]["raw"][..., 0].astype(np.float64)
    d = r[:, mapping]
    scale = np.asarray(reference["scale"], float)[:, None]
    axis = (d-r)/scale
    square_norm = np.sum(axis**2, axis=-1)
    assert np.all(square_norm >= 1e-12)
    narrow_difference = data["narrow_wording"]["raw"][..., 0].astype(float)-data["narrow_self"]["raw"][..., 0].astype(float)
    boot = ExplicitBootstrap(rows, seed=SEED, draws=draws)
    computed, decisions = {}, {}
    largest_mean_error = largest_interval_error = 0.
    for variant in ("broad", "matched"):
        computed[variant], decisions[variant] = {}, {}
        available = all(variant+"_"+kind in data for kind in ("wording", "self"))
        if available:
            # Derive each interaction from one difference-in-differences vector,
            # independently of the analyzer's separate S(E) computations.
            total_difference = (data[variant+"_wording"]["raw"][..., 0].astype(float)
                                - data[variant+"_self"]["raw"][..., 0].astype(float)-narrow_difference)/scale
            interaction = np.sum(total_difference*axis, axis=-1)/square_norm
        for i, ref in enumerate(REFS):
            saved = summary["primary"][variant][ref]
            if not available:
                assert not saved["valid"] and saved["mean"] is None and saved["multiplicity"] == MULTIPLICITY
                computed[variant][ref] = dict(valid=False, reason="matched comparison unavailable; fixed family retained")
                decisions[variant][ref] = "non_evaluable"
                continue
            value = boot.interval(interaction[i], multiplicity=MULTIPLICITY)
            for key in ("mean", "ci95", "adjusted_ci"):
                np.testing.assert_allclose(value[key], saved[key], rtol=0, atol=2e-10, err_msg=f"{variant}/{ref}/{key}")
            largest_mean_error = max(largest_mean_error, abs(value["mean"]-saved["mean"]))
            largest_interval_error = max(largest_interval_error, float(np.max(np.abs(np.asarray(value["adjusted_ci"])-saved["adjusted_ci"]))))
            lo, hi = value["adjusted_ci"]
            decision = "donorward_increase" if lo > 0 else ("donorward_decrease" if hi < 0 else "unresolved")
            decisions[variant][ref], computed[variant][ref] = decision, value
            for wording in (0, 1):
                for order in (0, 1):
                    mask = np.array([r["wording"] == wording and r["order"] == order for r in rows])
                    cell = f"w{wording}_o{order}"
                    secondary = boot.interval(interaction[i], mask, multiplicity=1)
                    for field in ("mean", "ci95"):
                        np.testing.assert_allclose(secondary[field], summary["primary_cellwise"][variant][ref][cell][field], rtol=0, atol=2e-10)
    assert decisions == summary["primary_decisions"]
    supported = all(decisions[v][ref] == "donorward_increase" for v in ("broad", "matched") for ref in REFS)
    assert supported == summary["all_four_donorward_increases"]
    return dict(draws=draws, seed=SEED, multiplicity=MULTIPLICITY, reproduced=computed,
                all_four_donorward_increases=supported, primary_cell_intervals_reproduced=True,
                maximum_point_error=largest_mean_error, maximum_adjusted_interval_error=largest_interval_error)


def calibration_audit(hasher, freeze, source_heads):
    record_path = HERE / "results/calibration.json"
    record = read(record_path)
    assert record["passed"] and record["tolerance"] == .0005 and record["freeze_sha256"] == freeze["sha256"]
    assert record["n"] == 8 and len(set(record["ids"])) == 8
    assert max(record["checks"].values()) < .0005
    results = dict(record_sha256=hasher(record_path), errors={})
    for name in ("broad_wording", "matched_wording"):
        path = HERE / "results/calibration" / (name + ".npz")
        a = load(path)
        errors = dict(signature=float(np.max(np.abs(a["original"]-a["prefix"]))),
                      behavior=float(np.max(np.abs(a["original_behavior"]-a["prefix_behavior"]))))
        assert max(errors.values()) < .0005
        np.testing.assert_allclose(errors["signature"], record["checks"][name+"_signature_error"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(errors["behavior"], record["checks"][name+"_behavior_error"], rtol=0, atol=1e-12)
        results["errors"][name] = errors
        results[name+"_sha256"] = hasher(path)
    source_path = HERE / "results/calibration/source-validation.npz"
    source = load(source_path)
    np.testing.assert_array_equal(source["layer_ids"], LAYERS)
    np.testing.assert_array_equal(source["selected_source_heads"], source_heads)
    assert source["ids"].tolist() == record["ids"]
    family_ids = read(BASE / "data/selection.json")["families"]
    calibration_rows = selected_rows(read(BASE / "data/candidates.json"), "calibration", family_ids)
    chosen = [next(r for r in calibration_rows if r["wording"] == w and r["order"] == o and r["role"] == role)
              for w in (0, 1) for o in (0, 1) for role in (0, 1)]
    assert [r["id"] for r in chosen] == record["ids"]
    np.testing.assert_array_equal(source["donor_indices"], donors(chosen))
    for kind in ("clean", "corrupt"):
        # NumPy's reduction is independent; tiny summation-order differences are
        # judged against the frozen numerical tolerance, never assumed identical.
        explicit_sum = np.sum(source[kind+"_projected_heads"], axis=2, dtype=np.float32)
        err = float(np.max(np.abs(explicit_sum-source[kind+"_layer_sums"])))
        assert err < .0005
        results["errors"][kind+"_explicit_sum_numpy"] = err
    explicit_delta = np.sum(source["clean_projected_heads"]-source["corrupt_projected_heads"], axis=2, dtype=np.float32)
    broad = source["clean_layer_sums"]-source["corrupt_layer_sums"]
    err = float(np.max(np.abs(explicit_delta-broad)))
    assert err < .0005
    results["errors"]["explicit_112_delta_numpy"] = err
    np.testing.assert_array_equal(actual_narrow(source["narrow_clean_heads"], source["narrow_corrupt_heads"], source_heads), source["narrow_layer_deltas"])
    err = float(np.max(np.abs(source["original_narrow_behavior"]-source["layer_narrow_behavior"])))
    assert err < .0005
    results["errors"]["narrow_layer_behavior"] = err
    results["source_validation_sha256"] = hasher(source_path)
    explicit_path = HERE / "results/calibration/explicit112-prefix.npz"
    explicit_check = load(explicit_path)
    signature_error = float(np.max(np.abs(explicit_check["explicit_raw"]-explicit_check["efficient_raw"])))
    behavior_error = float(np.max(np.abs(explicit_check["explicit_behavior"]-explicit_check["efficient_behavior"])))
    assert max(signature_error, behavior_error) < .0005
    np.testing.assert_allclose(signature_error, record["checks"]["explicit_112_operator_signature_error"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(behavior_error, record["checks"]["explicit_112_operator_behavior_error"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(explicit_check["explicit_layer_delta"], explicit_delta[source["donor_indices"]], rtol=0, atol=.0005)
    np.testing.assert_array_equal(explicit_check["efficient_layer_delta"], broad[source["donor_indices"]])
    results["errors"]["explicit_112_operator_signature"] = signature_error
    results["errors"]["explicit_112_operator_behavior"] = behavior_error
    results["explicit_operator_sha256"] = hasher(explicit_path)
    return results


def replay_audit(rows, data, deltas, hasher, freeze):
    path = HERE / "results/computational-replay.json"
    record = read(path)
    assert record["passed"] and record["fresh_engine"] and record["head_chunk"] == 1
    assert record["freeze_sha256"] == freeze["sha256"] and record["tolerance"] == .0005
    expected_indices = np.array([i for w in (0, 1) for o in (0, 1)
                                for i in [j for j, r in enumerate(rows) if r["wording"] == w and r["order"] == o][:2]])
    np.testing.assert_array_equal(record["indices"], expected_indices)
    assert record["n"] == 8 and record["ids"] == [rows[i]["id"] for i in expected_indices]
    result = dict(record_sha256=hasher(path), n=8, conditions={})
    for name, check in record["checks"].items():
        if check["status"] == "non_evaluable":
            assert name not in data
            marker = HERE / "results/test" / (name+"-non-evaluable.json")
            assert hasher(marker) == check["marker_sha256"]
            result["conditions"][name] = dict(status="non_evaluable", marker_sha256=check["marker_sha256"])
            continue
        saved = HERE / "results/verification" / (name+"-singleton.npz")
        assert hasher(saved) == check["saved_sha256"]
        fixture = load(saved)
        np.testing.assert_array_equal(fixture["indices"], expected_indices)
        variant, donor = name.split("_")
        expected_delta = deltas[donor]["layer_deltas" if variant == "matched" else "broad_layer_deltas"]
        np.testing.assert_array_equal(fixture["layer_deltas"], expected_delta[expected_indices])
        se = float(np.max(np.abs(fixture["raw"]-data[name]["raw"][:, expected_indices])))
        be = float(np.max(np.abs(fixture["behavior"]-data[name]["behavior"][expected_indices])))
        assert max(se, be) < .0005
        np.testing.assert_allclose([se, be], [check["signature_error"], check["behavior_error"]], rtol=0, atol=1e-12)
        result["conditions"][name] = dict(status="passed", signature_error=se, behavior_error=be, saved_sha256=hasher(saved))
    return result


def self_test():
    rows = [dict(id=f"{t}-{p}-{role}-{w}-{o}", template=t, pair=p, role=role, wording=w, order=o)
            for t in range(3) for p in range(4) for role in (0, 1) for w in (0, 1) for o in (0, 1)]
    n = len(rows)
    r = np.zeros((2, n, 2, 2), float)
    r[:, :, 0, 0] = [2*x["wording"] for x in rows]
    axis = r[:, donors(rows)][..., 0]-r[..., 0]
    data = dict(clean=dict(raw=r))
    for name, amount in (("narrow_wording", .2), ("narrow_self", .1), ("broad_wording", .4),
                         ("broad_self", .1), ("matched_wording", .3), ("matched_self", .1)):
        raw = r.copy(); raw[..., 0] += amount*axis; data[name] = dict(raw=raw)
    summary = dict(primary={}, primary_decisions={}, primary_cellwise={}, all_four_donorward_increases=True)
    for variant, expected in (("broad", .2), ("matched", .1)):
        summary["primary"][variant], summary["primary_decisions"][variant], summary["primary_cellwise"][variant] = {}, {}, {}
        for ref in REFS:
            summary["primary"][variant][ref] = dict(mean=expected, ci95=[expected, expected], adjusted_ci=[expected, expected], valid=True)
            summary["primary_decisions"][variant][ref] = "donorward_increase"
            summary["primary_cellwise"][variant][ref] = {f"w{w}_o{o}": dict(mean=expected, ci95=[expected, expected]) for w in (0, 1) for o in (0, 1)}
    verified = primary_audit(rows, dict(scale=np.ones((2, 2))), data, summary, draws=100)
    return dict(status="passed", scope="Synthetic known signed interactions; no model inference",
                maximum_point_error=verified["maximum_point_error"],
                maximum_adjusted_interval_error=verified["maximum_adjusted_interval_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results/verification.json")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    completed_path, summary_path = HERE / "results/inference-complete.json", HERE / "results/analysis/summary.json"
    if not completed_path.exists() or not summary_path.exists() or not (HERE / "results/computational-replay.json").exists():
        status_path = HERE / "results/status.json"
        if status_path.exists() and read(status_path).get("outcome") == "ineligible_base":
            assert not args.output.exists(), "Verification output already exists; use a fresh --output"
            status, design = read(status_path), read(HERE / "data/design.json")
            assert status["status"] == "completed" and not status["test_measured"] and not design["eligible"]
            hasher = Hashes()
            freeze = freeze_audit(hasher)
            assert status["freeze_sha256"] == freeze["sha256"]
            base_summary_path = BASE / "results/analysis/summary.json"
            base_summary = read(base_summary_path)
            base_audit = read(BASE / "results/verification.json")
            assert base_audit["status"] == "passed" and hasher(base_summary_path) == base_audit["analysis_summary_sha256"]
            assert not (base_summary["qualification"]["qualified"] and base_summary["test_qualification"]["qualified"])
            for directory in ("natural", "deltas", "test", "calibration"):
                assert not list((HERE / "results" / directory).glob("*.npz")), "Inference occurred on an ineligible base"
            assert not completed_path.exists() and not summary_path.exists()
            result = dict(status="passed", utc=datetime.now(timezone.utc).isoformat(), outcome="ineligible_base",
                          stopping_branch_verified=True, test_measured=False,
                          scope="Verified clean-eligibility stop; no source-coverage outcomes measured",
                          helper_sha256=hasher(Path(__file__)), computational_dependency_sha256=hasher(BASE / "code/verify007.py"),
                          base_analysis_summary_sha256=hasher(base_summary_path),
                          status_sha256=hasher(status_path), checks=dict(freeze=freeze))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
            print(json.dumps(dict(status="passed", outcome="ineligible_base", output=str(args.output)))); return
        print(json.dumps(dict(status="not_completed", report_written=False))); return
    assert not args.output.exists(), "Verification output already exists; use a fresh --output for replay"
    completed = read(completed_path)
    assert completed["status"] == "completed"
    hasher = Hashes()
    freeze = freeze_audit(hasher)
    assert completed["freeze_sha256"] == freeze["sha256"]
    base_summary = read(BASE / "results/analysis/summary.json")
    assert base_summary["qualified"], "The clean eligibility condition failed"
    selection = read(BASE / "data/selection.json")
    rows = selected_rows(read(BASE / "data/candidates.json"), "test", selection["families"])
    assert len(rows) == completed["n"] == 3200
    bank = load(BASE / "results/fit/bank.npz")["bank"]
    checks = dict(freeze=freeze, calibration=calibration_audit(hasher, freeze, selection["source_heads"]), runner_records=[])
    natural = {}
    for kind in ("clean", "corrupt"):
        rr = rows if kind == "clean" else [dict(r, token_ids=r["corrupt_tokens"]) for r in rows]
        natural[kind], record = compare_record(HERE / "results/natural" / (kind+".npz"), rr, freeze, None, None, hasher)
        checks["runner_records"].append(record)
        np.testing.assert_array_equal(natural[kind]["layer_ids_per_row"], np.tile(LAYERS, (len(rows), 1)))
    old_natural = {kind: load(BASE / "results/test" / ("natural-"+kind+".npz")) for kind in ("clean", "corrupt")}
    for kind in ("clean", "corrupt"):
        np.testing.assert_allclose(natural[kind]["behavior"], old_natural[kind]["behavior"], rtol=0, atol=.0005)
    narrow = actual_narrow(old_natural["clean"]["head_results"], old_natural["corrupt"]["head_results"], selection["source_heads"])
    broad = natural["clean"]["layer_sums"]-natural["corrupt"]["layer_sums"]
    matching_path = HERE / "results/deltas/matching-summary.json"
    matching = read(matching_path)
    assert matching["freeze_sha256"] == freeze["sha256"]
    mapped = donors(rows)
    deltas, checks["matching"] = {}, {}
    for kind, ix in (("wording", mapped), ("self", np.arange(len(rows)))):
        path = HERE / "results/deltas" / (kind+".npz")
        assert hasher(path) == matching["matching"][kind]["delta_sha256"]
        deltas[kind] = load(path)
        audited = audit_delta(deltas[kind], narrow[ix], broad[ix], rows)
        old = matching["matching"][kind]
        assert audited["evaluable"] == old["matched_evaluable"]
        for key in ("invalid_layer_count", "invalid_example_count", "invalid", "both_zero_count", "zero_narrow_nonzero_broad_count"):
            assert audited[key] == old[key], (kind, key)
        checks["matching"][kind] = dict(audited, artifact_sha256=hasher(path))
    checks["matching_summary_sha256"] = hasher(matching_path)
    data = {"narrow_"+kind: load(BASE / "results/test" / (kind+".npz")) for kind in ("wording", "self")}
    data["clean"] = load(BASE / "results/test/clean.npz")
    for variant in ("broad", "matched"):
        for kind in ("wording", "self"):
            name = variant+"_"+kind
            path = HERE / "results/test" / (name+".npz")
            if not path.exists():
                marker = read(path.with_name(name+"-non-evaluable.json"))
                assert variant == "matched" and not checks["matching"][kind]["evaluable"]
                assert completed["conditions"][name] == "non_evaluable" and marker["status"] == "non_evaluable"
                assert marker["freeze_sha256"] == freeze["sha256"]
                continue
            assert completed["conditions"][name] == "completed"
            delta = deltas[kind]["layer_deltas" if variant == "matched" else "broad_layer_deltas"]
            data[name], record = compare_record(path, rows, freeze, bank, delta, hasher)
            checks["runner_records"].append(record)
    summary = read(summary_path)
    checks["primary"] = primary_audit(rows, selection["reference"], data, summary)
    checks["fresh_engine_replay"] = replay_audit(rows, data, deltas, hasher, freeze)
    result = dict(status="passed", utc=datetime.now(timezone.utc).isoformat(),
                  scope="Independent saved-array/provenance verification; no new model inference or scientific endpoint",
                  helper_sha256=hasher(Path(__file__)), computational_dependency_sha256=hasher(BASE / "code/verify007.py"),
                  analysis_summary_sha256=hasher(summary_path), inference_completion_sha256=hasher(completed_path),
                  checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed", output=str(args.output), runner_records=len(checks["runner_records"]),
                         maximum_adjusted_interval_error=checks["primary"]["maximum_adjusted_interval_error"])))


if __name__ == "__main__":
    main()
