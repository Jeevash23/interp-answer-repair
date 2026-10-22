"""Prospective Bridge 007 estimands; NumPy-only, with no model inference.

The primary family has 16 quantities: four recovery means, eight recipient
distance margins, two mean absolute donor-axis differences, and two full
wording/self distances. The latter four average equally over family/order cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SEED = 2026100407
DRAWS = 50_000
MULTIPLICITY = 16
AXIS_BOUND = 0.10
FULL_BOUND = 0.50
REFS = ("zero", "fit_mean")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_for(split, families, candidates=None):
    candidates = read(HERE / "data/candidates.json") if candidates is None else candidates
    return [dict(r, wording=families.index(r["family"])) for r in candidates
            if r["split"] == split and r["family"] in families]


def cell_masks(rows):
    return {f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
            for w in range(2) for o in range(2)}


def validate_rows(rows):
    keys = [(r["template"], r["pair"], r["role"], r["wording"], r["order"]) for r in rows]
    assert len(set(keys)) == len(keys), "Duplicate crossed-design row"
    blocks = sorted({r["template"] for r in rows})
    triples = sorted({r["pair"] for r in rows})
    expected = {(t, p, role, w, o) for t in blocks for p in triples
                for role in range(2) for w in range(2) for o in range(2)}
    assert set(keys) == expected, "Missing or extra paired role/order/wording observations"
    index = {key: i for i, key in enumerate(keys)}
    donor = np.array([index[(t, p, role, 1 - w, o)] for t, p, role, w, o in keys])
    assert np.array_equal(donor[donor], np.arange(len(rows)))
    return donor


class Crossed:
    """Independently resample blocks and disjoint name triples, retaining pairings."""
    def __init__(self, rows, draws=DRAWS, seed=SEED):
        self.rows = rows
        self.blocks = sorted({r["template"] for r in rows})
        self.triples = sorted({r["pair"] for r in rows})
        self.t = np.array([self.blocks.index(r["template"]) for r in rows])
        self.p = np.array([self.triples.index(r["pair"]) for r in rows])
        self.draws, self.seed = draws, seed
        rng = np.random.default_rng(seed)
        self.tt = rng.integers(len(self.blocks), size=(draws, len(self.blocks)))
        self.pp = rng.integers(len(self.triples), size=(draws, len(self.triples)))
        self.wt = np.zeros((draws, len(self.blocks)))
        self.wp = np.zeros((draws, len(self.triples)))
        np.add.at(self.wt, (np.repeat(np.arange(draws), len(self.blocks)), self.tt.ravel()), 1)
        np.add.at(self.wp, (np.repeat(np.arange(draws), len(self.triples)), self.pp.ravel()), 1)

    def grid(self, values, mask=None):
        values = np.asarray(values, dtype=float)
        assert values.shape == (len(self.rows),)
        mask = np.ones(len(self.rows), dtype=bool) if mask is None else np.asarray(mask, bool)
        out = np.full((len(self.blocks), len(self.triples)), np.nan)
        for t in range(len(self.blocks)):
            for p in range(len(self.triples)):
                ix = mask & (self.t == t) & (self.p == p)
                assert ix.any(), "Incomplete block-by-triple grid"
                out[t, p] = values[ix].mean()
        return out

    def resample(self, grid):
        return np.einsum("bi,ij,bj->b", self.wt, grid, self.wp, optimize=True) / grid.size

    def interval(self, values, mask=None, multiplicity=1):
        grid = self.grid(values, mask)
        out = dict(n=int(len(self.rows) if mask is None else np.asarray(mask).sum()),
                   blocks=len(self.blocks), triples=len(self.triples), draws=self.draws,
                   seed=self.seed, multiplicity=multiplicity)
        if not np.isfinite(grid).all():
            return dict(out, mean=None, ci95=None, adjusted_ci=None,
                        valid=False, reason="Unresolvable endpoint; no rows omitted")
        boot = self.resample(grid)
        tail = 0.05 / (2 * multiplicity)
        return dict(out, mean=float(grid.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                    adjusted_ci=np.quantile(boot, [tail, 1 - tail]).tolist(), valid=True)


def fit_reference(raw, rows):
    """Same fit-only centroids and RMS residual scale as Bridge 005."""
    x = np.asarray(raw[..., 0], float)
    labels = np.array([2 * r["wording"] + r["order"] for r in rows])
    means = np.stack([x[:, labels == c].mean(1) for c in range(4)], 1)
    scale = np.maximum(np.sqrt(np.mean((x - means[:, labels]) ** 2, axis=1)), 0.1)
    return dict(scale=scale.tolist(), centers=(means / scale[:, None]).tolist())


def classification(raw, rows, reference):
    x = np.asarray(raw[..., 0], float) / np.asarray(reference["scale"])[:, None]
    centers = np.asarray(reference["centers"], float)
    out = np.zeros(x.shape[:2], dtype=bool)
    axes_valid = np.ones(x.shape[:2], dtype=bool)
    for i, row in enumerate(rows):
        a, b = centers[:, row["order"]], centers[:, 2 + row["order"]]
        axis = b - a
        den = (axis * axis).sum(-1)
        axes_valid[:, i] = den >= 1e-12
        numerator = ((x[:, i] - (a + b) / 2) * axis).sum(-1)
        # Preserve the historical midpoint classification rule; invalidate degenerate axes.
        out[:, i] = ((numerator > 0) == bool(row["wording"])) & axes_valid[:, i]
    return out, axes_valid


def behavior_cells(behavior, rows):
    behavior = np.asarray(behavior)
    result = {}
    for family in sorted({r["family"] for r in rows}):
        result[str(family)] = {}
        for order in range(2):
            mask = np.array([r["family"] == family and r["order"] == order for r in rows])
            b = behavior[mask]
            result[str(family)][str(order)] = dict(n=len(b), pairwise=float(b[:, 2].mean()),
                                                  top1=float(b[:, 3].mean()), logit=float(b[:, 0].mean()))
    return result


def behavior_pass(cells):
    return all(c["pairwise"] >= 0.95 and c["top1"] >= 0.85
               for orders in cells.values() for c in orders.values())


def separation(raw, rows, reference):
    correct, _ = classification(raw, rows, reference)
    return {cell: correct[:, mask].mean(1).tolist() for cell, mask in cell_masks(rows).items()}


def min_separation(cells):
    return min(value for values in cells.values() for value in values)


def qualification_summary(clean, rows, reference, draws=DRAWS):
    donors = validate_rows(rows)
    boot = Crossed(rows, draws=draws)
    behavior = np.asarray(clean["behavior"], float)
    assert behavior.shape[0] == len(rows) and behavior.shape[1] >= 4
    assert np.isfinite(behavior).all() and np.isfinite(clean["raw"]).all()
    correct, valid = classification(clean["raw"], rows, reference)
    raw = np.asarray(clean["raw"][..., 0], float)
    scale = np.asarray(reference["scale"], float)[:, None]
    assert np.isfinite(scale).all() and (scale > 0).all()
    pair_norm = np.linalg.norm((raw[:, donors] - raw) / scale, axis=-1)
    resolvable = pair_norm >= 1e-6
    cells = {}
    for cell, mask in cell_masks(rows).items():
        cells[cell] = dict(
            family=int(next(r["family"] for r, keep in zip(rows, mask) if keep)),
            pairwise=boot.interval(behavior[:, 2], mask),
            top1=boot.interval(behavior[:, 3], mask),
            classification={ref: boot.interval(correct[i].astype(float), mask) for i, ref in enumerate(REFS)})
    bp = all(c[key]["mean"] >= bound for c in cells.values()
             for key, bound in [("pairwise", 0.95), ("top1", 0.85)])
    sp = bool(valid.all() and all(v["mean"] >= 0.80 for c in cells.values()
                               for v in c["classification"].values()))
    return dict(n=len(rows), blocks=len(boot.blocks), triples=len(boot.triples), cells=cells,
                behavior_pass=bp, separation_pass=sp, qualified=bool(bp and sp and resolvable.all()),
                degenerate_classification_axes=int((~valid).sum()),
                unresolvable_reference_examples=int((~resolvable).sum()),
                pair_norm_minimum={ref: float(pair_norm[i].min()) for i, ref in enumerate(REFS)},
                gate_scope="Observed rates; pointwise crossed-bootstrap intervals describe uncertainty and are not gates")


def paired(edited, recipient, donor, scale):
    u = (np.asarray(edited, float) - recipient) / scale
    v = (np.asarray(donor, float) - recipient) / scale
    norm = np.linalg.norm(v, axis=-1)
    valid = norm >= 1e-6
    n = np.where(valid, norm, np.nan)
    score = (u * v).sum(-1) / n ** 2
    qr, qd = np.linalg.norm(u, axis=-1) / n, np.linalg.norm(u - v, axis=-1) / n
    return dict(score=score, q_recipient=qr, q_donor=qd, margin=qr - qd,
                orthogonal=np.linalg.norm(u - score[..., None] * v, axis=-1) / n,
                pair_norm=norm, resolvable=valid)


def passes(interval, side, bound):
    return bool(interval["valid"] and
                (interval["adjusted_ci"][0] > bound if side == "above"
                 else interval["adjusted_ci"][1] < bound))


def analyze_final(selection, rows, data, qualification, draws=DRAWS):
    donor_indices = validate_rows(rows)
    boot, masks = Crossed(rows, draws=draws), cell_masks(rows)
    for name in ("clean", "corrupt", "wording", "self"):
        assert data[name]["behavior"].shape[0] == len(rows)
        assert data[name]["raw"].shape[:2] == (2, len(rows))
        assert np.isfinite(data[name]["raw"]).all() and np.isfinite(data[name]["behavior"]).all()
    clean, corrupt = data["clean"], data["corrupt"]
    r = clean["raw"][..., 0].astype(float)
    d = r[:, donor_indices]
    scale = np.asarray(selection["reference"]["scale"], float)[:, None]
    assert np.isfinite(scale).all() and (scale > 0).all()
    geom = {name: paired(data[name]["raw"][..., 0], r, d, scale)
            for name in ("corrupt", "wording", "self")}
    delta_axis = geom["wording"]["score"] - geom["self"]["score"]
    axis_abs = np.abs(delta_axis)
    full = np.linalg.norm((data["wording"]["raw"][..., 0].astype(float)
                           - data["self"]["raw"][..., 0].astype(float)) / scale, axis=-1)
    full = full / np.where(geom["wording"]["resolvable"], geom["wording"]["pair_norm"], np.nan)
    gain = data["wording"]["behavior"][:, 0] - corrupt["behavior"][:, 0]
    primary = dict(recovery={cell: boot.interval(gain, mask, MULTIPLICITY) for cell, mask in masks.items()},
                   margin={}, mean_absolute_axis_difference={}, mean_full_signature_distance={})
    for i, ref in enumerate(REFS):
        primary["margin"][ref] = {cell: boot.interval(geom["wording"]["margin"][i], mask, MULTIPLICITY)
                                  for cell, mask in masks.items()}
        primary["mean_absolute_axis_difference"][ref] = boot.interval(axis_abs[i], multiplicity=MULTIPLICITY)
        primary["mean_full_signature_distance"][ref] = boot.interval(full[i], multiplicity=MULTIPLICITY)
    test_qualification = qualification_summary(clean, rows, selection["reference"], draws=draws)
    conditions = {}
    metrics = dict(wording_recovery=gain, wording_minus_self_axis=delta_axis,
                   absolute_axis_difference=axis_abs, full_signature_distance=full,
                   donor_indices=donor_indices)
    for name in ("corrupt", "wording", "self"):
        b = data[name]["behavior"]
        rec = b[:, 0] - corrupt["behavior"][:, 0]
        conditions[name] = dict(recovery=boot.interval(rec), top1=float(b[:, 3].mean()),
                                pairwise=float(b[:, 2].mean()), geometry={})
        for i, ref in enumerate(REFS):
            conditions[name]["geometry"][ref] = {key: boot.interval(geom[name][key][i])
                                                    for key in ("score", "q_recipient", "q_donor", "margin", "orthogonal")}
        metrics.update({name + "__" + key: value for key, value in geom[name].items()})
    heterogeneity = {}
    for i, ref in enumerate(REFS):
        heterogeneity[ref] = {cell: dict(signed_axis_difference=boot.interval(delta_axis[i], mask),
                                         absolute_axis_difference=boot.interval(axis_abs[i], mask),
                                         full_signature_distance=boot.interval(full[i], mask),
                                         wording_minus_self_recovery=boot.interval(
                                             data["wording"]["behavior"][:, 0] - data["self"]["behavior"][:, 0], mask))
                                for cell, mask in masks.items()}
    endpoint_checks = {}
    for label, endpoint, target in (("recipient", r, -1), ("donor", d, 1)):
        values = paired(endpoint, r, d, scale)["margin"]
        finite = np.isfinite(values)
        endpoint_checks[label + "_margin_max_error"] = float(np.max(np.abs(values[finite] - target))) if finite.any() else None
        if finite.any():
            np.testing.assert_allclose(values[finite], target, atol=1e-10, rtol=0)
    identity = geom["wording"]["q_donor"] ** 2 - geom["wording"]["q_recipient"] ** 2 - (1 - 2 * geom["wording"]["score"])
    finite = np.isfinite(identity)
    endpoint_checks["squared_distance_identity_max_error"] = float(np.max(np.abs(identity[finite]))) if finite.any() else None
    if finite.any():
        np.testing.assert_allclose(identity[finite], 0, atol=1e-8, rtol=0)
    outcome = dict(
        all_recovery_intervals_positive=all(passes(v, "above", 0) for v in primary["recovery"].values()),
        all_margin_intervals_recipient=all(passes(v, "below", 0) for cells in primary["margin"].values() for v in cells.values()),
        average_absolute_axis_bound_supported=all(passes(v, "below", AXIS_BOUND) for v in primary["mean_absolute_axis_difference"].values()),
        average_full_distance_bound_supported=all(passes(v, "below", FULL_BOUND) for v in primary["mean_full_signature_distance"].values()))
    qualified = bool(selection["status"] == "selected" and qualification["qualified"]
                     and test_qualification["qualified"] and geom["wording"]["resolvable"].all())
    result = dict(model="gpt2-medium", families=selection["families"], n=len(rows),
                  blocks=len(boot.blocks), triples=len(boot.triples), primary_multiplicity=MULTIPLICITY,
                  primary=primary, bounds=dict(mean_absolute_axis_difference=AXIS_BOUND, mean_full_signature_distance=FULL_BOUND),
                  bounds_interpretation="Prespecified assay-resolution bounds on equally weighted averages; not all-cell or algorithmic equivalence",
                  qualification=qualification, test_qualification=test_qualification, qualified=qualified,
                  outcomes=outcome, full_conjunction_supported=qualified and all(outcome.values()),
                  conditions=conditions, per_cell_heterogeneity=heterogeneity,
                  clean_top1=float(clean["behavior"][:, 3].mean()), corrupt_top1=float(corrupt["behavior"][:, 3].mean()),
                  clean_gap=boot.interval(clean["behavior"][:, 0] - corrupt["behavior"][:, 0]),
                  endpoint_checks=endpoint_checks, unresolvable_reference_examples=int((~geom["wording"]["resolvable"]).sum()),
                  uncertainty_scope="Crossed context-block/name-triple bootstrap conditional on selected pair, fixed heads and fitted references; secondary intervals pointwise")
    return result, metrics


def planning_from_old():
    """Development data only: conditional variance projection, not new-pair power."""
    base = ROOT / "studies/bridge-005"
    selection = read(base / "data/selection.json")
    rows = rows_for("test", selection["families"], read(base / "data/candidates.json"))
    donors = validate_rows(rows)
    data = {name: load(base / "results/test" / (name + ".npz")) for name in ("clean", "wording", "self")}
    r = data["clean"]["raw"][..., 0].astype(float)
    scale = np.asarray(selection["reference"]["scale"], float)[:, None]
    g = {name: paired(data[name]["raw"][..., 0], r, r[:, donors], scale) for name in ("wording", "self")}
    absolute = np.abs(g["wording"]["score"] - g["self"]["score"])
    full = np.linalg.norm((data["wording"]["raw"][..., 0].astype(float) - data["self"]["raw"][..., 0].astype(float)) / scale, axis=-1) / g["wording"]["pair_norm"]
    boot = Crossed(rows, draws=2)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * MULTIPLICITY))
    result = dict(scope="Old F6/F7 data used for declared planning only. Conditional precision projection; no new-pair power guarantee.",
                  primary_multiplicity=MULTIPLICITY, normal_critical_value=z, metrics={})
    for i, ref in enumerate(REFS):
        result["metrics"][ref] = {}
        for metric, values in (("mean_absolute_axis_difference", absolute[i]), ("mean_full_signature_distance", full[i])):
            grid = boot.grid(values)
            t, p = grid.shape
            residual = grid - grid.mean(0)[None, :] - grid.mean(1)[:, None] + grid.mean()
            ve = float(np.sum(residual ** 2) / ((t - 1) * (p - 1)))
            vt = max(0.0, float(np.var(grid.mean(1), ddof=1) - ve / p))
            vp = max(0.0, float(np.var(grid.mean(0), ddof=1) - ve / t))
            result["metrics"][ref][metric] = dict(
                old_mean=float(grid.mean()), variance_components=dict(block=vt, triple=vp, interaction=ve),
                projections=[dict(blocks=n, triples=n, standard_error=float(np.sqrt(vt / n + vp / n + ve / n ** 2)),
                                  approximate_adjusted_halfwidth=float(z * np.sqrt(vt / n + vp / n + ve / n ** 2)))
                             for n in (12, 16, 20, 24)])
    return result


def self_test():
    rows = [dict(id=f"{t}-{p}-{role}-{w}-{o}", template=t, pair=p, role=role, wording=w, order=o, family=10+w)
            for t in range(3) for p in range(4) for role in range(2) for w in range(2) for o in range(2)]
    boot = Crossed(rows, draws=100)
    values = np.array([r["template"] + 2 * r["pair"] for r in rows], float)
    grid = boot.grid(values)
    direct = grid[boot.tt[:, :, None], boot.pp[:, None, :]].mean((1, 2))
    np.testing.assert_allclose(boot.resample(grid), direct, rtol=0, atol=1e-12)
    assert boot.interval(np.full(len(rows), 7.0), multiplicity=16)["adjusted_ci"] == [7.0, 7.0]
    # Exact geometric fixtures test the scientific estimands, not only code paths.
    r = np.array([[0., 1.], [2., 1.]])
    d = r[::-1]
    scale = np.ones((1, 2))
    e = r + .25 * (d - r) + np.array([0., 1.])
    g = paired(e, r, d, scale)
    np.testing.assert_allclose(g["score"], .25)
    np.testing.assert_allclose(g["orthogonal"], .5)
    np.testing.assert_allclose(g["q_donor"]**2-g["q_recipient"]**2, 1-2*g["score"])
    assert not paired(r, r, r, scale)["resolvable"].any()
    # A signed mean of zero must not pass an absolute-discrepancy bound of 0.1.
    cancelling = np.array([.2, -.2] * (len(rows) // 2))
    assert abs(boot.interval(cancelling)["mean"]) < 1e-12
    assert not passes(boot.interval(abs(cancelling), multiplicity=16), "below", AXIS_BOUND)
    bad = rows[:-1]
    try:
        validate_rows(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing paired observation was not rejected")
    # Complete synthetic experiment: known recovery and a 0.02 axial difference.
    n = len(rows)
    raw = np.zeros((2, n, 2, 2))
    raw[:, :, 0, 0] = np.array([2 * x["wording"] for x in rows])[None, :]
    behavior = np.zeros((n, 6)); behavior[:, 0] = 2; behavior[:, 2:4] = 1
    data = {"clean": dict(raw=raw.copy(), behavior=behavior.copy())}
    donors = validate_rows(rows)
    axis = raw[:, donors][..., 0] - raw[..., 0]
    for name, fraction, score in (("corrupt", .5, 0), ("wording", .2, 1), ("self", .18, 1.1)):
        edited = raw.copy(); edited[..., 0] += fraction * axis
        b = behavior.copy(); b[:, 0] = score
        data[name] = dict(raw=edited, behavior=b)
    reference = fit_reference(raw, rows)
    selection = dict(status="selected", families=[10, 11], reference=reference)
    qual = qualification_summary(data["clean"], rows, reference, draws=100)
    result, _ = analyze_final(selection, rows, data, qual, draws=100)
    assert result["full_conjunction_supported"]
    for ref in REFS:
        np.testing.assert_allclose(result["primary"]["mean_absolute_axis_difference"][ref]["mean"], .02)
        np.testing.assert_allclose(result["primary"]["mean_full_signature_distance"][ref]["mean"], .02)
    degenerate = dict(raw=raw.copy(), behavior=behavior.copy())
    degenerate["raw"][:, 0] = degenerate["raw"][:, donors[0]]
    bad_qual = qualification_summary(degenerate, rows, reference, draws=100)
    assert bad_qual["unresolvable_reference_examples"] == 4 and not bad_qual["qualified"]
    return dict(status="passed", checks=["crossed_weights_vs_explicit_resampling", "constant_interval",
                "known_projection_and_orthogonal_geometry", "unresolvable_endpoints", "signed_cancellation_rejected",
                "missing_pair_rejected", "complete_known_synthetic_experiment", "qualification_rejects_degenerate_actual_pairs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final", "qualification", "planning", "self-test"), default="final")
    parser.add_argument("--selection", type=Path, default=HERE / "data/selection.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        result = self_test()
        if args.output: write(args.output, result)
        print(json.dumps(result)); return
    if args.mode == "planning":
        result = planning_from_old()
        target = args.output or HERE / "results/development-planning.json"
        assert not target.exists(), target
        write(target, result); print(json.dumps(dict(output=str(target), scope=result["scope"]))); return
    selection = read(args.selection)
    split = "qualification" if args.mode == "qualification" else "test"
    rows = rows_for(split, selection["families"])
    paths = [args.selection, HERE / "data/candidates.json"]
    directory = HERE / "results" / split
    names = ("clean",) if args.mode == "qualification" else ("clean", "corrupt", "wording", "self")
    data = {name: load(directory / (name + ".npz")) for name in names}
    paths += [directory / (name + ".npz") for name in names]
    if args.mode == "qualification":
        result = qualification_summary(data["clean"], rows, selection["reference"])
        result["input_hashes"] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in paths}
        target = args.output or HERE / "results/qualification-summary.json"
        assert not target.exists(), target
        write(target, result); print(json.dumps(dict(qualified=result["qualified"], output=str(target)))); return
    qp = HERE / "results/qualification-summary.json"
    result, metrics = analyze_final(selection, rows, data, read(qp))
    paths.append(qp)
    result["input_hashes"] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in paths}
    target = args.output or HERE / "results/analysis"
    assert not target.exists(), target
    target.mkdir(parents=True)
    write(target / "summary.json", result)
    np.savez_compressed(target / "per-example.npz", **metrics)
    print(json.dumps(dict(qualified=result["qualified"], outcomes=result["outcomes"],
                         full_conjunction_supported=result["full_conjunction_supported"], output=str(target))))


if __name__ == "__main__":
    main()
