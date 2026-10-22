"""Study 6 prespecified analysis; NumPy only, no model inference.

Primary family (six quantities): d(N1, R), d(N2, S_kappa) and d(O, R) under the zero and
fit-mean references, where d(E, E') = ||((E - E')/w)_K|| / ||((D - R)/w)_K|| and K excludes
the held heads. A control detects its change when the Bonferroni-adjusted lower limit
exceeds 1; an adjusted upper limit below 0.5 is "below AI-01's resolution".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
sys.path.insert(0, str(BASE / 'code'))
from analyze007 import read, write, load, validate_rows, cell_masks, paired, Crossed, qualification_summary  # noqa: E402

REFS = ('zero', 'fit_mean')
DRAWS = 50_000
BOOT_SEED = 2026101603
MULTIPLICITY = 6
DETECT = 1.0
RESOLUTION = 0.5
UNRESOLVED = 1e-6
PRIMARY = (('n1', 'clean'), ('n2', 'self_kappa'), ('order', 'clean'))
CONDITIONS = ('corrupt', 'wording', 'self', 'self_kappa', 'n1', 'n2', 'order', 'replay')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keep_mask(held):
    keep = np.ones(176, bool)
    for layer, head in held:
        keep[(layer - 13) * 16 + head] = False
    return keep


class Geometry:
    """Distances in units of each configuration's clean wording separation."""

    def __init__(self, clean, donors, scale, keep):
        self.R = np.asarray(clean, float)            # (ref, n, 176) logit-gap coordinates
        self.D = self.R[:, donors]
        self.w = np.asarray(scale, float)[:, None, :]
        self.keep = keep
        den = np.linalg.norm(((self.D - self.R) / self.w)[..., keep], axis=-1)
        self.resolvable = den >= UNRESOLVED
        self.den = np.where(self.resolvable, den, np.nan)

    def d(self, a, b):
        return np.linalg.norm(((np.asarray(a, float) - b) / self.w)[..., self.keep], axis=-1) / self.den

    def shape(self, a, b):
        x = (np.asarray(a, float) / self.w)[..., self.keep]
        y = (np.asarray(b, float) / self.w)[..., self.keep]
        coefficient = (x * y).sum(-1) / (y * y).sum(-1)
        return np.linalg.norm(x - coefficient[..., None] * y, axis=-1) / self.den, coefficient

    def paired(self, edited):
        return paired(np.asarray(edited, float)[..., self.keep], self.R[..., self.keep],
                      self.D[..., self.keep], self.w[..., self.keep])


def band(interval):
    if not interval['valid']:
        return 'invalid'
    low, high = interval['adjusted_ci']
    if low > DETECT:
        return 'detected'
    if high < RESOLUTION:
        return 'below_resolution'
    return 'intermediate'


def overall(bands):
    if all(b == 'detected' for b in bands):
        return 'detected'
    if any(b == 'detected' for b in bands):
        return 'reference_dependent'
    if all(b == 'below_resolution' for b in bands):
        return 'below_resolution'
    return 'not_detected'


def ratio_interval(boot, numerator, denominator, mask=None):
    gn, gd = boot.grid(numerator, mask), boot.grid(denominator, mask)
    if not (np.isfinite(gn).all() and np.isfinite(gd).all()):
        return dict(valid=False, mean=None, ci95=None)
    draws = boot.resample(gn) / boot.resample(gd)
    return dict(valid=True, mean=float(gn.mean() / gd.mean()), ci95=np.quantile(draws, [0.025, 0.975]).tolist())


def behaviour_block(boot, beh, corrupt):
    b = np.asarray(beh, float)
    return dict(gap=boot.interval(b[:, 0]), recovery=boot.interval(b[:, 0] - corrupt[:, 0]),
                pairwise=boot.interval(b[:, 2]), top1=boot.interval(b[:, 3]))


def analyse(rows, sig, attr, held, reference, draws=DRAWS, boot_indices=None, split='test'):
    """sig: name -> dict(raw (2, n, 176, 2), behavior (n, 6)); attr: name -> attribution arrays."""
    donors = validate_rows(rows)
    boot = Crossed(rows, draws=draws, seed=BOOT_SEED)
    if boot_indices is not None:
        np.testing.assert_array_equal(boot.tt, boot_indices['blocks'])
        np.testing.assert_array_equal(boot.pp, boot_indices['triples'])
    masks = cell_masks(rows)
    keep = keep_mask(held)
    scale = np.asarray(reference['scale'], float)
    logit = {name: np.asarray(value['raw'][..., 0], float) for name, value in sig.items()}
    geo, geo_all = Geometry(logit['clean'], donors, scale, keep), Geometry(logit['clean'], donors, scale, np.ones(176, bool))
    established = 'n1' in sig
    result = dict(split=split, n=len(rows), blocks=len(boot.blocks), triples=len(boot.triples), draws=draws,
                  seed=BOOT_SEED, held=[list(h) for h in held], retained_coordinates=int(keep.sum()),
                  route_controls_established=established, multiplicity=MULTIPLICITY,
                  thresholds=dict(detect=DETECT, resolution=RESOLUTION),
                  unresolvable_examples=int((~geo.resolvable).sum()))
    per_example = {}

    # Primary family.
    primary, decisions = {}, {}
    for control, comparator in PRIMARY:
        if control not in sig:
            decisions[control] = 'not_established'
            continue
        values = geo.d(logit[control], logit[comparator])
        per_example[f'd_{control}_vs_{comparator}'] = values
        primary[control] = {ref: boot.interval(values[i], multiplicity=MULTIPLICITY) for i, ref in enumerate(REFS)}
        for ref in REFS:
            primary[control][ref]['band'] = band(primary[control][ref])
        decisions[control] = overall([primary[control][ref]['band'] for ref in REFS])
    result.update(primary=primary, decisions=decisions,
                  comparators={c: k for c, k in PRIMARY})

    # 1. Distances from the recipient and donor, margins and donor-axis scores for every condition.
    geometry = {}
    for name in CONDITIONS:
        if name not in logit:
            continue
        g = geo.paired(logit[name])
        geometry[name] = {ref: {key: boot.interval(g[key][i]) for key in ('q_recipient', 'q_donor', 'margin', 'score')}
                          for i, ref in enumerate(REFS)}
        for key in ('q_recipient', 'q_donor', 'margin', 'score'):
            per_example[f'{name}__{key}'] = g[key]
    result['geometry'] = geometry
    anchor = np.abs(logit['replay'] - geo.D)
    result['anchor'] = dict(max_abs_error_vs_donor=float(anchor.max()),
                            q_recipient_mean=[float(np.nanmean(x)) for x in geo.d(logit['replay'], geo.R)],
                            q_donor_max=float(np.nanmax(geo.d(logit['replay'], geo.D))))

    # 2. Fresh-data replication of Study 2.
    c_beh = np.asarray(sig['corrupt']['behavior'], float)
    gain = np.asarray(sig['wording']['behavior'], float)[:, 0] - c_beh[:, 0]
    replication = dict(recovery={cell: boot.interval(gain, mask) for cell, mask in masks.items()})
    for label, g in (('retained', geo), ('all176', geo_all)):
        w, s = g.paired(logit['wording']), g.paired(logit['self'])
        axis = np.abs(w['score'] - s['score'])
        full = g.d(logit['wording'], logit['self'])
        per_example[f'F_{label}'], per_example[f'A_{label}'] = full, axis
        replication[label] = {ref: dict(
            margin={cell: boot.interval(w['margin'][i], mask) for cell, mask in masks.items()},
            mean_absolute_axis_difference=boot.interval(axis[i]),
            mean_full_signature_distance=boot.interval(full[i])) for i, ref in enumerate(REFS)}
    result['replication'] = replication

    # 3. Ratios to the wording/self discrepancy on the same rows.
    full = geo.d(logit['wording'], logit['self'])
    ratios = {}
    for control, comparator in PRIMARY[:2]:
        if control in sig:
            values = geo.d(logit[control], logit[comparator])
            ratios[f'{control}_over_F'] = {ref: ratio_interval(boot, values[i], full[i]) for i, ref in enumerate(REFS)}
    result['ratios_to_wording_self'] = ratios

    # 4. Amplitude: dose-only distance and shape distances.
    amplitude = {}
    if established:
        dose = geo.d(logit['self'], logit['self_kappa'])
        per_example['d_self_vs_self_kappa'] = dose
        amplitude['dose_only_distance'] = {ref: boot.interval(dose[i]) for i, ref in enumerate(REFS)}
        for label, a, b in (('n1_vs_clean', 'n1', 'clean'), ('n2_vs_self_kappa', 'n2', 'self_kappa'),
                            ('self_kappa_vs_self', 'self_kappa', 'self')):
            values, coefficient = geo.shape(logit[a], logit[b])
            per_example[f'shape_{label}'] = values
            amplitude[f'shape_{label}'] = {ref: dict(distance=boot.interval(values[i]),
                                                     scale_coefficient=boot.interval(coefficient[i]))
                                           for i, ref in enumerate(REFS)}
    result['amplitude'] = amplitude

    # 5 and 6. Primary quantities over all 176 heads, and within each wording/order cell.
    all176, cells = {}, {}
    for control, comparator in PRIMARY:
        if control not in sig:
            continue
        values = geo_all.d(logit[control], logit[comparator])
        all176[control] = {ref: boot.interval(values[i]) for i, ref in enumerate(REFS)}
        values = geo.d(logit[control], logit[comparator])
        cells[control] = {ref: {cell: boot.interval(values[i], mask) for cell, mask in masks.items()}
                          for i, ref in enumerate(REFS)}
    result['primary_all176'], result['primary_cells'] = all176, cells

    # 7. Behaviour and direct attribution.
    result['behaviour'] = {name: behaviour_block(boot, value['behavior'], c_beh) for name, value in sig.items()}
    attribution = {}
    if attr:
        mask_h = ~keep
        for name, value in attr.items():
            dla = np.asarray(value['dla'], float)
            gap = np.asarray(value['behavior'], float)[:, 0]
            attribution[name] = dict(
                held_dla=boot.interval(dla[:, mask_h].sum(1)), other_dla=boot.interval(dla[:, ~mask_h].sum(1)),
                gap=boot.interval(gap), held_share_of_gap=ratio_interval(boot, dla[:, mask_h].sum(1), gap),
                linearised_gap_max_error=float(np.abs(np.asarray(value['linearised_gap'], float) - gap).max()))
        for changed, base in (('n1', 'clean'), ('n2', 'self')):
            if changed in attr and base in attr:
                a, b = np.asarray(attr[changed]['dla'], float), np.asarray(attr[base]['dla'], float)
                lost = b[:, mask_h].sum(1) - a[:, mask_h].sum(1)
                gained = a[:, ~mask_h].sum(1) - b[:, ~mask_h].sum(1)
                attribution[f'compensation_{changed}_vs_{base}'] = dict(
                    held_dla_lost=boot.interval(lost), other_dla_gained=boot.interval(gained),
                    fraction_recovered=ratio_interval(boot, gained, lost))
    result['attribution'] = attribution

    # Clean gates of Study 2 on this bank (observed rates).
    gate = qualification_summary(sig['clean'], rows, reference, draws=min(draws, 2000))
    result['clean_gates'] = dict(behavior_pass=gate['behavior_pass'], separation_pass=gate['separation_pass'],
                                 qualified=gate['qualified'], cells=gate['cells'],
                                 unresolvable_reference_examples=gate['unresolvable_reference_examples'])
    return result, per_example


def self_test():
    """Exact synthetic geometry with known answers; must pass before the input freeze."""
    rows = [dict(id=f'{t}-{p}-{r}-{w}-{o}', template=t, pair=p, role=r, wording=w, order=o, family=10 + w)
            for t in range(3) for p in range(4) for r in range(2) for w in range(2) for o in range(2)]
    n = len(rows)
    held = [(13, 0)]
    v = np.zeros(176); v[1] = 1.0                      # wording axis on retained coordinate 1
    u = np.zeros(176); u[2] = 1.0                      # route direction, orthogonal to v
    base = np.zeros(176); base[3:] = 0.5
    wording = np.array([r['wording'] for r in rows], float)
    R = base[None] + wording[:, None] * v[None]
    R = np.stack([R, R])
    S = R + 0.5 * v * np.where(wording[:, None] > 0, -1, 1)
    def make(x, gap):
        raw = np.zeros((2, n, 176, 2)); raw[..., 0] = x
        beh = np.zeros((n, 6)); beh[:, 0] = gap; beh[:, 2:4] = 1
        return dict(raw=raw, behavior=beh)
    big = np.zeros(176); big[0] = 100.0                # excluded coordinate must not count
    sig = dict(clean=make(R, 4), corrupt=make(R + 1.5 * u, 0), wording=make(S + 0.2 * u, 2),
               self=make(S, 2), self_kappa=make(0.8 * S, 1.5), n1=make(R + 2 * u + big, 3),
               n2=make(0.8 * S + 0.4 * u + big, 1.5), order=make(R - 3 * u, -2))
    donors = validate_rows(rows)
    sig['replay'] = make(R[:, donors], 4)
    reference = dict(scale=np.ones((2, 176)).tolist(),
                     centers=np.stack([np.stack([R[0, [i for i, r in enumerate(rows) if 2 * r['wording'] + r['order'] == c]].mean(0)
                                                 for c in range(4)], 0)] * 2).tolist())
    def attribution(held_value, other_value, gap):
        dla = np.zeros((n, 176)); dla[:, 0] = held_value; dla[:, 1] = other_value
        beh = np.zeros((n, 6)); beh[:, 0] = gap
        return dict(dla=dla, behavior=beh, linearised_gap=beh[:, 0].copy())
    attr = dict(clean=attribution(2.0, 2.0, 4.0), n1=attribution(0.5, 2.6, 3.1),
                self=attribution(1.0, 1.0, 2.0), n2=attribution(0.5, 1.25, 1.75))
    result, per = analyse(rows, sig, attr, held, reference, draws=200)
    expected = dict(n1=2.0, n2=0.4, order=3.0)
    for control, value in expected.items():
        for ref in REFS:
            np.testing.assert_allclose(result['primary'][control][ref]['mean'], value, atol=1e-12)
    assert result['decisions'] == dict(n1='detected', n2='below_resolution', order='detected'), result['decisions']
    np.testing.assert_allclose(result['replication']['retained']['zero']['mean_full_signature_distance']['mean'], 0.2, atol=1e-12)
    np.testing.assert_allclose(result['anchor']['q_recipient_mean'], [1.0, 1.0], atol=1e-12)
    assert result['anchor']['max_abs_error_vs_donor'] == 0.0
    # A pure rescaling has zero shape distance; a perpendicular addition does not.
    np.testing.assert_allclose(result['amplitude']['shape_self_kappa_vs_self']['zero']['distance']['mean'], 0, atol=1e-12)
    np.testing.assert_allclose(result['amplitude']['shape_self_kappa_vs_self']['zero']['scale_coefficient']['mean'], 0.8, atol=1e-12)
    assert result['amplitude']['shape_n2_vs_self_kappa']['zero']['distance']['mean'] > 0.1
    np.testing.assert_allclose(result['ratios_to_wording_self']['n1_over_F']['zero']['mean'], 10.0, atol=1e-9)
    # The excluded coordinate does count over all 176 heads.
    assert result['primary_all176']['n1']['zero']['mean'] > 50
    np.testing.assert_allclose(result['attribution']['compensation_n1_vs_clean']['fraction_recovered']['mean'], 0.4, atol=1e-12)
    np.testing.assert_allclose(result['attribution']['compensation_n2_vs_self']['fraction_recovered']['mean'], 0.5, atol=1e-12)
    np.testing.assert_allclose(result['attribution']['clean']['held_share_of_gap']['mean'], 0.5, atol=1e-12)
    # Interval bands.
    assert band(dict(valid=True, adjusted_ci=[1.01, 2])) == 'detected'
    assert band(dict(valid=True, adjusted_ci=[0.2, 0.49])) == 'below_resolution'
    assert band(dict(valid=True, adjusted_ci=[0.4, 1.2])) == 'intermediate'
    assert overall(['detected', 'intermediate']) == 'reference_dependent'
    # Without established route controls, only the order donor is analysed.
    reduced = {k: v for k, v in sig.items() if k not in ('n1', 'n2', 'self_kappa')}
    r2, _ = analyse(rows, reduced, {}, [], reference, draws=200)
    assert r2['decisions']['n1'] == 'not_established' and r2['decisions']['order'] == 'detected'
    return dict(passed=True, checks=['known_primary_distances', 'held_coordinates_excluded', 'decision_bands',
                                     'replication_F', 'anchor_identity', 'shape_distance_rescaling',
                                     'ratio_to_F', 'all176_includes_held', 'attribution_compensation', 'not_established_path'])


def load_split(split, development):
    folder = HERE / 'results' / ('test' if split == 'test' else 'development')
    rows = [r for r in read(HERE / 'data/rows.json') if r['split'] == split]
    held = [tuple(x) for x in development['held']]
    names = ['clean', 'corrupt', 'wording', 'self', 'order', 'replay'] + (['self_kappa', 'n1', 'n2'] if held else [])
    sig = {name: load(folder / f'{name}.npz') for name in names}
    attr = {name: load(folder / f'attribution-{name}.npz') for name in ('clean', 'self', 'n1', 'n2')
            if (folder / f'attribution-{name}.npz').exists()}
    inputs = [folder / f'{name}.npz' for name in names] + [folder / f'attribution-{name}.npz' for name in attr]
    return rows, sig, attr, held, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('final', 'development', 'self-test'), default='final')
    args = parser.parse_args()
    if args.mode == 'self-test':
        print(json.dumps(self_test()))
        return
    development = read(HERE / 'results/development.json')
    split = 'test' if args.mode == 'final' else 'development'
    target = HERE / 'results' / ('analysis' if split == 'test' else 'analysis-development')
    assert not target.exists(), target
    rows, sig, attr, held, inputs = load_split(split, development)
    indices = load(HERE / f'data/bootstrap-{split}.npz')
    result, per = analyse(rows, sig, attr, held, read(BASE / 'data/selection.json')['reference'],
                          boot_indices=indices, split=split)
    inputs += [HERE / 'results/development.json', BASE / 'data/selection.json', HERE / f'data/bootstrap-{split}.npz']
    result['input_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    target.mkdir(parents=True)
    write(target / 'summary.json', result)
    np.savez_compressed(target / 'per-example.npz', **per)
    print(json.dumps(dict(split=split, decisions=result['decisions'],
                          primary={c: {r: [round(v['mean'], 4)] + [round(x, 4) for x in v['adjusted_ci']]
                                       for r, v in refs.items()} for c, refs in result['primary'].items()},
                          gates=result['clean_gates']['qualified'])))


if __name__ == '__main__':
    main()
