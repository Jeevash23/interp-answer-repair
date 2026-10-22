"""Study 7 prespecified analyses; NumPy only, no model inference.

Family A (Study 2 in GPT-2 Large): Study 2's frozen analyze_final and its secondary analysis,
unchanged (16 quantities; Bonferroni tails 0.05/32; Study 2's bootstrap seed). Family B (Study 6 in
GPT-2 Large): Study 6's analysis with the coordinate layout of GPT-2 Large (340 END heads, layers
19-35) and seed 2026101704. Both families' bootstrap indices were saved before final inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'studies/bridge-007/code'), str(ROOT / 'studies/bridge-011/code')]
import analyze007 as a7  # noqa: E402
import secondary007 as s7  # noqa: E402
import analyze011 as a11  # noqa: E402

SEED_B = 2026101704
LAYOUT = {'gpt2-large': dict(first=19, heads=20, count=340), 'gpt2-medium': dict(first=13, heads=16, count=176)}
OUT = HERE / 'results'


def keep_mask(held, layout):
    keep = np.ones(layout['count'], bool)
    for layer, head in held:
        keep[(layer - layout['first']) * layout['heads'] + head] = False
    return keep


def family_b(rows, sig, attr, held, reference, layout, draws, boot_indices=None, split='test'):
    """Study 6's primary and secondary quantities (analyze011.analyse), for any head layout."""
    R = a11
    donors = a7.validate_rows(rows)
    boot = a7.Crossed(rows, draws=draws, seed=SEED_B)
    if boot_indices is not None:
        np.testing.assert_array_equal(boot.tt, boot_indices['blocks'])
        np.testing.assert_array_equal(boot.pp, boot_indices['triples'])
    masks = a7.cell_masks(rows)
    keep = keep_mask(held, layout)
    scale = np.asarray(reference['scale'], float)
    logit = {name: np.asarray(value['raw'][..., 0], float) for name, value in sig.items()}
    geo = R.Geometry(logit['clean'], donors, scale, keep)
    geo_all = R.Geometry(logit['clean'], donors, scale, np.ones(layout['count'], bool))
    result = dict(split=split, n=len(rows), blocks=len(boot.blocks), triples=len(boot.triples), draws=draws, seed=SEED_B,
                  held=[list(h) for h in held], retained_coordinates=int(keep.sum()), route_controls_established='n1' in sig,
                  multiplicity=R.MULTIPLICITY, thresholds=dict(detect=R.DETECT, resolution=R.RESOLUTION),
                  unresolvable_examples=int((~geo.resolvable).sum()))
    per = {}
    primary, decisions = {}, {}
    for control, comparator in R.PRIMARY:
        if control not in sig:
            decisions[control] = 'not_established'
            continue
        values = geo.d(logit[control], logit[comparator])
        per[f'd_{control}_vs_{comparator}'] = values
        primary[control] = {ref: boot.interval(values[i], multiplicity=R.MULTIPLICITY) for i, ref in enumerate(R.REFS)}
        for ref in R.REFS:
            primary[control][ref]['band'] = R.band(primary[control][ref])
        decisions[control] = R.overall([primary[control][ref]['band'] for ref in R.REFS])
    result.update(primary=primary, decisions=decisions, comparators=dict(R.PRIMARY))
    geometry = {}
    for name in R.CONDITIONS:
        if name in logit:
            g = geo.paired(logit[name])
            geometry[name] = {ref: {k: boot.interval(g[k][i]) for k in ('q_recipient', 'q_donor', 'margin', 'score')}
                              for i, ref in enumerate(R.REFS)}
            for k in ('q_recipient', 'q_donor', 'margin', 'score'):
                per[f'{name}__{k}'] = g[k]
    result['geometry'] = geometry
    result['anchor'] = dict(max_abs_error_vs_donor=float(np.abs(logit['replay'] - geo.D).max()),
                            q_recipient_mean=[float(np.nanmean(x)) for x in geo.d(logit['replay'], geo.R)],
                            q_donor_max=float(np.nanmax(geo.d(logit['replay'], geo.D))))
    full = geo.d(logit['wording'], logit['self'])
    per['F_retained'] = full
    result['wording_self_F_retained'] = {ref: boot.interval(full[i]) for i, ref in enumerate(R.REFS)}
    ratios = {}
    for control, comparator in R.PRIMARY[:2]:
        if control in sig:
            values = geo.d(logit[control], logit[comparator])
            ratios[f'{control}_over_F'] = {ref: R.ratio_interval(boot, values[i], full[i]) for i, ref in enumerate(R.REFS)}
    result['ratios_to_wording_self'] = ratios
    amplitude = {}
    if 'n1' in sig:
        dose = geo.d(logit['self'], logit['self_kappa'])
        per['d_self_vs_self_kappa'] = dose
        amplitude['dose_only_distance'] = {ref: boot.interval(dose[i]) for i, ref in enumerate(R.REFS)}
        for label, x, y in (('n1_vs_clean', 'n1', 'clean'), ('n2_vs_self_kappa', 'n2', 'self_kappa'),
                            ('self_kappa_vs_self', 'self_kappa', 'self')):
            values, coefficient = geo.shape(logit[x], logit[y])
            per[f'shape_{label}'] = values
            amplitude[f'shape_{label}'] = {ref: dict(distance=boot.interval(values[i]),
                                                     scale_coefficient=boot.interval(coefficient[i])) for i, ref in enumerate(R.REFS)}
    result['amplitude'] = amplitude
    all_heads, cells = {}, {}
    for control, comparator in R.PRIMARY:
        if control in sig:
            values = geo_all.d(logit[control], logit[comparator])
            all_heads[control] = {ref: boot.interval(values[i]) for i, ref in enumerate(R.REFS)}
            values = geo.d(logit[control], logit[comparator])
            cells[control] = {ref: {cell: boot.interval(values[i], mask) for cell, mask in masks.items()}
                              for i, ref in enumerate(R.REFS)}
    result['primary_all_heads'], result['primary_cells'] = all_heads, cells
    c_beh = np.asarray(sig['corrupt']['behavior'], float)
    result['behaviour'] = {name: R.behaviour_block(boot, value['behavior'], c_beh) for name, value in sig.items()}
    attribution = {}
    mask_h = ~keep
    for name, value in attr.items():
        dla, gap = np.asarray(value['dla'], float), np.asarray(value['behavior'], float)[:, 0]
        attribution[name] = dict(held_dla=boot.interval(dla[:, mask_h].sum(1)), other_dla=boot.interval(dla[:, ~mask_h].sum(1)),
                                 gap=boot.interval(gap), held_share_of_gap=R.ratio_interval(boot, dla[:, mask_h].sum(1), gap),
                                 linearised_gap_max_error=float(np.abs(np.asarray(value['linearised_gap'], float) - gap).max()))
    for changed, base in (('n1', 'clean'), ('n2', 'self')):
        if changed in attr and base in attr:
            x, y = np.asarray(attr[changed]['dla'], float), np.asarray(attr[base]['dla'], float)
            lost, gained = y[:, mask_h].sum(1) - x[:, mask_h].sum(1), x[:, ~mask_h].sum(1) - y[:, ~mask_h].sum(1)
            attribution[f'compensation_{changed}_vs_{base}'] = dict(held_dla_lost=boot.interval(lost),
                                                                    other_dla_gained=boot.interval(gained),
                                                                    fraction_recovered=R.ratio_interval(boot, gained, lost))
    result['attribution'] = attribution
    return result, per


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run(test=False, draws=None):
    """Analyse the final test and the development bank; writes results/analysis once."""
    sys.path.insert(0, str(HERE / 'code'))
    from common012 import rows_for
    target = OUT / 'analysis'
    assert not (target / 'outcome.json').exists(), target
    draws = draws or (500 if test else a7.DRAWS)
    layout = LAYOUT['gpt2-medium' if test else 'gpt2-large']
    sel, qual, dev = (json.loads((OUT / f).read_text()) for f in ('selection.json', 'qualification-summary.json', 'development.json'))
    assert json.loads((OUT / 'inference-complete.json').read_text())['passed']
    split = 'calibration' if test else 'test'
    rows = rows_for(split, sel['families'])
    folder = OUT / 'test'
    held = [tuple(h) for h in dev['held']]
    names = ['clean', 'corrupt', 'wording', 'self', 'order', 'replay'] + (['self_kappa', 'n1', 'n2'] if held else [])
    sig = {n: load(folder / f'{n}.npz') for n in names}
    behaviour_only = ['zero', 'role', 'order_role'] + [f'{k}{i}' for k in ('residual', 'projected') for i in range(3)]
    data = dict(sig, **{n: load(folder / f'{n}.npz') for n in behaviour_only})
    natural = {k: load(folder / f'natural-{k}.npz') for k in ('clean', 'corrupt')}
    attr = {n: load(folder / f'attribution-{n}.npz') for n in ('clean', 'self', 'n1', 'n2') if (folder / f'attribution-{n}.npz').exists()}
    if not test:
        boot_a = load(HERE / 'data/bootstrap-test-a.npz')
        check = a7.Crossed(rows)
        np.testing.assert_array_equal(check.tt, boot_a['blocks'])
        np.testing.assert_array_equal(check.pp, boot_a['triples'])
    selection = dict(sel, status='selected')
    fam_a, per_a = a7.analyze_final(selection, rows, {k: sig[k] for k in ('clean', 'corrupt', 'wording', 'self')}, qual, draws=draws)
    sec_a, per_s = s7.analyze_secondary(selection, rows, data, natural, draws=draws)
    fam_b, per_b = family_b(rows, sig, attr, held, sel['reference'], layout, draws,
                            None if test else load(HERE / 'data/bootstrap-test-b.npz'))
    dev_rows = rows_for('calibration' if test else 'development', sel['families'])
    dsig = {n: load(OUT / f'development/{n}.npz') for n in names}
    dattr = {n: load(OUT / f'development/attribution-{n}.npz') for n in ('clean', 'self', 'n1', 'n2')
             if (OUT / f'development/attribution-{n}.npz').exists()}
    dev_b, _ = family_b(dev_rows, dsig, dattr, held, sel['reference'], layout, draws,
                        None if test else load(HERE / 'data/bootstrap-development-b.npz'), split='development')
    target.mkdir(parents=True, exist_ok=True)
    write(target / 'family-a.json', fam_a)
    write(target / 'secondary-a.json', sec_a)
    write(target / 'family-b.json', fam_b)
    write(target / 'development-b.json', dev_b)
    np.savez_compressed(target / 'per-example-a.npz', **{k: np.asarray(v) for k, v in per_a.items()})
    np.savez_compressed(target / 'per-example-secondary.npz', **{k: np.asarray(v) for k, v in per_s.items()})
    np.savez_compressed(target / 'per-example-b.npz', **{k: np.asarray(v) for k, v in per_b.items()})
    outcome = dict(code_test=test, draws=draws, n=len(rows), families=sel['families'],
                   family_a=dict(qualified=fam_a['qualified'], outcomes=fam_a['outcomes'],
                                 full_conjunction_supported=fam_a['full_conjunction_supported']),
                   family_b=dict(decisions=fam_b['decisions'],
                                 primary={c: {r: dict(mean=v['mean'], adjusted_ci=v['adjusted_ci'], band=v['band'])
                                              for r, v in refs.items()} for c, refs in fam_b['primary'].items()}),
                   development_b=dev_b['decisions'])
    write(target / 'outcome.json', outcome)
    return outcome
