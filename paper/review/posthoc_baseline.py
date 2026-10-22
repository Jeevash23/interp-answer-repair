"""POST HOC (version 3.2): where the unedited corrupted signature sits, and the self/wording position correlation.

(a) Distance margins M of the corrupted signature C, by recipient wording and mention order, in Study 1 and
    Study 2, beside the primary margins of the edited signature after wording transfer, and the paired change
    M(E) - M(C). The corrupted signatures were measured in the original runs, but neither frozen analysis
    summarised their margins by cell. Intervals are computed exactly as for the primary margins: Study 1 with
    its frozen interval function (seed 2026090905, 50,000 draws, twelve-quantity Bonferroni family) and Study 2
    with its frozen crossed bootstrap (seed 2026100407, 50,000 draws, sixteen-quantity family). The script
    first checks that it reproduces the saved primary margins exactly.
(b) Study 1: Pearson correlation of the per-configuration donor-axis positions S after self and wording
    transfer, under both replacement references (the zero-reference value is printed in Figure 3c).

Both analyses were defined after all results were known and carry no confirmatory weight.
Outputs: review/posthoc-baseline.json and sections/posthoc-baseline-table.tex. No model inference.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
B5, B7 = ROOT / 'studies/bridge-005', ROOT / 'studies/bridge-007'
sys.path[:0] = [str(ROOT / 'src'), str(B5 / 'code'), str(B7 / 'code'), str(ROOT / 'analysis')]
import common005 as s1  # noqa: E402  (frozen Study 1 helpers: rows, donors, paired geometry, intervals)
import analyze007 as s2  # noqa: E402  (frozen Study 2 rows, cells and crossed bootstrap)
from common import geometry, load, study_rows  # noqa: E402  (as used for Figure 3c)

REFS = ('zero', 'fit_mean')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def num(x, places=3):
    if abs(x) < .5 * 10 ** (-places):
        x = 0
    return f'{x:.{places}f}'


def ci(v, key='adjusted_ci'):
    return '$' + num(v['mean']) + r'\ci{' + num(v[key][0]) + '}{' + num(v[key][1]) + '}$'


def side(v, key='adjusted_ci'):
    lo, hi = v[key]
    return 'recipient' if hi < 0 else ('donor' if lo > 0 else 'inconclusive')


def keep(v):
    return dict(mean=v['mean'], ci95=list(v['ci95']), adjusted_ci=list(v['adjusted_ci']), side=side(v))


def study1():
    sel = s1.read(B5 / 'data/selection.json')
    rows = s1.rows_for('test', sel['families'])
    donors = s1.donor_indices(rows, 'wording')
    test = B5 / 'results/test'
    raw = {k: s1.load_arrays(test / f'{k}.npz')['raw'][..., 0] for k in ('clean', 'corrupt', 'wording')}
    r = raw['clean']
    d = r[:, donors]
    scale = np.asarray(sel['reference']['scale'])[:, None]
    margin = {k: s1.paired(raw[k], r, d, scale)['margin'] for k in ('corrupt', 'wording')}
    saved = s1.read(B5 / 'results/analysis/summary.json')['primary']['margin']
    labels = {f'w{w}_o{o}': f"F{sel['families'][w]}/order {o}" for w in range(2) for o in range(2)}
    masks = {f'w{w}_o{o}': np.array([x['wording'] == w and x['order'] == o for x in rows]) for w in range(2) for o in range(2)}
    out = {}
    for bi, ref in enumerate(REFS):
        out[ref] = {}
        for cell, mask in masks.items():
            edited = s1.intervals(margin['wording'][bi], rows, mask, 12)
            assert edited['mean'] == saved[ref][cell]['mean'] and edited['adjusted_ci'] == saved[ref][cell]['adjusted_ci'], (ref, cell)
            out[ref][labels[cell]] = dict(corrupted=keep(s1.intervals(margin['corrupt'][bi], rows, mask, 12)), edited=keep(edited),
                                          change=keep(s1.intervals(margin['wording'][bi] - margin['corrupt'][bi], rows, mask, 12)))
        out[ref]['pooled_corrupted'] = keep(s1.intervals(margin['corrupt'][bi], rows))
    return dict(n=len(rows), multiplicity=12, seed=s1.SEED, draws=s1.DRAWS, cells=out)


def study2():
    sel = s2.read(s2.HERE / 'data/selection.json')
    rows = s2.rows_for('test', sel['families'])
    s2.validate_rows(rows)
    per = np.load(B7 / 'results/analysis/per-example.npz')
    saved = s2.read(B7 / 'results/analysis/summary.json')['primary']['margin']
    boot, masks = s2.Crossed(rows, draws=s2.DRAWS), s2.cell_masks(rows)
    labels = {f'w{w}_o{o}': f"F{sel['families'][w]}/order {o}" for w in range(2) for o in range(2)}
    out = {}
    for bi, ref in enumerate(REFS):
        out[ref] = {}
        for cell, mask in masks.items():
            edited = boot.interval(per['wording__margin'][bi], mask, s2.MULTIPLICITY)
            assert edited['mean'] == saved[ref][cell]['mean'] and edited['adjusted_ci'] == saved[ref][cell]['adjusted_ci'], (ref, cell)
            out[ref][labels[cell]] = dict(
                corrupted=keep(boot.interval(per['corrupt__margin'][bi], mask, s2.MULTIPLICITY)), edited=keep(edited),
                change=keep(boot.interval(per['wording__margin'][bi] - per['corrupt__margin'][bi], mask, s2.MULTIPLICITY)))
        out[ref]['pooled_corrupted'] = keep(boot.interval(per['corrupt__margin'][bi]))
    return dict(n=len(rows), multiplicity=s2.MULTIPLICITY, seed=s2.SEED, draws=s2.DRAWS, cells=out)


def s_correlation():
    s = study_rows('bridge-005')
    R = load(s['test'] / 'clean.npz')['raw'][..., 0].astype(float)
    D = R[:, s['donor']]
    pos = {k: geometry(load(s['test'] / f'{k}.npz')['raw'][..., 0].astype(float), R, D, s['scale'])['S'] for k in ('self', 'wording')}
    return {ref: float(np.corrcoef(pos['self'][bi], pos['wording'][bi])[0, 1]) for bi, ref in enumerate(REFS)}


def table(result):
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Post hoc: distance margins of the unedited corrupted signature beside the primary margins after wording '
             r'transfer, by recipient wording and mention order. Means with Bonferroni-adjusted crossed-bootstrap intervals '
             r'computed exactly as for the primary margins and with the same draws (Study 1: twelve-quantity family; Study 2: '
             r'sixteen-quantity family). Negative margins favour the clean recipient. The corrupted signatures were measured in '
             r'the original runs; this comparison was defined after all results were known.}',
             r'\label{tab:baseline-margins}', r'\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{Zero replacement} & \multicolumn{2}{c}{Mean replacement} \\',
             r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
             r'Recipient / order & Corrupted $\Cc$ & Edited $\Ee$ & Corrupted $\Cc$ & Edited $\Ee$ \\', r'\midrule']
    for name, label in (('study1', 'Study 1'), ('study2', 'Study 2')):
        cells = [c for c in result[name]['cells']['zero'] if c != 'pooled_corrupted']
        lines.append(r'\multicolumn{5}{@{}l}{\emph{' + label + r'}} \\')
        for cell in cells:
            row = [cell.replace('/order ', ' / ')]
            for ref in REFS:
                v = result[name]['cells'][ref][cell]
                row += [ci(v['corrupted']), ci(v['edited'])]
            lines.append(' & '.join(row) + r' \\')
        if name == 'study1':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    header = ('% Generated by review/posthoc_baseline.py from saved Study 1 and Study 2 arrays (post hoc, version 3.2).\n')
    return header + '\n'.join(lines)


def main():
    result = dict(scope='Post hoc (version 3.2). Defined after all results were known; saved arrays only, no model inference.',
                  study1=study1(), study2=study2(), s_correlation_study1=s_correlation(),
                  inputs={p: sha(ROOT / p) for p in ('studies/bridge-005/results/analysis/summary.json',
                                                     'studies/bridge-007/results/analysis/summary.json',
                                                     'studies/bridge-007/results/analysis/per-example.npz')})
    for name in ('study1', 'study2'):
        cells = result[name]['cells']
        result[name]['corrupted_sides'] = {ref: {c: v['corrupted']['side'] for c, v in cells[ref].items() if c != 'pooled_corrupted'} for ref in REFS}
        result[name]['every_change_towards_recipient'] = all(v['change']['adjusted_ci'][1] < 0 for ref in REFS
                                                             for c, v in cells[ref].items() if c != 'pooled_corrupted')
    (REV / 'review/posthoc-baseline.json').write_text(json.dumps(result, indent=2) + '\n')
    (REV / 'sections/posthoc-baseline-table.tex').write_text(table(result))
    print(json.dumps({k: result[k]['corrupted_sides'] for k in ('study1', 'study2')}, indent=1))
    print('changes all towards recipient:', result['study1']['every_change_towards_recipient'], result['study2']['every_change_towards_recipient'])
    print('S correlation (Study 1):', {k: round(v, 3) for k, v in result['s_correlation_study1'].items()})


if __name__ == '__main__':
    main()
