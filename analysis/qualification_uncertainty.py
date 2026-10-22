"""Uncertainty of the clean-signature qualification rates, allowing for shared prompts.

Qualification cells contain 72 configurations that share 6 context blocks and 6 name
triples (final-bank cells of the other comparisons: 96 configurations, 8 blocks x 6 triples),
so the observations are not independent. This script re-derives each
configuration's classification from the saved arrays and frozen references, then
uses the crossed bootstrap (blocks x triples) to give intervals for:
  * every independent-qualification cell of F6/F7 (Bridge 005) and F4/F7 (Bridge 006);
  * the paired difference between the two references on the shared F7 prompts.
The frozen pass/fail decisions are unchanged; this is post hoc.
Writes results/qualification-uncertainty.json.
"""
import json

import numpy as np

from common import RESULTS, ROOT, Crossed, load, read, study_rows

B5, B6 = ROOT / 'studies/bridge-005', ROOT / 'studies/bridge-006'


def correct(raw, rows, ref):
    """Per-configuration correct classification, shape [reference, n] (nearest scaled centroid)."""
    x = np.asarray(raw[..., 0], np.float64) / np.asarray(ref['scale'])[:, None]
    c = np.asarray(ref['centers'])
    out = np.empty(x.shape[:2], bool)
    for i, r in enumerate(rows):
        a, b = r['order'], 2 + r['order']
        axis = c[:, b] - c[:, a]
        score = np.sum((x[:, i] - (c[:, a] + c[:, b]) / 2) * axis, axis=1) / np.sum(axis ** 2, axis=1)
        out[:, i] = (score > 0) == bool(r['wording'])
    return out


def study(rows, raw, ref, families):
    ok = correct(raw, rows, ref)
    boot = Crossed(rows)
    cells = {}
    for w in range(2):
        for o in range(2):
            m = np.array([r['wording'] == w and r['order'] == o for r in rows])
            cells[f'F{families[w]}/order {o}'] = {
                name: dict(k=int(ok[bi, m].sum()), n=int(m.sum()), **boot(ok[bi].astype(float), m))
                for bi, name in enumerate(['zero', 'mean'])}
    return ok, cells


def main():
    sel = read(B5 / 'data/selection.json')
    cand = read(B5 / 'data/candidates.json')
    rows5 = [dict(r, wording=[6, 7].index(r['family'])) for r in cand if r['split'] == 'qualification' and r['family'] in (6, 7)]
    raw5 = load(B5 / 'results/qualification/clean.npz')['raw']
    ok5, cells5 = study(rows5, raw5, sel['reference'], [6, 7])
    rows6 = [dict(r, wording=[4, 7].index(r['family'])) for r in read(B6 / 'data/qualification.json')]
    raw6 = load(B6 / 'results/qualification/clean.npz')['raw']
    ok6, cells6 = study(rows6, raw6, sel['scores']['4,7']['reference'], [4, 7])
    # paired comparison on the shared F7 prompts
    idx6 = {r['id']: i for i, r in enumerate(rows6)}
    pairs = [(i, idx6[r['id']]) for i, r in enumerate(rows5) if r['family'] == 7 and r['id'] in idx6]
    f7rows = [rows5[i] for i, _ in pairs]
    boot = Crossed(f7rows)
    paired = {}
    for o in range(2):
        m = np.array([r['order'] == o for r in f7rows])
        for bi, name in enumerate(['zero', 'mean']):
            d = ok5[bi, [i for i, _ in pairs]].astype(float) - ok6[bi, [j for _, j in pairs]].astype(float)
            paired[f'F7/order {o}, {name}'] = dict(n=int(m.sum()), **boot(d, m))
    # final-bank classification of the other comparisons (no separate qualification bank)
    others = {}
    for label, st, model in [('initial Medium F1/F2 (Bridge 003)', 'bridge-003', 'gpt2-medium'),
                             ('fresh Medium F1/F2 (Bridge 004)', 'bridge-004', 'gpt2-medium'),
                             ('fresh TinyLlama F2/F3 (Bridge 004)', 'bridge-004', 'tinyllama-1.1b-base')]:
        s = study_rows(st, model)
        _, cells = study(s['rows'], load(s['test'] / 'clean.npz')['raw'], s['selection']['reference'], s['families'])
        others[label] = cells
    result = dict(scope='Post hoc. Pointwise 95% crossed-bootstrap intervals over context blocks x name triples '
                        '(50,000 draws): 6 x 6 in the F6/F7 and F4/F7 qualification banks, 8 x 6 in the final banks '
                        'of the other comparisons. Frozen pass/fail decisions unchanged.',
                  f6_f7=cells5, f4_f7=cells6, other_comparisons_final_bank=others, shared_f7_difference_f6f7_minus_f4f7=paired,
                  shared_f7_prompts=len(pairs))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'qualification-uncertainty.json').write_text(json.dumps(result, indent=1) + '\n')
    fmt = lambda d: f"{100*d['mean']:.1f} [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}]"
    for name, cells in [('F6/F7', cells5), ('F4/F7', cells6)]:
        for c, v in cells.items():
            print(name, c, {k: f"{x['k']}/{x['n']} " + fmt(x) for k, x in v.items()})
    for name, cells in others.items():
        for c, v in cells.items():
            print(name, c, {k: f"{x['k']}/{x['n']} " + fmt(x) for k, x in v.items()})
    for c, v in paired.items():
        print('paired F7 (F6/F7 ref - F4/F7 ref)', c, fmt(v))


if __name__ == '__main__':
    main()
