"""Post hoc comparison of wording transfer with self transfer and with ideal repair.

The self-transfer signatures were measured in the original Bridge 005 (F6/F7) and
Bridge 004 (fresh Medium F1/F2, TinyLlama F2/F3) runs but were not analysed for
endpoint geometry. This script reads those saved arrays and writes
results/self-vs-wording.json. It performs no inference and changes nothing in studies/.

Run from the project root:  .venv/bin/python analysis/self_vs_wording.py
"""
import json

import numpy as np

from common import RESULTS, Crossed, geometry, load, study_rows

REFS = ['zero', 'mean']


def ratio_ci(boot, num, den):
    """Ratio of crossed-bootstrap means, e.g. recovery / clean-minus-corrupted gap."""
    def grid(x):
        g = np.full((len(boot.T), len(boot.P)), np.nan)
        for a in range(len(boot.T)):
            for b in range(len(boot.P)):
                g[a, b] = x[(boot.t == a) & (boot.p == b)].mean()
        return g
    gn, gd = grid(np.asarray(num, float)), grid(np.asarray(den, float))
    bn = gn[boot.tt[:, :, None], boot.pp[:, None, :]].mean((1, 2))
    bd = gd[boot.tt[:, :, None], boot.pp[:, None, :]].mean((1, 2))
    lo, hi = np.quantile(bn / bd, [0.025, 0.975])
    return dict(mean=float(gn.mean() / gd.mean()), ci95=[float(lo), float(hi)])


def projection_terms(h, axis):
    along = (h * axis).sum(-1) / (axis * axis).sum(-1)
    perp = h - along[..., None] * axis
    frac_perp = (perp ** 2).sum(-1) / np.maximum((h ** 2).sum(-1), 1e-24)
    cos = (h * axis).sum(-1) / np.linalg.norm(h, axis=-1) / np.linalg.norm(axis, axis=-1)
    return along, frac_perp, cos


def bridge005():
    s = study_rows('bridge-005')
    rows, donor, scale, test = s['rows'], s['donor'], s['scale'], s['test']
    boot = Crossed(rows)
    data = {k: load(test / f'{k}.npz') for k in ['clean', 'corrupt', 'wording', 'self']}
    R = data['clean']['raw'][..., 0].astype(np.float64)
    D = R[:, donor]
    C = data['corrupt']['raw'][..., 0].astype(np.float64)
    E = {k: data[k]['raw'][..., 0].astype(np.float64) for k in ['wording', 'self']}
    g = {'corrupt': geometry(C, R, D, scale), **{k: geometry(E[k], R, D, scale) for k in E}}
    v = (D - R) / scale
    out = dict(study='Bridge 005, GPT-2 Medium F6/F7', n=len(rows), draws=len(boot.tt),
               interval='pointwise 95% crossed bootstrap over 12 blocks x 12 triples', references={})
    cells = {f'F{s["families"][w]}/order {o}': np.array([r['wording'] == w and r['order'] == o for r in rows])
             for w in range(2) for o in range(2)}
    for bi, ref in enumerate(REFS):
        o = {}
        for cond in ['corrupt', 'self', 'wording']:
            o[cond] = {k: boot(g[cond][k][bi]) for k in ['S', 'qR', 'qD', 'M', 'O']}
        o['wording_minus_self'] = {k: boot(g['wording'][k][bi] - g['self'][k][bi]) for k in ['S', 'qR', 'qD', 'M', 'O']}
        o['wording_minus_self_by_cell'] = {
            c: {k: boot(g['wording'][k][bi] - g['self'][k][bi], m) for k in ['S', 'M']} for c, m in cells.items()}
        n = np.linalg.norm(v[bi], axis=-1)
        dist = lambda a, b: np.linalg.norm(((a - b) / scale)[bi], axis=-1) / n
        o['distance_wording_to_self'] = boot(dist(E['wording'], E['self']))
        o['distance_corrupt_to_wording'] = boot(dist(E['wording'], C))
        o['distance_corrupt_to_self'] = boot(dist(E['self'], C))
        o['distance_corrupt_to_recipient'] = boot(dist(R, C))
        rc = ((R - C) / scale)[bi]
        dc = ((D - C) / scale)[bi]
        _, ideal_perp, _ = projection_terms(rc, v[bi])
        o['ideal_repair_perpendicular_fraction'] = boot(ideal_perp)
        for k in ['wording', 'self']:
            h = ((E[k] - C) / scale)[bi]
            along_axis, perp, _ = projection_terms(h, v[bi])
            prog_r, _, cos_r = projection_terms(h, rc)
            prog_d, _, _ = projection_terms(h, dc)
            o[f'{k}_axial_change'] = boot(along_axis)
            o[f'{k}_perpendicular_fraction'] = boot(perp)
            o[f'{k}_progress_toward_recipient'] = boot(prog_r)
            o[f'{k}_progress_toward_donor'] = boot(prog_d)
            o[f'{k}_cosine_with_ideal_repair'] = boot(cos_r)
            o[f'{k}_recipient_distance_closed'] = boot(1 - g[k]['qR'][bi] / g['corrupt']['qR'][bi])
        out['references'][ref] = o
    yR, yC = data['clean']['behavior'][:, 0], data['corrupt']['behavior'][:, 0]
    beh = dict(clean_gap=boot(yR - yC))
    for k in ['wording', 'self']:
        rec = data[k]['behavior'][:, 0] - yC
        beh[f'{k}_recovery'] = boot(rec)
        beh[f'{k}_fraction_of_gap'] = ratio_ci(boot, rec, yR - yC)
        beh[f'{k}_top1'] = float(data[k]['behavior'][:, 3].mean())
    beh['wording_minus_self_recovery'] = boot(data['wording']['behavior'][:, 0] - data['self']['behavior'][:, 0])
    beh['clean_top1'] = float(data['clean']['behavior'][:, 3].mean())
    beh['corrupt_top1'] = float(data['corrupt']['behavior'][:, 3].mean())
    out['behavior'] = beh
    # How different are the injected signals themselves?
    nc, nk = load(test / 'natural-clean.npz'), load(test / 'natural-corrupt.npz')
    d_self = (nc['head_results'] - nk['head_results']).astype(np.float64).reshape(len(rows), -1)
    d_word = d_self[donor]
    rel = np.linalg.norm(d_word - d_self, axis=1) / np.linalg.norm(d_self, axis=1)
    cos = (d_word * d_self).sum(1) / np.linalg.norm(d_word, axis=1) / np.linalg.norm(d_self, axis=1)
    out['injected_delta'] = dict(relative_difference=boot(rel), cosine=boot(cos))
    return out


def bridge004(model, study='bridge-004'):
    s = study_rows(study, model)
    rows, donor, scale, test = s['rows'], s['donor'], s['scale'], s['test']
    boot = Crossed(rows)
    data = {k: load(test / f'{k}.npz') for k in ['clean', 'corrupt', 'wording', 'self']}
    R = data['clean']['raw'][..., 0].astype(np.float64)
    D = R[:, donor]
    g = {k: geometry(data[k]['raw'][..., 0].astype(np.float64), R, D, scale) for k in ['self', 'wording']}
    yC = data['corrupt']['behavior'][:, 0]
    out = dict(study=f'{study} comparison, {model}, families {s["families"]}', n=len(rows),
               draws=len(boot.tt), references={},
               recovery={k: boot(data[k]['behavior'][:, 0] - yC) for k in ['self', 'wording']})
    for bi, ref in enumerate(REFS):
        out['references'][ref] = {
            **{f'{c}_{k}': boot(g[c][k][bi]) for c in ['self', 'wording'] for k in ['S', 'qR', 'M']},
            **{f'wording_minus_self_{k}': boot(g['wording'][k][bi] - g['self'][k][bi]) for k in ['S', 'qR', 'M']}}
    return out


def main():
    result = dict(
        scope='Post hoc analysis of saved observations. Self-transfer signatures were measured in the original '
              'runs; this geometry was not part of any prespecified family. Intervals are secondary and pointwise.',
        f6_f7=bridge005(),
        fresh_medium_f1_f2=bridge004('gpt2-medium'),
        fresh_tinyllama_f2_f3=bridge004('tinyllama-1.1b-base'),
        initial_medium_f1_f2=bridge004('gpt2-medium', study='bridge-003'))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'self-vs-wording.json').write_text(json.dumps(result, indent=1) + '\n')
    f = result['f6_f7']
    for ref in REFS:
        r = f['references'][ref]
        print(ref, 'S self/word', round(r['self']['S']['mean'], 3), round(r['wording']['S']['mean'], 3),
              '| dS', r['wording_minus_self']['S'], '| dM', r['wording_minus_self']['M'])
    print('fraction of gap', f['behavior']['wording_fraction_of_gap'], f['behavior']['self_fraction_of_gap'])


if __name__ == '__main__':
    main()
