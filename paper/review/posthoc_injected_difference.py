"""POST HOC descriptive analysis of injected-vector differences (no model inference).

(a) Study 2: per-configuration relationship between the relative injected
    difference r = ||delta_w - delta_s|| / ||delta_s|| (twelve projected source
    heads, concatenated; saved prespecified secondary values) and the saved
    wording/self discrepancies F and A, under both replacement references.
    Spearman rank correlations with crossed context-block x name-triple bootstrap
    intervals, and decile-binned means. Because mention order shifts r and F in
    opposite directions, correlations and quintile-binned means are also given
    within each wording/order cell; this decomposition was chosen after the
    pooled correlation had been seen.
(b) Study 3: wording/self cosine and relative difference of the injected
    layer vectors (source heads summed within each of the seven S2 layers,
    concatenated) for narrow, broad and norm-matched broad coverage, next to the
    saved F and A of each condition.

Both analyses were defined after all results were known. They describe saved
arrays and carry no confirmatory weight. Outputs: review/posthoc-injected-difference.json,
figures/posthoc-injected-difference.pdf and sections/posthoc-coverage-table.tex.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.stats import rankdata

sys.dont_write_bytecode = True
REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
B7, B8 = ROOT / 'studies/bridge-007', ROOT / 'studies/bridge-008'
sys.path.insert(0, str(B7 / 'code'))
from analyze007 import Crossed, rows_for, validate_rows  # noqa: E402

SEED = 2026100511
DRAWS = 50_000
REFS = ('zero', 'fit_mean')
COVERAGE = (('narrow', 'narrow_layer_deltas'), ('broad', 'broad_layer_deltas'), ('matched', 'layer_deltas'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def spearman_rows(x, y):
    """Row-wise Spearman correlation with average ranks for ties."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))


def interval(point, draws):
    return dict(mean=float(point), ci95=np.quantile(draws, [.025, .975]).tolist())


def main():
    rows = rows_for('test', [10, 11])
    donors = validate_rows(rows)
    boot = Crossed(rows, draws=DRAWS, seed=SEED)
    cells = np.empty((len(boot.blocks), len(boot.triples), 8), int)
    for b in range(len(boot.blocks)):
        for t in range(len(boot.triples)):
            ix = np.flatnonzero((boot.t == b) & (boot.p == t)); assert len(ix) == 8
            cells[b, t] = sorted(ix, key=lambda i: (rows[i]['wording'], rows[i]['order'], rows[i]['role']))
    cell_of = np.array([2 * row['wording'] + row['order'] for row in rows])
    assert all((cell_of[cells[..., 2 * c:2 * c + 2]] == c).all() for c in range(4))

    # ------------------------------------------------------------ inputs
    inputs = [B7 / 'results/secondary-analysis/per-example.npz', B7 / 'results/analysis/per-example.npz',
              B7 / 'results/test/natural-clean.npz', B7 / 'results/test/natural-corrupt.npz',
              B8 / 'results/deltas/wording.npz', B8 / 'results/deltas/self.npz',
              B8 / 'results/analysis/per-example.npz', B8 / 'results/analysis/summary.json']
    secondary = np.load(inputs[0]); primary = np.load(inputs[1])
    r = secondary['injected__relative_difference'].astype(float)
    F = primary['full_signature_distance'].astype(float); A = primary['absolute_axis_difference'].astype(float)
    assert r.shape == (3200,) and F.shape == A.shape == (2, 3200) and np.isfinite(F).all() and np.isfinite(A).all()
    # Check the saved r against a recomputation from the saved head outputs.
    with np.load(inputs[2]) as clean, np.load(inputs[3]) as corrupt:
        ds = (clean['head_results'] - corrupt['head_results']).reshape(len(rows), -1).astype(float)
    dw = ds[donors]
    r_check = np.linalg.norm(dw - ds, axis=1) / np.linalg.norm(ds, axis=1)
    r_error = float(np.max(np.abs(r_check - r)))
    assert r_error < 1e-5, r_error
    del ds, dw

    # ------------------------------------------- (a) Study 2 relationship
    point = {ref: dict(F=float(spearman_rows(r[None], F[i][None])[0]), A=float(spearman_rows(r[None], A[i][None])[0]))
             for i, ref in enumerate(REFS)}
    edges = np.quantile(r, np.linspace(0, 1, 11)); edges[-1] += 1e-12
    label = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, 9)
    counts = np.bincount(label, minlength=10)
    bin_point = {ref: {m: [float(v[i][label == k].mean()) for k in range(10)] for m, v in (('F', F), ('A', A))}
                 for i, ref in enumerate(REFS)}
    CELLS = ['F10/order 0', 'F10/order 1', 'F11/order 0', 'F11/order 1']
    qlabel = np.empty(len(rows), int)
    for c in range(4):
        m = cell_of == c; q = np.quantile(r[m], np.linspace(0, 1, 6)); q[-1] += 1e-12
        qlabel[m] = np.clip(np.searchsorted(q, r[m], side='right') - 1, 0, 4)
    within_point = {ref: {m: [float(spearman_rows(r[cell_of == c][None], v[i][cell_of == c][None])[0]) for c in range(4)]
                          for m, v in (('F', F), ('A', A))} for i, ref in enumerate(REFS)}
    quint_point = {ref: [[float(F[i][(cell_of == c) & (qlabel == k)].mean()) for k in range(5)] for c in range(4)]
                   for i, ref in enumerate(REFS)}
    rho = {ref: {m: np.empty(DRAWS) for m in 'FA'} for ref in REFS}
    within = {ref: {m: np.empty((DRAWS, 4)) for m in 'FA'} for ref in REFS}
    bins = {ref: {m: np.empty((DRAWS, 10)) for m in 'FA'} for ref in REFS}
    quint = {ref: np.empty((DRAWS, 4, 5)) for ref in REFS}
    for start in range(0, DRAWS, 500):
        stop = min(start + 500, DRAWS)
        idx4 = cells[boot.tt[start:stop, :, None], boot.pp[start:stop, None, :]]
        idx = idx4.reshape(stop - start, -1)
        rr, lab = r[idx], label[idx]
        weights = np.stack([np.bincount(row, minlength=10) for row in lab])
        for i, ref in enumerate(REFS):
            for m, v in (('F', F[i]), ('A', A[i])):
                vv = v[idx]
                rho[ref][m][start:stop] = spearman_rows(rr, vv)
                sums = np.stack([np.bincount(row, weights=w, minlength=10) for row, w in zip(lab, vv)])
                bins[ref][m][start:stop] = sums / weights
                for c in range(4):
                    ic = idx4[..., 2 * c:2 * c + 2].reshape(stop - start, -1)
                    within[ref][m][start:stop, c] = spearman_rows(r[ic], v[ic])
                    if m == 'F':
                        ql, fv = qlabel[ic], v[ic]
                        quint[ref][start:stop, c] = (np.stack([np.bincount(a, weights=b, minlength=5) for a, b in zip(ql, fv)]) /
                                                     np.stack([np.bincount(a, minlength=5) for a in ql]))
    raw = F * primary['wording__pair_norm']
    cell_means = {CELLS[c]: dict(r=float(r[cell_of == c].mean()),
                                 F={ref: float(F[i][cell_of == c].mean()) for i, ref in enumerate(REFS)},
                                 clean_separation={ref: float(primary['wording__pair_norm'][i][cell_of == c].mean()) for i, ref in enumerate(REFS)},
                                 raw_discrepancy={ref: float(raw[i][cell_of == c].mean()) for i, ref in enumerate(REFS)})
                  for c in range(4)}
    study2 = dict(
        definition='r: saved per-configuration ||delta_w - delta_s|| / ||delta_s|| over the twelve projected S2 head differences; '
                   'F, A: saved Study 2 wording/self discrepancies (clean-wording separation units).',
        r_recomputation_max_error=r_error,
        r=dict(mean=float(r.mean()), quantiles=dict(zip(['min', 'p10', 'p50', 'p90', 'max'],
                                                        np.quantile(r, [0, .1, .5, .9, 1]).tolist()))),
        spearman={ref: {m: interval(point[ref][m], rho[ref][m]) for m in 'FA'} for ref in REFS},
        within_cell_spearman={ref: {m: dict(cells={CELLS[c]: interval(within_point[ref][m][c], within[ref][m][:, c]) for c in range(4)},
                                            mean=interval(np.mean(within_point[ref][m]), within[ref][m].mean(1)))
                                    for m in 'FA'} for ref in REFS},
        within_cell_quintiles={ref: {CELLS[c]: dict(median_r=[float(np.median(r[(cell_of == c) & (qlabel == k)])) for k in range(5)],
                                                    F=[interval(quint_point[ref][c][k], quint[ref][:, c, k]) for k in range(5)])
                                     for c in range(4)} for ref in REFS},
        cell_means=cell_means,
        deciles=dict(edges=edges.tolist(), counts=counts.tolist(),
                     median_r=[float(np.median(r[label == k])) for k in range(10)],
                     means={ref: {m: [interval(bin_point[ref][m][k], bins[ref][m][:, k]) for k in range(10)]
                                  for m in 'FA'} for ref in REFS}))

    # ----------------------------------------------- (b) Study 3 coverage
    p8 = np.load(inputs[6]); s8 = json.loads(inputs[7].read_text())
    coverage = {}
    with np.load(inputs[4]) as w8, np.load(inputs[5]) as self8:
        for name, key in COVERAGE:
            cos = np.empty(len(rows)); rel = np.empty(len(rows))
            wording_all, self_all = w8[key], self8[key]
            assert wording_all.shape == self_all.shape == (len(rows), 7, 1024)
            for start in range(0, len(rows), 400):
                vw = wording_all[start:start + 400].reshape(-1, 7 * 1024).astype(float)
                vs = self_all[start:start + 400].reshape(-1, 7 * 1024).astype(float)
                nw, ns = np.linalg.norm(vw, axis=1), np.linalg.norm(vs, axis=1)
                cos[start:start + 400] = (vw * vs).sum(1) / (nw * ns)
                rel[start:start + 400] = np.linalg.norm(vw - vs, axis=1) / ns
            assert np.isfinite(cos).all() and np.isfinite(rel).all()
            Fc = p8[name + '__full_signature_discrepancy'].astype(float)
            Ac = np.abs(p8[name + '__signed_axis_difference'].astype(float))
            entry = dict(cosine=interval(cos.mean(), boot.resample(boot.grid(cos))),
                         relative_difference=interval(rel.mean(), boot.resample(boot.grid(rel))),
                         F={ref: float(Fc[i].mean()) for i, ref in enumerate(REFS)},
                         A={ref: float(Ac[i].mean()) for i, ref in enumerate(REFS)})
            # The saved Study 3 summary must give the same pooled F.
            saved = s8['secondary_wording_self'][name]
            for i, ref in enumerate(REFS):
                assert abs(saved[ref]['full_signature_discrepancy']['average']['mean'] - entry['F'][ref]) < 1e-12
            coverage[name] = entry
    study3 = dict(definition='Wording and self injected layer vectors: source-head outputs summed within each of the seven S2 layers '
                             '(narrow: twelve heads; broad: all 112; matched: broad rescaled per layer to the narrow norm), '
                             'concatenated across layers. Cosine and ||v_w - v_s|| / ||v_s|| per configuration; pooled means with '
                             'pointwise crossed-bootstrap intervals. F and A: saved Study 3 pooled discrepancies.',
                  coverage=coverage)
    result = dict(label='POST HOC descriptive analysis; defined after all results were known; no model inference',
                  seed=SEED, draws=DRAWS, bootstrap='Crossed: 20 context blocks and 20 disjoint name triples resampled '
                  'independently with replacement, keeping all eight paired configurations in each cell; pointwise 95% intervals.',
                  study2=study2, study3=study3,
                  input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    (REV / 'review/posthoc-injected-difference.json').write_text(json.dumps(result, indent=2) + '\n')
    figure(result)
    table(result)
    print(json.dumps(dict(within={ref: {m: [round(v['mean']['mean'], 4), [round(x, 4) for x in v['mean']['ci95']],
                                             {c: round(e['mean'], 3) for c, e in v['cells'].items()}] for m, v in d.items()}
                                for ref, d in study2['within_cell_spearman'].items()},
                          pooled_ci={ref: {m: [round(x, 4) for x in v['ci95']] for m, v in d.items()} for ref, d in study2['spearman'].items()}), indent=1))
    print(json.dumps(dict(spearman={ref: {m: round(v['mean'], 4) for m, v in d.items()} for ref, d in study2['spearman'].items()},
                          coverage={k: dict(cos=round(v['cosine']['mean'], 4), rel=round(v['relative_difference']['mean'], 4),
                                            F=[round(x, 4) for x in v['F'].values()]) for k, v in coverage.items()}), indent=1))


def figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    blue, orange = '#2a78d6', '#eb6834'
    s2 = result['study2']; d = s2['deciles']; x = np.asarray(d['median_r'])
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
    def errorbars(ax, xs, values, **style):
        mean = np.array([v['mean'] for v in values]); lo, hi = np.array([v['ci95'] for v in values]).T
        ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], ms=3.2, lw=1, elinewidth=.7, capsize=0, **style)
    for ref, colour, marker, name in [('zero', blue, 'o', 'Zero replacement'), ('fit_mean', orange, 'D', 'Mean replacement')]:
        errorbars(axes[0], x, d['means'][ref]['F'], color=colour, marker=marker, label=name)
    axes[0].set_title('(a) Pooled, deciles', loc='left'); axes[0].legend(frameon=False, fontsize=7, loc='upper right')
    styles = {'F10/order 0': ('#1b9e77', 'o', '-'), 'F10/order 1': ('#1b9e77', 'D', '--'),
              'F11/order 0': ('#7570b3', 'o', '-'), 'F11/order 1': ('#7570b3', 'D', '--')}
    for ax, ref, title in [(axes[1], 'zero', '(b) Within cells, zero'), (axes[2], 'fit_mean', '(c) Within cells, mean')]:
        for cell, (colour, marker, line) in styles.items():
            q = s2['within_cell_quintiles'][ref][cell]
            errorbars(ax, np.asarray(q['median_r']), q['F'], color=colour, marker=marker, ls=line, label=cell.replace('order ', 'order\u00a0'))
        ax.set_title(title, loc='left')
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.66, 0.02))
    axes[0].set_ylabel('mean wording/self discrepancy $F$')
    for ax in axes:
        ax.set_xlabel(r'relative injected difference $\|\delta_w-\delta_s\|/\|\delta_s\|$', fontsize=7)
        ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=.6)
    fig.savefig(REV / 'figures/posthoc-injected-difference.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None, 'Subject': 'Post hoc; saved Study 2 arrays'})
    plt.close(fig)


def table(result):
    num = lambda v: f'{v:.3f}'
    # Study 2's prespecified secondary analysis summarises the same narrow edits as concatenated per-head vectors
    # (Table 3 and Section 5.6); the caption quotes those values so that the two summaries are not confused.
    per_head = json.loads((B7 / 'results/secondary-analysis/summary.json').read_text())['injected_delta']
    hc, hr = per_head['cosine']['pooled'], per_head['relative_difference']['pooled']
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Post hoc: wording/self similarity of the injected layer vectors at each Study 3 source coverage, next to the '
             r'saved signature discrepancies. Vectors sum the source-head outputs within each of the seven S2 layers and are '
             r'concatenated across layers; matched broad rescales each broad layer vector to the narrow norm. These per-layer sums '
             r'differ from the concatenated per-head vectors of \cref{tab:study-comparison} and \cref{sec:confirmation-results} '
             f"(cosine ${num(hc['mean'])}\\ci{{{num(hc['ci95'][0])}}}{{{num(hc['ci95'][1])}}}$, relative difference "
             f"{num(hr['mean'])}), so the narrow values differ slightly. Cosine and relative "
             r'difference are means over 3,200 configurations with pointwise 95\% crossed-bootstrap intervals; $F$ and $A$ are '
             r'zero / mean replacement.}',
             r'\label{tab:posthoc-coverage}', r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r'Coverage & Cosine & Relative difference & $F$ & $A$ \\', r'\midrule']
    names = dict(narrow='Narrow (12 heads)', broad='Broad (112 heads)', matched='Matched broad')
    for key, entry in result['study3']['coverage'].items():
        c, r = entry['cosine'], entry['relative_difference']
        lines.append(f"{names[key]} & ${num(c['mean'])}\\ci{{{num(c['ci95'][0])}}}{{{num(c['ci95'][1])}}}$ & "
                     f"${num(r['mean'])}\\ci{{{num(r['ci95'][0])}}}{{{num(r['ci95'][1])}}}$ & "
                     f"{num(entry['F']['zero'])} / {num(entry['F']['fit_mean'])} & {num(entry['A']['zero'])} / {num(entry['A']['fit_mean'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    header = ('% Generated by review/posthoc_injected_difference.py from saved Study 3 arrays (post hoc); the caption quotes\n'
              '% the per-head values of the saved Study 2 secondary analysis.\n')
    (REV / 'sections/posthoc-coverage-table.tex').write_text(header + '\n'.join(lines))


if __name__ == '__main__':
    main()
