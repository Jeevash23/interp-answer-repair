"""Independent Study 7 audit (runs where the raw arrays are, on the Modal Volume).

Checks every freeze, runner record and chunk hash, the stage records, and rebuilds with separate
plain-NumPy code the primary results of both families (Family A: 16 quantities; Family B: 6) from
the saved arrays and explicitly regenerated bootstrap indices. Imports no analysis code.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RES = HERE / 'results'
LAYOUT = {'gpt2-large': (19, 20, 340), 'gpt2-medium': (13, 16, 176)}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def draws_for(seed, nb, nt, draws):
    rng = np.random.default_rng(seed)
    return rng.integers(nb, size=(draws, nb)), rng.integers(nt, size=(draws, nt))


def summarise(values, cells, blocks, triples, tail):
    grid = np.array([[values[ix].mean() for ix in row] for row in cells])
    out = np.empty(len(blocks))
    for s in range(0, len(blocks), 5000):
        b, t = blocks[s:s + 5000], triples[s:s + 5000]
        out[s:s + 5000] = grid[b[:, :, None], t[:, None, :]].mean((1, 2))
    return float(grid.mean()), np.quantile(out, [tail, 1 - tail]).tolist()


def run(test=False, draws=None):
    target = RES / 'verification.json'
    assert not target.exists(), target
    draws = draws or (500 if test else 50_000)
    first, heads, count = LAYOUT['gpt2-medium' if test else 'gpt2-large']
    checked = 0
    freezes = sorted((RES / 'freezes').glob('*.json')) + ([] if test else [HERE / 'data/input-freeze.json', HERE / 'data/protocol-freeze.json'])
    for path in freezes:
        for rel, expected in read(path)['files'].items():
            if (ROOT / rel).exists():
                assert sha(ROOT / rel) == expected, (path.name, rel)
                checked += 1
            else:
                assert rel.startswith(('provenance/', 'studies/bridge-0')) and not rel.startswith('studies/bridge-012/results'), rel
    records = []
    for rp in sorted(RES.rglob('*-record.json')):
        state = read(rp)
        assert state['status'] == 'completed', rp
        assert sha(rp.with_name(rp.name.replace('-record.json', '.npz'))) == state['aggregate_sha256'], rp
        for chunk, expected in state['hashes'].items():
            assert sha(rp.parent / chunk) == expected, chunk
        records.append(str(rp.relative_to(RES)))
    complete = read(RES / 'inference-complete.json')
    for name, expected in complete['outputs'].items():
        assert sha(RES / 'test' / f'{name}.npz') == expected, name
    replays = {d: read(RES / f'{d}-replay.json') for d in ('gpu', 'cpu')}
    assert all(r['passed'] for r in replays.values())

    sel, dev = read(RES / 'selection.json'), read(RES / 'development.json')
    fam = sel['families']
    rows = [dict(r, wording=fam.index(r['family'])) for r in read(HERE / 'data/rows.json')
            if r['split'] == ('calibration' if test else 'test') and r['family'] in fam]
    key = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(rows)}
    donor = np.array([key[(r['template'], r['pair'], r['role'], 1 - r['wording'], r['order'])] for r in rows])
    blocks, trips = sorted({r['template'] for r in rows}), sorted({r['pair'] for r in rows})
    cells_ix = [[np.array([i for i, r in enumerate(rows) if r['template'] == t and r['pair'] == p]) for p in trips] for t in blocks]
    scale = np.asarray(sel['reference']['scale'], float)
    raw = {n: arrays(RES / 'test' / f'{n}.npz') for n in ('clean', 'corrupt', 'wording', 'self', 'order', 'replay', 'n1', 'n2', 'self_kappa')
           if (RES / 'test' / f'{n}.npz').exists()}
    sig = {n: v['raw'][..., 0].astype(np.float64) for n, v in raw.items()}
    R, D = sig['clean'], sig['clean'][:, donor]

    def dist(a, b, keep):
        num = np.sqrt((((a - b) / scale[:, None, :])[..., keep] ** 2).sum(-1))
        den = np.sqrt((((D - R) / scale[:, None, :])[..., keep] ** 2).sum(-1))
        assert (den >= 1e-6).all()
        return num / den

    def score(a, keep):
        u, v = ((a - R) / scale[:, None, :])[..., keep], ((D - R) / scale[:, None, :])[..., keep]
        return (u * v).sum(-1) / (v * v).sum(-1)

    rebuilt, errors = {}, []
    # Family A (Study 2's 16 quantities, all heads, Study 2's seed and draws).
    fa = read(RES / 'analysis/family-a.json')
    bt, bp = draws_for(2026100407, len(blocks), len(trips), draws)
    tail = 0.05 / (2 * 16)
    allk = np.ones(count, bool)
    gain = raw['wording']['behavior'][:, 0].astype(float) - raw['corrupt']['behavior'][:, 0].astype(float)
    cellmask = {f'w{w}_o{o}': np.array([r['wording'] == w and r['order'] == o for r in rows]) for w in range(2) for o in range(2)}

    def masked(values, mask):
        return [[ix[mask[ix]] for ix in row] for row in cells_ix]
    for cell, mask in cellmask.items():
        mean, ci = summarise(gain, masked(gain, mask), bt, bp, tail)
        saved = fa['primary']['recovery'][cell]
        errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci, saved['adjusted_ci']))))
    margin = dist(sig['wording'], R, allk) - dist(sig['wording'], D, allk)
    axis = np.abs(score(sig['wording'], allk) - score(sig['self'], allk))
    full = dist(sig['wording'], sig['self'], allk)
    for i, ref in enumerate(('zero', 'fit_mean')):
        for cell, mask in cellmask.items():
            mean, ci = summarise(margin[i], masked(margin[i], mask), bt, bp, tail)
            saved = fa['primary']['margin'][ref][cell]
            errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci, saved['adjusted_ci']))))
        for name, values in (('mean_absolute_axis_difference', axis[i]), ('mean_full_signature_distance', full[i])):
            mean, ci = summarise(values, cells_ix, bt, bp, tail)
            saved = fa['primary'][name][ref]
            errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci, saved['adjusted_ci']))))
            rebuilt[f'A_{name}_{ref}'] = mean
    # Family B (Study 6's 6 quantities, retained heads, seed 2026101704).
    fb = read(RES / 'analysis/family-b.json')
    keep = np.ones(count, bool)
    for layer, head in dev['held']:
        keep[(layer - first) * heads + head] = False
    bt, bp = draws_for(2026101704, len(blocks), len(trips), draws)
    tail = 0.05 / (2 * 6)
    for control, comparator in (('n1', 'clean'), ('n2', 'self_kappa'), ('order', 'clean')):
        if control not in sig:
            continue
        values = dist(sig[control], sig[comparator], keep)
        for i, ref in enumerate(('zero', 'fit_mean')):
            mean, ci = summarise(values[i], cells_ix, bt, bp, tail)
            saved = fb['primary'][control][ref]
            errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci, saved['adjusted_ci']))))
            label = 'detected' if ci[0] > 1 else 'below_resolution' if ci[1] < 0.5 else 'intermediate'
            assert label == saved['band'], (control, ref, label, saved['band'])
            rebuilt[f'B_{control}_{ref}'] = dict(mean=mean, adjusted_ci=ci, band=label)
    anchor = float(np.abs(sig['replay'] - D).max())
    errors.append(abs(anchor - fb['anchor']['max_abs_error_vs_donor']))
    worst = float(max(errors))
    value = dict(status='passed' if worst <= 1e-9 else 'failed', utc=datetime.now(timezone.utc).isoformat(), code_test=test,
                 draws=draws, max_reconstruction_error=worst, quantities_rebuilt=len(errors), rebuilt=rebuilt,
                 freeze_file_hashes_checked=checked, runner_records=records, anchor_max_abs_error=anchor,
                 replays={d: dict(passed=r['passed'], worst=max(max(v.values()) for v in r['checks'].values())) for d, r in replays.items()},
                 analysis_hashes={p.name: sha(p) for p in sorted((RES / 'analysis').glob('*'))})
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    assert value['status'] == 'passed', worst
    return dict(status=value['status'], max_reconstruction_error=worst, quantities=len(errors), records=len(records))
