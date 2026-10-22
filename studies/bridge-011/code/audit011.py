"""Independent Study 6 audit: record hashes, stage records, and a separate plain-NumPy
reconstruction of the primary results and selected secondary values from the saved arrays
and the saved bootstrap indices. It imports no Study 2 or Study 6 analysis code.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RES = HERE / 'results'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def records():
    checked = []
    for rp in sorted(RES.rglob('*-record.json')):
        state = read(rp)
        assert state['status'] == 'completed', rp
        aggregate = rp.with_name(rp.name.replace('-record.json', '.npz'))
        assert sha(aggregate) == state['aggregate_sha256'], aggregate
        for chunk, expected in state['hashes'].items():
            assert sha(rp.parent / chunk) == expected, chunk
        checked.append(dict(path=str(rp.relative_to(HERE)), chunks=len(state['hashes']),
                            aggregate_sha256=state['aggregate_sha256']))
    return checked


def bootstrap(grid, blocks, triples, tail):
    out = np.empty(len(blocks))
    for start in range(0, len(blocks), 5000):
        b, p = blocks[start:start + 5000], triples[start:start + 5000]
        out[start:start + 5000] = grid[b[:, :, None], p[:, None, :]].mean((1, 2))
    return float(grid.mean()), np.quantile(out, [0.025, 0.975]).tolist(), np.quantile(out, [tail, 1 - tail]).tolist()


def main():
    target = RES / 'verification.json'
    assert not target.exists(), target
    for name in ('protocol-freeze.json', 'input-freeze.json', 'development-freeze.json'):
        for rel, expected in read(HERE / 'data' / name)['files'].items():
            assert sha(ROOT / rel) == expected, (name, rel)
    assert read(RES / 'calibration.json')['passed']
    complete = read(RES / 'inference-complete.json')
    for name, expected in complete['outputs'].items():
        assert sha(RES / 'test' / f'{name}.npz') == expected, name
    replays = {d: read(RES / f'{d}-replay.json') for d in ('gpu', 'cpu')}
    assert all(r['passed'] for r in replays.values())
    rec = records()

    summary = read(RES / 'analysis/summary.json')
    dev = read(RES / 'development.json')
    rows = [r for r in read(HERE / 'data/rows.json') if r['split'] == 'test']
    scale = np.asarray(read(ROOT / 'studies/bridge-007/data/selection.json')['reference']['scale'], float)
    keep = np.ones(176, bool)
    for layer, head in dev['held']:
        keep[(layer - 13) * 16 + head] = False
    key = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(rows)}
    donor = np.array([key[(r['template'], r['pair'], r['role'], 1 - r['wording'], r['order'])] for r in rows])
    blocks = sorted({r['template'] for r in rows}); triples = sorted({r['pair'] for r in rows})
    cell = [[np.array([i for i, r in enumerate(rows) if r['template'] == t and r['pair'] == p]) for p in triples] for t in blocks]
    boot = arrays(HERE / 'data/bootstrap-test.npz')
    sig = {}
    for name in ('clean', 'self_kappa', 'n1', 'n2', 'order', 'wording', 'self', 'replay'):
        path = RES / 'test' / f'{name}.npz'
        if path.exists():
            sig[name] = arrays(path)['raw'][..., 0].astype(np.float64)
    R = sig['clean']; D = R[:, donor]

    def dist(a, b, mask):
        num = np.sqrt((((a - b) / scale[:, None, :])[..., mask] ** 2).sum(-1))
        den = np.sqrt((((D - R) / scale[:, None, :])[..., mask] ** 2).sum(-1))
        assert (den >= 1e-6).all()
        return num / den

    def grid(values):
        return np.array([[values[ix].mean() for ix in row] for row in cell])

    errors, rebuilt = [], {}
    tail = 0.05 / (2 * 6)
    for control, comparator in (('n1', 'clean'), ('n2', 'self_kappa'), ('order', 'clean')):
        if control not in sig:
            continue
        values = dist(sig[control], sig[comparator], keep)
        for i, ref in enumerate(('zero', 'fit_mean')):
            mean, ci95, adjusted = bootstrap(grid(values[i]), boot['blocks'], boot['triples'], tail)
            saved = summary['primary'][control][ref]
            errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci95, saved['ci95'])),
                              *np.abs(np.subtract(adjusted, saved['adjusted_ci']))))
            label = 'detected' if adjusted[0] > 1 else 'below_resolution' if adjusted[1] < 0.5 else 'intermediate'
            assert label == saved['band'], (control, ref, label, saved['band'])
            rebuilt[f'{control}_{ref}'] = dict(mean=mean, adjusted_ci=adjusted, band=label)
    # Secondary spot checks: the wording/self discrepancy, N1's margin and the anchor.
    full = dist(sig['wording'], sig['self'], keep)
    for i, ref in enumerate(('zero', 'fit_mean')):
        mean, ci95, _ = bootstrap(grid(full[i]), boot['blocks'], boot['triples'], 0.025)
        saved = summary['replication']['retained'][ref]['mean_full_signature_distance']
        errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci95, saved['ci95']))))
        rebuilt[f'F_{ref}'] = mean
        if 'n1' in sig:
            margin = dist(sig['n1'], R, keep)[i] - dist(sig['n1'], D, keep)[i]
            mean, ci95, _ = bootstrap(grid(margin), boot['blocks'], boot['triples'], 0.025)
            saved = summary['geometry']['n1'][ref]['margin']
            errors.append(max(abs(mean - saved['mean']), *np.abs(np.subtract(ci95, saved['ci95']))))
            rebuilt[f'n1_margin_{ref}'] = mean
    anchor = float(np.abs(sig['replay'] - D).max())
    errors.append(abs(anchor - summary['anchor']['max_abs_error_vs_donor']))
    worst = float(max(errors))
    value = dict(status='passed' if worst <= 1e-9 else 'failed', utc=datetime.now(timezone.utc).isoformat(),
                 max_reconstruction_error=worst, rebuilt=rebuilt, anchor_max_abs_error=anchor,
                 runner_records=rec, replays={d: dict(passed=r['passed'], worst=max(max(v.values()) for v in r['checks'].values()))
                                              for d, r in replays.items()},
                 analysis_summary_sha256=sha(RES / 'analysis/summary.json'),
                 scope='Freezes, runner records and chunk hashes, stage records, and an independent reconstruction of all '
                       'primary means, pointwise and adjusted intervals and bands, plus F, the N1 margin and the anchor')
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=value['status'], max_reconstruction_error=worst, records=len(rec))))
    assert value['status'] == 'passed'


if __name__ == '__main__':
    main()
