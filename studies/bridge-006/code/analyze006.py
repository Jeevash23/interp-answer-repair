"""Replay the post hoc F4/F7 extension under its prospectively saved analysis rules.

This analysis implementation was written after qualification inference started;
the pair, estimands, gates, seed and multiplicity were fixed in PROTOCOL.md first.
No Bridge 005 files are changed. Undefined primary cells cannot qualify.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'studies/bridge-005/code'))
from common005 import (read, write, digest, now, load_arrays, save_arrays, rows_for,
    donor_indices, paired, geometry, separation, min_separation, behavior_cells,
    behavior_pass, delta, verify, fit_reference)
from reanalyze005 import crossed, movement

HERE = ROOT / 'studies/bridge-006'
OLD = ROOT / 'studies/bridge-005'
CONDITIONS = ['clean', 'corrupt', 'wording', 'zero', 'self', 'order', 'role', 'order_role'] + [
    k + str(i) for k in ['residual', 'projected'] for i in range(3)]


def frozen_inputs():
    for name in ['protocol-freeze.json', 'implementation-freeze.json']:
        for path, sha in read(HERE / name)['files'].items():
            assert digest(ROOT / path) == sha, path
    for name in ['input-freeze.json', 'selection-freeze.json', 'test-freeze.json']:
        verify(name)
    assert read(HERE / 'results/inference-complete.json')['status'] == 'completed'
    assert read(HERE / 'results/numerical-validation.json')['passed']
    for split, names in [('qualification', ['clean']), ('test', ['clean', 'corrupt', 'wording'])]:
        rows = rows_for(split, [4, 7])
        assert rows == read(HERE / 'data' / (split + '.json'))
        for name in names:
            record = read(HERE / 'results' / split / (name + '-record.json'))
            assert record['status'] == 'completed'
            assert record['ids'] == [r['id'] for r in rows]
            for path, sha in record['hashes'].items():
                assert digest(HERE / 'results' / split / path) == sha, path
            assert digest(HERE / 'results' / split / (name + '.npz')) == record['aggregate_sha256']
    paths = sorted(list((HERE / 'results/qualification').glob('*.npz')) +
                   list((HERE / 'results/test').glob('*.npz')) +
                   list((HERE / 'results').glob('validation-*.npz')))
    record = HERE / 'measurement-manifest.json'
    if record.exists():
        for path, sha in read(record)['files'].items():
            assert digest(ROOT / path) == sha, path
    else:
        write(record, dict(utc=now(), scope='Post-inference measurement integrity manifest; not preregistration.',
                          files={str(p.relative_to(ROOT)): digest(p) for p in paths}))


def main(output=None):
    frozen_inputs()
    out = Path(output) if output else HERE / 'results/analysis'
    assert not out.exists(), out
    sel = read(OLD / 'data/selection.json')['scores']['4,7']
    fit = sel['reference']
    # Independently refit this pair, checking the indexing order explicitly.
    all_fit = rows_for('fit', read(OLD / 'results/screen.json')['eligible'])
    ids = [i for i, r in enumerate(all_fit) if r['family'] in [4, 7]]
    rr = [dict(all_fit[i], wording=int(all_fit[i]['family'] == 7)) for i in ids]
    raw = load_arrays(OLD / 'results/fit/clean.npz')['raw'][:, ids]
    assert raw.shape == (2, 288, 176, 2)
    assert fit_reference(raw, rr) == fit
    qualification = {}
    for split in ['qualification', 'test']:
        rr = rows_for(split, [4, 7])
        clean = load_arrays(HERE / 'results' / split / 'clean.npz')
        bc = behavior_cells(clean['behavior'], rr)
        sep = separation(clean['raw'], rr, fit)
        passed = bool(behavior_pass(bc) and min_separation(sep) >= .8)
        saved = read(HERE / 'results' / (split + '-qualification.json'))
        assert saved['behavior'] == bc and saved['separation'] == sep and saved['qualified'] == passed
        qualification[split] = dict(n=len(rr), behavior=bc, separation=sep, qualified=passed,
                                   worst_separation=min_separation(sep))
    rows = rows_for('test', [4, 7])
    p = HERE / 'results/test'
    data = {k: load_arrays(p / (k + '.npz')) for k in CONDITIONS}
    clean, corrupt, edited = [data[k] for k in ['clean', 'corrupt', 'wording']]
    natural = {}
    for k in ['clean', 'corrupt']:
        v = load_arrays(p / ('natural-' + k + '.npz'))
        natural[k + '_head_results'] = v['head_results']
        np.testing.assert_allclose(v['behavior'], data[k]['behavior'], atol=.0005, rtol=0)
    np.testing.assert_allclose(corrupt['behavior'], data['zero']['behavior'], atol=.0005, rtol=0)
    norms = delta(natural, rows, 'wording').norm(dim=-1).numpy()
    max_norm_error = 0.
    for kind in ['residual', 'projected']:
        for i in range(3):
            v = data[kind + str(i)]['head_norms']
            np.testing.assert_allclose(v, norms, atol=.0001, rtol=0)
            max_norm_error = max(max_norm_error, float(abs(v - norms).max()))
    donors = donor_indices(rows, 'wording')
    # A second explicit lookup checks paired donor identity and direction.
    for i, row in enumerate(rows):
        donor = rows[donors[i]]
        assert donor['wording'] == 1 - row['wording']
        for key in ['template', 'pair', 'order', 'role']:
            assert donor[key] == row[key]
    masks = {f'w{w}_o{o}': np.array([r['wording'] == w and r['order'] == o for r in rows])
             for w in range(2) for o in range(2)}
    scale = np.asarray(fit['scale'])[:, None]
    r = clean['raw'][..., 0].astype(float)
    d = r[:, donors]
    e = edited['raw'][..., 0].astype(float)
    g = paired(e, r, d, scale)
    result = dict(scope='Post hoc F4/F7 extension sharing blocks, names and F7 with Bridge 005; diagnostic after failed qualification.',
        families=[4, 7], n=len(rows), primary_multiplicity=12, bootstrap_seed=2026100106,
        bootstrap_draws=50000, qualification=qualification, original_selection_eligible=sel['eligible'],
        original_selection_worst=sel['worst'], conditions={}, primary={}, geometry={}, sensitivities={},
        clean_top1=float(clean['behavior'][:, 3].mean()), corrupt_top1=float(corrupt['behavior'][:, 3].mean()),
        validation=read(HERE / 'results/numerical-validation.json'), random_head_norm_max_error=max_norm_error)
    previous_rows = rows_for('test', [6, 7])
    previous_lookup = {row['id']: i for i, row in enumerate(previous_rows)}
    new_ids = [i for i, row in enumerate(rows) if row['family'] == 7]
    old_ids = [previous_lookup[rows[i]['id']] for i in new_ids]
    result['f7_shared_observation_checks'] = {}
    for name in ['clean', 'corrupt']:
        previous = load_arrays(OLD / 'results/test' / (name + '.npz'))
        error = float(np.max(abs(data[name]['raw'][:, new_ids] - previous['raw'][:, old_ids])))
        behavior_error = float(np.max(abs(data[name]['behavior'][new_ids] - previous['behavior'][old_ids])))
        assert max(error, behavior_error) < .0005
        record = read(p / (name + '-record.json'))
        result['f7_shared_observation_checks'][name] = dict(n=len(new_ids), signature_max=error,
            behavior_max=behavior_error, reused_chunks=len(record['reuse']),
            newly_computed_chunks=len(record['hashes'])-len(record['reuse']))
    arrays = {'wording__' + k: v for k, v in g.items()}
    for name in CONDITIONS:
        if name in ['clean', 'corrupt']:
            continue
        v = data[name]['behavior']
        gain = v[:, 0] - corrupt['behavior'][:, 0]
        arrays[name + '__recovery'] = gain
        result['conditions'][name] = dict(recovery=crossed(gain, rows),
            probability_recovery=crossed(v[:, 1] - corrupt['behavior'][:, 1], rows),
            pairwise=float(v[:, 2].mean()), top1=float(v[:, 3].mean()),
            cells={k: dict(recovery=crossed(gain, rows, m, multiplicity=12 if name == 'wording' else 1),
                           top1=float(v[m, 3].mean()), pairwise=float(v[m, 2].mean())) for k, m in masks.items()})
    result['primary']['recovery'] = {k: v['recovery'] for k, v in result['conditions']['wording']['cells'].items()}
    result['primary']['margin'] = {}
    for bi, ref in enumerate(['zero', 'fit_mean']):
        result['geometry'][ref] = {}
        for key in ['score', 'q_recipient', 'q_donor', 'margin', 'orthogonal', 'pair_norm']:
            z = g[key][bi]
            finite = np.isfinite(z)
            result['geometry'][ref][key] = dict(mean=float(z[finite].mean()) if finite.any() else None,
                finite_n=int(finite.sum()), ci=crossed(z, rows) if finite.all() else None)
        result['primary']['margin'][ref] = {k: crossed(g['margin'][bi], rows, m, multiplicity=12)
            if np.isfinite(g['margin'][bi, m]).all() else None for k, m in masks.items()}
    # Check the new seed through the older direct-indexing implementation as well.
    import common005
    previous_seed = common005.SEED
    common005.SEED = 2026100106
    bootstrap_errors = {}
    try:
        gain = edited['behavior'][:, 0] - corrupt['behavior'][:, 0]
        for cell, mask in masks.items():
            estimates = [('recovery', gain, result['primary']['recovery'][cell])]
            estimates += [(ref, g['margin'][bi], result['primary']['margin'][ref][cell])
                          for bi, ref in enumerate(['zero', 'fit_mean'])]
            for label, values, saved in estimates:
                if saved is None:
                    continue
                direct = common005.intervals(values, rows, mask, multiplicity=12)
                error = float(np.max(abs(np.array(direct['adjusted_ci'])-saved['adjusted_ci'])))
                assert error < 1e-10
                bootstrap_errors[label + '_' + cell] = error
    finally:
        common005.SEED = previous_seed
    result['independent_bootstrap_errors'] = bootstrap_errors
    direct_margin = (np.linalg.norm((e-r)/scale, axis=-1) - np.linalg.norm((e-d)/scale, axis=-1)) / np.linalg.norm((d-r)/scale, axis=-1)
    np.testing.assert_allclose(direct_margin, g['margin'], atol=1e-12)
    old = geometry(edited['raw'], clean['raw'], clean['raw'][:, donors], rows, fit, 'wording')
    unscaled = paired(e, r, d, np.ones_like(scale))
    for label, arr in [('unscaled_paired_margin', unscaled['margin']),
                       ('standardized_cosine_margin', g['cosine_margin']),
                       ('centroid_normalized_margin', old['recipient_distance']-old['donor_distance'])]:
        arrays[label] = arr
        result['sensitivities'][label] = {ref: {k: crossed(arr[bi], rows, m)
            if np.isfinite(arr[bi, m]).all() else None for k, m in masks.items()} for bi, ref in enumerate(['zero', 'fit_mean'])}
    result['movement'], mm = movement(e, corrupt['raw'][..., 0].astype(float), r, d, scale, rows)
    arrays.update({'movement__' + k: v for k, v in mm.items()})
    for kind in ['residual', 'projected']:
        avg = np.mean([data[kind + str(i)]['behavior'][:, 0] for i in range(3)], axis=0)
        result['wording_over_' + kind] = crossed(edited['behavior'][:, 0]-avg, rows)
    result['endpoint_errors'] = {}
    for label, endpoint, expected in [('recipient', r, [0., 0., 1., -1., 0.]), ('donor', d, [1., 1., 0., 1., 0.])]:
        ep = paired(endpoint, r, d, scale)
        errors = {k: float(np.nanmax(abs(ep[k]-v))) for k, v in zip(['score', 'q_recipient', 'q_donor', 'margin', 'orthogonal'], expected)}
        assert max(errors.values()) < 1e-11
        result['endpoint_errors'][label] = errors
    result['unresolvable_pairs'] = int((~g['resolvable']).sum())
    result['qualified'] = bool(sel['eligible'] and all(v['qualified'] for v in qualification.values()) and not result['unresolvable_pairs'])
    result['all_primary_recoveries_positive'] = all(v['adjusted_ci'][0] > 0 for v in result['primary']['recovery'].values())
    result['all_primary_margins_recipient'] = all(v is not None and v['adjusted_ci'][1] < 0 for cells in result['primary']['margin'].values() for v in cells.values())
    result['full_conjunction_supported'] = bool(result['qualified'] and result['all_primary_recoveries_positive'] and result['all_primary_margins_recipient'])
    out.mkdir(parents=True)
    write(out / 'summary.json', result)
    save_arrays(out / 'per-example.npz', **arrays)
    print('F4/F7', {k: result[k] for k in ['qualified', 'all_primary_recoveries_positive', 'all_primary_margins_recipient', 'full_conjunction_supported']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    main(parser.parse_args().output)
