"""Fixed-pair GPT-2 Small replication; never modifies earlier studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'studies/bridge-006/code'))
from small_engine import FullSequenceEngine
from common005 import (batches, behavior_cells, behavior_pass, delta, donor_indices,
                       hooks, separation, min_separation, fit_reference)
from runner009 import make_engine, measure_natural, measure_signatures, measure_behavior

SEED = 2026100409
TOL = 0.0005
FAMILIES = [10, 11]
BANK_FAMILIES = [10, 11, 12, 13]
PAIRS = [[10, 11]]
OUT = HERE / 'results'
INPUT = HERE / 'data/input-freeze.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = load(path)
        assert set(old) == set(values), path
        for key in old:
            np.testing.assert_array_equal(old[key], values[key])
        return
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(tmp, path)


def freeze(path, files):
    path = Path(path)
    if path.exists():
        verify(path)
        return
    files = sorted({Path(p).resolve() for p in files if Path(p).is_file()})
    write(path, {'utc': now(), 'kind': 'local prospective freeze; not public preregistration',
                 'files': {str(p.relative_to(ROOT)): digest(p) for p in files}})
    verify(path)


def verify(path):
    for name, expected in read(path)['files'].items():
        assert digest(ROOT / name) == expected, name


def rows_for(split, families):
    return [dict(r, wording=families.index(r['family']))
            for r in read(HERE / 'data/candidates.json')
            if r['split'] == split and r['family'] in families]


def input_freeze():
    assert (HERE / 'data/preparation-audit.json').exists()
    assert (HERE / 'docs/DESIGN-REVIEW.md').exists()
    audit = read(HERE / 'data/preparation-audit.json')
    assert audit['token_ids_match_small'] and audit['fixed_pair'] == FAMILIES
    for path, expected in {**audit['inherited_source_hashes'],
                           **audit['checkpoint_and_tokenizer_hashes']}.items():
        assert digest(ROOT / path) == expected, path
    if not INPUT.exists():
        assert not any((OUT / x).exists() for x in ['screen.json', 'fit', 'selection', 'qualification', 'test'])
    # Reporting and independent-verifier helpers may be added later. They do
    # not define scientific measurements, selection or analysis estimands.
    scientific_code = ['prepare009.py', 'small_engine.py', 'runner009.py',
                       'pipeline009.py', 'analyze009.py', 'secondary009.py']
    files = [HERE / 'code' / name for name in scientific_code]
    assert all(path.is_file() for path in files)
    files += [HERE / 'docs/PROTOCOL.md', HERE / 'docs/DESIGN-REVIEW.md', HERE / 'docs/DATA-REVIEW.md']
    files += [p for p in (HERE / 'data').glob('*.json') if 'freeze' not in p.name]
    files += [ROOT / 'configs/study.json', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    for folder in [ROOT / 'src', ROOT / 'studies/bridge-002/code', ROOT / 'studies/bridge-003/code',
                   ROOT / 'studies/bridge-004/code', ROOT / 'studies/bridge-005/code',
                   ROOT / 'studies/bridge-006/code', ROOT / 'studies/transport/src']:
        files.extend(folder.rglob('*.py'))
    files += [ROOT / 'studies/transport/results/experiment-001/exploratory/run-config.json',
              ROOT / 'studies/transport/data/model-manifest.json',
              HERE / 'provenance/code-origins.json']
    files += [ROOT / p for p in read(HERE / 'data/preparation-audit.json')['inherited_source_hashes']]
    files += [p for p in (ROOT / 'models/gpt2-small').rglob('*') if p.is_file()]
    freeze(INPUT, files)
    print('INPUT FROZEN', now(), digest(INPUT), flush=True)


def calibration(e):
    verify(INPUT)
    target = OUT / 'calibration.json'
    if target.exists():
        assert read(target)['passed']
        return
    rows = rows_for('calibration', FAMILIES)
    nat = measure_natural(e, rows, OUT / 'calibration/natural.npz', INPUT)
    bank = torch.tensor(nat['z'].mean(0))
    checks = {}
    # Both fixed templates, orders and roles on this new checkpoint path.
    chosen = [next(r for r in rows if r['family'] == f and r['order'] == o and r['role'] == role)
              for f in FAMILIES for o in [0, 1] for role in [0, 1]]
    for condition in ['clean', 'corrupt']:
        rr = chosen if condition == 'clean' else [dict(r, token_ids=r['corrupt_tokens']) for r in chosen]
        a, ab = FullSequenceEngine.signatures(e, rr, bank)
        b, bb = e.signatures(rr, bank)
        checks[condition] = {'signature': float(np.max(abs(a - b))), 'behavior': float(np.max(abs(ab - bb)))}
        assert max(checks[condition].values()) < TOL
        save(OUT / f'calibration/{condition}-validation.npz', original=a, prefix=b,
             original_behavior=ab, prefix_behavior=bb)
    # One fixed candidate pair, both directions and both mention orders, for edit semantics.
    rr = rows_for('calibration', PAIRS[0])
    rr = [next(r for r in rr if r['wording'] == w and r['order'] == o and r['role'] == 0)
          for w in [0, 1] for o in [0, 1]]
    a = measure_natural(e, rr, OUT / 'calibration/edit-clean.npz', INPUT)
    b = measure_natural(e, [dict(r, token_ids=r['corrupt_tokens']) for r in rr],
                        OUT / 'calibration/edit-corrupt.npz', INPUT)
    dd = delta({'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}, rr, 'wording')
    cr = [dict(r, token_ids=r['corrupt_tokens']) for r in rr]
    hs = hooks(e, dd, cr)
    x, xb = FullSequenceEngine.signatures(e, cr, bank, hs)
    y, yb = e.signatures(cr, bank, hs)
    checks['wording'] = {'signature': float(np.max(abs(x-y))), 'behavior': float(np.max(abs(xb-yb)))}
    assert max(checks['wording'].values()) < TOL
    save(OUT / 'calibration/edited-validation.npz', original=x, prefix=y)
    write(target, {'utc': now(), 'passed': True, 'tolerance': TOL, 'checks': checks,
                   'purpose': 'technical only; no candidate decision uses calibration outcomes'})
    print('CALIBRATION', checks, flush=True)


def screen(e):
    verify(INPUT)
    assert read(OUT / 'calibration.json')['passed']
    target = OUT / 'screen.json'
    if target.exists():
        freeze(HERE / 'data/screen-freeze.json', [INPUT, target, OUT / 'screen.npz', OUT / 'calibration.json'])
        return read(target)
    rows = rows_for('screen', FAMILIES)
    result = measure_behavior(e, rows, OUT / 'screen.npz', INPUT)
    cells = behavior_cells(result, rows)
    eligible = [int(f) for f, c in cells.items() if behavior_pass({f: c})]
    alignment = read(HERE / 'data/design.json')['alignment_eligible_pairs']
    pairs = [p for p in PAIRS if p in alignment and all(f in eligible for f in p)]
    value = {'utc': now(), 'eligible': eligible, 'pairs': pairs, 'cells': cells}
    write(target, value)
    freeze(HERE / 'data/screen-freeze.json', [INPUT, target, OUT / 'screen.npz', OUT / 'calibration.json'])
    print('SCREEN', value, flush=True)
    return value


def development(e, split):
    frozen = HERE / 'data/screen-freeze.json'
    verify(INPUT); verify(frozen)
    assert read(OUT / 'screen.json')['pairs'] == PAIRS
    rows = rows_for(split, FAMILIES)
    folder = OUT / split
    if split == 'fit':
        bank_rows = rows_for('fit', BANK_FAMILIES)
        assert len(bank_rows) == 1024 and len(rows) == 512
        nat = measure_natural(e, bank_rows, folder / 'natural.npz', frozen)
        bank = nat['z'].mean(0)
        save(folder / 'bank.npz', bank=bank)
    else:
        bank = load(OUT / 'fit/bank.npz')['bank']
    measure_signatures(e, rows, torch.tensor(bank), folder / 'clean.npz', frozen)


def select():
    verify(INPUT)
    target = HERE / 'data/selection.json'
    if target.exists():
        files = [INPUT, HERE / 'data/screen-freeze.json', target]
        for split in ['fit', 'selection']:
            files += list((OUT / split).glob('*.json')) + list((OUT / split).glob('*.npz'))
        freeze(HERE / 'data/selection-freeze.json', files)
        return read(target)
    scr = read(OUT / 'screen.json')
    assert scr['pairs'] == PAIRS
    train = rows_for('fit', FAMILIES); valid = rows_for('selection', FAMILIES)
    a = load(OUT / 'fit/clean.npz'); b = load(OUT / 'selection/clean.npz')
    scores = {}
    for pair in scr['pairs']:
        ii = np.array([i for i, r in enumerate(train) if r['family'] in pair])
        jj = np.array([i for i, r in enumerate(valid) if r['family'] in pair])
        rr = [dict(train[i], wording=pair.index(train[i]['family'])) for i in ii]
        vv = [dict(valid[i], wording=pair.index(valid[i]['family'])) for i in jj]
        ref = fit_reference(a['raw'][:, ii], rr)
        ts = separation(a['raw'][:, ii], rr, ref); vs = separation(b['raw'][:, jj], vv, ref)
        tc = behavior_cells(a['behavior'][ii], rr); vc = behavior_cells(b['behavior'][jj], vv)
        worst = min_separation(vs); average = float(np.mean(list(vs.values())))
        scores[','.join(map(str, pair))] = dict(families=pair, reference=ref, fit_behavior=tc,
            selection_behavior=vc, fit_separation=ts, selection_separation=vs, worst=worst,
            average=average, eligible=bool(behavior_pass(tc) and behavior_pass(vc)
                                         and min_separation(ts) >= .8 and worst >= .9))
    good = sorted([v for v in scores.values() if v['eligible']],
                  key=lambda v: (-v['worst'], -v['average'], v['families']))
    selected = good[0] if good else None
    value = dict(utc=now(), status='selected' if selected else 'no_qualified_pair', scores=scores,
                 families=selected['families'] if selected else None,
                 reference=selected['reference'] if selected else None,
                 fixed_pair_from='bridge-007', no_model_specific_pair_search=True,
                 bank_families=BANK_FAMILIES,
                 source_heads=read(ROOT / 'configs/study.json')['models']['gpt2-small']['source_heads'])
    write(target, value)
    files = [INPUT, HERE / 'data/screen-freeze.json', target]
    for split in ['fit', 'selection']:
        files += list((OUT / split).glob('*.json')) + list((OUT / split).glob('*.npz'))
    freeze(HERE / 'data/selection-freeze.json', files)
    print('SELECTION', {k: {x:v[x] for x in ['eligible','worst','average']} for k,v in scores.items()},
          'CHOSEN', value['families'], flush=True)
    return value


def selected_replay(e):
    verify(HERE / 'data/selection-freeze.json')
    target = OUT / 'selected-pair-replay.json'
    if target.exists():
        assert read(target)['passed']
        return
    sel = read(HERE / 'data/selection.json')
    rows = rows_for('calibration', sel['families']); mapping = donor_indices(rows, 'wording')
    bank = torch.tensor(load(OUT / 'fit/bank.npz')['bank'])
    errors = dict(signature=0., behavior=0.)
    nonzero = np.zeros(2, dtype=int)
    for ii in batches(rows, 8):
        dr = [rows[i] for i in ii]
        cr = [dict(rows[mapping[i]], token_ids=rows[mapping[i]]['corrupt_tokens']) for i in ii]
        assert all(len(x['token_ids']) == len(y['token_ids']) and x['s2'] == y['s2'] for x,y in zip(dr,cr))
        _, cache = e.forward(dr)
        a, ab = e.signatures(dr, bank)
        b, bb = e.signatures(cr, bank, boundary=cache[f'blocks.{e.site}.hook_resid_pre'])
        errors['signature'] = max(errors['signature'], float(np.max(abs(a-b))))
        errors['behavior'] = max(errors['behavior'], float(np.max(abs(ab-bb))))
        nonzero += np.count_nonzero(abs(a) > 1e-5, axis=(1,2,3))
    assert max(errors.values()) < TOL and np.all(nonzero > 0)
    write(target, dict(utc=now(), passed=True, n=len(rows), families=sel['families'],
                       directions='both', errors=errors, nonzero=nonzero.tolist(), tolerance=TOL))


def qualification(e):
    from analyze009 import qualification_summary
    frozen = HERE / 'data/selection-freeze.json'
    verify(INPUT); verify(frozen)
    sel = read(HERE / 'data/selection.json'); rows = rows_for('qualification', sel['families'])
    target = OUT / 'qualification-summary.json'
    if target.exists():
        return read(target)
    bank = torch.tensor(load(OUT / 'fit/bank.npz')['bank'])
    clean = measure_signatures(e, rows, bank, OUT / 'qualification/clean.npz', frozen)
    value = qualification_summary(clean, rows, sel['reference'])
    value.update(utc=now(), families=sel['families'], n=len(rows))
    write(target, value)
    print('QUALIFICATION', value['qualified'], flush=True)
    return value


def test_freeze():
    from secondary009 import verify_freeze as verify_secondary
    verify_secondary()
    assert read(OUT / 'qualification-summary.json')['qualified']
    assert read(OUT / 'selected-pair-replay.json')['passed']
    files = [INPUT, HERE / 'data/secondary-analysis-freeze.json',
             HERE / 'data/selection-freeze.json', HERE / 'data/selection.json',
             OUT / 'qualification-summary.json', OUT / 'selected-pair-replay.json']
    files += list((OUT / 'qualification').glob('*'))
    freeze(HERE / 'data/test-freeze.json', files)


def random_delta(e, dd, rows, kind, draw):
    result = []
    for hi, (layer, head) in enumerate(e.source):
        weight = e.weights[layer, head].cpu()
        vectors = []
        for r in rows:
            label = f'{SEED}:{kind}:{draw}:{r["id"]}:{layer}:{head}'
            seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little') % (2**63-1)
            vectors.append(torch.randn(weight.shape[0] if kind == 'projected' else weight.shape[1],
                                       generator=torch.Generator().manual_seed(seed)))
        v = torch.stack(vectors)
        if kind == 'projected':
            v = v @ weight
        v *= dd[:,hi].norm(dim=-1,keepdim=True) / v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        result.append(v)
    return torch.stack(result, 1)


def test(e):
    frozen = HERE / 'data/test-freeze.json'
    verify(INPUT); verify(frozen)
    sel = read(HERE / 'data/selection.json'); rows = rows_for('test', sel['families'])
    folder = OUT / 'test'; bank = torch.tensor(load(OUT / 'fit/bank.npz')['bank'])
    a = measure_natural(e, rows, folder / 'natural-clean.npz', frozen)
    cr = [dict(r, token_ids=r['corrupt_tokens']) for r in rows]
    b = measure_natural(e, cr, folder / 'natural-corrupt.npz', frozen)
    nat = {'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}
    measure_signatures(e, rows, bank, folder / 'clean.npz', frozen)
    measure_signatures(e, cr, bank, folder / 'corrupt.npz', frozen)
    for condition in ['wording', 'self']:
        dd = delta(nat, rows, condition)
        measure_signatures(e, rows, bank, folder / f'{condition}.npz', frozen, d=dd)
    for condition in ['zero', 'order', 'role', 'order_role'] + [k+str(i) for k in ['residual','projected'] for i in range(3)]:
        dd = delta(nat, rows, 'wording' if condition == 'zero' or condition.startswith(('residual','projected')) else condition)
        if condition == 'zero':
            dd *= 0
        elif condition.startswith(('residual','projected')):
            norm = dd.norm(dim=-1)
            dd = random_delta(e, dd, rows, condition[:-1], int(condition[-1]))
            assert float((dd.norm(dim=-1)-norm).abs().max()) < .0001
        measure_behavior(e, rows, folder / f'{condition}.npz', frozen, d=dd)
    write(OUT / 'inference-complete.json', dict(utc=now(), n=len(rows), status='completed',
                                               test_freeze=digest(frozen)))


def computational_replay(e):
    target = OUT / 'computational-replay.json'
    if target.exists():
        assert read(target)['passed']
        return
    frozen = HERE / 'data/test-freeze.json'; verify(frozen)
    rows = rows_for('test', read(HERE / 'data/selection.json')['families'])
    folder = OUT / 'test'
    a = load(folder / 'natural-clean.npz'); b = load(folder / 'natural-corrupt.npz')
    dd = delta({'clean_head_results':a['head_results'], 'corrupt_head_results':b['head_results']}, rows, 'wording')
    expected = load(folder / 'wording.npz')
    repeated = measure_behavior(e, rows, OUT / 'verification/wording.npz', frozen, d=dd)
    baseline = measure_behavior(e, rows, OUT / 'verification/corrupt.npz', frozen, d=dd*0)
    errors = {'wording_behavior': float(np.max(abs(repeated-expected['behavior']))),
              'corrupt_behavior': float(np.max(abs(baseline-b['behavior'])))}
    ii = np.array([i for w in [0,1] for o in [0,1]
                   for i in [j for j,r in enumerate(rows) if r['wording']==w and r['order']==o][:2]])
    rr = [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) for i in ii]
    old = e.config['head_chunk']; e.config['head_chunk'] = 1
    raw, behavior = e.signatures(rr, torch.tensor(load(OUT / 'fit/bank.npz')['bank']), hooks(e,dd[ii],rr))
    e.config['head_chunk'] = old
    errors['singleton_signature'] = float(np.max(abs(raw-expected['raw'][:,ii])))
    assert max(errors.values()) < TOL, errors
    save(OUT / 'verification/singleton.npz', indices=ii, raw=raw, behavior=behavior)
    write(target, dict(utc=now(), passed=True, errors=errors, singleton_n=len(ii), tolerance=TOL))


def pipeline():
    input_freeze()
    from secondary009 import freeze as freeze_secondary
    freeze_secondary()
    with torch.inference_mode():
        e = make_engine(threads=4)
        calibration(e)
        if not screen(e)['pairs']:
            write(OUT / 'status.json', dict(utc=now(), status='completed', outcome='no_screen_pair', test_measured=False))
            return
        development(e, 'fit'); development(e, 'selection')
        if select()['status'] != 'selected':
            write(OUT / 'status.json', dict(utc=now(), status='completed', outcome='no_selection_pair', test_measured=False))
            return
        selected_replay(e)
        if not qualification(e)['qualified']:
            write(OUT / 'status.json', dict(utc=now(), status='completed', outcome='independent_qualification_failed', test_measured=False))
            return
        test_freeze(); test(e)
        # Fresh model instance for a computational replication, not a new statistical sample.
        del e
        e = make_engine(threads=4)
        computational_replay(e)
    write(OUT / 'status.json', dict(utc=now(), status='inference_completed', outcome='final_test_completed', test_measured=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'pipeline', 'select', 'test_freeze', 'calibration',
                                         'screen', 'fit', 'selection', 'selected_replay', 'qualification',
                                         'test', 'computational_replay'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        input_freeze()
    elif args.stage in ['pipeline','select','test_freeze']:
        globals()[args.stage]()
    else:
        verify(INPUT)
        with torch.inference_mode():
            engine = make_engine(threads=4)
            if args.stage in ['fit','selection']:
                development(engine,args.stage)
            else:
                globals()[args.stage](engine)
