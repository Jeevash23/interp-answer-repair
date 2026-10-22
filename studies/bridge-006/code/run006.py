"""Frozen F4/F7 extension; every new inference lives in Bridge 006."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from cpu_engine import CPUEngine, Engine003, ROOT
from common005 import (rows_for, read, write, now, digest, load_arrays, save_arrays,
                       batches, donor_indices, delta, hooks, behavior_cells,
                       behavior_pass, separation, min_separation, verify)
from run005 import clean_outputs, behavior_only, random_delta

HERE = Path(__file__).resolve().parents[1]
OLD = ROOT / 'studies/bridge-005'
OUT = HERE / 'results'
FAMILIES = [4, 7]
TOL = 0.0005


def verify_inputs():
    for name, h in read(HERE / 'protocol-freeze.json')['files'].items():
        assert digest(ROOT / name) == h, name
    verify('input-freeze.json'); verify('selection-freeze.json'); verify('test-freeze.json')


def measure(e, rows, bank, folder, name, d=None, reuse=None):
    """Resume only hash-checked chunks; store records beside every aggregate."""
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / (name + '-record.json')
    state = read(record) if record.exists() else dict(started=now(), ids=[r['id'] for r in rows], hashes={}, reuse={})
    assert state['ids'] == [r['id'] for r in rows]
    raw = np.empty((2, len(rows), 176, 2), np.float32)
    behavior = np.empty((len(rows), 6), np.float32)
    for part, ids in enumerate(batches(rows, 8)):
        p = folder / f'{name}-{part:03d}.npz'
        if p.name in state['hashes']:
            assert digest(p) == state['hashes'][p.name]
            x = load_arrays(p)
        elif reuse and all(rows[i]['id'] in reuse[0] for i in ids):
            jj = [reuse[0][rows[i]['id']] for i in ids]
            x = dict(raw=reuse[1]['raw'][:, jj], behavior=reuse[1]['behavior'][jj])
            save_arrays(p, **x)
            state['reuse'][p.name] = str(reuse[2].relative_to(ROOT))
            state['hashes'][p.name] = digest(p)
        else:
            assert not p.exists(), p
            rr = [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if d is not None else rows[i] for i in ids]
            hs = hooks(e, d[ids], rr) if d is not None else ()
            a, b = e.signatures(rr, bank, hs)
            x = dict(raw=a, behavior=b)
            save_arrays(p, **x); state['hashes'][p.name] = digest(p)
        raw[:, ids], behavior[ids] = x['raw'], x['behavior']
        write(record, state)
        if part % 8 == 0:
            print(now(), name, part + 1, 'batches;', int(ids.max()) + 1, '/', len(rows), flush=True)
    target = folder / (name + '.npz')
    x = dict(raw=raw, behavior=behavior)
    if target.exists():
        old = load_arrays(target)
        for k in x: np.testing.assert_array_equal(old[k], x[k])
    else: save_arrays(target, **x)
    state.update(finished=now(), status='completed', aggregate_sha256=digest(target))
    write(record, state)
    return x


def validate(e, bank):
    target = OUT / 'numerical-validation.json'
    if target.exists():
        r = read(target); assert r['passed']; return
    rr = rows_for('calibration', FAMILIES)
    chosen = [next(r for r in rr if r['wording'] == w and r['order'] == o and r['role'] == role)
              for w in range(2) for o in range(2) for role in range(2)]
    nat_c = clean_outputs(e, chosen)
    nat_x = clean_outputs(e, [dict(r, token_ids=r['corrupt_tokens']) for r in chosen])
    dd = delta(dict(clean_head_results=nat_c['head_results'], corrupt_head_results=nat_x['head_results']), chosen, 'wording')
    checks = {}
    for name in ['clean', 'corrupt', 'edited']:
        rows = chosen if name == 'clean' else [dict(r, token_ids=r['corrupt_tokens']) for r in chosen]
        hs = hooks(e, dd, rows) if name == 'edited' else ()
        a, b = Engine003.signatures(e, rows, bank, hs)
        c, d = e.signatures(rows, bank, hs)
        checks[name] = dict(signature_max=float(abs(a - c).max()), behavior_max=float(abs(b - d).max()))
        assert max(checks[name].values()) < TOL, checks
        save_arrays(OUT / f'validation-{name}.npz', original_raw=a, fast_raw=c, original_behavior=b, fast_behavior=d)
        print('VALIDATED', name, checks[name], flush=True)
    # Recompute saved F7 signatures to validate reuse across CPU/GPU execution.
    old_rows = rows_for('qualification', [6, 7])
    ids = [next(i for i, r in enumerate(old_rows) if r['family'] == 7 and r['order'] == o and r['role'] == role)
           for o in range(2) for role in range(2)]
    saved = load_arrays(OLD / 'results/qualification/clean.npz')
    a, b = e.signatures([old_rows[i] for i in ids], bank)
    checks['saved_f7'] = dict(signature_max=float(abs(a - saved['raw'][:, ids]).max()),
                              behavior_max=float(abs(b - saved['behavior'][ids]).max()))
    assert max(checks['saved_f7'].values()) < TOL, checks
    # Both directions, all 32 calibration configurations, all-token layer-13 boundary.
    mapping = donor_indices(rr, 'wording')
    boundary_error, behavior_error, nonzero = 0., 0., np.zeros(2, dtype=int)
    for ids in batches(rr, 8):
        donors = [rr[i] for i in ids]
        recipients = [dict(rr[mapping[i]], token_ids=rr[mapping[i]]['corrupt_tokens']) for i in ids]
        assert all(len(a['token_ids']) == len(b['token_ids']) and a['s2'] == b['s2'] for a, b in zip(donors, recipients))
        _, cache = e.forward(donors)
        a, b = e.signatures(donors, bank)
        c, d = e.signatures(recipients, bank, boundary=cache['blocks.13.hook_resid_pre'])
        boundary_error = max(boundary_error, float(abs(a-c).max()))
        behavior_error = max(behavior_error, float(abs(b-d).max()))
        nonzero += np.count_nonzero(abs(a) > 1e-5, axis=(1, 2, 3))
    assert max(boundary_error, behavior_error) < TOL and (nonzero > 0).all()
    checks['boundary'] = dict(n=len(rr), layer=13, positions='all', coordinates=1024,
                             signature_max=boundary_error, behavior_max=behavior_error,
                             nonzero_by_reference=nonzero.tolist())
    # Directly observe additive semantics at every layer, after previous edits.
    corrupted = [dict(r, token_ids=r['corrupt_tokens']) for r in chosen]
    original_names = e.names
    e.names = original_names + [f'blocks.{l}.hook_attn_out' for l in sorted({l for l, _ in e.source})]
    captured = {}; hs = []
    for name, hook_fn in hooks(e, dd, corrupted):
        def instrument(value, hook, *, fn=hook_fn, name=name):
            out = fn(value, hook)
            captured[name] = (value.detach().clone(), out.detach().clone())
            return out
        hs.append((name, instrument))
    _, unedited = e.forward(corrupted)
    _, edited = e.forward(corrupted, hs)
    pos = torch.tensor([r['s2'] for r in corrupted]); ix = torch.arange(len(corrupted))
    layer_checks = {}
    for l in sorted({l for l, _ in e.source}):
        name = f'blocks.{l}.hook_attn_out'; before, after = captured[name]
        expected = dd[:, [i for i, h in enumerate(e.source) if h[0] == l]].sum(1)
        error = float(((after-before)[ix,pos]-expected).abs().max())
        unaffected = after-before; unaffected[ix,pos] = 0
        assert error < TOL and float(unaffected.abs().max()) == 0
        layer_checks[str(l)] = dict(addition_error=error, incoming_change_from_unedited=float((before-unedited[name]).abs().max()))
    assert any(x['incoming_change_from_unedited'] > 1e-5 for x in layer_checks.values())
    e.names = original_names
    checks['sequential_additions'] = layer_checks
    write(target, dict(passed=True, utc=now(), tolerance=TOL, checks=checks))


def run():
    OUT.mkdir(exist_ok=True); verify_inputs()
    files = [HERE/'code/run006.py', HERE/'code/cpu_engine.py', HERE/'protocol-freeze.json',
             ROOT/'src/transport_bridge/engine.py', ROOT/'src/transport_bridge/statistics.py']
    frozen = HERE/'implementation-freeze.json'
    if frozen.exists():
        for name, h in read(frozen)['files'].items(): assert digest(ROOT/name) == h, name
    else: write(frozen, dict(utc=now(), files={str(p.relative_to(ROOT)):digest(p) for p in files}))
    with torch.inference_mode():
        e = CPUEngine(); bank = torch.tensor(load_arrays(OLD/'results/fit/bank.npz')['bank'])
        validate(e, bank)
        fit = read(OLD/'data/selection.json')['scores']['4,7']['reference']
        for split in ['qualification', 'test']:
            rows = rows_for(split, FAMILIES)
            write(HERE/'data'/f'{split}.json', rows)
            folder = OUT/split
            old_rows = rows_for(split, [6, 7]); old_path = OLD/'results'/split/'clean.npz'
            reuse = ({r['id']:i for i,r in enumerate(old_rows) if r['family']==7}, load_arrays(old_path), old_path)
            clean = measure(e, rows, bank, folder, 'clean', reuse=reuse)
            bc = behavior_cells(clean['behavior'], rows); sep = separation(clean['raw'], rows, fit)
            summary = dict(utc=now(), n=len(rows), families=FAMILIES, behavior=bc, separation=sep,
                           worst_separation=min_separation(sep), qualified=bool(behavior_pass(bc) and min_separation(sep)>=.8))
            write(OUT/f'{split}-qualification.json', summary)
            print(split.upper(), 'QUALIFICATION', summary['qualified'], summary['worst_separation'], flush=True)
            if split == 'qualification': continue
            natural = {}
            for name in ['clean', 'corrupt']:
                p=folder/f'natural-{name}.npz'
                if p.exists(): v=load_arrays(p)
                else:
                    rr=rows if name=='clean' else [dict(r,token_ids=r['corrupt_tokens']) for r in rows]
                    v=clean_outputs(e, rr);save_arrays(p, **v)
                natural[name+'_head_results']=v['head_results']
                if name=='clean': np.testing.assert_allclose(v['behavior'], clean['behavior'], atol=TOL,rtol=0)
            dd=delta(natural, rows, 'wording')
            corrupt_rows=[dict(r,token_ids=r['corrupt_tokens']) for r in rows]
            old_corrupt_path=OLD/'results/test/corrupt.npz'
            reuse_corrupt=(reuse[0],load_arrays(old_corrupt_path),old_corrupt_path)
            measure(e,corrupt_rows,bank,folder,'corrupt',reuse=reuse_corrupt)
            measure(e,rows,bank,folder,'wording',d=dd)
            for name in ['zero','self','order','role','order_role']+[k+str(i) for k in ['residual','projected'] for i in range(3)]:
                p=folder/(name+'.npz')
                if p.exists():continue
                d=dd*0 if name=='zero' else (delta(natural, rows, name) if name in ['self','order','role','order_role'] else random_delta(e,dd,rows,name[:-1],int(name[-1])))
                save_arrays(p,**behavior_only(e,rows,d))
                print('CONTROL',name,flush=True)
    write(OUT/'inference-complete.json',dict(utc=now(),status='completed',implementation_freeze=digest(frozen)))


if __name__=='__main__': run()
