"""Prepare the approved design and actual random/mapping/bootstrap draws; no model run."""
import hashlib
import platform
import shutil
from datetime import datetime, timezone
import numpy as np
from common010 import *


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (SEED,)+parts)).encode()).digest()[:8], 'little')


def derange(n, rng):
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def save(path, **data):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    assert not FREEZE.exists()
    assert not (HERE/'data/design.json').exists()
    rows = rows_for('test', [10,11]); validate_rows(rows)
    calibration = rows_for('calibration', [10,11])
    calibration = [next(r for r in calibration if (r['wording'],r['order'],r['role'])==(w,o,role))
                   for w in range(2) for o in range(2) for role in range(2)]
    validate_rows(calibration)
    heads = read(ROOT/'configs/study.json')['models']['gpt2-medium']['source_heads']
    write(HERE/'data/rows.json', rows)
    write(HERE/'data/calibration-rows.json', calibration)
    replay = [next(i for i,r in enumerate(rows) if (r['wording'],r['order'],r['role'])==(w,o,role))
              for w in range(2) for o in range(2) for role in range(2)]
    blocks = sorted({r['template'] for r in rows}); triples = sorted({r['pair'] for r in rows})
    lookup = {(r['template'],r['pair'],r['role'],r['wording'],r['order']):i for i,r in enumerate(rows)}
    maps=[]; block_maps=[]; triple_maps=[]
    for k in range(REPLICATES):
        rng = np.random.default_rng(seed('mismatched', k))
        bt, pt = derange(len(blocks),rng), derange(len(triples),rng)
        mapping = np.array([lookup[(blocks[bt[blocks.index(r['template'])]],
                           triples[pt[triples.index(r['pair'])]],r['role'],r['wording'],r['order'])] for r in rows])
        assert len(set(mapping)) == len(rows)
        assert all(rows[j]['template'] != r['template'] and rows[j]['pair'] != r['pair']
                   for r,j in zip(rows,mapping))
        maps.append(mapping); block_maps.append(bt); triple_maps.append(pt)
        g = np.empty((len(rows),len(heads),64), np.float32)
        for i,r in enumerate(rows):
            for hi,h in enumerate(heads):
                g[i,hi] = np.random.default_rng(seed('random',k,r['id'],*h)).standard_normal(64)
        save(HERE/f'data/draws/random-r{k}.npz', z=g)
    save(HERE/'data/draws/mappings.npz', indices=np.stack(maps),
         block_maps=np.stack(block_maps), triple_maps=np.stack(triple_maps))
    boot = Crossed(rows, draws=DRAWS, seed=ANALYSIS_SEED)
    rng = np.random.default_rng(ANALYSIS_SEED+1)
    rr = rng.integers(REPLICATES,size=(len(CONTROLS),DRAWS,REPLICATES))
    save(HERE/'data/draws/bootstrap.npz', blocks=boot.tt.astype(np.int16),
         triples=boot.pp.astype(np.int16), replicates=rr.astype(np.int8))
    timing=[]
    for n in ('broad_self','broad_wording','matched_self','matched_wording'):
        p=ROOT/f'studies/bridge-008/results/test/{n}-record.json'; r=read(p)
        seconds=(datetime.fromisoformat(r['finished'])-datetime.fromisoformat(r['started'])).total_seconds()
        timing.append(dict(path=str(p.relative_to(ROOT)),seconds=seconds))
    minutes=float(np.mean([x['seconds'] for x in timing])/60)
    write(HERE/'data/design.json',dict(utc=datetime.now(timezone.utc).isoformat(),
          study=5, scope='Exploratory; previously inspected Study 2 final bank; not pooled',
          n=len(rows),blocks=20,triples=20,controls=list(CONTROLS),replicates=REPLICATES,
          source_heads=heads,random_seed=SEED,bootstrap_seed=ANALYSIS_SEED,
          replicate_bootstrap_seed=ANALYSIS_SEED+1,bootstrap_draws=DRAWS,
          multiplicity=MULTIPLICITY,geometric_contrasts=12,behaviour_contrasts=12,
          calibration_ids=[r['id'] for r in calibration],singleton_indices=replay,
          singleton_ids=[rows[i]['id'] for i in replay],
          runtime=dict(basis=timing,minutes_per_condition=minutes,
             estimated_inference_hours_4_replicates=minutes*12/60,
             estimated_inference_hours_8_replicates=minutes*24/60,
             additional_verification_hours=[.5,1.5],
             decision='Four replicates and all three controls; approximately four hours of full-bank inference plus verification. Eight would approximately double that CPU budget.',
             cpu=platform.processor(),threads=4,available_cpu_count=__import__('os').cpu_count(),
             disk_free_bytes=shutil.disk_usage(ROOT).free)))
    write(HERE/'data/pre-freeze-synthetic-checks.json',synthetic_test())
    print('Prepared actual draws, mappings, bootstrap draws, calibration and replay manifests.')


if __name__=='__main__': main()
