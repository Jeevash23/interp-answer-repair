"""Versioned helpers for the new study, without mutating old scientific files."""
from activation_transfer.paths import recorded_path, verify_frozen_file
import hashlib
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
PREV=ROOT/'studies/bridge-003'
sys.path.insert(0,str(PREV/'code'))
from common003 import Engine003,batches,delta,donor_indices,hooks,natural,signature,save_arrays,load_arrays
from transport_bridge.engine import random_matched,outcomes
from transport_bridge.statistics import fit_reference,axis_coordinates
SEED=2026090804
MODELS=['gpt2-medium','tinyllama-1.1b-base']
def read(p): return json.loads(p.read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def digest(p):
    h=hashlib.sha256()
    with recorded_path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.partial.json');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def freeze(name,paths):
    p=HERE/'data'/name
    assert not p.exists(),p
    write(p,{'utc':now(),'files':{os.path.relpath(x,HERE):digest(x) for x in sorted(set(paths))}})
def verify(name):
    for p,h in read(HERE/'data'/name)['files'].items(): verify_frozen_file(HERE/p,h)
def rows_for(model,split,families):
    return [dict(r,wording=families.index(r['family'])) for r in read(HERE/'data'/f'{model}-candidates.json') if r['split']==split and r['family'] in families]
def engine_for(model,source=None):
    if model=='gpt2-medium':
        e=Engine003(model)
    else:
        from llama004 import LlamaEngine
        e=LlamaEngine(source)
    return e
def interval(values,rows,mask=None,multiplicity=1):
    if mask is None:mask=np.ones(len(rows),bool)
    ts=sorted({r['template'] for r,k in zip(rows,mask) if k});ps=sorted({r['pair'] for r,k in zip(rows,mask) if k})
    x=np.asarray(values,dtype=np.float64)
    grid=np.array([[x[mask & np.array([r['template']==t and r['pair']==p for r in rows])].mean() for p in ps] for t in ts])
    assert np.isfinite(grid).all()
    rng=np.random.default_rng(SEED);tt=rng.integers(len(ts),size=(20000,len(ts)));pp=rng.integers(len(ps),size=(20000,len(ps)))
    boot=grid[tt[:,:,None],pp[:,None,:]].mean((1,2));tail=.05/(2*multiplicity)
    return dict(mean=float(grid.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),adjusted_ci=np.quantile(boot,[tail,1-tail]).tolist(),multiplicity=multiplicity,n=int(mask.sum()),blocks=len(ts),triples=len(ps),draws=20000)
