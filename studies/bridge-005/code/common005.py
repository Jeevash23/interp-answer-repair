"""Bridge 005 helpers; preceding studies remain immutable."""
from activation_transfer.paths import recorded_path, verify_frozen_file
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
PREV=ROOT/'studies/bridge-004'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'studies/bridge-003/code'))
from common003 import Engine003,batches,delta,donor_indices,hooks,load_arrays,save_arrays,signature
from transport_bridge.statistics import fit_reference,axis_coordinates,geometry
sys.path.insert(0,str(PREV/'code'))
from geometry004 import paired,self_test
SEED=2026090905
MODEL='gpt2-medium'
TOL=.0005
DRAWS=50000
def read(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):
    h=hashlib.sha256()
    with recorded_path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial.json')
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def freeze(name,paths):
    p=HERE/'data'/name;assert not p.exists(),p
    write(p,dict(utc=now(),files={os.path.relpath(x,HERE):digest(x) for x in sorted(set(paths))}))
def verify(name):
    for name,sha in read(HERE/'data'/name)['files'].items():verify_frozen_file(HERE/name,sha)
def rows_for(split,families):
    return [dict(r,wording=families.index(r['family'])) for r in read(HERE/'data/candidates.json') if r['split']==split and r['family'] in families]
def intervals(values,rows,mask=None,multiplicity=1):
    if mask is None:mask=np.ones(len(rows),bool)
    ts=sorted({r['template'] for r,k in zip(rows,mask) if k});ps=sorted({r['pair'] for r,k in zip(rows,mask) if k})
    x=np.asarray(values,np.float64)
    grid=np.array([[x[mask & np.array([r['template']==t and r['pair']==p for r in rows])].mean() for p in ps] for t in ts])
    assert np.isfinite(grid).all()
    rng=np.random.default_rng(SEED);tt=rng.integers(len(ts),size=(DRAWS,len(ts)));pp=rng.integers(len(ps),size=(DRAWS,len(ps)))
    boot=grid[tt[:,:,None],pp[:,None,:]].mean((1,2));tail=.05/(2*multiplicity)
    return dict(mean=float(grid.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),adjusted_ci=np.quantile(boot,[tail,1-tail]).tolist(),multiplicity=multiplicity,n=int(mask.sum()),blocks=len(ts),triples=len(ps),draws=DRAWS)
def behavior_cells(behavior,rows):
    result={}
    for family in sorted({r['family'] for r in rows}):
        result[str(family)]={}
        for order in range(2):
            ix=np.array([r['family']==family and r['order']==order for r in rows]);b=behavior[ix]
            result[str(family)][str(order)]=dict(n=len(b),pairwise=float(b[:,2].mean()),top1=float(b[:,3].mean()),logit=float(b[:,0].mean()))
    return result
def behavior_pass(cells):return all(v['pairwise']>=.95 and v['top1']>=.85 for f in cells.values() for v in f.values())
def separation(raw,rows,fit):
    coords=axis_coordinates(raw,rows,fit,'wording');target=np.array([r['wording'] for r in rows]);result={}
    for w in range(2):
        for o in range(2):
            mask=np.array([r['wording']==w and r['order']==o for r in rows]);result[f'w{w}_o{o}']=((coords[:,mask]>0)==target[mask]).mean(1).tolist()
    return result
def min_separation(cells):return min(x for cell in cells.values() for x in cell)

class Bundle:
    def __init__(self,path,rows,freeze_name):
        self.path=path;path.mkdir(parents=True,exist_ok=True);self.rp=path/'run.json'
        self.record=read(self.rp) if self.rp.exists() else dict(started=now(),status='running',hashes={},record_ids=[r['id'] for r in rows],freeze=digest(HERE/'data'/freeze_name))
        assert self.record['record_ids']==[r['id'] for r in rows]
        assert self.record['freeze']==digest(HERE/'data'/freeze_name)
        for p,h in self.record['hashes'].items():assert digest(path/p)==h,p
        write(self.rp,self.record)
    def get(self,name,fn):
        p=self.path/(name+'.npz')
        if p.name in self.record['hashes']:return load_arrays(p)
        assert not p.exists(),p
        data=fn();save_arrays(p,**data);self.record['hashes'][p.name]=digest(p);write(self.rp,self.record)
        print('SAVED',str(p.relative_to(HERE)),now(),flush=True);return data
    def done(self):self.record.update(status='completed',finished=now());write(self.rp,self.record)
