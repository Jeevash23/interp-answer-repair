"""Post hoc description of movement from the corrupted starting state."""
import numpy as np
from common004 import *
from geometry004 import paired

def main(model):
    verify('input-freeze.json');verify(model+'-test-freeze.json')
    out=HERE/'results'/model;sel=read(HERE/'data'/f'{model}-selection.json');rows=rows_for(model,'test',sel['families']);p=out/'test';donors=donor_indices(rows,'wording')
    raw={k:load_arrays(p/(k+'.npz'))['raw'][...,0] for k in ['clean','corrupt','wording']};scale=np.asarray(sel['reference']['scale'])[:,None];r=raw['clean'];d=r[:,donors]
    c=paired(raw['corrupt'],r,d,scale);e=paired(raw['wording'],r,d,scale);assert c['resolvable'].all() and e['resolvable'].all()
    report=dict(utc=now(),status='completed',scope='Post hoc descriptive comparison added after Medium and TinyLlama wording outcomes; no primary decision changes',baselines={})
    metrics={}
    for bi,b in enumerate(['zero','discovery_mean']):
        report['baselines'][b]={}
        for k in ['score','margin','q_donor','q_recipient']:
            change=e[k][bi]-c[k][bi];report['baselines'][b][k]=dict(corrupt=interval(c[k][bi],rows),edited=interval(e[k][bi],rows),change=interval(change,rows));metrics[b+'_'+k+'_change']=change
    write(out/'baseline-geometry.json',report);save_arrays(out/'baseline-geometry.npz',**metrics);print(model,{b:{k:x['change']['mean'] for k,x in v.items()} for b,v in report['baselines'].items()})
if __name__=='__main__':
    for m in MODELS:main(m)
