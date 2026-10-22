"""Post hoc projected-Gaussian controls and fresh behavior replay."""
import argparse,hashlib
import numpy as np
import torch
from common004 import *
from run004 import behavior_only

def main(model):
    verify('input-freeze.json');verify(model+'-test-freeze.json');verify('supplemental-controls-freeze.json')
    out=HERE/'results'/model;target=out/'projected-controls.json';assert not target.exists()
    sel=read(HERE/'data'/f'{model}-selection.json');rows=rows_for(model,'test',sel['families']);nat=load_arrays(out/'test/natural.npz');donor_delta=delta(nat,rows,'wording');saved=load_arrays(out/'test/wording.npz')['behavior'];results={};summary={'utc':now(),'status':'running','scope':'Post hoc secondary behavior controls, not the primary family','draws':{},'n':len(rows)}
    tol=.002 if model.startswith('tiny') else .0005
    with torch.inference_mode():
        e=engine_for(model,sel['source_heads']);wording=behavior_only(e,rows,donor_delta)['behavior'];base=behavior_only(e,rows,donor_delta*0)['behavior']
        replay_errors={'wording':float(np.max(np.abs(wording-saved))),'corrupt':float(np.max(np.abs(base-nat['corrupt_behavior'])))}
        assert max(replay_errors.values())<tol,replay_errors
        weights=[e.weights[l,h].cpu() for l,h in e.source]
        for draw in range(3):
            perhead=[]
            for hi,((l,h),w) in enumerate(zip(e.source,weights)):
                coords=[]
                for row in rows:
                    label=f'{SEED}:{model}:{row["id"]}:projected:{draw}:{l}:{h}'
                    seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')%(2**63-1)
                    coords.append(torch.randn(w.shape[0],generator=torch.Generator().manual_seed(seed)))
                v=torch.stack(coords)@w;v*=donor_delta[:,hi].norm(dim=-1,keepdim=True)/v.norm(dim=-1,keepdim=True).clamp_min(1e-12);perhead.append(v)
            d=torch.stack(perhead,1);norm_error=float((d.norm(dim=-1)-donor_delta.norm(dim=-1)).abs().max());assert norm_error<.0001,norm_error
            behavior=behavior_only(e,rows,d)['behavior'];results[f'draw{draw}_behavior']=behavior;results[f'draw{draw}_norms']=d.norm(dim=-1).numpy()
            summary['draws'][str(draw)]=dict(recovery=interval(behavior[:,0]-base[:,0],rows),top1=float(behavior[:,3].mean()),pairwise=float(behavior[:,2].mean()),max_norm_error=norm_error)
    average=np.mean([results[f'draw{i}_behavior'][:,0] for i in range(3)],axis=0)
    summary.update(status='passed',behavior_replay_errors=replay_errors,wording_over_projected_random=interval(wording[:,0]-average,rows),tolerance=tol)
    results.update(replayed_wording=wording,replayed_corrupt=base);save_arrays(out/'projected-controls.npz',**results);summary['raw_sha256']=digest(out/'projected-controls.npz');write(target,summary)
    print('PROJECTED CONTROLS',model,summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args();main(a.model)
