"""Pre-scientific numerical checks and matched donor endpoint replay."""
import argparse
import numpy as np
import torch
from common004 import *

def main(model):
    out=HERE/'results'/model;out.mkdir(parents=True,exist_ok=True)
    rows=rows_for(model,'calibration',list(range(8)));rr=[rows[i] for i in next(batches(rows,2))]
    with torch.inference_mode():
        e=engine_for(model)
        initial=out/'technical.json'
        if initial.exists():result=read(initial)
        else:
            result=e.validate(rr);result.update(record_ids=[r['id'] for r in rr],utc=now());write(initial,result)
        assert result.get('passed',True),result
        grouped={}
        for r in rows:grouped.setdefault((r['template'],r['pair'],r['role'],r['order']),[]).append(r)
        eligible=[];ineligible=[]
        for family_a in range(8):
            for family_b in range(family_a+1,8):
                matches=[]
                for group in grouped.values():
                    a=next(r for r in group if r['family']==family_a);b=next(r for r in group if r['family']==family_b)
                    if len(a['token_ids'])==len(b['token_ids']):matches.append((a,b))
                (eligible if matches else ineligible).append([family_a,family_b])
        # Prefer the main F1/F2 pair, else the first eligible technical pair.
        pair=[1,2] if [1,2] in eligible else eligible[0]
        group=next(iter(grouped.values()));a=next(r for r in group if r['family']==pair[0]);b=next(r for r in group if r['family']==pair[1]);assert len(a['token_ids'])==len(b['token_ids'])
        _,cache=e.forward([b]);bank=torch.stack([cache[f'blocks.{l}.attn.hook_z'][:,-1].mean(0).cpu() for l in e.layers])
        target,tb=e.signatures([b],bank);replay,rb=e.signatures([dict(a,token_ids=a['corrupt_tokens'])],bank,boundary=cache[f'blocks.{e.site}.hook_resid_pre'].cpu())
        err=dict(behavior=float(np.max(np.abs(tb-rb))),all_signature_coordinates=float(np.max(np.abs(target-replay))))
        tol=.002 if model.startswith('tiny') else .0005
        write(out/'matched-boundary-replay.json',dict(utc=now(),pair=pair,recipient=a['id'],donor=b['id'],eligible_family_pairs=eligible,ineligible_family_pairs=ineligible,errors=err,tolerance=tol,passed=max(err.values())<tol,scope='Whole boundary donor state, equal length only; not compact-source transfer.'))
        save_arrays(out/'matched-boundary-replay.npz',target_raw=target,replay_raw=replay,target_behavior=tb,replay_behavior=rb)
        assert max(err.values())<tol,err
    print('TECHNICAL',model,result,err,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args();main(a.model)
