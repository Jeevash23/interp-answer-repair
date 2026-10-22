"""Supplemental nontrivial-mean endpoint replay on calibration data only."""
import argparse
import numpy as np
import torch
from common004 import *

def main(model):
    verify('input-freeze.json');out=HERE/'results'/model;target=out/'calibration-multicase-replay.json';assert not target.exists()
    rows=rows_for(model,'calibration',list(range(8)))
    with torch.inference_mode():
        e=engine_for(model);zs=[]
        for ids in batches(rows):
            rr=[rows[i] for i in ids];_,cache=e.forward(rr);zs.append(e.extract(rr,cache)[1].numpy())
        bank=torch.tensor(np.concatenate(zs).mean(0))
        drows=rows_for(model,'calibration',[1,2]);indices=[i for i,r in enumerate(drows) if r['family']==2];donors=[drows[i] for i in indices]
        mapping=donor_indices(drows,'wording');recipients=[dict(drows[mapping[i]],token_ids=drows[mapping[i]]['corrupt_tokens']) for i in indices]
        expected=[];replayed=[];expected_b=[];replayed_b=[];order=[]
        for ids in batches(donors):
            dr=[donors[i] for i in ids];rr=[recipients[i] for i in ids]
            assert all(len(a['token_ids'])==len(b['token_ids']) for a,b in zip(dr,rr))
            _,cache=e.forward(dr);a,ab=e.signatures(dr,bank);b,bb=e.signatures(rr,bank,boundary=cache[f'blocks.{e.site}.hook_resid_pre'].cpu())
            expected.append(a);replayed.append(b);expected_b.append(ab);replayed_b.append(bb);order.extend(ids)
        ix=np.argsort(order);a=np.concatenate(expected,axis=1)[:,ix];b=np.concatenate(replayed,axis=1)[:,ix];ab=np.concatenate(expected_b)[ix];bb=np.concatenate(replayed_b)[ix]
    tol=.002 if model.startswith('tiny') else .0005;error=float(np.max(np.abs(a-b)));berr=float(np.max(np.abs(ab-bb)));nonzero=np.count_nonzero(np.abs(a)>1e-5,axis=(1,2,3)).tolist()
    assert all(v>0 for v in nonzero);assert max(error,berr)<tol
    save_arrays(out/'calibration-multicase-replay.npz',expected_raw=a,replayed_raw=b,expected_behavior=ab,replayed_behavior=bb,bank=bank.numpy())
    write(target,dict(utc=now(),status='passed',n=len(donors),donor_ids=[r['id'] for r in donors],recipient_ids=[r['id'] for r in recipients],mean_bank_configurations=len(rows),nonzero_coordinates_by_reference=nonzero,signature_error=error,behavior_error=berr,tolerance=tol,scope='Supplemental technical check added after initial Medium scientific inference; calibration bank only; no selections or primary estimands changed',raw_sha256=digest(out/'calibration-multicase-replay.npz')))
    print('MULTICASE REPLAY',model,error,berr,nonzero,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args();main(a.model)
