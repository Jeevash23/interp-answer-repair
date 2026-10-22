"""Independent saved-array arithmetic for the explicitly post hoc controls."""
import numpy as np
from common004 import *

def main():
    verify('supplemental-controls-freeze.json')
    report=dict(status='passed',utc=now(),scope='Saved-output verification, not new inference',models={})
    for model in MODELS:
        out=HERE/'results'/model
        sel=read(HERE/'data'/f'{model}-selection.json')
        rows=rows_for(model,'test',sel['families'])
        nat=load_arrays(out/'test/natural.npz')
        expected_norms=delta(nat,rows,'wording').norm(dim=-1).numpy()
        projected=read(out/'projected-controls.json'); arr=load_arrays(out/'projected-controls.npz')
        replay=read(out/'calibration-multicase-replay.json'); ref=load_arrays(out/'calibration-multicase-replay.npz')
        for stem,summary in [('projected-controls',projected),('calibration-multicase-replay',replay)]:
            assert summary['status']=='passed'
            assert digest(out/(stem+'.npz'))==summary['raw_sha256']
        for bundle in [arr,ref]:
            for a in bundle.values():assert np.isfinite(a).all()
        error=float(np.max(np.abs(ref['expected_raw']-ref['replayed_raw'])))
        berr=float(np.max(np.abs(ref['expected_behavior']-ref['replayed_behavior'])))
        assert error==replay['signature_error'] and berr==replay['behavior_error']
        counts=np.count_nonzero(np.abs(ref['expected_raw'])>1e-5,axis=(1,2,3)).tolist()
        assert counts==replay['nonzero_coordinates_by_reference'] and all(n>0 for n in counts)
        assert max(error,berr)<replay['tolerance']
        for key,expected in [('wording',load_arrays(out/'test/wording.npz')['behavior']),('corrupt',nat['corrupt_behavior'])]:
            err=float(np.max(np.abs(arr['replayed_'+key]-expected)))
            assert err==projected['behavior_replay_errors'][key] and err<projected['tolerance']
        for i in range(3):
            behavior=arr[f'draw{i}_behavior']; item=projected['draws'][str(i)]
            assert interval(behavior[:,0]-arr['replayed_corrupt'][:,0],rows)==item['recovery']
            assert float(behavior[:,3].mean())==item['top1']
            assert float(behavior[:,2].mean())==item['pairwise']
            err=float(np.max(np.abs(arr[f'draw{i}_norms']-expected_norms)))
            assert err==item['max_norm_error'] and err<.0001
        average=np.mean([arr[f'draw{i}_behavior'][:,0] for i in range(3)],axis=0)
        assert interval(arr['replayed_wording'][:,0]-average,rows)==projected['wording_over_projected_random']
        # Recompute paired movement without using geometry004.paired.
        raw={k:load_arrays(out/'test'/(k+'.npz'))['raw'][...,0].astype(np.float64) for k in ['clean','corrupt','wording']}
        scale=np.asarray(sel['reference']['scale'])[:,None]
        r=raw['clean']; d=r[:,donor_indices(rows,'wording')]; v=(d-r)/scale
        norm=np.linalg.norm(v,axis=-1); assert (norm>1e-6).all()
        direct={}
        for condition in ['corrupt','wording']:
            u=(raw[condition]-r)/scale
            qr=np.linalg.norm(u,axis=-1)/norm; qd=np.linalg.norm(u-v,axis=-1)/norm
            direct[condition]=dict(q_recipient=qr,q_donor=qd,margin=qr-qd,score=(u*v).sum(-1)/norm**2)
        bg=read(out/'baseline-geometry.json'); saved=load_arrays(out/'baseline-geometry.npz')
        for bi,b in enumerate(['zero','discovery_mean']):
            for k in ['score','margin','q_donor','q_recipient']:
                change=direct['wording'][k][bi]-direct['corrupt'][k][bi]
                np.testing.assert_allclose(saved[b+'_'+k+'_change'],change,rtol=1e-11,atol=1e-11)
                # Floating-point order can differ from the original vectorized formula.
                for label,values in [('corrupt',direct['corrupt'][k][bi]),('edited',direct['wording'][k][bi]),('change',change)]:
                    a=interval(values,rows); z=bg['baselines'][b][k][label]
                    for field in ['mean','ci95','adjusted_ci']:np.testing.assert_allclose(a[field],z[field],rtol=1e-11,atol=1e-11)
        report['models'][model]=dict(endpoint_replay=True,nontrivial_mean_reference=True,raw_hashes=True,projected_norms=True,all_behavior_replay=True,exact_projected_intervals=True,independent_baseline_geometry=True)
    write(HERE/'results/supplemental-verification.json',report)
    print('Supplemental saved-output verification passed',flush=True)

if __name__=='__main__':main()
