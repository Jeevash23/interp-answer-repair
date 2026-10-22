"""Pre-inference design, metric and estimand checks."""
from common004 import *
from geometry004 import self_test
import numpy as np

def main():
    report=self_test();counts={}
    for m in MODELS:
        counts[m]={}
        allrows=read(HERE/'data'/f'{m}-candidates.json');assert len({r['id'] for r in allrows})==len(allrows)
        for split,n in [('calibration',16),('screen',64),('discovery',128),('test',384)]:
            rows=rows_for(m,split,[1,2]);assert len(rows)==n,(split,len(rows),n)
            counts[m][split]=len(rows)
            for r in rows:
                assert len(r['token_ids'])==len(r['corrupt_tokens'])
                assert [i for i,(a,b) in enumerate(zip(r['token_ids'],r['corrupt_tokens'])) if a!=b]==[r['s2']]
                assert r['token_ids'][r['s2']]==r['subject_id'];assert r['corrupt_tokens'][r['s2']]==r['alt_id']
                assert r['token_ids'][r['io_position']]==r['io_id'];assert r['token_ids'][r['subject_position']]==r['subject_id']
            for f in ['wording','order','role','order_role','self']:
                ids=donor_indices(rows,f);assert len(set(ids))==len(rows);assert np.array_equal(ids[ids],np.arange(len(rows)))
        rows=rows_for(m,'test',[1,2]);v=np.array([r['template']+.2*r['pair'] for r in rows]);ci=interval(v,rows,multiplicity=24)
        assert ci==interval(v,rows,multiplicity=24)
        assert ci['adjusted_ci'][0]<=ci['ci95'][0]<=ci['mean']<=ci['ci95'][1]<=ci['adjusted_ci'][1]
    # Four regimes: attenuation equals noising, and interaction is noising minus denoising.
    c,e,egc,cge=1.,4.,2.,1.5
    attenuation=(e-c)-(egc-c);nie=e-egc;pie=cge-c;interaction=e-cge-egc+c
    assert attenuation==nie and interaction==nie-pie
    report.update(counts=counts,donor_involutions=True,semantic_corruption_positions=True,bootstrap_replay=True,mediation_identity=True,utc=now())
    write(HERE/'results/design-tests.json',report);print('DESIGN TESTS PASS',counts)
if __name__=='__main__':main()
