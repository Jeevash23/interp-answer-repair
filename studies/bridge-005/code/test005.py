"""Structural and arithmetic regressions before scientific inference."""
from common005 import *

def main():
    design=read(HERE/'data/design.json');rows=read(HERE/'data/candidates.json');banks=design['names'];names=sum(banks.values(),[])
    assert len(names)==108 and len(set(names))==108 and not set(names)&set(design['excluded_names'])
    blocks=[tuple(x) for v in design['blocks'].values() for x in v];assert len(blocks)==len(set(blocks))
    texts=[r['text'] for r in rows];assert len(texts)==len(set(texts))
    old_texts=set()
    def collect(x):
        if isinstance(x,list):
            for v in x:collect(v)
        elif isinstance(x,dict):
            for k,v in x.items():
                if isinstance(v,str) and k in ['text','clean_text','corrupt_text','alternative_corrupt_text']:old_texts.add(v)
                elif isinstance(v,(list,dict)):collect(v)
    for source,sha in design['prior_name_sources'].items():
        p=HERE/source;assert digest(p)==sha
        if p.suffix=='.jsonl':
            for line in p.read_text().splitlines():
                if line.strip():collect(json.loads(line))
        else:collect(read(p))
    for p in [ROOT/'data/recipients.json',ROOT/'studies/bridge-002/data/candidates.json',ROOT/'studies/bridge-002/data/candidates-v2.json',ROOT/'studies/bridge-003/data/gpt2-medium-candidates.json',PREV/'data/gpt2-medium-candidates.json']:collect(read(p))
    assert not set(texts)&old_texts
    for r in rows:
        assert r['token_ids'][r['s2']]==r['subject_id']
        assert r['token_ids'][r['io_position']]==r['io_id']
        assert r['token_ids'][r['subject_position']]==r['subject_id']
        assert r['corrupt_tokens'][r['s2']]==r['alt_id']
        assert len({r['io_id'],r['subject_id'],r['alt_id']})==3
        assert [i for i,(a,b) in enumerate(zip(r['token_ids'],r['corrupt_tokens'])) if a!=b]==[r['s2']]
    for split in banks:
        rr=rows_for(split,[1,2])
        for factor in ['self','wording','order','role','order_role']:
            ids=donor_indices(rr,factor);np.testing.assert_equal(ids[ids],np.arange(len(rr)))
            for i,j in enumerate(ids):
                for k in ['template','pair']:assert rr[i][k]==rr[j][k]
                assert rr[j]['wording']==rr[i]['wording']^int('wording' in factor)
                assert rr[j]['order']==rr[i]['order']^int('order' in factor)
                assert rr[j]['role']==rr[i]['role']^int('role' in factor)
    rng=np.random.default_rng(17);raw=rng.normal(size=(2,32,7,2));ids=np.arange(32)[::-1]
    direct=np.stack([np.stack([raw[b,int(i),:,0] for i in ids]) for b in range(2)])
    np.testing.assert_equal(raw[...,0][:,ids],direct)
    rr=rows_for('fit',[1,2]);raw=np.zeros((2,len(rr),7,2));raw[...,0]=np.array([r['wording'] for r in rr])[None,:,None]*2-1
    fit=fit_reference(raw,rr);assert min_separation(separation(raw,rr,fit))==1
    interval=intervals(np.ones(len(rr))*3,rr,multiplicity=12);assert interval['mean']==3 and interval['adjusted_ci']==[3,3]
    result=dict(status='passed',utc=now(),n_rows=len(rows),old_prompt_strings_checked=len(old_texts),names_disjoint=True,prompt_strings_fresh=True,semantic_positions=True,donor_pairing=True,reference_example_axis_regression=True,classification_fixture=True,crossed_constant_fixture=True,geometry=self_test())
    write(HERE/'results/design-tests.json',result);print('DESIGN AND METRIC TESTS PASSED',len(rows),flush=True)
if __name__=='__main__':main()
