"""Post-run verification of frozen decisions, saved outputs and preserved manuscripts."""
from common005 import *
from analyze005 import CONDITIONS
import argparse

def main(skip_checkpoint_bytes=False,replay_output='results/audit-reproduced',verification_output='results/verification.json'):
    verify('input-freeze.json');verify('screen-freeze.json');status=read(HERE/'results/status.json');assert status['status']=='completed'
    checks=[]
    def check(name,value):assert value,name;checks.append(name)
    for module in ['common005','common003','assay','geometry004','transport_bridge.common','transport_bridge.statistics']:
        check('local source '+module,Path(sys.modules[module].__file__).resolve().is_relative_to(ROOT))
    timeline=['data/input-freeze.json','results/technical.json','results/screen.json','data/screen-freeze.json','results/fit/run.json','results/selection/run.json','data/selection.json','data/selection-freeze.json','results/selected-pair-replay.json','results/qualification/run.json','results/qualification-summary.json','data/test-freeze.json','results/test/run.json','results/computational-replay.json','results/status.json']
    last=None
    for name in timeline:
        if not (HERE/name).exists():continue
        entry=read(HERE/name);stamp=datetime.fromisoformat(entry.get('utc',entry.get('finished',entry.get('started'))))
        check('ordered stage '+name,last is None or stamp>=last);last=stamp
    prior=read(ROOT/'output/combined-artifact-manifest-v4.json')
    for name,sha in prior['files'].items():check('prior '+name,digest(ROOT.parent/name)==sha)
    preserved=read(HERE/'data/prior-preservation.json')
    for ext,sha in preserved['draft_v2'].items():check('draft manuscript '+ext,digest(ROOT/'paper/draft-v2'/('manuscript.'+ext))==sha)
    if not skip_checkpoint_bytes:
        for name,record in read(HERE/'data/checkpoint.json').items():check('checkpoint '+name,digest(HERE/name)==record['sha256'])
    design=read(HERE/'data/design.json');screen=read(HERE/'results/screen.json');arr=load_arrays(HERE/'results/screen.npz')['behavior'];rows=rows_for('screen',list(range(10)))
    check('screen raw hash',digest(HERE/'results/screen.npz')==screen['raw_sha256']);check('screen cells',screen['cells']==behavior_cells(arr,rows))
    eligible=[int(f) for f,c in screen['cells'].items() if behavior_pass({f:c})]
    check('screen decision',eligible==screen['eligible']);check('screen pairs',screen['pairs']==[p for p in design['alignment_eligible_pairs'] if all(f in eligible for f in p)])
    # Verify every saved scientific bundle, including resumable pieces.
    for p in (HERE/'results').glob('*/run.json'):
        rec=read(p);check(str(p.relative_to(HERE))+' complete',rec['status']=='completed')
        for name,sha in rec['hashes'].items():
            q=p.parent/name;check(str(q.relative_to(HERE))+' hash',digest(q)==sha)
            for a in load_arrays(q).values():check(str(q.relative_to(HERE))+' finite',np.isfinite(a).all())
    selected=(HERE/'data/selection.json').exists()
    if selected:
        verify('selection-freeze.json');sel=read(HERE/'data/selection.json')
        # Recompute all candidate fits and external validation scores from arrays.
        train=rows_for('fit',eligible);valid=rows_for('selection',eligible);a=load_arrays(HERE/'results/fit/clean.npz');b=load_arrays(HERE/'results/selection/clean.npz');rank=[]
        check('fit mean bank',np.array_equal(load_arrays(HERE/'results/fit/bank.npz')['bank'],load_arrays(HERE/'results/fit/natural.npz')['z'].mean(0)))
        for fa,fb in screen['pairs']:
            ids=np.array([i for i,r in enumerate(train) if r['family'] in [fa,fb]]);jds=np.array([i for i,r in enumerate(valid) if r['family'] in [fa,fb]])
            rr=[dict(train[i],wording=int(train[i]['family']==fb)) for i in ids];vv=[dict(valid[i],wording=int(valid[i]['family']==fb)) for i in jds]
            fit=fit_reference(a['raw'][:,ids],rr);ts=separation(a['raw'][:,ids],rr,fit);vs=separation(b['raw'][:,jds],vv,fit)
            tc=behavior_cells(a['behavior'][ids],rr);vc=behavior_cells(b['behavior'][jds],vv);key=f'{fa},{fb}';saved=sel['scores'][key]
            check(key+' exact fit',fit==saved['reference']);check(key+' fit cells',ts==saved['fit_separation'] and tc==saved['fit_behavior']);check(key+' selection cells',vs==saved['selection_separation'] and vc==saved['selection_behavior'])
            good=behavior_pass(tc) and behavior_pass(vc) and min_separation(ts)>=.8 and min_separation(vs)>=.9
            check(key+' eligibility',good==saved['eligible'])
            if good:rank.append((-min_separation(vs),-float(np.mean(list(vs.values()))),[fa,fb]))
        expected=sorted(rank)[0][2] if rank else None;check('fixed pair ranking',expected==sel['families'])
    if (HERE/'results/selected-pair-replay.json').exists():
        s=read(HERE/'results/selected-pair-replay.json');a=load_arrays(HERE/'results/selected-pair-replay.npz')
        check('selected replay raw hash',digest(HERE/'results/selected-pair-replay.npz')==s['raw_sha256'])
        check('selected replay arrays',np.max(np.abs(a['expected_raw']-a['replayed_raw']))==s['signature_error'] and np.max(np.abs(a['expected_behavior']-a['replayed_behavior']))==s['behavior_error'])
        check('replay nontrivial means',all(n>0 for n in s['nonzero_by_reference']))
    if (HERE/'results/qualification-summary.json').exists():
        q=read(HERE/'results/qualification-summary.json');rr=rows_for('qualification',sel['families']);a=load_arrays(HERE/'results/qualification/clean.npz')
        cells=behavior_cells(a['behavior'],rr);sep=separation(a['raw'],rr,sel['reference'])
        check('independent qualification arrays',q['behavior']==cells and q['separation']==sep)
        check('independent qualification gate',q['qualified']==(behavior_pass(cells) and min_separation(sep)>=.8))
    if status['test_measured']:
        verify('test-freeze.json');check('qualification before test',q['qualified'])
        rows=rows_for('test',sel['families']);path=HERE/'results/test';a=load_arrays(path/'wording.npz');c=load_arrays(path/'clean.npz');donors=donor_indices(rows,'wording');scale=np.asarray(sel['reference']['scale'])[:,None]
        e=a['raw'][...,0].astype(np.float64);r=c['raw'][...,0].astype(np.float64);d=np.stack([np.stack([r[bi,int(j)] for j in donors]) for bi in range(2)])
        u=(e-r)/scale;v=(d-r)/scale;norm=np.linalg.norm(v,axis=-1);safe=np.where(norm>=1e-6,norm,np.nan)
        direct=(np.linalg.norm(u,axis=-1)-np.linalg.norm(u-v,axis=-1))/safe
        saved=load_arrays(HERE/'results/analysis/per-example.npz');np.testing.assert_allclose(saved['wording__margin'],direct,atol=1e-12,rtol=1e-12,equal_nan=True)
        np.testing.assert_allclose(saved['wording__q_donor']**2-saved['wording__q_recipient']**2,1-2*saved['wording__score'],atol=1e-12,rtol=1e-12,equal_nan=True)
        check('independent paired arithmetic',True)
        # An alternative bootstrap implementation uses multinomial count weights
        # and an explicit per-cell grid, rather than indexed replicate tensors.
        original=read(HERE/'results/analysis/summary.json');rng=np.random.default_rng(SEED)
        draws_t=rng.integers(12,size=(DRAWS,12));draws_p=rng.integers(12,size=(DRAWS,12))
        wt=np.stack([(draws_t==i).sum(1) for i in range(12)],axis=1)
        wp=np.stack([(draws_p==i).sum(1) for i in range(12)],axis=1)
        for w in range(2):
            for o in range(2):
                mask=np.array([r['wording']==w and r['order']==o for r in rows]);key=f'w{w}_o{o}'
                ts=sorted({r['template'] for r in rows});ps=sorted({r['pair'] for r in rows})
                quantities=[('recovery',saved['wording__recovery'],original['primary']['recovery'][key])]
                quantities += [(b,direct[bi],original['primary']['margin'][b][key]) for bi,b in enumerate(['zero','fit_mean'])]
                for label,values,expected in quantities:
                    if expected is None:
                        check(key+label+' unresolved interval',not np.isfinite(values[mask]).all());continue
                    grid=np.empty((12,12))
                    for ti,t in enumerate(ts):
                        for pi,p in enumerate(ps):
                            ids=[i for i,r in enumerate(rows) if mask[i] and r['template']==t and r['pair']==p]
                            assert len(ids)==2;grid[ti,pi]=np.asarray(values[ids],np.float64).mean()
                    boot=np.einsum('bi,ij,bj->b',wt,grid,wp,optimize=True)/144
                    expected_numbers=[expected['mean']]+expected['ci95']+expected['adjusted_ci']
                    actual_numbers=[grid.mean()]+np.quantile(boot,[.025,.975,.05/24,1-.05/24]).tolist()
                    np.testing.assert_allclose(actual_numbers,expected_numbers,atol=1e-12,rtol=1e-12)
                    check(key+label+' independent bootstrap',True)
        from analyze005 import main as analyze
        dest=replay_output;assert not (HERE/dest).exists();analyze(dest)
        check('exact analysis replay',read(HERE/dest/'summary.json')==read(HERE/'results/analysis/summary.json'))
        redo=load_arrays(HERE/dest/'per-example.npz')
        for k in saved:np.testing.assert_equal(saved[k],redo[k])
        check('exact metric arrays',True)
        replay=read(HERE/'results/computational-replay.json');z=load_arrays(HERE/'results/computational-replay.npz')
        check('computational replay status',replay['status']=='passed' and digest(HERE/'results/computational-replay.npz')==replay['raw_sha256'])
        check('wording inference replay',np.max(np.abs(z['wording_behavior']-a['behavior']))<TOL)
        check('corrupt inference replay',np.max(np.abs(z['corrupt_behavior']-load_arrays(path/'corrupt.npz')['behavior']))<TOL)
        check('singleton signature replay',np.max(np.abs(z['singleton_raw']-a['raw'][:,z['singleton_indices']]))<TOL)
    else:
        check('stopped before test',not (HERE/'results/test').exists() and not (HERE/'data/test-freeze.json').exists())
        if status['outcome']=='no_screen_pair':check('screen stop rule',not screen['pairs'])
        elif status['outcome']=='no_selection_pair':check('selection stop rule',sel['families'] is None)
        else:check('qualification stop rule',not q['qualified'])
    result=dict(status='passed',utc=now(),checks=len(checks),check_names=checks,outcome=status['outcome'],test_measured=status['test_measured'],prior_files_unchanged=len(prior['files']),draft_manuscript_unchanged=True,checkpoint_bytes='explicitly skipped for saved-output replay' if skip_checkpoint_bytes else 'all verified',scope='Preservation and saved-array replay; computational inference replay is separately recorded when test is reached')
    write(HERE/verification_output,result);print('AUDIT PASSED',len(checks),status['outcome'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skip-checkpoint-bytes',action='store_true');p.add_argument('--replay-output',default='results/audit-reproduced');p.add_argument('--verification-output',default='results/verification.json');a=p.parse_args();main(a.skip_checkpoint_bytes,a.replay_output,a.verification_output)
