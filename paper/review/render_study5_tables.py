"""Render manuscript tables from completed, independently checked Study 5 outputs."""
from pathlib import Path
import hashlib
import json
import re

REV=Path(__file__).resolve().parents[1];ROOT=REV.parent
STUDY=ROOT/'studies/bridge-010'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x,places=3):
    if abs(x)<.5*10**(-places):x=0
    return f'{x:.{places}f}'
def ci(v,key='ci95',factor=1,places=3):
    assert v['valid']
    p=num(v['mean']*factor,places);a,b=[num(x*factor,places) for x in v[key]]
    return '$'+p+r'\ci{'+a+'}{'+b+'}$'
def limits(v,key,places=3):
    a,b=[num(x,places) for x in v[key]];return '$['+a+', '+b+']$'
def scientific(x):
    if x==0:return '$0$'
    mantissa,exponent=f'{x:.3e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
def table(label,caption,headers,rows,placement='H',font='footnotesize',tabcolsep='3pt',natural=False):
    if natural:
        begin,end=r'\begin{tabular}{@{}l'+'c'*(len(headers)-1)+'@{}}',r'\end{tabular}'
    else:
        begin,end=r'\begin{tabularx}{\linewidth}{@{}l'+'Y'*(len(headers)-1)+'@{}}',r'\end{tabularx}'
    return '\n'.join([r'\begin{table}['+placement+']','\\centering\\'+font,
          r'\setlength{\tabcolsep}{'+tabcolsep+'}',r'\caption{'+caption+'}',r'\label{'+label+'}',
          begin,r'\toprule',
          ' & '.join(headers)+r' \\',r'\midrule']+
          [' & '.join(row)+r' \\' for row in rows]+[r'\bottomrule',end,r'\end{table}',''])


def main():
    audit=read(STUDY/'results/verification.json');assert audit['passed']
    summary_path=STUDY/'results/analysis/summary.json'
    assert sha(summary_path)==audit['analysis_summary_sha256']
    data=read(summary_path);controls=['random','mismatched','natural'];names={'random':'Random','mismatched':'Mismatched','natural':'Natural'}
    refs=['zero','fit_mean'];rnames={'zero':'zero','fit_mean':'mean'}
    study2_path=ROOT/'studies/bridge-007/results/analysis/summary.json'
    study2=read(study2_path)
    provenance=('% Generated from saved, independently verified Study 5 outputs.\n% Summary SHA-256: '+sha(summary_path)+
                '\n% Study 2 cell values: '+str(study2_path.relative_to(ROOT))+' SHA-256: '+sha(study2_path)+'\n')
    word=data['wording']
    rows=[]
    for c in controls:
        for ref in refs:
            g=data['controls'][c]['geometry'][ref]
            rows.append([names[c]+' / '+rnames[ref],num(g['F']['null']['mean']),ci(g['F']['word_over_null'],'adjusted_ci'),
                         num(g['A']['null']['mean']),ci(g['A']['word_over_null'],'adjusted_ci')])
    main_table=table('tab:geometric-null-ratios',
       r'Study 5 (exploratory; inspected Study 2 bank). Control/self discrepancies and the ratio of the wording/self discrepancy to them, for the full signature ($F$) and the absolute donor-axis position ($A$), in clean-wording separations. Wording/self values are $F='+
       num(word['zero']['F']['mean'])+'/'+num(word['fit_mean']['F']['mean'])+r'$ and $A='+num(word['zero']['A']['mean'])+'/'+num(word['fit_mean']['A']['mean'])+
       r"$ (zero/mean replacement). Ratios of pooled means with Bonferroni-adjusted intervals over 24 paired contrasts; the 50,000 crossed bootstrap draws also resample the four replicates of each control. Every control perturbs self transfer while keeping, for each source head, the wording edit's norm and angle to self (\cref{sec:geometric-null-results}). A ratio above 1 means that wording directions change the signature more than the control directions.",
       ['Control / reference',r'$F_{\rm null}$',r'$F_{\rm word}/F_{\rm null}$',r'$A_{\rm null}$',r'$A_{\rm word}/A_{\rm null}$'],rows,placement='t',font='small',natural=True)
    (REV/'sections/geometric-null-main-table.tex').write_text(provenance+main_table)
    out=[];rows=[]
    for c in controls:
        for ref in refs:
            g=data['controls'][c]['geometry'][ref]
            rows.append([names[c]+' / '+rnames[ref],ci(g['F']['null']),ci(g['A']['null']),ci(g['signed_axis']),ci(g['q_recipient'])])
    out.append(table('tab:null-means',
       r'Study 5 pooled control/self discrepancies, signed donor-axis change from self and distance to the clean recipient. Means and pointwise 95\% intervals. Four replicates are averaged within each of the 3,200 configurations. Units are matched clean-wording separations; all cases are retained.',
       ['Control / reference',r'$F_{\rm null}$',r'$A_{\rm null}$',r'$S_{\rm null}-S_{\rm self}$',r'$q_R$'],rows))
    rows=[];rounded=[]
    for c in controls:
        for ref in refs:
            for m in ['F','A']:
                e=data['controls'][c]['geometry'][ref][m]
                rows.append([names[c]+' / '+rnames[ref]+' / '+m,ci(e['word_over_null']),limits(e['word_over_null'],'adjusted_ci'),ci(e['null_minus_word']),limits(e['null_minus_word'],'adjusted_ci')])
                for key,kind in [('ci95','pointwise'),('adjusted_ci','adjusted')]:
                    for side,x in zip(['lower','upper'],e['null_minus_word'][key]):
                        if x!=0 and num(x)=='0.000':
                            rounded.append(f"the {kind} {side} difference limit for {names[c].lower()}/{rnames[ref]}/{m} is ${x:+.4f}$")
    note=(' Limits printed as 0.000 are rounded: '+'; '.join(rounded)+'.') if rounded else ''
    out.append(table('tab:null-contrasts',
       r'Study 5 geometric contrasts, retaining pointwise 95\% and 24-contrast Bonferroni intervals. Ratios and differences express the same paired comparison, with nulls 1 and 0. Difference is control/self minus wording/self discrepancy. The numerator and denominator are paired in every bootstrap draw.'+note,
       ['Control / ref. / metric',r'Ratio [95\% CI]','Adjusted CI',r'Difference [95\% CI]','Adjusted CI'],rows,natural=True))
    rows=[]
    for c in controls:
        b=data['controls'][c]['behaviour']
        rows.append([names[c],ci(b['recovery']['null']),ci(b['top1']['null'],factor=100,places=1),ci(b['pairwise']['null'],factor=100,places=1)])
    out.append(table('tab:null-behaviour',
       r'Study 5 behaviour with pointwise 95\% intervals. Recovery is the IO-minus-subject logit gain over corruption. Accuracy columns are percentages over the full vocabulary (top-1) or intended name versus subject (pairwise). Same four-replicate averages and crossed bootstrap as the geometry summaries.',
       ['Control','Recovery (logits)',r'Top-1 (\%)',r'Pairwise (\%)'],rows))
    rows=[]
    for c in controls:
        for metric in ['recovery','top1']:
            for base in ['self','wording']:
                v=data['controls'][c]['behaviour'][metric]['minus_'+base]
                if metric=='top1':v={**v,'mean':100*v['mean'],'ci95':[100*x for x in v['ci95']],'adjusted_ci':[100*x for x in v['adjusted_ci']]}
                places=3 if metric=='recovery' else 2
                rows.append([names[c],('Recovery' if metric=='recovery' else 'Top-1'),base,ci(v,places=places),limits(v,'adjusted_ci',places)])
    out.append(table('tab:null-behaviour-contrasts',
       r'Study 5 paired behavioural changes: control minus the named comparator. Recovery is in logits; top-1 change is in percentage points. Pointwise and 24-contrast adjusted intervals retain behavioural disruption rather than selecting accuracy-matched draws.',
       ['Control','Measure','Comparator',r'Change [95\% CI]','Adjusted CI'],rows))
    rows=[]
    for ref in refs:
        for cell in ['w0_o0','w0_o1','w1_o0','w1_o1']:
            w,o=map(int,re.findall(r'\d',cell))
            rows.append([f'F{10+w} / {o} / '+rnames[ref],ci(study2['per_cell_heterogeneity'][ref][cell]['full_signature_distance'])]+
                        [ci(data['controls'][c]['cells'][cell][ref]['F']) for c in controls])
    out.append(table('tab:null-cell-comparison',
       r'Full-signature discrepancy from self transfer by wording/order cell: wording transfer (saved Study 2 cell summaries) and the three Study 5 controls, in clean-wording separations (800 configurations per cell; four replicates averaged per control). Brackets are pointwise 95\% intervals from the separate saved bootstrap analyses of each study; they are descriptive and are not paired cell contrasts.',
       ['Family / order / ref.','Wording','Random','Mismatched','Natural'],rows))
    for c in controls:
        rows=[]
        for cell,entry in data['controls'][c]['cells'].items():
            w,o=map(int,re.findall(r'\d',cell));label=f'F{10+w} / {o}'
            for ref in refs:
                g=entry[ref]
                rows.append([label+' / '+rnames[ref],ci(g['F']),ci(g['A']),ci(g['signed_axis']),ci(g['q_recipient'])])
        out.append(table('tab:null-cells-'+c,
           names[c]+r' control: all wording/order/reference cells, pointwise 95\% intervals. Four replicates per configuration; 800 configurations per wording/order cell. These are descriptive summaries, not additional adjusted cell tests.',
           ['Family / order / ref.',r'$F_{\rm null}$',r'$A_{\rm null}$',r'$S_{\rm null}-S_{\rm self}$',r'$q_R$'],rows))
    rows=[]
    for c in controls:
        for cell,entry in data['controls'][c]['cells'].items():
            w,o=map(int,re.findall(r'\d',cell));b=entry['behaviour']
            rows.append([names[c]+f' / F{10+w} / {o}',ci(b['recovery']),ci(b['top1'],factor=100,places=1)])
    out.append(table('tab:null-cell-behaviour',r'Study 5 behaviour in every wording/order cell. Recovery is in logits and top-1 in percent, with pointwise 95\% intervals. No cell or disruption is removed.',
             ['Control / family / order','Recovery','Top-1'],rows))
    rows=[]
    for c in controls:
        for entry in data['controls'][c]['replicates']:
            g=entry['geometry'];k=entry['replicate'];dis=data['controls'][c]['disruptions']
            rows.append([names[c]+f' / {k+1}',num(g['zero']['F'])+' / '+num(g['fit_mean']['F']),
                 num(g['zero']['A'])+' / '+num(g['fit_mean']['A']),num(entry['recovery']),num(entry['top1']*100,1),
                 f"{dis['gap_below_self_per_replicate'][k]:,}"+' / '+f"{dis['lost_top1_from_self_per_replicate'][k]:,}"+' / '+f"{dis['recovery_below_zero_per_replicate'][k]:,}"])
    out.append(table('tab:null-replicates',
       r'All four replicates of every Study 5 control. Paired geometry values are zero / mean reference. Disruption counts are: lower logit gap than self / loss of a correct self-transfer top-1 prediction / negative recovery relative to corruption, each out of 3,200. Accuracy is in percent. Replicates are Monte Carlo repeats of the same configurations.',
       ['Control / repeat',r'$F$',r'$A$','Recovery','Top-1','Disruption counts'],rows))
    for key,label,description in [
        ('configuration_distribution','tab:null-distributions',
         'per-configuration four-replicate mean discrepancies (3,200 values per row)'),
        ('all_draw_distribution','tab:null-draw-distributions',
         'individual configuration/replicate discrepancies (12,800 values per row)')]:
        rows=[]
        for c in controls:
            for ref in refs:
                for metric in ['F','A']:
                    dist=data['controls'][c]['geometry'][ref][metric][key]
                    q=dist['quantiles']
                    rows.append([names[c]+' / '+rnames[ref]+' / '+metric,num(dist['minimum']),num(q['p05']),num(q['p50']),num(q['p95']),num(dist['maximum'])])
        out.append(table(label,
           'Distributions of the '+description+r' in Study 5. Quantiles describe heterogeneity rather than confidence limits. Every configuration and replicate is retained.',
           ['Control / ref. / metric','Minimum','5th pct.','Median','95th pct.','Maximum'],rows))
    cal=read(STUDY/'results/calibration.json');replay=read(STUDY/'results/singleton-replay.json')
    largest=max(max(x.values()) for x in replay['checks'].values())
    checks=[]
    for key,label in [('norm','Joint injection norm'),('cross','Cross term with self'),('perturbation','Perturbation norm')]:
        checks.append([label,scientific(max(x[key]['max_absolute_error'] for x in cal['geometry'].values())),
             num(max(x[key]['max_tolerance_fraction'] for x in cal['geometry'].values()),4),
             str(sum(x[key]['violations'] for x in cal['geometry'].values()))])
    out.append(table('tab:null-calibration',
        r'Study 5 matching checks over all 460,800 realised head injections. The tolerance for each target is $10^{-6}+10^{-5}|\mathrm{target}|$; the largest fraction of that tolerance is also reported. Zero and reconstructed-wording perturbations reproduce their respective self and wording signatures and behaviours exactly. Full-sequence signature error is $6.676\times10^{-6}$, below $5\times10^{-4}$.',
        ['Check','Maximum absolute error','Maximum tolerance fraction','Violations'],checks))
    out.append(r'\noindent\textit{Completed verification.} Independent reconstruction checked '+f"{audit['chunks_checked']:,}"+
               ' saved measurement chunks and '+str(audit['intervals_checked'])+
               r' headline and behavioural intervals. Fresh-engine singleton replays of eight fixed cases in every condition agreed within '+scientific(largest)+
               r', below $5\times10^{-4}$. No calibration case, control replicate or disruptive observation was removed.'+'\n')
    (REV/'sections/geometric-null-tables.tex').write_text(provenance+'\n'.join(out))
    (REV/'review/study5-table-provenance.json').write_text(json.dumps(dict(summary_sha256=sha(summary_path),
       verification_sha256=sha(STUDY/'results/verification.json'),calibration_sha256=sha(STUDY/'results/calibration.json'),
       singleton_max_error=largest,source=str(summary_path.relative_to(ROOT)),
       study2_cell_source=str(study2_path.relative_to(ROOT)),study2_cell_source_sha256=sha(study2_path)),indent=2)+'\n')
    print('Rendered Study 5 manuscript tables from verified saved outputs.')


if __name__=='__main__':main()
