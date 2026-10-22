"""Readable results and handoff notes from the completed analyses."""
from common004 import *

def fmt(x,key='ci95'):
    a,b=x[key];return f"{x['mean']:.3f} [{a:.3f}, {b:.3f}]"
def yes(x):return 'Pass' if x else 'Fail'
def main():
    results={m:read(HERE/'results'/m/'analysis/summary.json') for m in MODELS};clamps=read(HERE/'results/medium-cancellation/analysis/summary.json')
    for d in [HERE/'results'/m for m in MODELS]+[HERE/'results/medium-cancellation']:assert read(d/'verification.json')['status']=='passed'
    text=['# Bridge 004 results\n','8 September 2026. Completed fresh measurements; all prior results and failures retained.\n',
    'This study tests paired-endpoint calibration, a new GPT-2 Medium replication, fixed downstream groups on the same wording pair, and the public TinyLlama 1.1B base model. TinyLlama is a separately trained Llama-architecture model, not Meta Llama 3.2; the latter required unavailable gated access.\n',
    '## Qualification and effects\n',
    '| Model | Discovery separation | Test behavior | Test separation | Positive primary recovery cells | Recipient-favoring primary margins | Full conjunction |',
    '| --- | --- | --- | --- | --- | --- | --- |']
    for m,s in results.items():
        positives=sum(x['adjusted_ci'][0]>0 for x in s['primary']['recovery'].values());recipient=sum(x is not None and x['adjusted_ci'][1]<0 for c in s['primary']['margin'].values() for x in c.values())
        text.append(f"| {m} | {yes(s['discovery_qualified'])} | {yes(s['behavior_valid'])} | {yes(s['test_separated'])} | {positives}/4 | {recipient}/8 | {yes(s['repair_with_recipient_proximity_supported'])} |")
    text+=['\nThe primary intervals use one 24-quantity Bonferroni family across both models. Qualification failures prevent the full claim even if effect intervals support recovery or relative proximity.\n']
    for m,s in results.items():
        base=sum(v['corrupt_top1'] for v in s['natural'].values())/4;w=s['conditions']['wording']
        text += [f'## {m}\n',f"Selected families: {s['families']}. Source heads: {s['source_heads']}. Test configurations: {s['n']}.\n",f"Wording recovery: **{fmt(w['recovery'])} logits** (pointwise 95% interval). Top-1: **{100*base:.1f}% → {100*w['top1']:.1f}%**. Wording minus the average random control: {fmt(s['wording_over_random'])}.\n",'| Cell | Clean pairwise | Clean top-1 | Zero / mean classification |','| --- | --- | --- | --- |']
        for cell,v in s['natural'].items():text.append(f"| {cell} | {100*v['pairwise']:.1f}% | {100*v['top1']:.1f}% | {' / '.join(f'{100*x:.1f}%' for x in v['wording_classification'])} |")
        text+=['\n| Cell | Adjusted recovery | Adjusted zero margin | Adjusted mean margin |','| --- | --- | --- | --- |']
        for cell,x in s['primary']['recovery'].items():
            a,b=[s['primary']['margin'][k][cell] for k in ['zero','discovery_mean']]
            text.append(f"| {cell} | {fmt(x,'adjusted_ci')} | {fmt(a,'adjusted_ci') if a else 'Unresolved'} | {fmt(b,'adjusted_ci') if b else 'Unresolved'} |")
        text+=['\nNegative margins favor the recipient. They are relative distances in measured coordinates, not evidence of endpoint equality or a complete circuit.\n']
    text+=['## Direction and measurement sensitivity\n','TinyLlama has four adjusted margins favoring the recipient and four spanning zero with donor-favoring point estimates. Uniform recipient proximity does not replicate in this architecture extension. TinyLlama uses F2/F3; Medium uses F1/F2, so their difference cannot be attributed exclusively to architecture.\n','The distance conclusions also depend on coordinate weighting. Under unscaled paired Euclidean distance, pointwise intervals give five recipient-favoring, two donor-favoring and one inconclusive Medium cell; TinyLlama gives four recipient-favoring and four donor-favoring cells. These secondary intervals are not the adjusted primary family. The result is conditional on the declared standardization, not a metric-independent property of the mechanism.\n','Editing can reduce distance to both clean endpoints while retaining greater recipient proximity. Mean edited distances can still exceed the donor–recipient separation. Relative proximity therefore does not mean endpoint matching, absence of donor influence or an unchanged complete mechanism. See the post hoc baseline-geometry records.\n','## Same pair Medium group tests\n','The groups and controls were fixed from the older handed study, then tested on F1/F2 in new cases. Adjusted primary intervals use one 20-quantity family. Order values below use the inherited −1 orientation.\n','| Factor | A | B | Union | All opposing/control requirements | Union equivalence ±0.2 |','| --- | --- | --- | --- | --- | --- |']
    for f,v in clamps['factors'].items():text.append(f"| {f} | {fmt(v['primary']['A'],'simultaneous_ci')} | {fmt(v['primary']['B'],'simultaneous_ci')} | {fmt(v['primary']['union'],'simultaneous_ci')} | {yes(v['opposing_contributions_supported'])} | {yes(v['union_equivalent_within_0p2'])} |")
    text+=['\nFailure of equivalence does not prove non-equivalence. Joint nonadditivity and the four-run interaction are distinct contrasts.\n','| Factor / group | Noising N | Denoising P | Interaction N−P |','| --- | --- | --- | --- |']
    for f,v in clamps['factors'].items():
        for g in ['A','B','union']:
            d=v['four_run_decomposition'][g];text.append(f"| {f} / {g} | {fmt(d['NIE_noising'])} | {fmt(d['PIE_denoising'])} | {fmt(d['INT_difference'])} |")
    text+=['\nThese decomposition intervals are pointwise 95% and preserve raw signs. They apply existing mediation contrasts to the upstream edit, without claiming exclusive paths.\n','## Calibration and record integrity\n','The old centroid switch gate remains failed and cannot reject full mechanism transfer. The new paired measurement passes algebraic endpoint checks and measured whole-boundary F1/F2 donor replay. Replay errors are zero in behavior and all measured signature coordinates for both models. The maximum intervention discrepancies are 0.0000066757 for Medium and 0.0000057220 for TinyLlama.\n','A documented analysis indexing correction preserves the original frozen code and changes no estimand or threshold. Read [the correction record](docs/ANALYSIS-CORRECTION.md). Saved-output replay, direct distance arithmetic and the four-run identities pass. All 810 preceding research snapshot files are unchanged.\n','## Contribution and review\n','The [related-work audit](docs/CONTRIBUTION-AUDIT.md) narrows the claim against circuit reuse, faithfulness sensitivity, statistical variance and mediator-interaction studies. The paired metric and decomposition are not claimed as new methods. The candidate contribution is this controlled empirical comparison and its boundary conditions.\n','This is a more complete work-in-progress record for feedback. Additional experiments do not by themselves establish journal readiness. A reader should judge whether the measured effects and failed qualification teach a sufficiently specific lesson, and review the conditional estimands before choosing further experiments or a TMLR submission framing.\n']
    text+=['## Supplemental controls\n','These controls were added after initial outcomes were known and remain explicitly post hoc. The nontrivial mean-bank endpoint replay passed in both models, with zero behavior/signature discrepancy. TinyLlama’s selected F2/F3 prompts have unequal length; its whole-boundary replay therefore uses equal-length F1/F2 calibration prompts, while semantic-position audits cover its selected test pair.\n']
    for m in MODELS:
        s=read(HERE/'results'/m/'projected-controls.json');assert s['status']=='passed'
        text.append(f"- {m}: wording minus the mean of three output-projected, norm-matched random controls is {fmt(s['wording_over_projected_random'])} logits, pointwise 95%. Each random draw’s own recovery interval spans zero. All 384 corrupted and wording behavior outputs independently replay with zero discrepancy.\n")
    assert read(HERE/'results/supplemental-verification.json')['status']=='passed'
    text+=['\nSaved-array audits verify the supplemental hashes, norms, intervals and independent baseline-distance arithmetic. These checks do not constitute a new independent rerun of all downstream signature inference.\n']
    (HERE/'RESULTS.md').write_text('\n'.join(text))
    print('Wrote Bridge 004 results')
if __name__=='__main__':main()
