# Completed follow-up: valid wording and causal explanation

8 September 2026. Both GPT-2 Medium and GPT-2 Small completed every planned test
condition and passed the independent saved-output audit. The earlier task-validity
gap is resolved for these two selected formulations. The causal explanation is
stronger, but partial: the four-head output-blocking result succeeds in Small
and fails to meet the success criterion in Medium. That difference is retained
as a main result.

## What was run

The two grammatical families are “After … walked into … handed … to” and
“When … arrived at … handed … to”. They differ in the prefix/location clause
and share the transfer verb. A failed initial screen and a prospectively frozen
second-screen amendment are retained. Family selection used clean behavior,
before discovery or test transfer outcomes.

Each model used 128 discovery cells and the same 384 held-out test cells: eight
location/object blocks × six exclusive name triples × two role assignments ×
two wordings × two orders. One name per triple supplies the corruption. Names
and blocks are separated across screening, discovery and test; the new name
triples avoid the earlier overlapping corruption-name problem.

All 13 fixed edits were evaluated on every test cell. The 9,984 edited-recipient
evaluations across models are repeated paired conditions, not independent samples.
Three additional trace bundles per model test own-clean repair, wording transfer
and order transfer with selected-head and control clamps. The signatures measure
176 later heads in Medium and 60 in Small, under zero and discovery-clean mean
replacement. The study uses fixed pretrained models and the existing locked
bridge environment.

Scientific code and inputs were frozen before discovery. References, selected
heads, controls, effect orientation and the analyzer were frozen before test.
The 10,000-draw bootstrap crosses template blocks and name triples, preserving
paired conditions. Intervals condition on the discovery fits and selected heads.

## 1. Both prompt families are reliably solved

Every clean family/order cell has 100% pairwise accuracy: the intended name
outranks the other actor. The stronger measure asks whether it is the highest
scoring token in the entire vocabulary:

| Recipient formulation / order | Medium top-1 | Small top-1 |
|---|---:|---:|
| walked / ABBA | 97.9% | 95.8% |
| walked / BABA | 93.8% | 96.9% |
| arrived / ABBA | 97.9% | 92.7% |
| arrived / BABA | 93.8% | 93.8% |

All eight cells pass the frozen 95% pairwise / 85% top-1 thresholds. This directly
addresses Bridge 001's 0–4.7% altered-wording top-1 performance. It establishes
validity for this selected comparison, not arbitrary paraphrases.

## 2. Wording transfer repairs the answer; order transfer damages it

Recovery is the gain in intended-name minus subject logit difference relative
to the corrupted recipient. The pooled table uses secondary pointwise 95%
intervals. Full reports contain all conditions and the primary adjusted tests.

| Source-head edit | Medium recovery [95% interval] | Small recovery [95% interval] |
|---|---|---|
| Own clean prompt | 2.017 [1.803, 2.250] | 2.883 [2.315, 3.449] |
| Other wording, same order and roles | 1.968 [1.760, 2.191] | 2.870 [2.309, 3.431] |
| Opposite order, same roles | −1.546 [−1.736, −1.335] | −1.852 [−2.594, −1.315] |
| Same order, exchanged roles | 1.546 [1.335, 1.736] | 1.852 [1.315, 2.594] |
| Opposite order and exchanged roles | −2.017 [−2.250, −1.803] | −2.883 [−3.449, −2.315] |

All four wording-transfer recovery cells remain positive after the prespecified
eight-quantity adjustment in each model. Wording transfer reaches 92.2% top-1
in Medium and 87.0% in Small; order transfer reaches only 8.3% and 13.5%.
Wording transfer also exceeds the mean of its three norm-matched random controls:
1.969 [1.760, 2.192] logits in Medium and 2.832 [2.243, 3.407] in Small.

The four-donor factorial test shows a much larger order effect than role effect,
but both matter. For walked recipients, Medium's order effect is −3.344 with
adjusted interval [−3.901, −2.791], versus a role effect of −0.438
[−0.708, −0.235]. Small's corresponding effects are −4.512
[−6.326, −3.028] and −0.984 [−1.428, −0.516]. Arrived recipients show the
same qualitative pattern. No role effect meets the fixed ±0.2-logit equivalence
criterion. Position compatibility is the larger factor in these interventions;
answer identity is not irrelevant.

The apparent sign symmetries in the pooled donor table are partly algebraic.
Opposite-order/role-swapped recipients have identical corrupted input tokens and
reversed scoring labels. They are not independent replications of the same result.
The diagnostic reports verify this pairing explicitly. The interventions also
change concrete token features; the factorial does not isolate a pure abstract
semantic variable.

## 3. Answer recovery does not establish a complete dependency switch

The new families' clean dependency signatures are not reliably distinguishable
in every held-out cell. Medium misses the ≥75% wording-classification gate in
two reference/cell combinations; Small misses it in four. This limits what a
failed switch test can mean even though clean answer behavior is now strong.

Medium does show donor-oriented movement for some recipients. In walked/ABBA,
the donor-axis position S is 0.999 [0.736, 1.260] under zero replacement and
0.741 [0.502, 0.992] under mean replacement, using adjusted intervals. However,
orthogonal movement is approximately 2.17 and 1.99 discovery-gap units, and the
actual matched-donor proximity intervals include zero. A projection near one
therefore does not establish the donor's full measured signature.

In Medium's arrived/BABA cell, the direction changes with the replacement
reference: S is −0.318 [−0.549, −0.103] under zero and +0.380
[0.067, 0.711] under mean replacement. This is direct reference dependence in
the measurement. Small's edited signatures are closer on average to the matched
clean recipient in every wording cell under both references, with negative
pointwise donor-margin intervals.

No wording condition passes the complete switch criterion. This is not proof
of an unchanged mechanism, absence of donor influence, or impossibility of any
router. The result is that these successful repairs do not establish the stronger
operational switch claim.

## 4. The causal follow-up gives a partial, model-dependent explanation

Discovery selected the same four-head set for wording and order within each
model: Small 10.7, 8.6, 7.9, 11.10; Medium 15.14, 22.14, 17.12, 13.13.
Selection used absolute changes in individual-head effects, so opposing
contributions could enter the same group. All head IDs and three layer/size-matched
control groups were frozen before test.

The output clamp holds these heads at their corrupted-recipient values while
the upstream edit is applied. Oriented attenuation is positive when it reduces
the edit's repair or damage, with direction fixed from discovery.

| Model / transfer | Oriented attenuation [adjusted interval] | Exceeds all three controls under the adjusted criterion? |
|---|---|---|
| Small wording | 0.996 [0.502, 1.510] | Yes |
| Small order | 0.722 [0.361, 1.284] | Yes |
| Medium wording | −0.044 [−0.323, 0.229] | No |
| Medium order | 0.048 [−0.216, 0.303] | No |

Small therefore supports a causal contribution of the selected four-head group
to both effects. Medium does not support that same compact-group blocking claim.
A nonsignificant net effect is not evidence of exact zero contribution. The
separately selected own-clean repair control succeeds in both models: output
clamping attenuates repair by 1.075 [0.903, 1.245] logits in Medium and 2.483
[1.953, 3.037] in Small, exceeding all three controls with pointwise intervals.
This confirms that the assay can detect repair attenuation; it is not a known
compact-router positive control.

Secondary component interventions reveal opposing contributions:

| Model / transfer | Query clamp signed attenuation [95%] | Value clamp signed attenuation [95%] |
|---|---|---|
| Medium wording | −0.440 [−0.688, −0.187] | 0.399 [0.335, 0.462] |
| Small wording | −2.550 [−2.965, −2.068] | 1.975 [1.558, 2.434] |
| Medium order | 0.318 [0.082, 0.560] | −0.359 [−0.431, −0.283] |
| Small order | 1.767 [1.297, 2.301] | −1.353 [−1.911, −0.982] |

For wording, holding values at their corrupted values reduces repair, whereas
holding queries increases it. For order, the value clamp reduces damage and
the query clamp increases it. Thus the query changes oppose part of the net
edit effect under these interventions. Keys also contribute in the edit's
direction. These are controlled intervention interactions with secondary
pointwise uncertainty, not additive mediated fractions or unique causal paths.

The individual-head breakdowns show opposing changes within both selected
groups, and joint-head effects are non-additive and replacement-dependent.
Together these observations give a plausible account of why a large dependency
change need not yield a large net group-blocking effect, especially in Medium.
They do not mathematically decompose the whole result into an exhaustive mechanism.

Pooled corrupted non-additivity cancels by the paired scoring design. It must
not be interpreted as absence of interactions: nonzero within-cell values are
reported in the [Medium](results/gpt2-medium/report/DIAGNOSTICS.md) and
[Small](results/gpt2-small/report/DIAGNOSTICS.md) secondary breakdowns. These
displays were added after Small's test outcomes and do not revise primary tests.

## Assessment for one combined project

The requested follow-up is complete. It resolves the poorly solved wording
comparison and adds an empirical repair/blocking control, a factorial donor
study, component clamps and joint-head measurements. It strengthens a single
empirical project around **what activation transport establishes about downstream
dependence**. The main new caution within this dataset is that large changes in
individual-head effects need not identify a group with a robust net blocking
effect; the latter must be tested and was model-dependent here.

Do not frame this as discovery of a universal router, complete circuit
equivalence, or a new positional IOI mechanism. The role/position distinction,
IOI head classes and broad limitations of patching already have substantial
[prior art](docs/MECHANISTIC-CONTEXT.md). This study's contribution would be its
controlled matched evaluation and the specific practical lessons it supports.

The evidence is sufficient to develop a bounded combined manuscript. It is not
yet a submission-ready manuscript, and the stronger claim of transfer between
distinct solved circuits remains unsupported. For TMLR, the remaining essential
work is a coherent paper, precise comparison with prior findings, and an author
review of the methods and evidence. Broader templates or another model family
would strengthen generality; they are necessary if broad generality is claimed,
not an automatic prerequisite to reporting these narrower results. TMLR assesses
evidence and a clearly communicated lesson; completing experiments alone does
not establish acceptance. See its [criteria](https://jmlr.org/tmlr/acceptance-criteria.html).

## Verification and files

Both models passed numerical full-forward checks with maximum error below
1.5×10⁻⁵ against a 5×10⁻⁴ tolerance. Each completed test has 17 raw files;
every recorded hash was checked. The independent audit reproduced each summary
byte-for-byte and all 72 per-example arrays exactly, verified factorial effects
through a separate linear solve, checked geometry and clamp algebra, and
confirmed that all 549 tracked earlier files remain unchanged. This is saved-output
verification, not another full inference replication. Five new finite-design
and clamp tests passed before the scientific freeze.

The [Medium report](results/gpt2-medium/report/RESULTS.md),
[Small report](results/gpt2-small/report/RESULTS.md),
[reproduction guide](docs/REPRODUCIBILITY.md) and
[combined project guide](../../COMBINED-PROJECT.md) provide the complete results,
figures and execution context. The full combined research snapshot is
`../../output/combined-bridge-research-v2.zip`; it includes all three study
directories' scientific records and historical manuscripts, with a checked
per-file manifest. Model weights and installed environments are separate.
