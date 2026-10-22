# Bridge 003 findings

8 September 2026. Both planned arms are complete. Every fixed edit and clamp
condition was retained, and all three saved-output audits passed. The project is
stronger, with a valid and distinguishable Medium wording comparison plus a fresh
causal follow-up. It still has important scope and measurement limits.

| Arm / model | Test cases | Clean behavior valid | Signatures separated | Interpretation |
| --- | --- | --- | --- | --- |
| A / Medium | 384 | True | Not this arm's selection target | Opposing-group causal follow-up |
| B / Medium | 384 | True | True | Qualified F1/F2 comparison |
| B / Pythia | 384 | False | False | Unqualified F0/F1 diagnostic |

Each test comprises eight location/object blocks × six disjoint name triples ×
two roles × two wordings × two orders. These are paired cells, not independent
intervention samples. Arm A uses a separate name/block bank from arm B.
The design uses local prospective hash freezes, not public preregistration.

## 1. Medium's opposing effects replicate, but simple cancellation does not

The hypothesis came from Bridge 002's test outcomes; A/B membership came only
from its discovery outputs. A = L15H14/L13H13 and B = L22H14/L17H12 (zero-based).
On fresh prompts both groups pass the predeclared opposing-sign criterion and
all six layer/size-matched control comparisons per factor.

| Oriented clamp effect | Wording | Order (orientation reversed) |
| --- | --- | --- |
| A | +0.870 [+0.587, +1.210] | +0.798 [+0.557, +1.065] |
| B | -1.996 [-2.487, -1.416] | -1.824 [-2.238, -1.241] |
| union | -0.489 [-0.662, -0.182] | -0.360 [-0.543, -0.015] |
| A_minus_B | +2.867 [+2.080, +3.663] | +2.622 [+1.907, +3.283] |

These intervals use the fixed 20-quantity Bonferroni family. Attenuation compares
the upstream edit's effect before and after clamping the selected component to
the corrupted recipient's value, including the unedited clamp identity control.
The order table reverses the sign according to the discovery-fixed orientation.

The joint clamp has a negative oriented effect and fails the ±0.2-logit
equivalence criterion in both factors. Oriented union-minus-A-minus-B is
+0.637 [+0.542, +0.714] for wording and
+0.665 [+0.546, +0.753] for order (secondary,
pointwise intervals). Thus the earlier near-zero joint effect does not generalize
as exact cancellation. These are conditional, nonadditive clamp effects, not
additive mediation fractions or identification of unique causal paths.

Secondary component interventions reproduce B's large opposing effect mainly
through query clamps. A responds to both queries and values, with a smaller key
effect. All 19 clamp variants per factor and their per-example outcomes are retained.
This arm uses the older walked/arrived wording pair; it does not directly test
the causal explanation for arm B's new gave/passed pair.

[Full Medium clamp tables and figures](results/medium-cancellation/report/RESULTS.md)

## 2. Broader wording closes the validity/separation gap in Medium

Eight fixed families were screened on 32 cases each per model. Six passed Medium's
observed-accuracy gates, and the fixed first-four rule retained F0/F1/F2/F4.
Among discovery-qualified pairs, F1/F2 won the fixed worst-cell accuracy rule and
lexicographic tie-break: “When … went … gave … to” versus
“When … walked … passed … to”. Their worst discovery classification accuracy was
93.75%. These are multi-feature wording changes, not isolated transfer-verb edits.

Medium passes every held-out clean behavior and separability gate. Pythia had no
eligible screen candidate and therefore uses the predetermined F0/F1 fallback.
The models use different selected texts, so their results cannot be treated as an
exactly matched cross-architecture replication.

| Model | Cell | Pairwise | Top-1 | Classification zero / mean |
| --- | --- | --- | --- | --- |
| gpt2-medium | w0_o0 | 100.00% | 97.92% | 100.00% / 100.00% |
| gpt2-medium | w0_o1 | 100.00% | 100.00% | 100.00% / 100.00% |
| gpt2-medium | w1_o0 | 100.00% | 100.00% | 89.58% / 89.58% |
| gpt2-medium | w1_o1 | 100.00% | 100.00% | 87.50% / 94.79% |
| pythia-410m-deduped | w0_o0 | 100.00% | 96.88% | 72.92% / 77.08% |
| pythia-410m-deduped | w0_o1 | 98.96% | 93.75% | 94.79% / 88.54% |
| pythia-410m-deduped | w1_o0 | 98.96% | 87.50% | 92.71% / 88.54% |
| pythia-410m-deduped | w1_o1 | 94.79% | 87.50% | 89.58% / 96.88% |

The gates are 95% pairwise, 85% vocabulary top-1 and 80% signature classification
in every cell/reference. Pythia's held-out top-1 is much better than its screen,
but its 94.79% pairwise cell and 72.9–77.1% classification cell still fail the fixed
criteria. These observations show sensitivity to the name/block population,
not universal task inability. No threshold or family was changed after outcomes.

[All eight screen results, including failures](reports/SCREEN.md)

## 3. Answer repair and donor-like dependencies are different observations

| Edit | Medium recovery [pointwise 95% CI] | Pythia diagnostic recovery [pointwise 95% CI] |
| --- | --- | --- |
| self | +2.416 [+1.835, +3.065] | +2.517 [+2.174, +2.942] |
| wording | +2.415 [+1.837, +3.061] | +2.491 [+2.162, +2.908] |
| order | -2.063 [-2.544, -1.650] | -1.805 [-2.480, -1.255] |
| role | +2.063 [+1.650, +2.544] | +1.805 [+1.255, +2.480] |
| order_role | -2.416 [-3.065, -1.835] | -2.517 [-2.942, -2.174] |
| random0 | -0.006 [-0.028, +0.018] | -0.019 [-0.061, +0.024] |
| random1 | +0.001 [-0.016, +0.019] | -0.005 [-0.054, +0.037] |
| random2 | +0.005 [-0.016, +0.026] | -0.018 [-0.057, +0.021] |

Medium wording transfer increases pooled top-1 from 24.5% to 93.0% (clean 99.5%),
and pairwise accuracy from 50.0% to 95.1%. All four cell recoveries remain positive
under the prespecified adjustment. Its wording advantage over the mean random
control is +2.414 [+1.835, +3.059] logits.

In every Medium cell/reference, the edited signature remains closer on average
to the matched clean recipient than to the matched clean donor. All eight
secondary pointwise distance-margin intervals are negative. Some donor-axis
movement is positive, with reference/direction dependence. This is evidence about
these measured conditional signatures; it is not evidence that the full circuit
is unchanged or that the donor has no causal influence.

Pythia wording transfer yields 86.7% top-1 and
91.9% pairwise accuracy. Its wording advantage
over random edits is +2.505 [+2.172, +2.929] logits. Its geometry
also favors the clean recipient on average in all eight cell/reference comparisons,
with negative secondary pointwise distance intervals. Every cell's recovery is
positive after the original adjustment. The comparison remains diagnostic because
selection, behavior and separability requirements fail;
the full per-cell outcomes are retained in its report rather than pooled into
Medium's inference. The order/role factorial is secondary in this broader arm.

![Medium repair and donor proximity](results/gpt2-medium/report/repair-and-proximity.png)

[Medium's complete broader results](results/gpt2-medium/report/RESULTS.md) ·
[Pythia's complete diagnostic results](results/pythia-410m-deduped/report/RESULTS.md)

## 4. The strict switch gate has an endpoint-calibration limitation

Both frozen analyses report no all-cell wording switch. That output cannot stand
alone as evidence against circuit transfer: a post hoc calibration rejects the
actual clean donor in both models because of its off-axis displacement. The original
gate therefore does not reliably recognize its own intended endpoint on this test.
Its failure is an assay limitation and remains disclosed in the record.

A separate post hoc paired-axis diagnostic has exact clean-recipient/donor
endpoints 0/1. For Medium its cell/reference mean projections range from 0.001
to 0.380, with 2.1–32.3% of examples closer to the donor. Those numbers are
descriptive, use held-out clean references, and do not replace the primary tests.
They are another view of the same saved measurements, not independent evidence.

[Dated diagnostic rationale](docs/GEOMETRY-CALIBRATION-NOTE.md) ·
[Data and estimand definitions](docs/DATA-AND-ESTIMANDS.md)

## 5. Publication assessment

The results support drafting a narrower empirical TMLR manuscript about what
behavioral transfer establishes under a specified dependency assay. They do not
justify a broad circuit-transfer claim across architectures, a claim that all
dependencies are preserved, or discovery of nonadditivity/self-repair itself.

The contribution must be articulated against closely related work, particularly
[Vaidyanathan et al. (2026)](https://arxiv.org/abs/2606.27510). See the
[contribution audit](docs/RELATED-WORK.md). [TMLR's criteria](https://jmlr.org/tmlr/acceptance-criteria.html) permit useful empirical and
reproducibility studies, but accurate evidence and a clear lesson remain necessary.
The completed run is not an acceptance prediction or a finished submission.

Before submission, integrate the manuscripts, review the causal/metric definitions,
and make the primary/secondary/post hoc distinctions explicit. If a revised switch
criterion is central to the claim, validate it prospectively on new data; do not
silently replace the failed criterion on this test. A manuscript about the measured
distances and the assay limitation can describe the current evidence as it stands.

## Verification and scope

The three new test audits check 25 raw test bundles in total (3 + 11 + 11),
42 exactly replayed metric arrays (4 + 19 + 19), exact summary bytes, clamp algebra,
geometry, factorial contrasts and preservation of 700 tracked earlier files.
Screen/discovery outputs and their freezes are also retained. Four structural
tests and source checks pass. The package audit additionally replays the post hoc
diagnostics and both Bridge 002 analyses from extracted files.

Pythia's original-model calibration had zero maximum logit discrepancy on the
recorded prompts. Its vectorized intervention-equivalence tolerance was disclosed
as 0.002 logits before screening, with maximum observed error 0.001139; Medium
kept 0.0005 and reached 0.00000954. This is sampled numerical validation, not a
proof for every input or hook. All checkpoints/config/tokenizer files are hashed.

The bootstrap has 10,000 crossed block/triple draws and conditions on fixed
discovery fits. Only eight test blocks and six triples, two small pretrained
architectures, selected grammatical skeletons, single-head END signatures and
two intervention references limit generalization. The new names exclude Bridge
001/002 names, not pretraining or every historical source study.

[Reproduction instructions](REPRODUCIBILITY.md) ·
[Current combined-project guide](../../FOLLOWUP-003.md)
