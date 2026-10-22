# Fresh replication with independent clean qualification

9 September 2026. Bridge 005. Experimental addendum to draft manuscript v2; the manuscript being reviewed is unchanged.

**Outcome: the qualified replication supports the central result.** The final test was completed on 1,152 configurations after the selected pair passed separate clean qualification. All planned test conditions were retained.

The selected pair is F6/F7. Clean test top-1 is 93.1%; wording transfer changes corrupted top-1 from 35.8% to 86.7%. Pooled logit recovery is 2.125 [1.832, 2.431], with a pointwise 95% interval.

## Why this study was run

Bridge 004 preserved strong behavioral recovery but failed its full qualification in both new models. Medium’s lowest clean zero-reference classification was 69.8%; TinyLlama’s was 77.1%. This follow-up prioritizes an independent qualified comparison in Medium rather than adding another model. It is informed by those failures and is not an independent origin of the hypothesis.

## What the experiment does

The model completes short stories containing two people and an object. The expected answer names the person to whom the object is given or shown. Two wording families must both support reliable answers and distinguishable patterns of response to later-head ablations. A corruption replaces the repeated subject name with an unrelated third name. The transfer edit adds the donor wording’s clean-minus-corrupted signal from twelve fixed attention heads to the corrupted recipient.

The experiment then asks two separate questions: does that edit restore the correct answer, and is the resulting pattern of later-head ablation effects closer to the clean recipient or the clean donor? These measurements describe a specified intervention and distance metric. They do not identify every component of a circuit.

The selected wording skeletons are:

- **F6:** At the {place}, {a} and {b} met. {s} gave a {obj} to
- **F7:** While {a} and {b} were at the {place}, {s} showed a {obj} to

The final blank receives the answer name; `{s}` is one of the two earlier names. Since this pair differs from the earlier F1/F2 pair, this is a conceptual replication of the phenomenon.

## What was fixed before inference

The complete candidate set, tokenization-only names, contexts, selection rules, model and source heads, numerical tolerance, measurements, control distributions, inference family and stopping rules were locally frozen before new GPU inference. This is a local timing record, not public preregistration. Read [the protocol](docs/PROTOCOL.md).

The study uses 108 fresh names in six disjoint banks. Earlier named actors/corruptions and exact complete prompt strings were checked. Location/object pairs are new relative to Bridges 003–004 and disjoint across splits; some individual words and eight grammar skeletons recur. Two additional fixed grammar candidates were included. No name or prompt was filtered using model answers.

| Stage | Names | Context blocks | Configurations per family | Role |
| --- | --- | --- | --- | --- |
| calibration | 6 | 2 | 16 | Numerical checks only |
| screen | 12 | 3 | 48 | Clean behavioral eligibility |
| fit | 18 | 6 | 144 | Fit mean references, scales and centroids |
| selection | 18 | 6 | 144 | Choose one pair on fresh clean data |
| qualification | 18 | 6 | 144 | Independent clean gate for chosen pair |
| test | 36 | 12 | 576 | One final transfer test, only if qualified |

Mean-ablation references, coordinate scales and centroids use fit data only. Selection requires 90% classification in every family/order/reference cell; independent qualification and test retain the original 80% requirement. Every clean behavior gate requires 95% pairwise and 85% full-vocabulary top-1 accuracy in each order. Selection never uses a scientific upstream-transfer outcome.

## Screening

Eligible families: F0, F1, F2, F3, F4, F6, F7, F8. Aligned eligible pairs: 21. All ten candidates were evaluated; no additional candidates were introduced.

| Family | Order 0 pairwise / top-1 | Order 1 pairwise / top-1 | Behavioral eligibility |
| --- | --- | --- | --- |
| F0 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F1 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F2 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F3 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F4 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F5 | 95.8% / 4.2% | 100.0% / 4.2% | Fail |
| F6 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F7 | 100.0% / 100.0% | 100.0% / 100.0% | Pass |
| F8 | 100.0% / 91.7% | 100.0% / 95.8% | Pass |
| F9 | 100.0% / 58.3% | 100.0% / 41.7% | Fail |

## Pair selection

A separate selection bank evaluated references fitted on the fit bank. The table retains all behaviorally screened, token-aligned pairs. Eligibility requires both fit/selection behavior gates, 80% in-fit separation and 90% selection separation in every cell. Ties use mean selection classification, then family identifiers.

| Pair | Worst in-fit class | Worst selection class | Fit / selection behavior | All selection requirements |
| --- | --- | --- | --- | --- |
| F6/F7 **selected** | 95.8% | 90.3% | Pass / Pass | Pass |
| F4/F7 | 90.3% | 90.3% | Pass / Pass | Pass |
| F2/F8 | 95.8% | 87.5% | Pass / Pass | Fail |
| F2/F6 | 100.0% | 86.1% | Pass / Pass | Fail |
| F0/F7 | 91.7% | 86.1% | Pass / Pass | Fail |
| F0/F2 | 88.9% | 84.7% | Pass / Pass | Fail |
| F1/F6 | 95.8% | 83.3% | Pass / Pass | Fail |
| F2/F7 | 95.8% | 81.9% | Pass / Pass | Fail |
| F0/F4 | 91.7% | 81.9% | Pass / Pass | Fail |
| F1/F2 | 94.4% | 80.6% | Pass / Pass | Fail |
| F7/F8 | 93.1% | 79.2% | Pass / Pass | Fail |
| F6/F8 | 97.2% | 77.8% | Pass / Pass | Fail |
| F1/F7 | 86.1% | 76.4% | Pass / Pass | Fail |
| F0/F6 | 90.3% | 76.4% | Pass / Pass | Fail |
| F0/F8 | 91.7% | 75.0% | Pass / Pass | Fail |
| F0/F1 | 84.7% | 75.0% | Pass / Pass | Fail |
| F1/F4 | 88.9% | 73.6% | Pass / Pass | Fail |
| F4/F8 | 90.3% | 73.6% | Pass / Pass | Fail |
| F2/F4 | 94.4% | 72.2% | Pass / Pass | Fail |
| F1/F8 | 88.9% | 69.4% | Pass / Pass | Fail |
| F4/F6 | 90.3% | 66.7% | Pass / Pass | Fail |

## Independent clean qualification

Qualification: **Pass**. This bank was not used to choose the pair or refit any reference. It contains 288 configurations.

| Recipient family / order | Pairwise / top-1 | Zero / mean classification |
| --- | --- | --- |
| F6 / 0 | 100.0% / 100.0% | 100.0% / 100.0% |
| F6 / 1 | 100.0% / 97.2% | 100.0% / 95.8% |
| F7 / 0 | 98.6% / 88.9% | 97.2% / 100.0% |
| F7 / 1 | 100.0% / 94.4% | 84.7% / 90.3% |

## Final held-out outcomes

The final test crosses 12 context blocks and 12 name triples with two roles, orders and wordings. The 1,152 configurations share those units. All four recovery and eight paired-distance comparisons form one 12-quantity Bonferroni family, using 50,000 crossed-bootstrap draws. Negative distance margins favor the clean recipient.

Final behavior qualification: **Pass**. Final dependency separation: **Pass**. Unresolvable paired endpoints: **0**.

| Recipient family / order | Clean pairwise / top-1 | Zero / mean classification | Adjusted recovery |
| --- | --- | --- | --- |
| F6 / 0 | 100.0% / 93.1% | 99.3% / 99.3% | 1.682 [1.273, 2.153] |
| F6 / 1 | 100.0% / 94.1% | 97.2% / 95.1% | 3.324 [2.653, 4.028] |
| F7 / 0 | 100.0% / 92.7% | 97.6% / 99.7% | 1.093 [0.758, 1.443] |
| F7 / 1 | 100.0% / 92.7% | 99.7% / 99.7% | 2.399 [1.789, 3.072] |

| Recipient family / order | Adjusted zero margin | Adjusted mean margin |
| --- | --- | --- |
| F6 / 0 | -0.243 [-0.335, -0.144] | -0.344 [-0.430, -0.253] |
| F6 / 1 | -0.119 [-0.211, -0.032] | -0.274 [-0.358, -0.189] |
| F7 / 0 | -0.363 [-0.442, -0.277] | -0.308 [-0.391, -0.230] |
| F7 / 1 | -0.500 [-0.551, -0.443] | -0.410 [-0.474, -0.348] |

Adjusted distance intervals: 8/8 recipient-favoring, 0/8 donor-favoring, 0/8 inconclusive or unresolved. All adjusted recoveries positive: True. Full conjunction supported: True.

## Controls and sensitivities

All intervals in this section are secondary pointwise 95%, not additional primary confirmations. Both random families were declared before inference; projected vectors stay inside the source-head output span but are not matched to natural activations.

| Condition | Recovery in logits | Top-1 |
| --- | --- | --- |
| order | -1.913 [-2.178, -1.671] | 8.2% |
| order_role | -2.208 [-2.527, -1.906] | 4.6% |
| projected0 | 0.008 [-0.013, 0.028] | 36.9% |
| projected1 | 0.007 [-0.018, 0.030] | 35.9% |
| projected2 | 0.013 [-0.008, 0.035] | 36.8% |
| residual0 | 0.000 [-0.010, 0.011] | 35.2% |
| residual1 | 0.000 [-0.012, 0.014] | 35.7% |
| residual2 | 0.006 [-0.005, 0.017] | 35.7% |
| role | 1.913 [1.671, 2.178] | 80.4% |
| self | 2.208 [1.906, 2.527] | 83.8% |
| wording | 2.125 [1.832, 2.431] | 86.7% |
| zero | 0.000 [0.000, 0.000] | 35.8% |

Wording minus the average residual random control: 2.122 [1.830, 2.429] logits. Wording minus the average projected random control: 2.115 [1.824, 2.422] logits.

| Distance sensitivity | Recipient-favoring | Donor-favoring | Inconclusive / unresolved |
| --- | --- | --- | --- |
| centroid_normalized_margin | 8 | 0 | 0 |
| standardized_cosine_margin | 8 | 0 | 0 |
| unscaled_paired_margin | 8 | 0 | 0 |

These counts use pointwise intervals and must not be compared with the primary counts as though their adjustment were identical. Full per-cell intervals remain in the saved summary. Each conclusion concerns a cell mean, not every individual example. Any change in direction limits a metric-independent interpretation.

| Reference | Change in donor distance | Change in recipient distance |
| --- | --- | --- |
| fit_mean | -0.365 [-0.442, -0.292] | -0.580 [-0.679, -0.489] |
| zero | -0.321 [-0.383, -0.257] | -0.512 [-0.596, -0.433] |

Changes are measured from the corrupted starting state. An edit can move closer to both clean endpoints. Relative recipient proximity therefore does not mean no donor influence, actual recipient matching, or preservation of an entire mechanism.

View the [primary comparison figure](figures/primary-outcomes.png) and [all-pair selection figure](figures/selection-qualification.png).

## Verification and scope

Numerical validation maximum discrepancy: 0.00001526, below the unchanged 0.0005 tolerance. The structural checks cover all 10,720 candidate configurations. Saved-output and preservation verification passed; all 987 previous release files and both draft-v2 document hashes remain unchanged.

The actual selected-pair full-boundary replay covers both directions on 32 calibration configurations, with a 160-configuration mean bank. Both references have nonzero target effects. Maximum behavior/signature discrepancies: 0.00000000 / 0.00000000. This is a complete-state technical control, not evidence that the compact edit transfers a circuit.

A newly loaded engine repeats corrupted and wording behavior on all 1152 test configurations. A fixed 8-example subset also repeats every measured head with singleton ablation execution. Maximum errors: {'corrupt_behavior': 0.0, 'singleton_signature': 4.100799560546875e-05, 'wording_behavior': 0.0}. All pass the fixed tolerance. Saved-array analyses also reproduce exactly. These are computational checks, not new statistical replications.

The study conditions on one selected pair, one historical source set and one fitted reference. It does not estimate selection-pipeline variance, generalize to arbitrary language, or establish complete circuit identity. The new 12-quantity primary family belongs to this one-model study; Bridge 004’s 24-quantity family is unchanged. An unsuccessful stop or test remains part of the evidence.

## Implication for the manuscript

This supplies a new independently qualified replication: donor signals substantially improve answers while the measured dependency pattern remains closer, on average, to the clean recipient in every primary cell. All three prespecified alternative distance measures agree in this comparison, using secondary pointwise intervals. These results strengthen the specific empirical claim. The earlier failures and metric reversals remain part of the evidence, and complete circuit identity is not established. Whether the combined empirical lesson merits TMLR submission remains a scientific judgment for review.

The manuscript prepared for feedback has not been edited. Further experiments, if desired after feedback, require a separately justified design; this study must not be continued by relaxing its gates or replacing a failed selection.
