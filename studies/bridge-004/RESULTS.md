# Bridge 004 results

8 September 2026. Completed fresh measurements; all prior results and failures retained.

This study tests paired-endpoint calibration, a new GPT-2 Medium replication, fixed downstream groups on the same wording pair, and the public TinyLlama 1.1B base model. TinyLlama is a separately trained Llama-architecture model, not Meta Llama 3.2; the latter required unavailable gated access.

## Qualification and effects

| Model | Discovery separation | Test behavior | Test separation | Positive primary recovery cells | Recipient-favoring primary margins | Full conjunction |
| --- | --- | --- | --- | --- | --- | --- |
| gpt2-medium | Fail | Pass | Fail | 4/4 | 8/8 | Fail |
| tinyllama-1.1b-base | Pass | Pass | Fail | 4/4 | 4/8 | Fail |

The primary intervals use one 24-quantity Bonferroni family across both models. Qualification failures prevent the full claim even if effect intervals support recovery or relative proximity.

## gpt2-medium

Selected families: [1, 2]. Source heads: [[6, 4], [6, 15], [7, 2], [7, 4], [7, 11], [8, 15], [9, 3], [10, 1], [10, 12], [11, 6], [11, 14], [12, 14]]. Test configurations: 384.

Wording recovery: **2.083 [1.598, 2.505] logits** (pointwise 95% interval). Top-1: **28.6% → 97.4%**. Wording minus the average random control: 2.080 [1.595, 2.504].

| Cell | Clean pairwise | Clean top-1 | Zero / mean classification |
| --- | --- | --- | --- |
| w0_o0 | 100.0% | 100.0% | 99.0% / 95.8% |
| w0_o1 | 100.0% | 100.0% | 99.0% / 96.9% |
| w1_o0 | 100.0% | 100.0% | 69.8% / 90.6% |
| w1_o1 | 100.0% | 100.0% | 89.6% / 97.9% |

| Cell | Adjusted recovery | Adjusted zero margin | Adjusted mean margin |
| --- | --- | --- | --- |
| w0_o0 | 1.476 [0.757, 2.144] | -0.234 [-0.388, -0.103] | -0.235 [-0.364, -0.125] |
| w0_o1 | 2.616 [1.935, 3.313] | -0.255 [-0.337, -0.159] | -0.302 [-0.373, -0.234] |
| w1_o0 | 1.517 [0.756, 2.150] | -0.204 [-0.333, -0.071] | -0.235 [-0.324, -0.141] |
| w1_o1 | 2.721 [1.891, 3.303] | -0.122 [-0.195, -0.059] | -0.097 [-0.165, -0.023] |

Negative margins favor the recipient. They are relative distances in measured coordinates, not evidence of endpoint equality or a complete circuit.

## tinyllama-1.1b-base

Selected families: [2, 3]. Source heads: [[8, 12], [8, 13], [10, 26], [11, 13], [14, 10], [15, 1], [15, 18], [15, 19]]. Test configurations: 384.

Wording recovery: **1.476 [1.204, 1.820] logits** (pointwise 95% interval). Top-1: **40.4% → 74.2%**. Wording minus the average random control: 1.473 [1.201, 1.821].

| Cell | Clean pairwise | Clean top-1 | Zero / mean classification |
| --- | --- | --- | --- |
| w0_o0 | 100.0% | 99.0% | 94.8% / 87.5% |
| w0_o1 | 100.0% | 100.0% | 77.1% / 81.2% |
| w1_o0 | 100.0% | 88.5% | 93.8% / 91.7% |
| w1_o1 | 100.0% | 87.5% | 100.0% / 100.0% |

| Cell | Adjusted recovery | Adjusted zero margin | Adjusted mean margin |
| --- | --- | --- | --- |
| w0_o0 | 1.483 [0.999, 2.103] | 0.128 [-0.088, 0.393] | 0.074 [-0.157, 0.396] |
| w0_o1 | 1.613 [1.137, 2.446] | 0.189 [-0.007, 0.428] | 0.098 [-0.121, 0.365] |
| w1_o0 | 1.140 [0.784, 1.611] | -0.467 [-0.633, -0.280] | -0.409 [-0.616, -0.227] |
| w1_o1 | 1.670 [1.224, 2.282] | -0.474 [-0.614, -0.334] | -0.388 [-0.561, -0.236] |

Negative margins favor the recipient. They are relative distances in measured coordinates, not evidence of endpoint equality or a complete circuit.

## Direction and measurement sensitivity

TinyLlama has four adjusted margins favoring the recipient and four spanning zero with donor-favoring point estimates. Uniform recipient proximity does not replicate in this architecture extension. TinyLlama uses F2/F3; Medium uses F1/F2, so their difference cannot be attributed exclusively to architecture.

The distance conclusions also depend on coordinate weighting. Under unscaled paired Euclidean distance, pointwise intervals give five recipient-favoring, two donor-favoring and one inconclusive Medium cell; TinyLlama gives four recipient-favoring and four donor-favoring cells. These secondary intervals are not the adjusted primary family. The result is conditional on the declared standardization, not a metric-independent property of the mechanism.

Editing can reduce distance to both clean endpoints while retaining greater recipient proximity. Mean edited distances can still exceed the donor–recipient separation. Relative proximity therefore does not mean endpoint matching, absence of donor influence or an unchanged complete mechanism. See the post hoc baseline-geometry records.

## Same pair Medium group tests

The groups and controls were fixed from the older handed study, then tested on F1/F2 in new cases. Adjusted primary intervals use one 20-quantity family. Order values below use the inherited −1 orientation.

| Factor | A | B | Union | All opposing/control requirements | Union equivalence ±0.2 |
| --- | --- | --- | --- | --- | --- |
| order | 0.772 [0.544, 0.984] | -1.226 [-1.454, -0.999] | 0.064 [-0.180, 0.281] | Pass | Fail |
| wording | 0.808 [0.563, 1.055] | -1.308 [-1.616, -1.047] | -0.006 [-0.215, 0.179] | Pass | Fail |

Failure of equivalence does not prove non-equivalence. Joint nonadditivity and the four-run interaction are distinct contrasts.

| Factor / group | Noising N | Denoising P | Interaction N−P |
| --- | --- | --- | --- |
| order / A | -0.772 [-0.916, -0.618] | -0.640 [-0.811, -0.473] | -0.132 [-0.224, -0.046] |
| order / B | 1.226 [1.075, 1.378] | 1.258 [1.092, 1.423] | -0.031 [-0.106, 0.038] |
| order / union | -0.064 [-0.211, 0.095] | 0.026 [-0.214, 0.251] | -0.090 [-0.205, 0.037] |
| wording / A | 0.808 [0.642, 0.968] | 0.640 [0.464, 0.811] | 0.168 [0.051, 0.299] |
| wording / B | -1.308 [-1.507, -1.127] | -1.318 [-1.528, -1.132] | 0.010 [-0.071, 0.093] |
| wording / union | -0.006 [-0.143, 0.118] | -0.082 [-0.306, 0.163] | 0.075 [-0.071, 0.203] |

These decomposition intervals are pointwise 95% and preserve raw signs. They apply existing mediation contrasts to the upstream edit, without claiming exclusive paths.

## Calibration and record integrity

The old centroid switch gate remains failed and cannot reject full mechanism transfer. The new paired measurement passes algebraic endpoint checks and measured whole-boundary F1/F2 donor replay. Replay errors are zero in behavior and all measured signature coordinates for both models. The maximum intervention discrepancies are 0.0000066757 for Medium and 0.0000057220 for TinyLlama.

A documented analysis indexing correction preserves the original frozen code and changes no estimand or threshold. Read [the correction record](docs/ANALYSIS-CORRECTION.md). Saved-output replay, direct distance arithmetic and the four-run identities pass. All 810 preceding research snapshot files are unchanged.

## Contribution and review

The [related-work audit](docs/CONTRIBUTION-AUDIT.md) narrows the claim against circuit reuse, faithfulness sensitivity, statistical variance and mediator-interaction studies. The paired metric and decomposition are not claimed as new methods. The candidate contribution is this controlled empirical comparison and its boundary conditions.

This is a more complete work-in-progress record for feedback. Additional experiments do not by themselves establish journal readiness. A reader should judge whether the measured effects and failed qualification teach a sufficiently specific lesson, and review the conditional estimands before choosing further experiments or a TMLR submission framing.

## Supplemental controls

These controls were added after initial outcomes were known and remain explicitly post hoc. The nontrivial mean-bank endpoint replay passed in both models, with zero behavior/signature discrepancy. TinyLlama’s selected F2/F3 prompts have unequal length; its whole-boundary replay therefore uses equal-length F1/F2 calibration prompts, while semantic-position audits cover its selected test pair.

- gpt2-medium: wording minus the mean of three output-projected, norm-matched random controls is 2.073 [1.593, 2.493] logits, pointwise 95%. Each random draw’s own recovery interval spans zero. All 384 corrupted and wording behavior outputs independently replay with zero discrepancy.

- tinyllama-1.1b-base: wording minus the mean of three output-projected, norm-matched random controls is 1.480 [1.203, 1.824] logits, pointwise 95%. Each random draw’s own recovery interval spans zero. All 384 corrupted and wording behavior outputs independently replay with zero discrepancy.


Saved-array audits verify the supplemental hashes, norms, intervals and independent baseline-distance arithmetic. These checks do not constitute a new independent rerun of all downstream signature inference.
