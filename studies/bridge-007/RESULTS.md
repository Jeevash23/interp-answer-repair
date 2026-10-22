# Bridge 007 results

Completed final test: the full prespecified conjunction is supported. The clean gates and four component decisions are reported separately below.

This study uses new GPT-2 Medium prompts and the frozen four-pair search. It was informed by earlier experiments, and its local prospective freeze is not public preregistration. The earlier self-versus-wording analysis remains post hoc.

## Behavioral screen and complete candidate disposition

**All four families at behavioral screening**

| Family / order | n | Pairwise | Top-1 | Family decision |
| --- | --- | --- | --- | --- |
| F10 / 0 | 24 | 100.0% | 100.0% | eligible |
| F10 / 1 | 24 | 100.0% | 100.0% | eligible |
| F11 / 0 | 24 | 100.0% | 100.0% | eligible |
| F11 / 1 | 24 | 100.0% | 100.0% | eligible |
| F12 / 0 | 24 | 100.0% | 100.0% | eligible |
| F12 / 1 | 24 | 100.0% | 100.0% | eligible |
| F13 / 0 | 24 | 100.0% | 100.0% | eligible |
| F13 / 1 | 24 | 100.0% | 100.0% | eligible |

Each family must reach 95% pairwise and 85% top-1 accuracy in both orders.

**Complete fixed pair budget and observed clean-selection scores**

| Pair | Worst fitting cell | Worst selection cell | Mean selection cell | Decision |
| --- | --- | --- | --- | --- |
| F10/F11 | 97.7% | 96.9% | 99.0% | selected |
| F10/F13 | 98.4% | 91.4% | 97.9% | eligible |
| F11/F12 | 97.7% | 93.0% | 96.9% | eligible |
| F12/F13 | 93.8% | 71.9% | 91.4% | ineligible |

Unmeasured entries identify actual screen exclusions. Eligibility additionally requires fit/selection behavior gates; separation thresholds are 80% fitting and 90% selection. Ranking uses worst selection, then mean selection, then family IDs.

**Behavior gates and full eligibility for measured pairs**

| Pair | Fit behavior | Selection behavior | All eligibility requirements |
| --- | --- | --- | --- |
| F10/F11 | pass | pass | pass |
| F10/F13 | pass | pass | pass |
| F11/F12 | pass | pass | pass |
| F12/F13 | pass | pass | fail |

## Independent qualification

The fixed ranking selected F10/F11 before independent qualification. No reference was refitted on qualification observations and no replacement pair was selected.

**Independent clean qualification**

| Family / order | Pairwise | Top-1 | Classification, zero | Classification, fit mean |
| --- | --- | --- | --- | --- |
| F10 / 0 | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 97.9% [93.1%, 100.0%] | 98.6% [95.1%, 100.0%] |
| F10 / 1 | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 91.3% [81.6%, 98.6%] | 91.7% [81.2%, 99.7%] |
| F11 / 0 | 100.0% [100.0%, 100.0%] | 93.1% [77.1%, 100.0%] | 98.3% [94.1%, 100.0%] | 99.7% [97.9%, 100.0%] |
| F11 / 1 | 100.0% [100.0%, 100.0%] | 94.8% [82.3%, 100.0%] | 100.0% [100.0%, 100.0%] | 95.5% [87.8%, 100.0%] |

Observed rates with pointwise 95% crossed-bootstrap intervals. Gates apply to observed rates, not interval bounds.

Independent clean qualification: pass. Behavior gate: pass; clean separation gate: pass; unresolved reference/example endpoints: 0. Sampling: 12 context blocks by 12 name triples, 1152 configurations.

## Final clean checks and all primary outcomes

**Final-bank clean qualification**

| Family / order | Pairwise | Top-1 | Classification, zero | Classification, fit mean |
| --- | --- | --- | --- | --- |
| F10 / 0 | 100.0% [100.0%, 100.0%] | 99.9% [99.2%, 100.0%] | 98.0% [94.8%, 100.0%] | 99.1% [97.2%, 100.0%] |
| F10 / 1 | 100.0% [100.0%, 100.0%] | 99.9% [99.2%, 100.0%] | 96.0% [90.4%, 99.8%] | 94.1% [86.6%, 99.8%] |
| F11 / 0 | 99.9% [99.2%, 100.0%] | 93.4% [85.0%, 99.5%] | 99.9% [99.2%, 100.0%] | 100.0% [100.0%, 100.0%] |
| F11 / 1 | 100.0% [100.0%, 100.0%] | 94.2% [85.5%, 100.0%] | 99.9% [99.2%, 100.0%] | 98.9% [96.9%, 100.0%] |

Observed rates with pointwise 95% crossed-bootstrap intervals. Gates apply to observed rates, not interval bounds.

Final-bank clean qualification: pass. Behavior gate: pass; clean separation gate: pass; unresolved reference/example endpoints: 0. Sampling: 20 context blocks by 20 name triples, 3200 configurations.

The final test contains 3,200 configurations from 20 context blocks and 20 disjoint name triples. All sixteen primary quantities use 50,000 crossed-bootstrap draws and one Bonferroni family (tails 0.05/(2*16)). The next two tables show adjusted intervals; all other intervals in this report are pointwise 95% unless stated.

**Twelve primary recovery and recipient-proximity quantities**

| Family / order | Recovery (logits) | Margin, zero | Margin, fit mean |
| --- | --- | --- | --- |
| F10 / 0 | 1.854 [1.571, 2.170] | -0.309 [-0.380, -0.237] | -0.387 [-0.449, -0.327] |
| F10 / 1 | 3.591 [2.971, 4.293] | -0.211 [-0.275, -0.142] | -0.310 [-0.376, -0.238] |
| F11 / 0 | 1.103 [0.875, 1.359] | -0.255 [-0.343, -0.164] | -0.249 [-0.325, -0.168] |
| F11 / 1 | 2.780 [2.164, 3.448] | -0.299 [-0.372, -0.229] | -0.230 [-0.297, -0.166] |

Recovery lower bounds must exceed zero. Margin upper bounds must be below zero; negative margins favor the recipient.

**Four prospective wording-versus-self average-bound quantities**

| Quantity | Reference | Mean [adjusted interval] | Criterion | Outcome |
| --- | --- | --- | --- | --- |
| Mean absolute axis difference | zero | 0.027 [0.021, 0.035] | upper < 0.10 | supported |
| Mean absolute axis difference | fit mean | 0.034 [0.026, 0.044] | upper < 0.10 | supported |
| Mean full signature discrepancy | zero | 0.159 [0.137, 0.182] | upper < 0.50 | supported |
| Mean full signature discrepancy | fit mean | 0.174 [0.150, 0.200] | upper < 0.50 | supported |

Distances use matched clean-wording separation units. Absolute differences are taken before averaging; these are pooled average bounds, not all-cell equivalence.

**Separate component decisions**

| Component | Outcome |
| --- | --- |
| All four recovery intervals positive | supported |
| All eight margins favor recipient | supported |
| Both absolute-axis bounds supported | supported |
| Both full-discrepancy bounds supported | supported |
| All clean gates and resolvable endpoints | pass |
| Complete prespecified conjunction | supported |

## All-cell wording-versus-self heterogeneity

**Every family/order cell under both references**

| Family / order | Reference | Signed axis | Absolute axis | Full discrepancy | Recovery difference (logits) |
| --- | --- | --- | --- | --- | --- |
| F10 / 0 | zero | 0.008 [0.000, 0.016] | 0.021 [0.017, 0.026] | 0.101 [0.091, 0.111] | 0.009 [-0.046, 0.063] |
| F10 / 1 | zero | 0.033 [0.018, 0.050] | 0.046 [0.034, 0.059] | 0.241 [0.209, 0.277] | -0.460 [-0.575, -0.347] |
| F11 / 0 | zero | 0.009 [0.005, 0.013] | 0.014 [0.012, 0.016] | 0.095 [0.086, 0.104] | -0.019 [-0.056, 0.019] |
| F11 / 1 | zero | 0.010 [0.001, 0.020] | 0.029 [0.023, 0.035] | 0.198 [0.174, 0.223] | 0.143 [0.035, 0.245] |
| F10 / 0 | fit mean | 0.009 [0.002, 0.016] | 0.021 [0.017, 0.026] | 0.105 [0.095, 0.116] | 0.009 [-0.046, 0.063] |
| F10 / 1 | fit mean | 0.058 [0.041, 0.078] | 0.063 [0.048, 0.082] | 0.273 [0.237, 0.314] | -0.460 [-0.575, -0.347] |
| F11 / 0 | fit mean | 0.010 [0.006, 0.015] | 0.016 [0.014, 0.019] | 0.097 [0.088, 0.107] | -0.019 [-0.056, 0.019] |
| F11 / 1 | fit mean | 0.028 [0.018, 0.039] | 0.037 [0.030, 0.045] | 0.219 [0.193, 0.246] | 0.143 [0.035, 0.245] |

Wording minus self, except that absolute and full discrepancies are nonnegative. All intervals are pointwise 95%.

## Behavioral controls and paired comparisons

**All declared behavioral conditions**

| Condition | Logit recovery | Top-1 | Pairwise |
| --- | --- | --- | --- |
| clean | 4.601 [4.354, 4.854] | 96.8% [92.6%, 99.8%] | 100.0% [99.8%, 100.0%] |
| corrupt | 0.000 [0.000, 0.000] | 33.9% [29.8%, 38.2%] | 50.0% [50.0%, 50.0%] |
| wording | 2.332 [2.072, 2.608] | 90.7% [84.1%, 96.3%] | 95.1% [91.1%, 98.3%] |
| self | 2.414 [2.157, 2.685] | 90.4% [83.7%, 96.1%] | 95.3% [91.6%, 98.3%] |
| zero | 0.000 [0.000, 0.000] | 33.9% [29.8%, 38.2%] | 50.0% [50.0%, 50.0%] |
| order | -2.244 [-2.501, -2.001] | 3.9% [1.0%, 8.0%] | 5.0% [1.8%, 9.3%] |
| role | 2.244 [2.001, 2.501] | 89.9% [82.8%, 95.9%] | 95.0% [90.7%, 98.2%] |
| order_role | -2.414 [-2.685, -2.157] | 3.9% [1.2%, 7.5%] | 4.7% [1.7%, 8.4%] |
| residual0 | 0.004 [-0.003, 0.011] | 33.9% [29.7%, 38.1%] | 49.6% [48.6%, 50.5%] |
| residual1 | 0.001 [-0.007, 0.009] | 34.1% [29.9%, 38.5%] | 50.1% [49.3%, 50.9%] |
| residual2 | 0.001 [-0.006, 0.009] | 33.6% [29.2%, 38.1%] | 49.8% [49.0%, 50.7%] |
| projected0 | 0.006 [-0.007, 0.020] | 34.4% [30.3%, 38.8%] | 50.2% [49.0%, 51.3%] |
| projected1 | 0.009 [-0.007, 0.027] | 34.9% [30.4%, 39.3%] | 50.2% [49.0%, 51.5%] |
| projected2 | 0.001 [-0.011, 0.014] | 34.0% [29.6%, 38.4%] | 50.0% [48.6%, 51.3%] |

Recovery is relative to corrupted behavior. Pointwise 95% intervals; random controls have three fixed draws of each kind. Order/role symmetries are not independent corroborations.

**Paired wording-minus-control differences**

| Comparator | Logit difference | Top-1 difference |
| --- | --- | --- |
| order | 4.576 [4.075, 5.106] | 86.8% [77.9%, 94.5%] |
| order_role | 4.746 [4.229, 5.292] | 86.8% [78.3%, 94.3%] |
| projected0 | 2.326 [2.068, 2.600] | 56.3% [49.8%, 62.2%] |
| projected1 | 2.322 [2.065, 2.595] | 55.8% [49.5%, 61.9%] |
| projected2 | 2.330 [2.072, 2.604] | 56.7% [50.1%, 62.8%] |
| projected_mean | 2.326 [2.069, 2.599] | 56.3% [49.9%, 62.2%] |
| residual0 | 2.328 [2.067, 2.606] | 56.8% [50.7%, 62.6%] |
| residual1 | 2.331 [2.071, 2.607] | 56.6% [50.3%, 62.4%] |
| residual2 | 2.331 [2.068, 2.610] | 57.1% [50.6%, 63.2%] |
| residual_mean | 2.330 [2.069, 2.607] | 56.9% [50.6%, 62.7%] |
| role | 0.088 [0.036, 0.144] | 0.8% [-0.5%, 2.2%] |
| self | -0.082 [-0.102, -0.063] | 0.3% [-0.5%, 1.1%] |
| zero | 2.332 [2.072, 2.608] | 56.8% [50.4%, 62.7%] |

Top-1 differences are percentage points. The two mean controls average their three fixed draws within configuration before comparison; intervals are pointwise 95%.

**Fraction of the clean logit gap restored**

| Condition | Restored fraction |
| --- | --- |
| wording | 0.507 [0.455, 0.560] |
| self | 0.525 [0.474, 0.577] |

Ratio of mean recovery to mean clean-minus-corrupted logit gap; paired bootstrap ratio intervals.

## Secondary distance and movement checks

**Three declared alternative distance margins**

| Metric | Reference | Wording transfer | Self transfer |
| --- | --- | --- | --- |
| unscaled paired l2 | zero | -0.230 [-0.251, -0.208] | -0.241 [-0.263, -0.217] |
| unscaled paired l2 | fit mean | -0.255 [-0.278, -0.231] | -0.278 [-0.303, -0.252] |
| standardized cosine | zero | -0.119 [-0.132, -0.105] | -0.124 [-0.137, -0.109] |
| standardized cosine | fit mean | -0.142 [-0.160, -0.122] | -0.151 [-0.170, -0.129] |
| centroid normalized l2 | zero | -0.406 [-0.443, -0.369] | -0.426 [-0.467, -0.386] |
| centroid normalized l2 | fit mean | -0.433 [-0.476, -0.391] | -0.469 [-0.515, -0.424] |

Pooled pointwise 95% intervals; negative margins favor the recipient. Per-cell results remain in the secondary summary. These sensitivities do not replace the primary metric.

**Movement relative to corruption and full recipient repair**

| Quantity | Reference | Wording transfer | Self transfer |
| --- | --- | --- | --- |
| progress toward recipient | zero | 0.460 [0.420, 0.500] | 0.465 [0.427, 0.503] |
| progress toward recipient | fit mean | 0.512 [0.468, 0.556] | 0.523 [0.482, 0.565] |
| recipient distance closed | zero | 0.296 [0.271, 0.321] | 0.303 [0.278, 0.329] |
| recipient distance closed | fit mean | 0.320 [0.292, 0.350] | 0.331 [0.302, 0.361] |
| cosine with ideal repair | zero | 0.712 [0.686, 0.736] | 0.717 [0.692, 0.742] |
| cosine with ideal repair | fit mean | 0.725 [0.698, 0.751] | 0.733 [0.706, 0.758] |
| perpendicular fraction | zero | 0.941 [0.932, 0.951] | 0.938 [0.928, 0.948] |
| perpendicular fraction | fit mean | 0.945 [0.936, 0.955] | 0.941 [0.931, 0.951] |

Pooled pointwise 95% intervals. The secondary summary also contains the ideal-repair perpendicular baseline and all cellwise values.

**Actual injected wording/self differences**

| Quantity | Pooled estimate [pointwise 95% interval] |
| --- | --- |
| self norm | 37.788 [37.017, 38.570] |
| wording norm | 37.788 [37.017, 38.570] |
| relative difference | 0.388 [0.374, 0.402] |
| cosine | 0.925 [0.920, 0.931] |

Norms flatten the twelve source-head output differences; relative difference is divided by the self-delta norm.

## Interpretation and generality limits

The average bounds describe this assay's specified resolutions. They do not establish every-cell or every-example similarity, exact signature identity, equivalent algorithms, or absence of donor influence. Failure to establish a bound is not automatically evidence that the population discrepancy exceeds it.

The evidence concerns one GPT-2 Medium checkpoint, a finite synthetic wording generator, fixed historical source heads and fitting references. It is not a replication in another model or an estimate of selection-pipeline uncertainty. Context blocks and disjoint name triples are the crossed sampling dimensions; paired roles, orders and control draws do not supply additional independent samples.

## Verification and provenance

Independent saved-observation verification passed for the final_test_completed branch at 2026-10-04T20:09:12.920340+00:00. It checked the stopping branch, unchanged freezes, measurement records, numerical calibration and the applicable selection/qualification computations. For a final test it independently reproduces all sixteen primary intervals and decisions; secondary calculations are separately identified in their summary and implementation freeze.

This report only formats saved summaries. Its source-summary hashes and the report-helper hash are recorded in results/report-manifest.json. No model inference, resampling, source selection or statistical fitting occurs here.
