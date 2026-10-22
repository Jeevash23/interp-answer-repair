# Bridge 008 — source coverage and donor-axis sensitivity

The specified four-contrast donorward-increase criterion is not met. Raw and norm-matched outcomes are reported separately below. An interval spanning zero is unresolved, not evidence of invariance or equivalent computations.

This correlated explanatory follow-up reuses Bridge 007's qualified F10/F11 benchmark: 3,200 configurations from 20 context blocks and 20 disjoint name triples. All 112 broad source heads occupy the same seven S2 layers (6–12) as the twelve narrow heads. Fitted references, clean endpoints, corruption and downstream measurement heads are unchanged.

For each configuration, D is wording-transfer donor-axis position minus self-transfer position. The interaction is D at broad or matched-broad coverage minus D at narrow coverage. Positive values mean an increased donorward differential. The primary intervals use 50,000 paired crossed-bootstrap draws with seed 2026100408 and a four-quantity Bonferroni family. The criterion requires all four adjusted lower bounds above zero.

Clean top-1 accuracy is 96.8%; corrupted top-1 is 33.9%.

## Primary contrasts

Four primary paired source-coverage interactions. Intervals use the fixed four-quantity Bonferroni family.

| Coverage | Reference | Interaction [adjusted CI] | Decision |
| --- | --- | --- | --- |
| Broad | Zero | 0.016 [0.009, 0.023] | Increase |
| Broad | Fit mean | 0.018 [0.009, 0.027] | Increase |
| Matched broad | Zero | 0.007 [0.001, 0.013] | Increase |
| Matched broad | Fit mean | 0.006 [-0.001, 0.013] | Unresolved |

## Every family/order cell

All sixteen prespecified cell interactions, with pointwise 95% intervals. Pooled effects need not hold in every cell.

| Coverage | Reference | Recipient / order | Interaction [95% CI] |
| --- | --- | --- | --- |
| Broad | Zero | F10 / 0 | 0.014 [0.008, 0.020] |
| Broad | Zero | F10 / 1 | 0.024 [0.012, 0.036] |
| Broad | Zero | F11 / 0 | 0.012 [0.005, 0.018] |
| Broad | Zero | F11 / 1 | 0.012 [0.003, 0.022] |
| Broad | Fit mean | F10 / 0 | 0.017 [0.012, 0.023] |
| Broad | Fit mean | F10 / 1 | 0.031 [0.015, 0.047] |
| Broad | Fit mean | F11 / 0 | 0.009 [0.002, 0.016] |
| Broad | Fit mean | F11 / 1 | 0.013 [0.001, 0.026] |
| Matched broad | Zero | F10 / 0 | 0.005 [-0.001, 0.010] |
| Matched broad | Zero | F10 / 1 | 0.022 [0.009, 0.035] |
| Matched broad | Zero | F11 / 0 | 0.004 [-0.001, 0.008] |
| Matched broad | Zero | F11 / 1 | -0.003 [-0.012, 0.006] |
| Matched broad | Fit mean | F10 / 0 | 0.004 [-0.001, 0.009] |
| Matched broad | Fit mean | F10 / 1 | 0.025 [0.009, 0.041] |
| Matched broad | Fit mean | F11 / 0 | 0.001 [-0.004, 0.005] |
| Matched broad | Fit mean | F11 / 1 | -0.007 [-0.016, 0.002] |

## Behavioral recovery

Recovery in IO-minus-subject logits, with pointwise 95% intervals. Accuracy uses the same retained configurations.

| Coverage / donor | Recovery [95% CI] | Top-1 | Pairwise |
| --- | --- | --- | --- |
| Narrow / wording | 2.332 [2.071, 2.606] | 90.7% | 95.1% |
| Narrow / self | 2.414 [2.156, 2.683] | 90.4% | 95.3% |
| Broad / wording | 2.439 [2.149, 2.745] | 92.4% | 96.2% |
| Broad / self | 2.517 [2.233, 2.813] | 92.5% | 96.7% |
| Matched broad / wording | 1.873 [1.607, 2.155] | 87.7% | 92.1% |
| Matched broad / self | 1.942 [1.677, 2.220] | 87.0% | 92.8% |

## Signed, absolute and full signature differences

Wording-versus-self differences at each coverage level, in clean-separation units. All intervals are pointwise 95%.

| Coverage | Reference | Signed ΔS [95% CI] | Mean \|ΔS\| [95% CI] | Full distance [95% CI] |
| --- | --- | --- | --- | --- |
| Narrow | Zero | 0.015 [0.009, 0.021] | 0.027 [0.023, 0.032] | 0.159 [0.144, 0.174] |
| Narrow | Fit mean | 0.026 [0.020, 0.033] | 0.034 [0.029, 0.041] | 0.174 [0.157, 0.191] |
| Broad | Zero | 0.030 [0.022, 0.040] | 0.042 [0.036, 0.049] | 0.242 [0.223, 0.262] |
| Broad | Fit mean | 0.044 [0.033, 0.057] | 0.052 [0.043, 0.063] | 0.263 [0.241, 0.288] |
| Matched broad | Zero | 0.022 [0.014, 0.030] | 0.037 [0.032, 0.044] | 0.204 [0.186, 0.223] |
| Matched broad | Fit mean | 0.032 [0.024, 0.041] | 0.043 [0.036, 0.051] | 0.220 [0.199, 0.242] |

## Distances to clean endpoints

Edited distances from the clean recipient (qR) and donor (qD), in clean-separation units; pointwise 95% intervals.

| Coverage / donor | Reference | qR [95% CI] | qD [95% CI] |
| --- | --- | --- | --- |
| Narrow / wording | Zero | 1.077 [1.024, 1.130] | 1.345 [1.309, 1.382] |
| Narrow / wording | Fit mean | 0.965 [0.916, 1.014] | 1.259 [1.226, 1.293] |
| Narrow / self | Zero | 1.064 [1.011, 1.118] | 1.346 [1.311, 1.382] |
| Narrow / self | Fit mean | 0.947 [0.898, 0.996] | 1.267 [1.234, 1.300] |
| Broad / wording | Zero | 1.088 [1.034, 1.143] | 1.346 [1.308, 1.385] |
| Broad / wording | Fit mean | 0.982 [0.931, 1.034] | 1.261 [1.227, 1.298] |
| Broad / self | Zero | 1.068 [1.013, 1.124] | 1.353 [1.317, 1.390] |
| Broad / self | Fit mean | 0.954 [0.902, 1.007] | 1.274 [1.241, 1.309] |
| Matched broad / wording | Zero | 1.175 [1.120, 1.233] | 1.404 [1.363, 1.445] |
| Matched broad / wording | Fit mean | 1.079 [1.027, 1.133] | 1.328 [1.291, 1.367] |
| Matched broad / self | Zero | 1.161 [1.104, 1.220] | 1.406 [1.367, 1.447] |
| Matched broad / self | Fit mean | 1.061 [1.007, 1.117] | 1.336 [1.299, 1.375] |

## Layerwise dose and matching diagnostics

Layerwise dose diagnostics. Norms are averages of the summed residual-vector norms; alpha is the matched-broad scale, with its observed range.

| Donor | Layer | Narrow norm | Broad norm | Alpha range | Undefined |
| --- | --- | --- | --- | --- | --- |
| wording | 6 | 17.194 | 21.383 | 0.466 to 1.068 | 0 |
| wording | 7 | 19.478 | 20.536 | 0.713 to 1.040 | 0 |
| wording | 8 | 10.164 | 16.626 | 0.119 to 0.923 | 0 |
| wording | 9 | 16.996 | 18.726 | 0.219 to 1.365 | 0 |
| wording | 10 | 12.518 | 18.450 | 0.427 to 0.974 | 0 |
| wording | 11 | 12.200 | 15.823 | 0.219 to 1.055 | 0 |
| wording | 12 | 2.403 | 14.139 | 0.027 to 0.567 | 0 |
| self | 6 | 17.194 | 21.383 | 0.466 to 1.068 | 0 |
| self | 7 | 19.478 | 20.536 | 0.713 to 1.040 | 0 |
| self | 8 | 10.164 | 16.626 | 0.119 to 0.923 | 0 |
| self | 9 | 16.996 | 18.726 | 0.219 to 1.365 | 0 |
| self | 10 | 12.518 | 18.450 | 0.427 to 0.974 | 0 |
| self | 11 | 12.200 | 15.823 | 0.219 to 1.055 | 0 |
| self | 12 | 2.403 | 14.139 | 0.027 to 0.567 | 0 |

wording: 0 undefined layer matches across 0 configurations; 0 near-zero narrow and 0 near-zero broad layer norms; self: 0 undefined layer matches across 0 configurations; 0 near-zero narrow and 0 near-zero broad layer norms.

Matching uses the actual summed residual vector in each layer, with float64 normalization and float32 injection. The norm check allows 1e-6 absolute plus 1e-5 relative error. Undefined matching makes the entire corresponding matched condition unavailable; no configuration is omitted and the primary family remains four.

## Interpretation and evidence

The new interventions were specified before their measurement, but this is an outcome-informed follow-up on an inspected benchmark, not independent replication. Uncertainty conditions on the benchmark, fixed references and selected hypotheses. Even positive raw and norm-controlled interactions establish only coverage-dependent donor-axis sensitivity under this assay. They do not establish donor algorithm transfer or a complete information bottleneck. A null signed interaction does not establish scope invariance: opposite effects can cancel, and donor wording information shared by clean and corrupted source states can cancel in the injected difference. Recovery and endpoint distances must be considered alongside the signature contrast; disrupted examples are retained.

All secondary cell estimates, configuration distributions, recovery changes relative to narrow, and endpoint-distance changes are retained in [the complete analysis](results/analysis/summary.json). The [per-example arrays](results/analysis/per-example.npz) permit direct reanalysis.

The [independent verification](results/verification.json) rechecks frozen inputs, source vectors, calibration, chunk aggregation, fresh-engine replay and all four primary intervals. The [protocol](docs/PROTOCOL.md) defines the limits and stopping rules.
