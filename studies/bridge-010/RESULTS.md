# Bridge 010 (Study 5): results

**Completed and independently verified.** All twelve full-bank conditions finished on 5 October 2026 at 09:15:40 UTC; the frozen analysis, fresh-engine singleton replays and the independent reconstruction passed (audit 09:49:30 UTC). This exploratory study reuses the inspected Study 2 (Bridge 007) final bank and is not pooled with Study 2.

## Headline ratios

| Control | Reference | F ratio [adjusted CI] | A ratio [adjusted CI] |
| --- | --- | --- | --- |
| Mismatched configuration | Zero | 0.997 [0.898, 1.106] | 0.992 [0.840, 1.142] |
| Mismatched configuration | Fit mean | 0.997 [0.898, 1.104] | 0.998 [0.859, 1.145] |
| Natural name/context | Zero | 1.067 [0.944, 1.195] | 1.087 [0.863, 1.366] |
| Natural name/context | Fit mean | 1.104 [0.973, 1.243] | 1.256 [1.013, 1.537] |
| Random head-subspace | Zero | 1.279 [1.124, 1.427] | 1.191 [0.891, 1.518] |
| Random head-subspace | Fit mean | 1.320 [1.156, 1.478] | 1.399 [1.066, 1.793] |

## Paired differences (control/self minus wording/self)

| Control | Reference | F control − F wording [adjusted CI] | A control − A wording [adjusted CI] |
| --- | --- | --- | --- |
| Mismatched configuration | Zero | 0.001 [-0.016, 0.017] | 0.000 [-0.004, 0.005] |
| Mismatched configuration | Fit mean | 0.000 [-0.018, 0.018] | 0.000 [-0.005, 0.005] |
| Natural name/context | Zero | -0.010 [-0.028, 0.008] | -0.002 [-0.009, 0.004] |
| Natural name/context | Fit mean | -0.016 [-0.037, 0.004] | -0.007 [-0.014, -0.000] |
| Random head-subspace | Zero | -0.035 [-0.052, -0.016] | -0.004 [-0.010, 0.003] |
| Random head-subspace | Fit mean | -0.042 [-0.062, -0.022] | -0.010 [-0.017, -0.002] |

## Control/self discrepancies and endpoint distances

Means with pointwise 95% intervals; four replicates averaged within each configuration. Wording/self signed shift (from the Study 3 saved analysis of the same arrays) is 0.0150 / 0.0264.

| Control | Reference | F control/self [95% CI] | A control/self [95% CI] | Signed S shift [95% CI] | q_R [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mismatched configuration | Zero | 0.159 [0.147, 0.172] | 0.028 [0.023, 0.033] | 0.015 [0.009, 0.021] | 1.082 [1.030, 1.135] |
| Mismatched configuration | Fit mean | 0.174 [0.160, 0.189] | 0.034 [0.029, 0.041] | 0.026 [0.020, 0.033] | 0.971 [0.922, 1.020] |
| Natural name/context | Zero | 0.149 [0.138, 0.160] | 0.025 [0.022, 0.030] | 0.009 [0.005, 0.014] | 1.082 [1.032, 1.133] |
| Natural name/context | Fit mean | 0.157 [0.145, 0.170] | 0.027 [0.023, 0.032] | 0.012 [0.007, 0.017] | 0.964 [0.918, 1.011] |
| Random head-subspace | Zero | 0.124 [0.114, 0.135] | 0.023 [0.019, 0.029] | 0.010 [0.006, 0.016] | 1.088 [1.035, 1.140] |
| Random head-subspace | Fit mean | 0.132 [0.121, 0.143] | 0.025 [0.020, 0.031] | 0.012 [0.007, 0.018] | 0.969 [0.920, 1.019] |

## Behaviour

| Condition | Recovery (logits) [95% CI] | Top-1 (%) [95% CI] |
| --- | --- | --- |
| Self transfer | 2.414 [2.155, 2.683] | 90.4 [83.6, 96.1] |
| Wording transfer | 2.332 [2.070, 2.606] | 90.7 [84.0, 96.2] |
| Mismatched configuration | 2.288 [2.028, 2.559] | 90.2 [83.4, 95.8] |
| Natural name/context | 2.335 [2.086, 2.597] | 90.4 [83.6, 96.0] |
| Random head-subspace | 2.258 [1.997, 2.529] | 89.3 [82.3, 95.2] |

Paired behavioural contrasts, 24-contrast adjusted intervals:

| Control | Measure | Minus self [adjusted CI] | Minus wording [adjusted CI] |
| --- | --- | --- | --- |
| Mismatched configuration | Recovery (logits) | -0.126 [-0.162, -0.095] | -0.044 [-0.065, -0.023] |
| Mismatched configuration | Top-1 (points) | -0.25 [-1.45, 0.95] | -0.53 [-1.62, 0.38] |
| Natural name/context | Recovery (logits) | -0.079 [-0.169, 0.014] | 0.003 [-0.089, 0.102] |
| Natural name/context | Top-1 (points) | 0.00 [-1.41, 1.71] | -0.28 [-2.12, 2.02] |
| Random head-subspace | Recovery (logits) | -0.156 [-0.195, -0.121] | -0.074 [-0.099, -0.047] |
| Random head-subspace | Top-1 (points) | -1.13 [-2.69, -0.01] | -1.41 [-2.86, -0.36] |

## Disruptions (per replicate, out of 3,200)

| Control | Gap below self | Lost self-correct top-1 | Negative recovery |
| --- | --- | --- | --- |
| Mismatched configuration | 2019, 1950, 1967, 2011 | 25, 30, 32, 24 | 15, 12, 15, 18 |
| Natural name/context | 2021, 1894, 2017, 1953 | 35, 28, 32, 29 | 27, 28, 22, 14 |
| Random head-subspace | 2336, 2364, 2362, 2356 | 49, 47, 44, 51 | 14, 18, 15, 16 |

## Interpretation and limits

The prespecified reading treats a ratio interval below 1 as evidence that wording directions are discounted relative to same-geometry turns, an interval around 1 as similarity generic to the edit geometry, and an interval above 1 as greater effect of wording directions. No interval lies below 1. The mismatched control retains template-level wording and tests configuration specificity; the natural control compares wording with name/context turns. Random head-subspace directions are unlike natural activation differences, so their smaller effect may reflect weaker downstream readout of such directions. The study reuses an inspected bank, manipulates neither recipient context nor source coverage, and does not identify an algorithm. Cell results are descriptive; all cells, replicates and distributions are in [results/analysis/summary.json](results/analysis/summary.json) and the manuscript appendix.
