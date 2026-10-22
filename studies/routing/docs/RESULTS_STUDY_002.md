# Study 002: completed results

**The measured state predicts prompt-conditioned causal dependencies, but neither tested edit demonstrates a robust circuit-family switch under the frozen criteria.** This is a bounded negative/mixed result for one selected direction and one full-vector comparison; it does not rule out other compact or distributed mechanisms.

The study used GPT-2 Small, 64 discovery recipients, 64 development recipients and 128 test recipients. Names and templates were disjoint across splits. Test data comprise eight templates and four unordered name pairs, crossed with role assignment and ABBA/BABA order. All 30 conditions completed. No prompt was removed after observing model outcomes.

## Natural behavior and prediction

Natural test pairwise accuracy was 100.00%; full-vocabulary top-1 accuracy was 96.09%. These measure different things: ranking the correct name above the repeated subject versus ranking it above every vocabulary token.

| Signature baseline | Test family classification | Natural axis separation | 95% block interval |
|---|---:|---:|---|
| s2_corrupt | 82.81% | 0.9019 | [0.6663, 1.1904] |
| zero | 100.00% | 0.8267 | [0.6432, 1.0624] |

The residual state before layer 6 at S2 was selected using development prediction only. Its development R² was 0.3402 and test R² was 0.6243. A post-hoc discovery-fitted prompt-order mean achieved test R² 0.6363. Thus no predictive advantage over that simple input baseline is established. The diagnostic was added after partial test outputs were observed and changed no selection or causal test criterion.

## Causal transfer and disruption

The following comparison keeps the recipient's answer and wording fixed while changing the donor family. S is movement toward the opposite-family centroid along the fixed discovery axis. O is perpendicular movement, divided by the same centroid distance. A positive S alone is not a circuit switch.

| Edit | S, zero | O, zero | S, corruption | O, corruption | Pairwise accuracy | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| rank1 | 0.0031 | 0.1064 | 0.0767 | 0.1609 | 100.00% | 94.53% |
| full | 0.0432 | 0.9208 | 0.2330 | 1.3995 | 72.66% | 63.28% |

## Frozen decision criteria

Primary contrasts compare opposite-family minus same-family edits at fixed answer and wording. There are 16 per edit mode: two directions × two answers × two wording strata × two ablation baselines. The table counts positive Bonferroni simultaneous lower bounds. All components must pass; a few positive comparisons do not establish the proposed result.

| Edit | Positive primary bounds | Behavior preserved | Orthogonal bounded | All location/random controls | Overall |
|---|---:|---|---|---|---|
| full | 4/16 | No | No | No | Fail |
| rank1 | 4/16 | Yes | No | No | Fail |

The four positive adjusted contrasts in each mode are all BABA→ABBA under S2-corrupt replacement, across both answer and wording conditions. No zero-baseline contrast has a positive adjusted lower bound. This is evidence of a conditional directional effect, but it fails the stronger bidirectional, baseline-robust switching claim.

Behavior preservation permits at most ten percentage points of pairwise accuracy loss. The perpendicular-movement bound compares mean O to mean positive S. Uncertainty uses 10,000 crossed template/name-pair bootstrap draws, keeping paired roles and donor cells together. It conditions on the discovery fit. Eight templates and four name pairs cannot support broad population or language generalization claims.

## Complete condition inventory

Condition names encode wording / family / answer. `other` means alternate template; `cross` means opposite family. `same_same_same` is an identity control. Random and misplaced controls are norm-matched to the same-wording opposite-family same-answer edit within their own mode. The two edit modes are not matched to each other's norms. The alternative-name diagnostic always uses the recipient's four-donor alternative C; it need not be a shuffled donor's actual answer.

| Condition | Mean S, zero | Mean S, corrupt | Pairwise accuracy | Top-1 | Alternative C > IO |
|---|---:|---:|---:|---:|---:|
| rank1__same_same_same | 0.0000 | 0.0000 | 100.00% | 96.09% | 0.00% |
| rank1__same_same_different | 0.0004 | 0.0014 | 100.00% | 96.09% | 0.00% |
| rank1__same_cross_same | 0.0031 | 0.0767 | 100.00% | 94.53% | 0.00% |
| rank1__same_cross_different | 0.0036 | 0.0798 | 100.00% | 94.53% | 0.00% |
| rank1__other_same_same | 0.0001 | 0.0005 | 100.00% | 96.09% | 0.00% |
| rank1__other_same_different | 0.0004 | 0.0019 | 100.00% | 96.09% | 0.00% |
| rank1__other_cross_same | 0.0033 | 0.0769 | 100.00% | 94.53% | 0.00% |
| rank1__other_cross_different | 0.0037 | 0.0800 | 100.00% | 94.53% | 0.00% |
| rank1__shuffled | 0.0040 | 0.0786 | 100.00% | 94.53% | 0.00% |
| rank1__random_0 | 0.0011 | 0.0050 | 100.00% | 96.88% | 0.00% |
| rank1__random_1 | -0.0010 | -0.0001 | 100.00% | 96.09% | 0.00% |
| rank1__random_2 | 0.0013 | 0.0032 | 100.00% | 96.88% | 0.00% |
| rank1__wrong_position | 0.0011 | 0.0078 | 100.00% | 96.88% | 0.00% |
| rank1__wrong_layer | -0.0006 | 0.0447 | 100.00% | 96.09% | 0.00% |
| rank1__mlp_control | 0.0001 | 0.0202 | 100.00% | 96.09% | 0.00% |
| full__same_same_same | 0.0000 | 0.0000 | 100.00% | 96.09% | 0.00% |
| full__same_same_different | 0.0003 | 0.0056 | 100.00% | 96.09% | 0.00% |
| full__same_cross_same | 0.0432 | 0.2330 | 72.66% | 63.28% | 0.00% |
| full__same_cross_different | 0.0362 | 0.2275 | 70.31% | 60.94% | 0.00% |
| full__other_same_same | 0.0005 | -0.0094 | 100.00% | 97.66% | 0.00% |
| full__other_same_different | 0.0007 | -0.0055 | 100.00% | 98.44% | 0.00% |
| full__other_cross_same | 0.0426 | 0.2318 | 73.44% | 62.50% | 0.00% |
| full__other_cross_different | 0.0352 | 0.2240 | 69.53% | 60.16% | 0.00% |
| full__shuffled | 0.0098 | 0.2735 | 23.44% | 20.31% | 0.00% |
| full__random_0 | 0.0098 | 0.0549 | 100.00% | 92.97% | 0.00% |
| full__random_1 | 0.0086 | 0.0464 | 100.00% | 96.09% | 0.00% |
| full__random_2 | 0.0130 | 0.0568 | 100.00% | 95.31% | 0.00% |
| full__wrong_position | 0.0137 | 0.0468 | 100.00% | 97.66% | 0.00% |
| full__wrong_layer | 0.0661 | 0.2341 | 92.97% | 84.38% | 0.00% |
| full__mlp_control | 0.0189 | 0.1600 | 100.00% | 93.75% | 0.00% |

## Scope, evidence and reproduction

The 72 head coordinates at END, scales, centroids, site-selection rule and test analysis were fixed before test inference. They measure single-head causal dependencies, not a complete circuit or all possible routes. No new axes or candidates were selected to improve these test results. The earlier 64-recipient pilot remains unchanged and exploratory.

The transfer run took 24.51 minutes on CPU; peak process memory was 2.90 GiB. GPU checking and public-source retrieval were blocked by the app's automatic approval-review service error despite the user's approval. No GPU inference, fresh novelty audit, official-code reproduction or Pythia replication is claimed. The original protocol made replication conditional on a successful core result.

See [reproduction instructions](STUDY-002-REPRODUCIBILITY.md), the [frozen protocol](STUDY-002-PROTOCOL.md), [complete statistics](../results/study-002/analysis/summary.json), [verification record](../results/study-002/verification.json) and [manuscript](../output/pdf/causal-dependencies-research-draft.pdf).

Analysis summary SHA-256: `8f40fb691871a48f5608e521514daef15984a5751e029ab565589bee3ddec090`.
