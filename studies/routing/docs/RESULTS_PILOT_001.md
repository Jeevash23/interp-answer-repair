# Pilot 001: results and decision

Completed 2026-09-05. **Proceed with investigation; do not yet claim causal routing.**

GPT-2 Small ran in CPU float32 with four compute threads and four-example batches. The run took
189.9 seconds and recorded peak process RSS of 2,991 MiB (about 2.9 GiB). This establishes that
the bounded pilot is feasible locally without GPU access.

## Measurements

64 IOI prompts: 32 discovery and 32 held out, with disjoint template IDs and name vocabularies.
Each split contains two hand-selected templates and four unordered name pairs, both role
assignments, and both ABBA/BABA orders. Matched family conditions preserve the correct answer.

Before inference, we froze 72 downstream END-head coordinates, each measured using zero and
S2-corrupt replacement. A nearest-centroid classifier was fit to discovery causal effects only.
The split was not chosen after inspecting outcomes, and no failed prompts were removed.

| Replacement | Held-out family classification | Natural axis separation [bootstrap 95%] |
|---|---:|---:|
| Zero | 32/32 (100%) | 0.912 [0.839, 0.975] |
| S2-corrupt | 25/32 (78.1%) | 1.179 [0.994, 1.322] |

The separation statistic compares *unedited* ABBA and BABA prompts along the discovery axis.
**It is not a transfer or circuit-switch score.** Both baselines met the small pilot's frozen
numerical rules; its recorded status is `provisional_pass`.

Classification depends on the ablation baseline, falling from 100% to 78.1%. Within-family
variation is substantial: average standardized within-family distances were 12.55 versus 16.34
between families under zero replacement, and 7.48 versus 9.30 under S2-corrupt replacement.
Supervised discrimination does not by itself establish two discrete clusters or complete circuits.

## Behavioral limitation

The correct answer beat the repeated subject on every prompt, but open-vocabulary behavior
was weaker:

| Template | Split | Correct name is top next token |
|---|---|---:|
| d-museum | Discovery | 16/16 (100%) |
| d-cafe | Discovery | 15/16 (93.8%) |
| h-station | Held out | 15/16 (93.8%) |
| h-studio | Held out | 1/16 (6.3%) |

Held-out full-vocabulary top-1 accuracy is 16/32 (50%). The studio template often predicts
` the` rather than a name. It remains in every saved analysis. The frozen gate uses the
IO-versus-subject contrast and passes, but a separate behavior-screened validation study is
needed before making a strong natural-task claim.

Corruption reduced the IO-minus-subject logit gap by 1.99 to 4.94 logits across split/family cells,
above the predeclared 0.25-logit efficacy threshold.

## Uncertainty and verification

There are only two held-out template clusters and four unordered name-pair clusters. The
bootstrap is conditional on the discovery fit and resamples fixed, hand-selected items.
The zero-ablation accuracy interval collapses to [100%, 100%] because all observed cases were
correct; this does **not** imply zero uncertainty or perfect generalization. Conditional sign-flip
diagnostics in the JSON do not resolve shared-template/shared-name dependence.

Five correctness tests passed. All 72 identity head interventions reproduced the original
logit difference exactly on the validation batch. Hook cleanup also had zero error.
Native-versus-compatibility centered logits differed by at most 0.0000801, below the frozen
0.005 tolerance. Package compatibility passed. Saved-effect statistical replay reproduced the
summary and discovery fit byte-for-byte.

All 348 recorded files from the previous project remained unchanged. Cached weights and
tokenizer files were independently hashed and rechecked after inference. See
[verification.json](../results/pilot-001/verification.json).

## Decision

The result warrants further family validation. First complete the blocked canonical-source/code
audit, then freeze an independent validation dataset with more templates and a clear behavioral
screen. Preserve this pilot, including its weak studio-template behavior and baseline sensitivity.

No router edits, full circuit discovery, ACC++ implementation reproduction, full factorial study
or Pythia experiment were performed. Current evidence does not establish a localized, distributed,
token-bound or transferable router; those alternatives have not been causally tested.

See the [generated report](../results/pilot-001/analysis/REPORT.md),
[statistics](../results/pilot-001/analysis/summary.json),
[interpretation review](PILOT_REVIEW.md), and [reproduction instructions](../REPRODUCIBILITY.md).
