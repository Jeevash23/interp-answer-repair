# Project charter

## Working title

**Circuit Equivalence Beyond Overlap: Causal Transport Across Prompt Distributions**

## Central question

Does component or edge overlap across prompt distributions imply that the model is reusing the same causal computation?

## Proposed answer test

Treat circuit reuse as causal interoperability. A causal intermediate produced in source condition `A` should restore the predicted local and end-to-end behavior when inserted into a role-matched corrupted run from target condition `B`.

## Unit of analysis

A unit is a matched tuple containing:

- model and checkpoint;
- source and target prompt condition;
- entity assignment and role mapping;
- clean and controlled-corruption prompts;
- circuit site and intervention;
- local downstream outcome;
- final bounded answer outcome.

Raw activation transfer is performed only within the same model. Cross-model experiments replicate the qualitative claim rather than patching incompatible activation spaces.

## Primary causal quantity

For source condition `A`, target condition `B`, and site `s`:

```text
delta_A(s) = activation_A_clean(s) - activation_A_corrupt(s)

CTS(A -> B, s) =
  [mean m(B_corrupt + delta_A) - mean m(B_corrupt)]
  ---------------------------------------------------
  [mean m(B_clean)          - mean m(B_corrupt)]
```

The numerator is cross-condition recovery. The denominator is the target's within-condition clean effect. The score is reported with numerator and denominator estimates, not as an uninterpreted ratio alone.

The preferred bounded final outcome is `P(IO) - P(repeated subject)`, in `[-1, 1]`. Logit difference is secondary.

## Hypotheses

1. **Overlap is not equivalence.** Some high-overlap condition pairs have weak causal transport.
2. **Stage dependence.** Relational/duplicate signals transport more widely than later selection or copying signals.
3. **Mechanism families.** The directed transport matrix clusters prompt conditions into a small number of families.
4. **Predictive validity.** Transport predicts held-out circuit faithfulness better than node overlap, edge overlap, attention similarity, or representational similarity.
5. **Asymmetry.** Normalized `A -> B` transport can differ from `B -> A` transport.

## Intended contribution

The intended contribution is a measurement and robustness study:

1. a directional causal-transport measure;
2. a paired IOI distribution-shift benchmark;
3. controlled comparisons with conventional circuit-similarity measures;
4. held-out predictive evaluation;
5. a reproducible intervention and uncertainty protocol.

## What this project is not

- not merely a demonstration that GPT-2 behaves differently on gibberish;
- not a claim that shared attention heads automatically imply a shared mechanism;
- not cross-model raw activation transplantation;
- not a blind search over every model, task, head, and metric;
- not a paper written around results before the evaluation is frozen.

## TMLR-level evidence target

The GPT-2 Small study is the pilot. The intended paper adds at least one other model family, a held-out prediction test, multiple corruption controls, uncertainty estimates, and an anonymous reproducible release. Multiple model families are a scientific recommendation, not an official TMLR rule.

## Evidence status after Experiment 004

The planned evidence target is met for this research edition: GPT-2 Medium provides the cross-scale
result, Experiment 003 provides the frozen 16-template prediction, and independently localized
Pythia-410M provides a GPT-NeoX-family replication. The next expansion should test whether transport
predicts graph faithfulness beyond structural and representational baselines on additional tasks; it
should not retune the completed IOI result.

## 2026-09-05 scope correction (original charter retained above)

The original numerical Studies 001--004 are complete, but they do not complete
all the hypotheses or intended contributions listed above. Mechanism-family
clustering and predictive comparison with overlap, representation similarity,
and circuit-faithfulness baselines remain unperformed. They are excluded from
the revised paper's completion claim.

The current title is **Activation Transport Across Prompt Distributions:
Recovery, Normalization, and Specificity**. The supported scope is a measurement
study of cross-condition endpoint effects, target-normalized recovery, diagnostic
local projections, and control sensitivity. The September 5 saved-output audit
corrects claims about random head controls, batch invariance, directional raw
effects, local-pathway identification, and public preregistration; see
[`AUDIT-2026-09-05.md`](AUDIT-2026-09-05.md).

Experiment 005 is a separately specified post-hoc robustness extension on the
already inspected templates. It preserves the original studies while testing
sampled controls, matched groups/norms, and controlled node blocking. It does not
provide untouched-data confirmation, identify a natural indirect effect, or
fulfil the deferred predictive-faithfulness and clustering aims. New prospective
generalization claims require a separately specified untouched evaluation set.
