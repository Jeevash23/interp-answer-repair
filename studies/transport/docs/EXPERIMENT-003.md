# EXPERIMENT-003: frozen multi-template transport prediction

**Status:** design frozen before any Experiment 003 model evaluation  
**Frozen on:** 1 September 2026  
**Model and sites:** the exact GPT-2 Medium revision and selected heads frozen by Experiment 002  
**Purpose:** test whether the Experiment 002 transport result predicts causal behavior on new prompt
templates rather than only the original wording family

## Motivation and evidential status

Experiment 002 independently localized Medium heads and found partial component overlap,
directional held-out transport, and an early-role/late-entity distinction. Its development and test
splits nevertheless contained only one template pair each. Circuit-generalization work warns that
mechanistic conclusions found on narrow prompt formats may not survive prompt variation, while
recent circuit-stability work explicitly relates stability across input partitions to model
generalization. Experiment 003 therefore makes a forward prediction using frozen heads on new
templates; it performs no localization, head replacement, threshold tuning, or example filtering.

## Frozen data

- Sixteen new paired template families: eight development and eight untouched test.
- Ten new entity triples per template, each rendered in ABBA and BABA order: 20 examples per
  template, 160 per split, and 320 total.
- Every prompt remains exactly 14 GPT-2 tokens with entity positions 1/3/9 and prediction position
  13, making the intervention sites role-matched without padding or token alignment.
- No unordered entity triple overlaps Experiment 001/002 or another Experiment 003 template.
- Development and test templates are distinct. Test is not inspected for model behavior or
  intervention outcomes before implementation is frozen.
- Manifest seed: `20260903`.

## Frozen predictions

The test split is primary. Development is diagnostic and cannot change the model, selected heads,
controls, analysis, or test inclusion.

1. **Behavioral generalization:** aggregate clean accuracy is at least 70% with positive mean
   probability difference in both conditions, and at least six of eight controlled templates pass
   the same per-template gate.
2. **Transport generalization:** canonical-to-controlled correct patching has a positive hierarchical
   95% interval at both S2 and END on the primary probability numerator.
3. **Directional prediction:** canonical-to-controlled CTS exceeds controlled-to-canonical CTS at
   both sites under a template-resampling bootstrap.
4. **END specificity:** canonical-to-controlled END transport exceeds random-head,
   random-direction, off-position, and mismatched-entity controls after correction across four
   exact template-level tests.
5. **Stage prediction:** mismatched entities retain more of the correct effect at S2 than at END;
   the END correct-minus-mismatch contrast exceeds the S2 contrast.

A failed prediction is retained as a template-generalization boundary. Heads, templates, entities,
or thresholds will not be changed after outcomes are observed.

## Frozen interventions and outcomes

- Use the Experiment 002 canonical- and controlled-localized top-eight head groups at S2 and END.
- Evaluate the four correct source-target pairs at both sites.
- Evaluate canonical-to-controlled random-head, norm-matched random-direction, off-position, and
  mismatched-entity controls at both sites.
- Use only the primary corruption. Experiment 002 already tests the alternative corruption; omitting
  it here allocates compute to template replication without adding a researcher degree of freedom.
- Primary final outcome: patched-minus-corrupt `P(IO)-P(subject)`, always reported beside the target
  clean-minus-corrupt denominator and CTS.
- Secondary outcomes: logit difference and the same local S2/END outcomes as Experiment 002.
- Do not suppress absolute effects. Suppress only ratios whose aggregate absolute probability
  denominator is below 0.01 or logit denominator below 0.1.

## Hierarchical inference

The population of interest is prompt templates, not merely the 160 realized examples. The primary
95% intervals use a balanced two-stage bootstrap with seed `20260903`: resample eight templates,
then resample the ten adjacent ABBA/BABA entity pairs within each sampled template. Use 5,000 draws.
Preserve each pair as the within-template sampling unit.

For the four END correct-minus-control comparisons, average paired effects within each of the eight
test templates and enumerate all `2^8` sign flips. Apply Holm correction across the four tests. The
directional and stage contrasts use the hierarchical bootstrap and are reported with absolute
numerators even if a ratio denominator is small.

## Gates and claim boundary

Mechanistic interpretation at a site requires both aggregate conditions to pass behavior, positive
within-source effects, a non-small target denominator, and the predicted local sign. Generalization
additionally requires the relevant hierarchical interval to exclude zero. Passing this experiment
supports template-family transport for the frozen sparse Medium groups; it does not establish a full
graph, MLP or subspace equivalence, a non-IOI task result, or cross-architecture validity.
