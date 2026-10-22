# EXPERIMENT-001: canonical-to-controlled-context transport

**Status:** frozen, run, reproduced, and analyzed on 31 August 2026  
**Model:** GPT-2 Small  
**Purpose:** decide whether the causal-transport intervention is measurable and nontrivial before expanding the benchmark

## Question

Can the duplicate-name signal produced in canonical IOI restore the predicted downstream mechanism in a position-matched, semantically degraded IOI prompt?

## Conditions

### A: canonical

Natural IOI templates with two single-token names and a repeated subject.

### B: controlled context

The same entity tokens, roles, ABB/BAB order, and token positions as condition A. Non-entity context is replaced by prevalidated token-level fillers. The primary manipulation changes context while preserving the causal roles needed to interpret activation transfer.

## Paired examples

For each condition and entity assignment:

- `clean`: the subject appears in the initial pair and again at the subject-2 position;
- `corrupt`: the subject-2 occurrence is replaced by a third single-token name.

The corruption is an intervention baseline, not necessarily a naturally occurring IOI sentence. We will repeat the result with at least one alternative corruption before making a substantive claim.

## Sites

1. Duplicate/Induction head output at the repeated-name position.
2. S-Inhibition head output at the final prediction position.
3. Name Mover result at the final prediction position.

The primary head groups are fixed from Figure 2 of
Wang et al. (2023), using only the non-parenthesized heads:

- Duplicate Token: `(0, 1)`, `(3, 0)`;
- Induction: `(5, 5)`, `(6, 9)`;
- S-Inhibition: `(7, 3)`, `(7, 9)`, `(8, 6)`, `(8, 10)`;
- Name Mover: `(9, 9)`, `(9, 6)`, `(10, 0)`.

The primary early-stage intervention patches Duplicate Token and Induction outputs together. Fuzzy
Duplicate/Induction heads `(0, 10)`, `(5, 8)`, and `(5, 9)` are robustness-only and cannot replace
the primary list after outcomes are observed.

## Outcomes

### Primary final outcome

`P(IO) - P(repeated subject)`, bounded to `[-1, 1]`.

### Secondary final outcome

IO-minus-subject logit difference.

### Required local outcomes

- After Duplicate/Induction intervention: predicted change in S-Inhibition response to the repeated subject.
- After S-Inhibition intervention: predicted change in Name Mover attention away from the repeated subject and toward the IO.
- After Name Mover intervention: predicted direct contribution toward the IO token.

An end-to-end recovery without the predeclared local effect is classified as nonspecific, not successful transport.

The early-stage local outcome is the projection of the patched change in summed S-Inhibition
output onto the paired target clean-minus-corrupt S-Inhibition direction. The S-Inhibition local
outcome is mean Name Mover attention to IO minus attention to S1. The Name Mover local outcome is
the summed head-result projection onto the IO-minus-subject unembedding direction.

## Comparisons

- A clean signal into A corrupt: positive control.
- B clean signal into B corrupt: positive control.
- A clean signal into B corrupt: primary direction.
- B clean signal into A corrupt: reverse direction.
- Random head at matched layer/position.
- Norm-matched random direction.
- Correct signal at an off-target position.
- Mismatched entity-role pairing.

Random-head mappings and random-direction seeds are written to the result configuration before
the test split is run. The off-target position is one token before the role-matched target position.

## Frozen sample and split

- 240 paired role assignments: 80 exploratory, 80 development, and 80 test.
- Each split is balanced 40/40 between ABBA and BABA order.
- Entity triples are assigned to exactly one split.
- Canonical/control template pairs are disjoint across development and test.
- The manifest is generated with seed `1729` and frozen by SHA-256 before model outcomes.
- The first 100 rows are rendered into a structured audit sheet before test evaluation.

The controlled-context strings replace every non-entity token with a one-token filler while
preserving entity positions and sequence length. Exact text, token IDs, role positions, split, and
template identifiers are authoritative in the frozen JSONL manifest.

## Frozen behavioral and statistical rules

- A condition passes the pilot behavioral floor when at least 70% of clean examples have a
  positive IO-minus-subject logit difference and the mean clean probability difference is positive.
- Primary outcomes use float32 GPT-2 Small.
- Effects are paired by manifest row.
- Report the transport numerator, target clean denominator, and their ratio.
- Report 95% percentile intervals from 2,000 paired bootstrap resamples with seed `20260831`.
- Ratios are not reported when the absolute target denominator is below `0.01` probability-difference
  units or `0.1` logit-difference units.
- No examples are filtered by model correctness after outcomes are observed.

## Dataset checks

- Every entity is one token including its leading-space form.
- Clean/corrupt sequences have equal token length.
- Role positions are identical within paired interventions.
- ABB and BAB orders are balanced.
- Entity triples do not cross train/development/test splits.
- Templates or filler sequences do not cross development/test splits.
- Generated metadata records every token ID and role position.
- At least 100 examples are manually audited before the test split is frozen.

The structured 100-row audit may be [redacted]-assisted, but the paper must not describe it as an
independent human audit unless a human signs the audit record.

## Go/no-go criteria

Proceed to the full source-by-target matrix only if:

1. GPT-2 Small solves both clean conditions above a declared behavioral floor;
2. within-condition positive controls recover a substantial fraction of the clean effect;
3. canonical head roles reproduce qualitatively;
4. cross-condition estimates are distinguishable from negative controls or provide a clear falsification of the intervention;
5. the result is stable under one alternative corruption.

## Interpretation table

| Result | Interpretation |
|---|---|
| High overlap, high transport | evidence for reusable computation |
| High overlap, low transport | component reuse overstates causal reuse |
| Early transport only | reusable relational module with changing downstream computation |
| All patches fail, including positive controls | intervention or dataset is invalid; no mechanism conclusion |
| Strong dependence on corruption | intervention-distribution sensitivity becomes the finding |
