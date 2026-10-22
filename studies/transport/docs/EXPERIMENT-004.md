# EXPERIMENT-004: Pythia cross-architecture circuit transport

**Status:** frozen before Pythia localization outcomes  
**Model:** EleutherAI Pythia-410M-deduped at revision
`c4fc8d586d62df497f1f9b69d66d3ca419992d3e`  
**Data:** the 16 template families and 320 texts retokenized in
`experiment-004-manifest.jsonl`

## Scientific question

Does the late, directional canonical-to-controlled activation transport found in GPT-2 Medium
replicate after components are independently localized in a size-matched GPT-NeoX-family model?
Raw activations and head identities are never compared across architectures. Pythia heads are
localized from scratch and only qualitative causal claims are compared.

## Frozen split and selection

- Use all 160 development records (eight templates, ten entity triples, adjacent ABBA/BABA rows)
  for exhaustive head localization.
- At both the second subject mention (`S2`) and final prompt token (`END`), patch each of the 384
  attention heads separately from clean into primary-corrupt runs.
- Score a head by mean restoration of the IO-minus-subject final logit difference across all
  development examples. Select the eight highest strictly positive heads separately for each
  condition and site. If fewer than eight are positive, the corresponding localization fails.
- Choose deterministic, layer-matched random heads outside the union of the canonical and
  controlled selected sets.
- Never use test outcomes to select heads, sites, counts, controls, or thresholds.

## Frozen intervention

For a selected source group, compute the clean-minus-corrupt head-result vector at the source
condition's corresponding S2 or END position. Add that vector to the corrupt target run at the
target condition's corresponding position. Evaluate all four source/target directions and the
following canonical-to-controlled controls: layer-matched random heads, equal-norm random
directions, one-token-earlier placement, and entity-mismatched vectors shifted to another adjacent
ABBA/BABA pair. Use primary corruption only and do not filter examples by model correctness.

The endpoint metric is the change in `P(IO)-P(subject)`. Circuit transport score (CTS) divides that
change by the target clean-minus-corrupt endpoint difference and is reported only when the absolute
denominator is at least 0.01. Local metrics are (a) projection of the patched END-group change onto
its target clean causal direction for S2 patches and (b) selected-head direct contribution to the
IO-minus-subject logit direction for END patches.

## Frozen inference

Average adjacent ABBA/BABA rows into entity pairs. Use a two-stage percentile bootstrap that first
resamples templates and then entity pairs within sampled templates (5,000 draws, seed 20260904).
For the four control contrasts, enumerate all 256 template-level sign assignments and apply Holm
family-wise correction. Development transport is descriptive; all confirmatory decisions use the
untouched eight-template test split.

## Frozen predictions and decision rule

The primary cross-architecture replication passes only if all four held-out END rules pass:

1. canonical-to-controlled endpoint probability restoration has a 95% CI strictly above zero;
2. canonical-to-controlled CTS exceeds controlled-to-canonical CTS with a 95% CI strictly above
   zero;
3. the correct endpoint effect exceeds every control with a positive 95% CI and Holm-adjusted
   `p < 0.05`;
4. the local END direct-effect numerator has a 95% CI strictly above zero.

The preregistered stage extension asks whether entity mismatch is retained more at S2 than END and
whether the correct-minus-mismatch contrast is larger at END; both differences must have positive
95% CIs. S2 endpoint and local effects are reported regardless of sign, but no S2 pathway claim is
allowed unless both its endpoint and local-numerator CIs are positive.

## Reproducibility gate

The localization selection and every development/test scientific result file must reproduce
byte-for-byte in an independent rerun. The frozen manifest, model revision and weight hash, loader,
intervention code, runners, analysis code, tests, package lock, and feasibility decision are all
hashed before localization begins.

## Implementation amendment 001 (before outcomes)

The first localization process failed on its first patched forward pass before writing any result.
TransformerLens' `hook_result` implementation materialized an expanded
`[batch, position, head, d_head, d_model]` tensor and requested 17.5 GiB. The frozen 20-example
batch is retained, but localization patches each head's `hook_z` value instead. For head `h`,
adding `z_clean - z_corrupt` before its fixed linear `W_O[h]` projection is algebraically identical
to adding the corresponding clean-minus-corrupt head result after that projection. Transport runs
and local metrics remain at `hook_result`; no data, score, selection rule, prediction, threshold,
seed, model value, or test procedure changed. The amendment and amended hashes are recorded in
`experiment-004-implementation-amendment-001.json` before the localization rerun.

## Implementation amendment 002 (before transport outcomes)

The first development transport process likewise failed on its first template before writing any
result: four full `hook_result` caches plus a patched forward left insufficient memory for a 20-row
batch. Transport therefore uses four-row batches. Four rows contain two complete adjacent
ABBA/BABA entity pairs, so the two-row entity-mismatch shift remains a different matched pair.
Batching changes only execution granularity; every example, selected head, intervention, control,
metric, seed rule, prediction, and inference unit remains frozen. This amendment is recorded before
either development or test transport produces an outcome.
