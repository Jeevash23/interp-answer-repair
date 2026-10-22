# Experiment plan

Version: pilot-001, 2026-09-05. Locally frozen protocol; **not a public preregistration**.

## First experiment: fixed downstream dependencies

Use the cached, revision-recorded GPT-2 Small model, float32 evaluation without gradient tracking.
The clean endpoint is `logit(IO) - logit(subject)` at END. Record both probabilities, their
difference, pairwise correctness and the full-vocabulary top-1 token. Pairwise correctness is
not open-vocabulary generation accuracy.

For names A (indirect object) and B (repeated subject):

- ABBA: `When A and B ... , B ... to` predicts A.
- BABA: `When B and A ... , B ... to` predicts A.

The last letter denotes the correct completion and is *not* in the input. Both orders have the
same correct answer for a given role assignment. Token lengths and entity positions are checked.
Use explicit GPT-2 BOS and one-token leading-space names; do not silently drop failed prompts.

The minimal pilot has 64 clean prompts: two discovery templates and two held-out templates,
each crossed with four unordered name pairs, both IO/subject assignments, and both families.
Discovery and held-out sets use disjoint name vocabularies and template IDs. Each clean prompt has
a matched S2-corrupted prompt replacing the repeated subject with a distinct third name from its
own split. The original answer/subject contrast is retained when evaluating corrupt behavior.
All prompts, token IDs, roles, split labels and hashes are saved before model inference.

Only 2 templates and 4 name pairs per split means this is a small feasibility sample. It cannot
provide a reliable population-level estimate over all natural-language templates.

## Signature measurement

Predetermine every head in layers 6–11 (zero-indexed), all 12 heads per layer: 72 coordinates.
At END only, set each head's `hook_z` to (i) zero, or (ii) that head's cached END value from the
recipient's S2-corrupted prompt. Recompute the downstream network normally, one head at a time.
`hook_z` precedes the fixed output projection, so this replaces the corresponding head write
without materializing a `[batch, positions, heads, model_width]` result tensor.

The two signatures are separate analyses. This is exact single-node intervention, not a gradient
proxy or a claim to recover the full IOI circuit. Heads in layers 0–5 and non-END effects remain
outside the measurement; an early router may be tested later upstream of this fixed signature.

See `FAMILY_SWITCH_METRIC.md` for standardization and later nested interventions.

## Frozen feasibility rules

The pilot is promising only if all these checks pass:

1. Clean IO-vs-subject accuracy is at least 0.75 in each split × family cell.
2. A nearest-centroid classifier fit to discovery causal signatures achieves at least 0.75 held-out
   balanced accuracy for *each* ablation baseline, with each held-out template above chance.
3. Discovery-fitted family separation has a positive held-out paired axis projection for both
   baselines, and a 95% hierarchical bootstrap interval excludes zero.
4. Implementation invariants, frozen-file hashes, finite measurements, and identity interventions
   pass. Mean clean-minus-corrupt IO–subject logit difference must be at least 0.25 in each
   split × family cell. Ineffective corruption prevents strong conclusions about its corresponding
   replacement baseline even if zero ablation separates the families.

Bootstrap uncertainty resamples template and unordered-name-pair blocks, retaining both role
assignments and the matched ABBA/BABA pair. Report its extremely small number of clusters.
Use a 10,000-draw seeded bootstrap and a conditional sign-flip check on the eight held-out
template × name-pair blocks; the latter conditions on the sampled templates and is not an
independent population-level test. Do not treat 72 heads as 72 independent observations.

Report within-family distances, cross-family distances, raw effect heatmaps, per-template
accuracy, and control-baseline sensitivity. Supervised ABBA/BABA separation is not unsupervised
evidence of exactly two circuit families. A passing pilot is **Gate 1 provisional**, pending
larger family validation and comparison with the actual source method. No threshold tuning on
held-out outcomes is permitted; any change starts a new numbered pilot.

## Four-donor dataset for the later causal test

For each recipient with IO=A and subject=B, choose C distinct from both. Construct all four cells
with B fixed, the same donor wording within each quadruplet, and matched token lengths:

| Donor family | Donor IO token | Purpose |
|---|---|---|
| Same as recipient | A | Family and answer matched |
| Same as recipient | C | Answer changes, family fixed |
| Opposite | A | Family changes, answer fixed |
| Opposite | C | Both change |

Use two predeclared wording strata: the recipient template and an alternate template. The
same-template/same-family/same-answer cell is an identity technical control; it is not independent
support for invariance. The alternate-template stratum supplies non-identical same-family donors.
For the different-answer cells, preserve the subject and replace only IO mentions. Keep donor
templates within the recipient's split. An independently reserved confirmatory split will be
frozen before candidate selection. Names, donor choices, roles, token alignment and answer IDs
must be recorded. Never choose donors based on their observed effect.

The present turn specifies this dataset; it does not execute donor-transfer experiments.

## Subsequent gates

- **Gate 1:** stable causal dependency families, independently validated beyond the small pilot.
- **Gate 2:** predict downstream signature values on unseen names/templates; compare simple prompt
  variables, answer/token identity, and internal representations. Predicting word order alone is
  not success. Freeze centroids, scales, predictor, and test manifest before evaluation.
- **Gate 3:** select an early/middle candidate using discovery data. Separate answer identity,
  wording, role binding and family information. Probe accuracy alone is insufficient.
- **Gate 4:** bidirectional edits produce reliable movement in the frozen downstream signature;
  quantify raw effects, orthogonal movement, endpoint damage and donor-answer intrusion.
- **Gate 5:** pass same-family, shuffled-donor, wrong-position, wrong-layer, unrelated-component,
  norm-matched random-direction, answer-matched and wording-matched controls across baselines.
  Use fresh held-out names/templates and block bootstrap intervals. Correct selection/multiple
  testing, and add joint or pathway-specific validation before making circuit-level claims.
- **Gate 6:** replicate in Pythia-160M or a different task after a behavioral screen. Do not assume
  Pythia's meaningful partition is ABBA/BABA. Its weights are not requested for this pilot.

Optional backup/self-repair and developmental studies remain deferred. No full factorial dataset
starts before a stable Gate 1 result. Even a promising small pilot does not authorize claiming
that all six gates passed.
