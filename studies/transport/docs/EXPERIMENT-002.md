# EXPERIMENT-002: independently localized transport in GPT-2 Medium

**Status:** design frozen before any GPT-2 Medium circuit outcomes  
**Frozen on:** 1 September 2026  
**Model:** GPT-2 Medium, revision `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`  
**Purpose:** test whether Experiment 001's directional transport survives independent component
localization at a larger model scale

## Motivation and evidential status

Experiment 001 used published GPT-2 Small heads. Its held-out controlled prompts failed the frozen
Small behavioral gate, while a behavior-only follow-up showed that GPT-2 Medium passed on the same
prompts. Experiment 002 therefore uses Medium, but does not copy Small head coordinates. The
Medium behavior results were known when this design was written; no Medium head-localization or
activation-transport outcomes had been computed.

The frozen Experiment 001 manifest is reused so that model scale and component localization change
without also changing prompts. This is a new held-out test of circuit outcomes, not a blinded
replication of the already-observed behavioral result.

## Questions

1. Do independently localized Medium heads differ between canonical and controlled contexts?
2. Can the clean-minus-corrupt signal from heads localized in one context causally improve the
   other context at the same role-matched position?
3. Is any transport directional, position-specific, entity-specific, and stronger than matched
   negative controls?

## Frozen hypotheses

- **H1, localization:** canonical and controlled contexts will have positively associated head-effect
  rankings, but their top-head sets will not be identical.
- **H2, transport:** source-localized correct patches will have positive final effects in both
  directions, conditional on the behavioral and source-group validation gates.
- **H3, asymmetry:** canonical-to-controlled transport will exceed controlled-to-canonical transport,
  reproducing Experiment 001's directional ordering.
- **H4, specificity:** correct canonical-to-controlled patches will exceed random-head,
  norm-matched random-direction, off-position, and mismatched-entity controls.
- **H5, stage:** entity mismatch will retain more of the repeated-subject-position effect than of the
  final-position effect, consistent with entity specificity emerging late.

Failure of H2-H5 is retained as evidence against scale-robust causal interoperability; heads or
thresholds will not be changed after development/test outcomes are seen.

## Model and numerical configuration

- Load the cached safetensors snapshot locally through TransformerLens `3.8.0` compatibility mode.
- Process compatibility weights on CPU, then move the resulting model to CUDA. This path passed a
  measured smoke test at approximately 931 MiB peak allocated VRAM on the 6 GB NVIDIA GPU.
- Use `float32` model weights, logits, probabilities, accumulated effects, and summaries. A two-record,
  two-layer implementation dry run in bfloat16 showed 0.0625-logit quantization, which is too coarse
  for ranking small individual-head effects. The float32 CPU-processing path passed at approximately
  1,855 MiB peak allocated VRAM before the full localization was run.
- Use deterministic inference mode; record PyTorch, CUDA, TransformerLens, Transformers, model,
  source, manifest, and configuration hashes.

## Data and splits

- Reuse `data/experiment-001-manifest.jsonl` byte-for-byte.
- `exploratory` is used only for Medium head localization and implementation validation.
- `development` and `test` are not used for selection, tuning, or head replacement.
- Primary and alternative corruptions retain their Experiment 001 definitions.
- No example is filtered after model behavior or intervention outcomes are observed.

## Phase A: exhaustive exploratory localization

For every one of the model's 384 attention heads, compute an exact denoising activation-patching
effect separately in canonical and controlled contexts at two role-matched positions:

1. **S2 site:** head result at the repeated-subject position;
2. **END site:** head result at the final prediction position.

For each exploratory example, replace one head result in a primary-corrupt run by adding that
head's clean-minus-corrupt result from the same condition and position. The localization effect is
the patched minus corrupt IO-minus-subject logit difference. Heads are ranked by mean paired effect;
negative means are retained in the output rather than clipped.

For each condition and site, freeze the eight heads with the largest mean effect. Ties are broken by
layer and then head index. Report Spearman correlation over all 384 head scores plus top-8 and top-32
Jaccard overlap. These localization comparisons are descriptive because selection and measurement
use the same exploratory split.

Before development or test is run, write the selected coordinates, all localization scores, random
head mappings, seeds, manifest hash, and implementation hashes to an immutable freeze record.

## Phase B: frozen group interventions

For each site and source condition, patch the eight-head group localized in that source condition.
Run four correct source-target pairs:

- canonical into canonical;
- controlled into controlled;
- canonical into controlled;
- controlled into canonical.

The intervention adds each source example's clean-minus-corrupt head-result delta to the target
corrupt run at the same layer, head, and role-matched position. It does not replace the entire target
residual stream.

Required controls for canonical into controlled are:

- eight random heads matched one-for-one by layer and not selected by either condition at that site;
- a norm-matched isotropic random direction for each selected head and example;
- the correct group applied one token before the target role position;
- correct deltas circularly shifted by two rows, ensuring different entity triples because adjacent
  ABBA/BABA rows share a triple.

Random head mappings and random-direction seeds are frozen after exploratory localization and before
development/test execution.

## Outcomes

### Final outcomes

- **Primary:** `P(IO) - P(repeated subject)` at the final token.
- **Secondary:** IO-minus-subject logit difference.
- Report clean, corrupt, patched, paired numerator, full target clean-minus-corrupt denominator, and
  their ratio. Omit ratios when the absolute probability denominator is below `0.01` or the absolute
  logit denominator is below `0.1`.

### Local outcomes

- **S2-site patch:** projection of the patched change in the target condition's frozen END-group
  result onto that group's target clean-minus-corrupt direction.
- **END-site patch:** direct contribution of the actually patched group in the IO-minus-subject
  unembedding direction.

End-to-end recovery without the declared local response is labeled nonspecific rather than successful
mechanistic transport.

## Gates and decision rules

Interpret a split/site/direction mechanistically only when all applicable rules pass:

1. Both source and target clean conditions have at least 70% pairwise IO accuracy and positive mean
   probability difference on that split.
2. The source-localized group has a positive within-source patch effect whose 95% paired bootstrap
   interval excludes zero.
3. The target clean-minus-corrupt denominator exceeds the ratio threshold.
4. The correct cross-condition patch has the predicted local-effect sign.

The test split is the primary confirmatory split; development is reported separately and cannot
substitute for a failed test gate.

## Statistical analysis

- Use 2,000 paired percentile bootstrap samples with seed `20260901` for means, transport ratios,
  directional differences, and correct-minus-control contrasts.
- For the test primary probability outcome, compute paired sign-flip randomization p-values with
  20,000 draws and seed `920260901`.
- Control false discovery rate at 0.05 with Benjamini-Hochberg separately for the two confirmatory
  families: two directional-asymmetry tests and eight correct-minus-control tests.
- Report unadjusted estimates and intervals even when a gate fails, but do not reinterpret a failed
  gate as evidence for equivalence.

## Go/no-go interpretation

| Outcome | Interpretation |
|---|---|
| Similar localization and valid bidirectional transport | scale-robust reusable computation |
| Similar localization but weak transport | component agreement does not imply causal interoperability |
| Different localization but positive transport | different component sets encode interoperable signals |
| Positive canonical-to-controlled only | directional compatibility or target-context fragility |
| Behavioral or within-source gate failure | no mechanism conclusion for that split/site |
| Correct patch does not beat controls | nonspecific activation injection |

## Explicit limitations

The two fixed token positions are role hypotheses, not a full circuit graph. Exhaustive localization
tests head outputs but not MLPs, edges, or distributed subspaces. Top-eight selection is a sparse,
predeclared operationalization, not proof that exactly eight heads constitute the Medium IOI circuit.
The prompt manifest still contains only one controlled template in each held-out split; a later
template-family experiment is required for hierarchical generalization claims.
