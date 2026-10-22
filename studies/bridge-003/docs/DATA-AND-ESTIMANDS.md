# Data and estimand notes

## Unit of analysis and data separation

The task is next-token indirect object identification. In an ABBA prompt, the
second mentioned person appears again as the subject; the correct completion is
the first person. In BABA the first mentioned person is repeated; the correct
completion is the second. Both actor-role assignments are retained. Corruption
uses the third name in an exclusive triple. Exact texts, actor identities, target
tokens and semantic token positions are saved in the per-model candidate JSONs.

Sixty names were chosen by deterministic tokenization checks only, without model
accuracy filtering. All are single tokens with their leading space in both
tokenizers. Screening has four triples, discovery four, arm A six, and arm B test
six; the four banks are disjoint, including corruption names. They exclude Bridge
001/002 names, not every historical experiment or pretraining occurrence.

The arm A test and each selected-pair arm B test have 384 paired cells:
8 location/object blocks × 6 triples × 2 roles × 2 wordings × 2 orders.
These are 48 block-by-triple combinations with eight related configurations each.
Bootstrap resampling crosses blocks and triples and preserves the paired edits.
It does not treat each intervention as an independent observation or include
uncertainty from selecting the discovery fit. Blocks vary locations and objects;
they are not eight independent grammatical constructions.

The broader screen has 32 cells per family/model and discovery 64 per retained
family/model. Test names and location/object blocks differ from both. The selected
pair's grammatical skeletons necessarily recur from screening to test; the holdout
tests new combinations within those selected families, not arbitrary language.

## Intervention and recovery

For a donor matched in block, name triple and relevant factorial conditions, form
the donor's clean-minus-corrupted source-head output delta at the repeated-subject
position S2. Add that delta to the corresponding heads of the corrupted recipient.
The fixed source heads are historical selections; the study does not discover a
new complete circuit. For random controls, replace each head delta's direction
while preserving that head's norm. Three control draws are retained.

Recovery is the edited-minus-corrupted change in the intended-name-minus-subject
logit difference. It is not an accuracy percentage or a fraction of causal mediation.
Pairwise accuracy checks the two candidate names; vocabulary top-1 additionally
requires the intended name to beat every token. Neither metric is a sampled
multi-token generation test.

## Conditional dependency signature and geometry

For each later head, replace its output at END with zero or its discovery-clean
mean and record the logit/probability effect. The collection of single-head effects
is the dependency signature. It omits earlier heads, other positions, MLPs, joint
effects and path structure. Different references define different interventions.

Discovery-only within-cell residual scales (floor 0.1 logits) standardize the
signature coordinates. Discovery family/order centroids define a donor axis.
The reported geometry uses the logit-effect channel. Probability effects are
retained as behavioral/signature outputs, but are not a second full geometry analysis.
S projects the edited-minus-clean-recipient displacement onto that axis, normalized
by squared centroid separation. Orthogonal displacement measures the remainder,
scaled by axis length. Actual donor proximity compares the edited signature with
the matched clean recipient and matched clean donor, beyond centroid projection.
Positive S can coexist with large off-axis movement or greater recipient proximity.

Clean separability is an operational classification check, not a proof of two
different algorithms. Wording families change multiple surface features together:
for example F1 uses “went” and “gave”, whereas F2 uses “walked” and “passed”.
The study cannot assign the effect to one verb independently.

## Causal group follow-up

In arm A, compare the upstream edit's effect with and without a downstream clamp
to the corrupted recipient's value. The unedited clamp is also measured and must
be an identity control. A group can support the upstream edit's net effect, oppose
it, or have an uncertain conditional effect. The union need not equal the sum of
separate clamps, especially for heads at different layers with shared downstream
computation. Query, key and value clamps intervene on different components and
positions; they should not be interpreted as additive percentages of explanation.

Arm A's directional tests use a discovery-fixed sign (+ wording, − order); the
secondary component table additionally preserves original signed logit effects.
The arm A wording pair is inherited from Bridge 002; arm B selects a broader pair.
Arm A's clamp explanation cannot be assumed to hold for arm B without measuring it.
The union's equivalence margin is ±0.2 logits. Failing that criterion does not
establish an unchanged mechanism, and a failed broad switch criterion does not
prove an absence of donor influence or reject all possible circuit representations.

## Post hoc paired-axis diagnostic

After the strict off-axis rule failed a true-donor endpoint check, a separate
descriptive projection was added. Let u be the standardized edited-minus-clean-
recipient signature and v the standardized clean-donor-minus-clean-recipient
signature for the matched example. The paired score is S_pair = (u·v)/(v·v).
The squared-distance difference satisfies

`||u||² − ||u−v||² = ||v||² (2 S_pair − 1)`.

Thus a particular edited example is closer to the donor exactly when S_pair > 0.5.
The actual clean recipient and donor have exact scores 0 and 1, respectively.
The saved code checks that identity. This algebra does not make the measurement a
complete circuit comparison or convert an exploratory diagnostic into a primary
test. Read the dated `GEOMETRY-CALIBRATION-NOTE.md` for why it was added.
