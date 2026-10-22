# Experiment 005: post-hoc robustness of activation transport

Specified 5 September 2026 after inspecting Studies 1--4. This is an explicitly
post-hoc robustness extension on the existing eight test templates (160 examples)
per model, not an untouched confirmatory replication or public preregistration.
No original study, freeze, selection, or result is overwritten. The runtime
records source/data hashes before inference and refuses existing output folders.

## Questions and interventions

Models remain cached GPT-2 Medium and Pythia-410M-deduped, float32. No training,
model downloads, new dependencies, or system changes are required. CPU execution
is supported. Original condition-specific groups reproduce all four native
source-target directions at S2 and END; endpoints are checked against saved
original results with historical probability tolerance 0.001 and logit tolerance
0.01, separately from the stricter same-computation hook-placement check below.

1. Do canonical-to-controlled effects exceed genuinely sampled alternative heads
   and norm-matched random directions? Draw five independent layer-matched head
   sets, uniformly without replacement within layer, excluding the union of
   selected groups. Average draws per example before template inference; report
   between-draw ranges. Random residual directions are generated per immutable
   model/example/site/head/draw key and match each head's native delta norm.
2. Is directionality stable when group identity and total injection norm are
   held fixed? Use the union of canonical and controlled selected heads in both
   directions. For each paired example, set both concatenated head-result delta
   norms to the geometric mean of their native norms, then use multipliers
   0.5, 1, 2. A zero source norm sets both matched doses to zero and is counted.
   This matches an explicit input norm, not downstream gain or semantic content.
3. Does holding the target's selected END nodes at their corrupt values reduce
   the effect of canonical S2 injection? Run all four combinations of S2
   injection and END clamp, including five matched random END clamp groups.
   Report attenuation, remaining gain, and attenuation beyond the random clamps.
   These are controlled node-blocking/dependence diagnostics, not edge/path
   isolation or natural indirect effects. Some S2 and END groups share layers.

Entity-mismatch controls use a seeded disjoint donor permutation in each full
template pool, preserving ABBA/BABA and requiring disjoint IO, subject, and
corrupt names. A deterministic augmenting-path matching with seeded candidate
order finds a one-to-one assignment shared by both patterns. Save explicit donor
IDs before target minibatching. This replaces both historical donor mappings.
An off-position control adds the original vector one token earlier.

Per-layer attention-output additions equal summing the original per-head result
additions, without materializing large head-result tensors. Downstream selected
layers keep their live target computation after upstream changes: this is
multilayer additive steering, not simultaneous clean-state replacement. Source
head values are cached before minibatching; END clamping replaces head values
before their fixed linear output projections.

## Measurements and checks

The primary endpoint is raw change in P(IO)-P(subject); logit change and CTS are
secondary descriptions. CTS divides by the target's whole-model clean-corrupt
effect, not by the effect of the selected group. No claim about causal influence
ordering follows from normalized ordering alone. The old raw END unembedding
projection is not reused as a mechanistic gate in this extension.

There are 56 intervention specifications and 160 examples per model. Retain all
examples, including negative outcomes, without correctness-based selection.
Bootstrap whole templates 5,000 times with seed 20260905, preserving donor-linked
examples inside their template. Average control seeds within example; seeds and
prompt rows do not inflate the eight independent template clusters. Intervals
are descriptive, pointwise, and exploratory, not multiplicity-adjusted tests.
Report every template's result. Original two-stage intervals remain historical.

Required implementation checks: immutable-ID controls under order/batch changes;
same head union in both directions; matched norm equality; no-op zero injection;
no-op END clamping without injection; original/native endpoint reproduction;
complete row coverage and finite values. Full-run hashes and numerical errors
are saved. A smoke result is explicitly not a completed study.

## Scope

This study can establish robustness or sensitivity of measured interventions.
It cannot establish algorithmic equivalence, a complete circuit graph, a natural
indirect effect, superiority over overlap/representation predictors, or
generalization beyond the reused synthetic template pool. Those claims are
removed or deferred rather than marked achieved. Subsequent work may use new
untouched templates after a separate prospective specification.

## Pre-full-run implementation checks

The first four-example GPT-2 smoke run passed. A manifest-only check then showed
that the originally proposed previous-pair mapping did not guarantee disjoint
names across every template. Before any full Study 5 run, it was replaced with
the specified disjoint matching; the first smoke folder is retained as a
superseded implementation check, not a study result. This change used only
entity identities, not transport outcomes. Pythia's first CPU baseline check
also exceeded the initial probability tolerance; execution numerics are audited
before accepting a full run. No failed smoke output is treated as evidence.

## CPU numerical amendment before the full runs

The recovered task independently reran all 160 Pythia test examples with both
the original per-head attention contraction and the memory-efficient contraction.
The original computation on CPU itself differs from the historical GPU baselines
by at most 0.000324131 probability units and 0.002351761 logit units. The new
contraction differs from those historical baselines by at most 0.000244052 and
0.002546311, respectively. Switching only the CPU contraction differs by at most
0.000352636 and 0.002389908. Thus the initial historical probability tolerance
of 0.0001 fails even for the original computation on CPU.

Before either full Study 5 control run, historical endpoint agreement is given
separate absolute tolerances of 0.001 probability and 0.01 logit units. These
bound the measured cross-device and reduction-order numerical envelope; they
are not thresholds for statistical significance or effect selection. All actual
baseline and native-patch errors are retained. A separate diagnostic isolates
hook placement under the same per-head target computation, retaining the original
0.0001 probability / 0.005 logit limits for that equivalence check. It also
separates source projection order from target contraction. No historical code,
result, or failed diagnostic is overwritten to make it pass.

Full runs use CPU float32, four threads and 20-example batches (one complete
template) for throughput. Donor IDs, sampled head sets, and per-ID random seeds
are independent of this batch choice. Floating-point endpoints need not be
bit-identical across batch sizes. Full native agreement checks and the no-op
checks therefore apply at the actual execution batch size. The runtime records
all model weight, model config, inference helper, data and protocol hashes.

The isolated hook-placement diagnostic completed on 16 examples (one ABBA/BABA
pair from each of eight templates), all four source/target combinations, and
both sites. Its maximum errors are 0.000099406 probability and 0.001409531 logit
units, passing the isolated limits without an amendment. The complete new CPU
implementation versus original CPU computation differs by at most 0.000180662
and 0.001545907 on this diagnostic; versus original saved GPU patch outcomes,
it differs by at most 0.000116997 and 0.001127244. These subset diagnostics do
not replace the full baseline and native-patch validation in each final run.
