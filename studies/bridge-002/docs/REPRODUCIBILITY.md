# Reproducing Bridge 002

Run commands from `bridge-project`, using its existing locked
`.venv`. The follow-up adds no model download, trained weights or dependency
upgrade. The original source modules and the completed Bridge 001 are preserved.

## Retained development history

The original `code/design.py` and `code/screen.py` generated and screened four
grammatical families. The mandatory canonical family failed the stricter Medium
top-1 threshold. `data/family-selection.json` retains that failure. No follow-up
discovery or test inference preceded the amended screen.

`code/amendment.py` and `code/screen_v2.py` add three grammatical candidates and
four new screening name triples/blocks. The second screen selected families 1
and 4 in the fixed candidate order. The successful-screen record is
`data/family-selection-v2.json`; all unsuccessful and unused candidates remain
in the data. See the protocol and `SCREENING-AMENDMENT-001.md` for the decision
rule and its limited scope. Do not rerun builders into these retained paths.

## Original execution sequence

Numerical validation ran on already screened prompts. Source and inputs were
then frozen before discovery. The fit stage measured discovery only and froze
scales, centroids, selected heads, layer-matched controls, effect orientation and
the analyzer before test. For each model:

```bash
.venv/bin/python studies/bridge-002/code/run.py technical --model gpt2-medium --device cuda
.venv/bin/python studies/bridge-002/code/run.py fit --model gpt2-medium --device cuda
.venv/bin/python studies/bridge-002/code/run.py test --model gpt2-medium --device cuda
.venv/bin/python studies/bridge-002/code/analyze.py --model gpt2-medium
```

Use `gpt2-small` for the replication. `run.py freeze` was run once after both
technical checks and before either fit. It refuses to overwrite an existing
scientific freeze. The primary and replication ran in separate processes on the
same local GPU. Random edit vectors are deterministic per recipient and draw,
independent of execution order. Inference uses the unchanged shared engine's seed;
random edit labels include the new study ID. Head-control selection uses seed
2026090802 with fixed factor offsets. The shared deterministic bootstrap uses
the original engine package's seed 20260908.

The runner records each complete file and its hash, and writes arrays atomically.
Rerunning a completed stage preserves outputs and exits. An interrupted stage can
resume from recorded files; an unrecorded pre-existing file causes an error rather
than being overwritten. A complete test has 17 raw files: natural measurements,
13 transfers and three causal-trace bundles. Each trace includes seven component
or control clamps under edited and unedited contexts, plus joint reference ablations.

## Model-free replay

Use a new output path for each replay:

```bash
.venv/bin/python studies/bridge-002/code/analyze.py --model gpt2-medium --output reproduced/medium-audit
.venv/bin/python studies/bridge-002/code/verify_results.py --model gpt2-medium --replay reproduced/medium-audit
```

The summary must match byte-for-byte and every per-example array must match
exactly. The separate verifier recomputes geometry, recovers factorial effects
by a saturated linear solve, checks attenuation algebra, verifies reference-clamp
identity, checks timestamps/hashes and audits preservation of earlier files.
The analyzer does not load models. The extra verifier checks external checkpoint
and historical-file hashes and therefore requires the original workspace layout.
Statistical replay is not independent inference replication.

## Fresh inference rerun

Make a new sibling copy of the bridge project, preserving the shared source,
lock, configs, original frozen inputs and the `studies/bridge-002` source/data.
Exclude virtual environments, caches, outputs and reproduced directories. In
the follow-up results directory retain only the screening records and technical
check records; remove fit/test/analysis from the new copy only. Set up the locked
environment there and run fit, test and analysis. The cached model path remains
`../transport-project/models/{model}`. Exact checkpoint hashes are in the
original bridge's `data/source-freeze.json`. The original full study directories
are additionally needed only for the preservation audit. New timestamps and fit
hashes will differ. Previously observed prompts do not become scientifically
unseen through a technical rerun.

The scientific freeze protects the listed files, not subsequently added reporting
and auditing helpers. The original second-screen builder retains two cosmetic
lint findings (import order and a dictionary-loop style) because it was already
frozen before screening; its execution and freeze are preserved. The active
assay, runner, analyzer and finite-design tests passed their code checks before
the scientific freeze. Five new tests exercise donor construction, name separation,
query/key/value clamp scope, per-layer joint references and dose scaling.

## Measurement interpretation

Source-head edits precede every signature head. The shared optimized suffix
measurement is used only for those upstream edits. Downstream q/k/v/z clamps
are evaluated with full forwards in both terms of the interaction, so later
clamps cannot be accidentally dropped by a cached suffix. Joint clamping is a
controlled intervention interaction and can be negative or exceed the original
edit effect. It is not an additive mediated fraction.

Reports and figures render frozen analyses. Descriptive per-head displays, if
included, are labeled secondary and do not change selected groups or tests.

## Combined research snapshot

After both models complete and pass verification, `code/package_combined.py`
creates `../../output/combined-bridge-research-v2.zip`. It includes the source,
inputs, raw results, scientific freezes, reports, dependency locks and historical
manuscripts from all three project directories. `bridge-project/
COMBINED-PROJECT.md` is the current entry point. The earlier manuscripts are not
a combined submission draft.

The package includes the actual routing pilot input that its historical
clean-reconstruction helper omitted. The frozen helper itself is unchanged;
its original clean-reproduction issue is not claimed repaired. Virtual
environments, model checkpoints, duplicate replay outputs and older output
archives are excluded. Exact inference weights must be obtained separately.
The full 549-file preservation audit still needs the original workspace's
excluded archives; the archive's own manifest independently verifies every
included file.

For model-free replay after extraction, install the bridge's locked packages
and ensure the extracted `src` directory is the imported package (an editable
environment from another copy could otherwise resolve that copy). From the
extracted bridge directory, for example:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python studies/bridge-002/code/analyze.py --model gpt2-medium --output reproduced/package-medium
```

Repeat for Small with another new output path. Compare each regenerated
`summary.json` and all arrays in `per-example.npz` with the retained originals.
This does not require checkpoint files or a GPU. Archive CRC and per-file SHA-256
checks are recorded alongside the package, separately from statistical replay.
`code/audit_snapshot.py` performs extraction, confirms imports resolve to the
extracted project, and runs both model-free analyses with the existing locked
dependencies. It compares the summaries and all metric arrays exactly and writes
`../../output/combined-extracted-replay-v2.json`. This exercises portable saved-data
analysis; it is not a dependency reinstall or a fresh inference run.
