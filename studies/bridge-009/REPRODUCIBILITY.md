# Reproducing Bridge 009

Bridge 009 is complete at the verified `no_screen_pair` stopping outcome.
Calibration and a 96-row behavioral screen were measured; scientific fitting,
selection signatures, qualification and final transfer were not. Run checks
from the `<project>` project root using its existing `.venv`.

## Safe checks on retained observations

Independently recheck all applicable evidence and the stopping decision using
a fresh output path:

```bash
bridge009_audit_dir=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-009/code/verify009.py \
  --output "$bridge009_audit_dir/verification.json"
```

The canonical audit passed for 635 unique files and four measurement records,
including unchanged inputs, exact Small tokenization, the preserved amendment,
numerical calibration and behavioral screen. Passing the audit verifies this
bounded failure; it does not establish scientific transfer success.
Do not omit `--output` or overwrite canonical records during routine checks.

The common saved-observation replay explicitly skips nonexistent final
analyses for this verified stopping branch:

```bash
.venv/bin/python scripts/replay_extensions.py --only 009
```

A `SKIP` for `no_screen_pair` is expected. Do not run the final primary or
secondary analyzers or fabricate outputs for stages never measured. After all
extensions have terminal canonical audits, `make verify-extensions` repeats
each independent audit in temporary paths and compares stable content.
`make replay-extensions` separately reruns applicable frozen original analyses
in temporary directories. Neither workflow runs a model or adds observations.

Synthetic fixtures remain safe to run:

```bash
.venv/bin/python studies/bridge-009/code/analyze009.py --mode self-test
.venv/bin/python studies/bridge-009/code/secondary009.py --mode self-test
.venv/bin/python studies/bridge-009/code/report009.py --check-ready
```

The reporting check reads completeness without writing. Existing `RESULTS.md`,
`tables/summary.tex` and `results/report-manifest.json` already describe the
verified stop and bind their source summaries. Avoid unnecessary regeneration
after release hashes are recorded.

## Frozen inputs and amendment

Preparation used local tokenizer files, retained the exact Bridge 007 candidate
manifest and checked inherited input/selection hashes. It refuses preparation
after this study's input freeze exists. Do not delete the freeze or rerun
preparation in the retained tree. The planned all-four-family mean bank and
F10/F11 fitting signatures remain design quantities, not observed datasets.

The effective input freeze contains 156 files, SHA-256
`703e5e1edf64432debd40f25013320dc641d2eca53f0c2f0004447e5ae62e7c4`.
The secondary implementation/plan was separately frozen before calibration,
SHA-256 `a64da08bf278bd0b8ac5d01dbd454f870948f6e5fa48c94d7540a58c8ad256b3`.
Both effective freezes date to 4 October 2026 at 18:52 UTC. Their scientific
code inventories exclude independent verification and reporting helpers, which
are recorded separately in audit/report and release provenance. Local freezes
are not public preregistration.

`data/pre-inference-amendment.json` records the exact descriptive seed-label
correction, confirms the numerical seed was already 2026100409, and maps the
archived source and original manifests to preserved paths. The verifier checks
those archived manifests through their path overrides and checks the effective
current manifests. Do not apply a superseded manifest to the corrected live
source or delete it as a mismatch.

## Model execution and interruption recovery

The retained run used CPU float32, four threads, batches of eight, exact
causal-prefix reuse and the original GPT-2 Small source heads/configuration.
Package versions are locked by the project environment; model/tokenizer bytes
are locally hash-bound and model weights are not redistributed. Calibration
compares the prefix engine with full-sequence `assay.CausalEngine` using the
Small configuration and GPT-2 suffix. Historical `Engine003('gpt2-small')`
loads Pythia and must not be substituted.

An unfinished run interrupted before a terminal outcome can resume only after
its previous process stops, with unchanged inputs and verified chunk hashes.
That recovery facility does not make completed re-execution safe:
**rerunning the completed pipeline in this retained tree can rewrite status
or completion timestamps and invalidate later hashes.** Do not do so.

Full model re-execution requires a separate preserved copy of the complete
project, the same environment/checkpoint inputs and new run provenance. The
inherited dependency and stage-freeze structure needs care; no tested
one-command clean-rerun recipe is provided. Never delete freezes or reset
canonical outputs to force execution. Keep model processes sequential and
use four CPU threads.

The exact cases are shared with Medium, so cross-model comparisons remain
correlated. This attempt supplies no additional qualified transfer comparison,
no new independent language cases and no evidence across architectures.
