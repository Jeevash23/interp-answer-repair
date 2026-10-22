# Reproducing Bridge 007

Bridge 007 is complete: all sixteen primary criteria passed, the independent
saved-observation audit passed, and deterministic primary/secondary replays
matched the retained outputs. Run the commands below from the
`<project>` project root using its existing `.venv`.

## Safe checks on the retained study

The preferred analysis replay reruns the frozen primary and secondary analyzers
in fresh temporary directories and compares their outputs with the canonical
files. It checks JSON recursively except `utc` timestamps, and array keys,
shapes, dtypes and numerical values exactly, including aligned NaNs:

```bash
.venv/bin/python scripts/replay_extensions.py --only 007
```

This reproduces the original analysis of saved observations. It does not load a
model, repeat inference or add independent samples. The separate verifier
independently checks freezes, measurement/chunk identities, donor mappings,
clean gates and primary statistics. Give it a new output path:

```bash
bridge007_audit_dir=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-007/code/verify007.py \
  --output "$bridge007_audit_dir/verification.json"
```

Do not omit `--output`: the default is the canonical audit, whose bytes are
bound by Bridge 008. After every extension has a terminal canonical audit,
`make verify-extensions` repeats each independent audit in temporary paths and
compares stable content. It is distinct from the original-analysis replay.
A passing integrity audit can verify either scientific success or a correctly
executed stopping outcome.

NumPy-only fixtures are also safe and perform no inference:

```bash
.venv/bin/python studies/bridge-007/code/analyze007.py --mode self-test
.venv/bin/python studies/bridge-007/code/secondary007.py --mode self-test
```

To inspect fresh analysis files rather than the wrapper's temporary outputs,
use new, nonexistent destinations:

```bash
bridge007_replay_dir=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-007/code/analyze007.py \
  --mode qualification --output "$bridge007_replay_dir/qualification.json"
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-007/code/analyze007.py \
  --mode final --output "$bridge007_replay_dir/primary"
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-007/code/secondary007.py \
  --mode final --output "$bridge007_replay_dir/secondary"
```

The retained `results/analysis/` and `results/secondary-analysis/` are the
canonical outputs. Reporting helpers format those saved summaries without
performing analysis; `report007.py --check-ready` is a read-only completeness
check. Do not regenerate canonical reports unnecessarily after release hashing.

## Input provenance

The input freeze covers 148 files: protocol, design/data reviews, four original
scientific code files, candidate data, inherited code/source-head selection,
environment lock and checkpoint/tokenizer bytes. Its SHA-256 is
`921b174ea9b292bc9bd8d35aed6f03a2041780dc6b6d3b69849ab5890d44c43c`.
The secondary implementation has a separate pre-final-analysis freeze. The
independent verifier and reporting helpers were excluded from the scientific
input freeze and are recorded in audit/report and release provenance instead.
The completed independent audit checked 7,953 unique files and 26 measurement
records. Local timestamps and hashes document ordering, not public
preregistration.

The preparation audit records 74 historical input-source hashes and three
name-source hashes. The retained candidate file is the authoritative inference
input. Recreating the original eligible name/context pool requires the original
project snapshot; rerunning preparation after later studies exist can change
which prior inputs are excluded. The pre-freeze article correction and original
drafts remain under `provenance/pre-freeze-drafts/article-repair/`; allocation,
row IDs and sampling sizes were unchanged.

## Model execution and interruption recovery

The retained run used CPU float32, four CPU threads and batches of eight
aligned prompts. Package versions are locked by `pyproject.toml` and `uv.lock`;
checkpoint bytes are locally hash-bound and are not redistributed. Calibration
compares exact causal-prefix inference with the historical full-sequence
implementation. The completed fresh-engine replay reproduced all 3,200
wording/corrupt behaviors exactly and checked singleton-ablation signatures
on eight fixed cases below the 0.0005 tolerance. These are computational checks
on the same observations.

An interrupted, unfinished run can resume only with unchanged input identities
and verified chunk hashes, after its previous process has stopped. That
recovery behavior is not a safe rerun procedure for this completed release:
**re-executing the completed pipeline can rewrite status/completion timestamps
and invalidate downstream hashes even when it reuses measurements.** Do not
rerun it in the retained evidence tree.

Full model re-execution belongs in a separate preserved copy of the complete
project, with the same environment/checkpoint inputs and new run provenance.
Inherited paths, stage freezes and dependencies require care; no clean-rerun
command has been validated here. Do not delete freezes or reset canonical
outputs to make a command proceed. Run model processes sequentially, using
four CPU threads; the safe routine reproduction path is saved-observation
verification and replay above.

## Statistical scope

Behavior arrays have shape `[n,6]`, with logit gap, probability gap, pairwise
correctness and top-1 correctness in the first four columns. Signatures have
shape `[2,n,176,2]`: zero/fit-mean reference, manifest row, downstream head,
then logit/probability effect. Primary analysis uses logit outcomes, fit-only
scales and the full balanced crossed block/name-triple resampling design.

Primary and secondary analyses are separately declared and frozen. Secondary
controls, heterogeneity and geometry checks do not replace a primary endpoint.
The development-planning record uses historical Bridge 005 observations;
it is not new confirmatory evidence. Two qualified pairs in one checkpoint,
related templates and pooled similarity bounds do not establish broad model,
syntactic or computational equivalence.
