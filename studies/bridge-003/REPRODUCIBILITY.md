# Bridge 003 reproduction

Run commands from `bridge-project`. Use the existing Python 3.12
environment, installed from `uv.lock` with the `experiment` extra and development
dependencies (`uv sync --locked --extra experiment`). No new packages, model
training or checkpoint downloads were needed for the recorded study.

## Replay saved statistics without a GPU or model weights

Use new output names; the analysis deliberately refuses to overwrite a directory.

```sh
.venv/bin/python studies/bridge-003/code/analyze_medium003.py --output reproduced/my-medium
.venv/bin/python studies/bridge-003/code/verify003.py medium --replay reproduced/my-medium
.venv/bin/python studies/bridge-003/code/analyze_broad003.py --model gpt2-medium --output reproduced/my-broad-medium
.venv/bin/python studies/bridge-003/code/verify003.py broad --model gpt2-medium --replay reproduced/my-broad-medium
.venv/bin/python studies/bridge-003/code/analyze_broad003.py --model pythia-410m-deduped --output reproduced/my-broad-pythia
.venv/bin/python studies/bridge-003/code/verify003.py broad --model pythia-410m-deduped --replay reproduced/my-broad-pythia
```

The verification helper also audits 700 unchanged historical workspace files.
That preservation audit needs old output archives deliberately excluded from the
new package. For an extracted package, use `code/audit_package003.py` from the
original workspace to extract, check its manifest and replay all three Bridge 003
analyses plus both Bridge 002 analyses. It checks exact summary bytes and array
values with imports isolated to the extracted project. This uses the existing
locked dependencies, not a freshly reinstalled environment or fresh inference.

```sh
.venv/bin/python studies/bridge-003/code/audit_package003.py
```

The two post hoc diagnostics can also be replayed individually with new output
paths, for example:

```sh
.venv/bin/python studies/bridge-003/code/calibrate_geometry003.py --model gpt2-medium --output reproduced/my-calibration.json
.venv/bin/python studies/bridge-003/code/paired_geometry003.py --model gpt2-medium --output reproduced/my-paired-axis
```

## Recorded inference sequence

Input design was built before inference, with local hash freezes rather than public
preregistration. `docs/PROTOCOL.md` records the scientific rules. The completed
input design and selection records must be preserved. The original stage commands
are recorded below; running them on completed outputs generally preserves them
or refuses duplicate selection, so this is not a command to generate a second run
in place.

```sh
.venv/bin/python studies/bridge-003/code/build003.py
.venv/bin/python studies/bridge-003/code/screen003.py --model gpt2-medium --technical
.venv/bin/python studies/bridge-003/code/medium003.py freeze
.venv/bin/python studies/bridge-003/code/medium003.py test
.venv/bin/python studies/bridge-003/code/screen003.py --model pythia-410m-deduped --technical
.venv/bin/python studies/bridge-003/code/screen003.py --model gpt2-medium
.venv/bin/python studies/bridge-003/code/screen003.py --model pythia-410m-deduped
.venv/bin/python studies/bridge-003/code/broad003.py freeze
.venv/bin/python studies/bridge-003/code/broad003.py fit --model gpt2-medium
.venv/bin/python studies/bridge-003/code/broad003.py fit --model pythia-410m-deduped
.venv/bin/python studies/bridge-003/code/broad003.py select
.venv/bin/python studies/bridge-003/code/broad003.py test --model gpt2-medium
.venv/bin/python studies/bridge-003/code/broad003.py test --model pythia-410m-deduped
```

For a fresh test-inference reproduction, work in a separate full project copy.
Keep the frozen input/selection/discovery records, obtain the model files with the
hashes in `data/checkpoints.json` under the sibling project's `models/` paths, and
move only the copy's `results/medium-cancellation` and the two models' `test`,
`analysis`, `report`, `paired-geometry` directories and `verification.json` and
`geometry-calibration.json` records into a separate historical-results directory.
Then run the three `test` commands, analyses, and both post hoc geometry scripts
for each broader-arm model in that copy. The post hoc scripts' default outputs
are regenerated when no `--output` is supplied. Regenerate reports only after the
corresponding audits. Do not alter the original workspace or its frozen records. A new
screen/discovery study requires its own new records and freezes; it is a different
selection run, not replay of the fixed reported selection.

Model runners resume from complete, hash-recorded files. An interruption during
an unfinished file may require inspection; an unexpected unrecorded file is not
silently trusted. Run GPU models sequentially on the 6 GB GPU, especially Pythia's
raw-model comparison, which temporarily holds both model representations.

## Technical scope and record interpretation

`data/config.json` preserves the inherited Bridge 001 configuration plus the
Pythia head specification. Its old `study` label and 75% thresholds are historical
metadata; Bridge 003's actual 95% pairwise / 85% top-1 / 80% separability gates are
explicit in the frozen protocol and Bridge 003 screen/analysis code. The bootstrap
and norm-matched random generator retain the inherited seed 20260908; the new
design and A/B control selection use 2026090803.

The Pythia arithmetic adapter is inference-only and preserves native checkpoint
weights. Raw-model comparison had zero maximum logit discrepancy on the recorded
GPU calibration prompts. Vectorized intervention equivalence used a documented
0.002-logit tolerance (observed maximum 0.001139); Medium kept 0.0005 (observed
maximum 0.00000954). See `docs/NUMERICAL-CALIBRATION.md` for the pre-screen amendment
and its limitations. Calibration is sampled validation, not a proof over every
possible input or downstream hook.

Four structural tests check disjoint name banks, token positions/donor mapping,
upstream source heads and identical cross-architecture prompt text. Independent
saved-output audits check clamp attenuation, geometry, factorial algebra, run
completeness, freeze ordering and exact statistical replay. Report generation is
post-analysis and never changes selection or raw results. The explicitly post hoc
endpoint calibration and paired-axis diagnostic have separate scripts and output
records. The extracted-package audit replays these too; they are not part of the
original primary analysis or an independent experiment.

```sh
.venv/bin/python -m pytest studies/bridge-003/code/test003.py -q
MPLCONFIGDIR=/tmp/bridge003-matplotlib .venv/bin/python studies/bridge-003/code/report003.py all
```

The package excludes model weights, installed environments, caches, duplicate
replay outputs and older zip files. Their absence must not be described as a
self-contained environment or full-inference reproduction.
