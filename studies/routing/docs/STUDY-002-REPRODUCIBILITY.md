# Reproducing study 002

Use the same project `.venv` and `uv.lock` as the pilot. No new environment was created. All study
inference is CPU float32; four threads and batches of 16 are fixed in `configs/study-002.json`.

## Original sequence

From the project root:

```bash
.venv/bin/python scripts/freeze_study2.py
.venv/bin/python scripts/run_study2_family.py
.venv/bin/python scripts/run_study2_transfer.py
.venv/bin/python scripts/analyze_study2.py
```

The first command created the 256-recipient/768-prompt library and source/data freeze. The second
evaluated only discovery and development and created the candidate-selection freeze. The analysis
code received a separate SHA-256 freeze before the third command could access test outcomes. That
record is `data/study-002-analysis-freeze.json`; it verifies the exact analyzer.

Do not rerun those commands into populated original output directories. They refuse overwrite.
For a fresh end-to-end rerun:

```bash
.venv/bin/python scripts/reproduce_study2.py --output-dir reproduced/full-study-002-new
```

This helper extracts and hash-checks the original frozen source and data into a new directory,
adds the separately frozen analyzer, and arranges a link to the same read-only model cache. It
uses the existing interpreter and explicitly points imports at the recovered source; it does not
create or modify an environment. It reruns discovery/development, freezes that rerun's selected
predictor and analysis before test inference, then executes all transfers and analyses. Runtime
timestamps and run-record hashes will differ. Original outputs are never overwritten. The raw
test data remain scientifically observed: a reproduction does not make them a new holdout.

Archive recovery can be checked without model inference:

```bash
.venv/bin/python scripts/reproduce_study2.py --output-dir reproduced/source-check-new --prepare-only
```

Recovery was checked for all 18 frozen files. A second full model sweep has not been run merely
to regenerate the manuscript; the saved-statistics replay provides the reported numerical check.

## Recompute statistics without model inference

```bash
.venv/bin/python scripts/analyze_study2.py --output-dir reproduced/study-002-statistics
.venv/bin/python scripts/verify_study2_results.py --replay reproduced/study-002-statistics
```

Use a new output directory for each replay. The analyzer verifies all frozen model/data/source
files, the selection and analysis freezes, and every raw result hash. It then replays the same
paired block bootstrap and gate rules. Compare `summary.json` with the original analysis; the
summary should reproduce byte-for-byte.

## Manuscript

```bash
make -C paper paper
```

This regenerates manuscript result sections and figures from saved statistics and compiles the
PDF. It does not run new model experiments. The final PDF is
`output/pdf/causal-dependencies-research-draft.pdf`; `paper/main.tex` is the editable manuscript.
The manuscript uses LaTeX because its equations and linked scientific tables benefit from the
existing local TeX toolchain. The PDF is rendered and visually inspected before delivery.

## Evidence

- `data/study-002-freeze.json`: prompt/model/source hashes before study inference.
- `data/study-002-frozen-source.tar.gz`: corresponding source snapshot.
- `data/study-002-selection-freeze.json`: coefficients/site frozen before test outcomes.
- `data/study-002-analysis-freeze.json`: analysis source frozen before test inference.
- `results/study-002/family/measurements.npz`: discovery/development signatures and activations.
- `results/study-002/family/selection.json`: all six candidate predictors and the frozen choice.
- `results/study-002/transfer/natural.npz`: every test recipient's natural signature and behavior.
- `results/study-002/transfer/*__*.npz`: all 30 condition outputs, including identity and failed edits.
- Matching JSON files: actual donor IDs, edit locations, control seeds, durations and hashes.
- `results/study-002/hook-verification.json`: additional native/suffix checks on old pilot prompts.
- `results/study-002/analysis/`: complete contrasts, controls, per-example metrics and provenance.
- `results/study-002/verification.json`: raw/source hashes, exact statistical replay, signature
  geometry, identity conditions, random-control norms and prior-project integrity checks.

The descriptive prompt-order prediction baseline is explicitly post hoc. Its timing, method and
output are documented in `docs/POSTHOC-PREDICTION-DIAGNOSTIC.md` and
`results/study-002/prediction-diagnostic.json`; it changes none of the frozen causal criteria.

The additional plot/manuscript builders and explanatory documents were written after the initial
source freeze. They do not change the frozen measurement/selection/analysis programs. Original
pilot 001 files and the earlier completed project remain unchanged.
