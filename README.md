# Answer repair and downstream dependencies in activation transfer

Code and results for the preprint *Answer repair and downstream dependencies in activation transfer*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved summaries behind it; the saved outputs and
the frozen protocols are attached to its v1.0 release (see [Release files](#release-files)). The tables of Studies
2, 5, 6 and 7 and two post hoc analyses can be regenerated from the saved outputs on a CPU without downloading a
model, every freeze record can be checked, and the model runs can be repeated with the pinned checkpoints.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 29.2 MB | The saved outputs: the 14 compressed archives in `data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 245 kB | The studies' protocols, predictions, amendments, approvals and freeze records (77 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-answer-repair
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-answer-repair
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included, and the checks below run as described.

## What the paper asks

Activation transfer copies the difference between a donor's and a recipient's activations into the
recipient. In indirect object identification (IOI) this can repair the model's answer. The paper asks whether
a repaired answer also makes the model's later computation resemble the donor's, measured by a signature of
downstream attention-head dependencies, mainly in GPT-2 Medium. It compares transfer from a reworded donor
with transfer from the recipient's own clean run, tests the comparison prospectively on fresh cases, varies how
many source heads are copied, tests whether the signature detects a change of route known in advance, and
attempts the same design in GPT-2 Small and GPT-2 Large.

## Studies

Folder names are the original bridge numbers; the manuscript numbers the studies as below.

| Manuscript | Folder | Model | What it did | From this release |
|---|---|---|---|---|
| Study 1 | `studies/bridge-005` | GPT-2 Medium | F6/F7 wording pair, the original qualified experiment (1,152 final configurations) | Summaries, four test arrays for the post hoc baseline |
| Study 2 | `studies/bridge-007` | GPT-2 Medium | Prospective F10/F11 confirmation (3,200 configurations); sixteen primary criteria | Summaries; tables regenerated; analysis self-tests |
| Study 3 | `studies/bridge-008` | GPT-2 Medium | Source coverage (12 versus 112 heads); full criterion not met | Summaries |
| Study 4 | `studies/bridge-009` | GPT-2 Small | Fixed F10/F11 attempt; stopped at the behavioural screen | All records |
| Study 5 | `studies/bridge-010` | GPT-2 Medium | Exploratory geometric null on the Study 2 cases | Summaries; tables regenerated |
| Study 6 | `studies/bridge-011` | GPT-2 Medium | Prospective positive control for the signature (GPU) | Summaries; tables regenerated |
| Study 7 | `studies/bridge-012` | GPT-2 Large | Study 2's pipeline on rented GPUs (Modal); stopped at pair selection | Every record pulled back from the run; tables regenerated |
| F4/F7 follow-up | `studies/bridge-006` | GPT-2 Medium | Second wording pair; fails independent qualification | Summaries |
| Historical comparisons | root `scripts/`, `results/` (Bridge 001), `studies/bridge-002` to `bridge-004` | GPT-2 Small and Medium, Pythia-410M, TinyLlama-1.1B | The earlier joint experiments | Summaries |
| Earlier experiments | `studies/transport` (T1–T5), `studies/routing` (R1–R2) | GPT-2, Pythia-410M | Transport and dependency-prediction experiments merged into this project | Code, records, summaries |

Each study folder keeps its own protocol (`docs/`), code, frozen inputs (`data/`) and records; frozen
measurements were never edited. Local freezes are not public preregistrations.

## What is in this folder

| Path | Contents |
|---|---|
| `src/`, `studies/transport/src/`, `studies/routing/src/` | The shared packages (`transport_bridge` engine and statistics, `activation_transfer` path checks, `circuit_transport`, `causal_circuit_routing`) |
| `studies/*/code/`, `scripts/` | Each study's pipeline, analysis and verifier; `scripts/` holds Bridge 001 and the historical replay entry point |
| `analysis/` | Post hoc analyses, the figure scripts and the plain-NumPy re-implementation check; `analysis/results/` holds their saved outputs |
| `paper/review/`, `paper/sections/` | The table renderers and two post hoc analyses, and the table fragments they write (the rest of the manuscript is not included) |
| `tests/`, `studies/*/tests/` | Unit tests |
| `data/saved-outputs/` | The saved summaries as compressed archives (29.2 MB); `make unpack` places them in the study folders (attached to the release: `per-run-outputs.tar.gz`) |
| `provenance/source-relocations.json` | The ledger of source files whose paths changed when the earlier projects were merged, with digests before and after |
| `tools/` | Unpacking, freeze checking, model download and comparison helpers |
| `pyproject.toml`, `uv.lock` | The exact environment of the runs (see `ENVIRONMENT.md`) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # uv sync --frozen --extra experiment --group dev: creates .venv from uv.lock
```

This downloads about 3.0 GB of packages (most of it PyTorch's CUDA libraries); `ENVIRONMENT.md` lists them and
gives a CPU-only route. To use another environment with the same packages, pass `PYTHON=/path/to/python` to any
`make` target. The Makefile puts the shared packages on `PYTHONPATH`, so they need not be installed.

## Quick check (CPU, a few minutes, no model, no network)

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make quick-check
```

It unpacks the saved outputs and compressed inputs (checking every SHA-256), then:

1. checks every freeze record against the files it names (`tools/check_freezes.py`): each file is unchanged,
   changed for the release in a documented way, relocated with recorded digests, or documented as not shipped;
2. runs the freeze verifiers of Studies 5, 6 and 7 (Study 7's also checks its documented navigation updates);
3. regenerates the Study 6 and Study 7 tables and checks them against the shipped fragments (`--check`), and
   regenerates the Study 5 and Study 2 secondary tables and compares them byte for byte
   (`tools/compare_regenerated.py`);
4. runs the self-tests of Study 2's frozen primary and secondary analyses.

Expected end of the output (about half a minute on a CPU that was busy with other work):

```
total: 37 changed for the release (documented), 1 frozen copy matches, 2634 not shipped (documented), 87 relocated (recorded with digests before and after), 984 unchanged
Study 7 freezes verified (131 file hashes; 8 documented navigation update; 28 listed files kept on the Modal Volume)
Study 6 freezes verified (1348 file hashes)
Existing Study 5 freeze verified
Study 6 tables match the saved summaries
Study 7 tables match the saved records
Rendered Study 5 manuscript tables from verified saved outputs.
wrote sections/study2-secondary-table.tex
identical  paper/sections/geometric-null-main-table.tex
identical  paper/sections/geometric-null-tables.tex
identical  paper/review/study5-table-provenance.json
identical  paper/sections/study2-secondary-table.tex
Study 2 primary analysis self-test passed
Study 2 secondary analysis self-test passed
```

## More checks from the saved outputs

```sh
make verify       # quick-check, then the post hoc qualification intervals and the post hoc corrupted-signature
                  # baseline rerun from saved arrays and compared byte for byte (about a minute more)
make test         # the 93 unit tests; they read the GPT-2 Small and Pythia-410M tokenisers (make tokenisers, a few MB)
make figures      # Figures 3, 4, 5 and 7 into paper/figures/
```

The study verifiers count a frozen file that the release documents as not shipped as checked;
`make check-freezes` gives the exact breakdown for every record.

Some checks of the original project need raw arrays that are not included because of their size: the
independent saved-observation audits of Studies 1–3, the historical evidence manifest (1.4 GB), the
configuration-level distributions of Study 2 (Table 30; 31 MB of test arrays), the post hoc injected-difference analysis
(1.1 GB) and the post hoc self-versus-wording analysis of Study 1 (214 MB). They can be run again after the
arrays are regenerated with the study pipelines. The manuscript's own number checkers read the manuscript's
LaTeX source, which is not part of this release.

## Models

```sh
make models                                   # GPT-2 Medium and Small (about 2.1 GB)
python tools/get_models.py --list             # every checkpoint: repository, revision, size, which studies use it
make independent-check                        # plain-NumPy GPT-2 Medium on 12 Study 1 configurations, compared
                                              # with the saved record (about 15 minutes on a busy CPU)
```

The independent check passes when every behaviour and signature coordinate is within the manuscript's bound of
4.6 × 10⁻⁵ logits of the original pipeline, with identical top-1 predictions. Float32 sums depend on the number of
BLAS threads, so its differences can change between runs in the seventh decimal place.

`models/README.md` explains the rest. Every file is checked against the SHA-256 recorded before the runs, and
the freeze records name the same digests. The historical transport scripts read checkpoints through
`studies/transport/models`, which was a link to `models/` in the original project; `make link-models` recreates it.

## Full reruns

Every model stage refuses to overwrite its outputs and verifies its freeze before running, so a rerun starts
from a copy of the release with the study's `results/` removed and new freezes written by the study's
`freeze*.py`. Each study's `REPRODUCIBILITY.md` (or its README) gives the stage order. The original runs used
one 6 GB NVIDIA GPU in float32, except Study 7. Study 6 ran at 8 prompts and 8 heads per pass,
about 0.04 s per row per condition; the frozen pipelines of Studies 2, 3 and 5 are of the same kind and size
(3,200 configurations each).

### Study 7 on Modal (optional, your own account)

Study 7 ran GPT-2 Large on Modal L4 GPUs from `studies/bridge-012/code/modal_app012.py`, with the image built
from this release's `pyproject.toml` and `uv.lock` and the weights downloaded inside it at the pinned revision.
With the Modal CLI authenticated to your own account (`modal setup`), from the release root:

```sh
AI01_STUDY7_VOLUME=my-study7-codetest modal run studies/bridge-012/code/modal_app012.py --stages all --test   # code test, separate Volume
modal run studies/bridge-012/code/modal_app012.py --stages calibration,calibration_cpu,screen,sources,fit_select
modal volume get <volume> <path> studies/bridge-012/results/<path>                                         # pull records back
```

`AI01_STUDY7_VOLUME` names the Volume (default `ai01-study7`) and `AI01_STUDY7_GPU` the GPU (default `L4`). A
rerun needs new freezes; the shipped ones must not be overwritten. The original run cost $0.30 on the GPU
($0.89 with the code tests and the smoke test).

## Tables and figures

| Manuscript | Source |
|---|---|
| Table 4 and Tables 39–51 (Study 5) | `paper/review/render_study5_tables.py` → `paper/sections/geometric-null-main-table.tex`, `geometric-null-tables.tex` |
| Table 6 and Tables 52–55 (Study 6) | `paper/review/render_study6_tables.py` → `positive-control-main-table.tex`, `positive-control-tables.tex` |
| Tables 56–57 (Study 7) | `paper/review/render_study7_tables.py` → `large-attempt-tables.tex` |
| Table 32 (Study 2 secondary geometry) | `paper/review/render_study2_secondary.py` → `study2-secondary-table.tex` |
| Table 18 (post hoc corrupted-signature margins) | `paper/review/posthoc_baseline.py` → `posthoc-baseline-table.tex`, `paper/review/posthoc-baseline.json` |
| Table 5 and Figure 6 (post hoc injected difference) | `paper/review/posthoc_injected_difference.py` (needs Study 2 and 3 raw arrays) |
| Tables 26–31, 33–37 and 38 (Studies 2–4) | `analysis/make_extension_tables.py` from the verified summaries → `extension-tables-007.tex`, `-008.tex`, `-009.tex` (shipped as built; editorial formatting added later) |
| Table 30 (Study 2 distributions) | `analysis/describe_confirmation.py` (needs Study 2 test arrays) |
| Table 10 (intervals for the worst qualification cells) | `analysis/qualification_uncertainty.py` → `analysis/results/qualification-uncertainty.json` |
| The Study 1 column of Table 3, Tables 15–17 and Appendix E (post hoc) | `analysis/self_vs_wording.py` (needs Study 1 test arrays) → `analysis/results/self-vs-wording.json` |
| Figures 3 and 7 | `analysis/make_figures.py` |
| Figures 4 and 5 | `analysis/make_extension_figures.py` |
| Figures 1 and 2 | diagrams drawn in the manuscript's source |
| Other tables | transcribed in the manuscript from the verified study reports (`studies/*/RESULTS.md`, `results/analysis/summary.json`) |

## What is not included

- Model weights and tokenisers (download them; see Models).
- Raw arrays and per-chunk records of every study (several GB); the summaries, analyses and verification
  records are included. Study 5's seeded random draws (37 MB) can be regenerated with `code/prepare010.py`.
- Logs, the manuscript and its source, rendered figures, packaging scripts, design and data reviews, working
  notes, and the original project's provenance history. Frozen files that were not shipped are listed with their
  SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, with three kinds of exception, all listed in `RELEASE-CHANGES.md` with the
original's SHA-256:

- records had personal paths, machine details and the names of software tools in working records replaced by placeholders such as `<project>`, `<home>` and `[redacted]`, and the names under which the earlier projects were recorded replaced by the tokens `transport-project`, `routing-project` and `bridge-project` (the assistants used are named under "Use of AI assistants" below); the frozen source archives were repacked with
  neutral owner fields;
- the path module (`src/activation_transfer/paths.py`) maps those tokens to the same folders as before and
  accepts the documented release changes, and the freeze verifiers of Studies 5–7 accept them too;
- two archived source copies of a study's smoke test now find the release root from their own location.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. The checkpoints carry their own licences (MIT for GPT-2, Apache 2.0
for Pythia and TinyLlama).

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | this repository |
| AI-08 | not public yet |

## Use of AI assistants

As the preprint states:

> Language-model assistants (Claude, Anthropic; Codex, OpenAI) were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`; the assistants used are the ones named above.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_answer_repair_2026,
  author  = {Jeevash, Richard},
  title   = {{Answer repair and downstream dependencies in activation transfer: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-answer-repair}
}
```
