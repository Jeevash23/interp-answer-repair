# Analyses, figures and checks used in the manuscript

These scripts read saved arrays and verified summaries only. They run no model inference (except
the independent re-implementation check, which runs its own NumPy forward pass) and never write
into `studies/`. Table and figure numbers refer to manuscript version 3.2 (`paper/manuscript.pdf`).

| Script | Output | Manuscript |
|---|---|---|
| `self_vs_wording.py` | `results/self-vs-wording.json` | Post hoc self-transfer and movement analyses: Sections 5.4–5.5, the Study 1 column of Table 3, Tables 9 and 14–16, Figure 3 and Appendix E |
| `qualification_uncertainty.py` | `results/qualification-uncertainty.json` | Post hoc crossed-bootstrap intervals for qualification rates (Section 5.9, Table 9) and the paired comparison of the F6/F7 and F4/F7 references on shared F7 prompts (Appendix I) |
| `make_figures.py` | `../paper/figures/signature-map.pdf`, `recovery-and-margins.pdf` | Figures 3 and 7 |
| `make_extension_figures.py` | `../paper/figures/prospective-self-wording.pdf`, `source-coverage-interactions.pdf` | Figures 4 and 5 (Studies 2 and 3) |
| `describe_confirmation.py` | `results/confirmation-distributions.json` | Study 2 configuration-level distributions (Table 29); `--check` re-derives them without writing |
| `make_extension_tables.py` | Study 2–4 table fragments (`--output-dir` for review copies) | Tables 25–30 and 32–37 in Appendices J–L, whose fragments in `../paper/sections/` carry later editorial formatting; `make analysis` runs it with `--check-ready`, which validates inputs and writes nothing |
| `independent_check.py` | one JSON line per configuration; reported run in `results/independent-check.jsonl` | Appendix H and the Reproducibility paragraph (Section 8) |
| `check_paper_numbers.py` | pass/fail report | Rebuilds the rows of the Study 1 and historical comparison tables and selected quoted values (63 checks); run by `make check-paper` |
| `check_extension_numbers.py` | pass/fail report | Checks the rows of the Study 2–4 primary tables within their named tables (24 checks) and reconstructs the Study 2 distributions; run by `make check-paper` |
| `release_manifest.py` | `../provenance/release-manifest.json` | One hash manifest over the manuscript, figures, analyses, documentation and Bridge 007–012 records (`make release`, `make release-check`) |
| `common.py` | shared helpers (row loading, crossed bootstrap, paired geometry, Wilson interval) | |

The Study 5 tables (Tables 4 and 38–50) are rendered by `../paper/review/render_study5_tables.py`,
the Study 2 prespecified secondary geometry (Table 31) by `../paper/review/render_study2_secondary.py`,
the post hoc injected-difference analysis (Table 5, Figure 6) by
`../paper/review/posthoc_injected_difference.py`, and the post hoc corrupted-signature margins of
Studies 1 and 2 (Table 17) and the Figure 3c correlation under both references by
`../paper/review/posthoc_baseline.py`. `../paper/review/check_revision.py` wraps the two
number checkers above for `make check-paper`, and `../paper/review/check_references.py` checks the
bibliography (`make references`).

`self_vs_wording.py` compares wording transfer with self transfer (both measured in the
original Bridge 005 and Bridge 004 runs) and with ideal repair. Intervals are pointwise 95%
crossed-bootstrap intervals (50,000 draws, seed 2026100107) over context blocks and name
triples, the same resampling scheme as the original analyses. These analyses were defined
after the primary results and are not part of any prespecified family.

`independent_check.py` re-implements GPT-2 Medium from `models/gpt2-medium/model.safetensors`
in plain NumPy. It reproduces the saved behaviour and all 176 x 2 signature coordinates for
clean, corrupted, wording and self conditions to within 4.6e-5 logits (largest 4.52e-5) on the 12 reported
configurations. The original pipeline's "zero" ablation is zero in TransformerLens's folded
coordinates; the script documents the equivalent raw-coordinate value.

Run from the project root, e.g. `make analysis` or `.venv/bin/python analysis/self_vs_wording.py`.
