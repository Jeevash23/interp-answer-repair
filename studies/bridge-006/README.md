# F4/F7 extension and manuscript audit

This directory records the 1 October 2026 review of the refined AI-01 manuscript.
Bridge 005's original data, source, thresholds and primary analysis remain intact.

The F4/F7 comparison is a **post hoc wording extension**. It was the only other
originally eligible selection pair. The follow-up procedure was saved before new
inference, but it shares blocks, names and F7 with the inspected F6/F7 study and
is not a fresh independent replication.

Independent qualification fails for F7/order 1 (75.0% zero-reference and 76.4%
mean-reference classification, versus the 80% requirement). The final clean bank
passes. Final diagnostic accuracy improves from 29.9% to 84.9%, against 95.0%
clean. Four adjusted recovery intervals are positive; seven of eight adjusted
paired margins favor the recipient, and one includes zero. The full conjunction
therefore fails. No pair substitution or threshold revision was made.

For the original F6/F7 comparison, explicit corrupted-to-edited decomposition
finds mean per-example squared perpendicular fractions of 91.5% and 92.8%.
Distance from the clean-pair axis decreases, as do distances to both clean
endpoints. The six tested scale floors all retain eight recipient-favoring
pointwise intervals. Perpendicular dominance alone is not a special semantic
direction or evidence against a high-dimensional geometric null.

## Records

- `docs/PROTOCOL.md`, `protocol-freeze.json`: fixed extension and post hoc audit scope.
- `implementation-freeze.json`: inference implementation frozen before qualification.
- `results/inference.log`, `results/execution-events.json`: execution history, including one SIGTERM/resume with unchanged code. A resume revalidates completed chunks and updates aggregate completion metadata; the log retains the first completion order.
- `results/numerical-validation.json`: full/optimized suffix comparisons, actual-pair full-boundary replay and sequential source-addition checks.
- `results/{qualification,test}/`: per-example measurements, chunk hashes and control arrays.
- `results/analysis/`: complete extension estimates and normalized per-example geometry.
- `measurement-manifest.json`: post-inference integrity hashes, not a preregistration record.
- `results/methods-audit.json`: fifteen independent implementation/provenance checks.
- `results/f6f7-movement-and-sensitivity.json`: direct movement, six floors, and an independent reproduction of the original twelve adjusted intervals.
- `results/saved-replay-consolidated.json`: twelve original saved-output components pass, combining initial successes with three reruns after restoring exact-hash archived helpers. Initial failures are retained in a separate record.
- `results/routing-recovery-verification.json`: recovered historical engine smoke checks; full historical inference was not rerun.
- `docs/LITERATURE-AUDIT.md`: focused primary-source comparison, official repository availability and explicit external-reproduction limits.

The original project verification reports 787 unchanged evidence files and 42
documented relocations. Ninety-three existing tests pass. Two missing frozen
packaging helpers were restored from the portfolio archive with their expected
hashes; `provenance/20261001-recovery-repair.json` records the repair. No hash check
was bypassed. Root and historical README files remain at their frozen revisions;
this directory and the current manuscript describe the extension.

## Replay

From the project root:

```bash
.venv/bin/python studies/bridge-006/code/audit_methods.py
.venv/bin/python studies/bridge-006/code/reanalyze005.py
.venv/bin/python studies/bridge-006/code/analyze006.py --output /tmp/new-extension-analysis
.venv/bin/python scripts/reproduce_routing.py --output /tmp/new-routing-recovery --smoke
```

Choose new output directories for replay. The original twelve-component command
remains `scripts/project.py replay --output /tmp/new-original-replay`.
`analyze006.py` was implemented after qualification began; its formulas, seed,
gates and multiplicity follow the earlier protocol. It reconstructs the pair's
fit and both qualification banks, checks two bootstrap implementations and
compares shared F7 observations with the original arrays.

`run006.py` resumes this specific frozen inference record using existing verified
chunks. It does not create an independent replication. The local checkpoint and
locked root environment are required. `write_extension_section.py` generates
Appendix I from the completed analysis. The current PDF and editable Word paper
are in `../../paper/`.

Public archival release, independent reproduction of external repositories and
a complete mechanism explanation remain outside the completed numerical audit.
