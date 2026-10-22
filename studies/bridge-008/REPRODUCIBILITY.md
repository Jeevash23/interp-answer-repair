# Reproducing Bridge 008

All four new conditions and fresh-engine replay are complete, and the canonical
independent audit passed on 4 October 2026 at 21:32:10 UTC. Both raw interactions
and the matched zero-reference interaction had positive adjusted intervals;
the matched fit-mean interval spans zero, so the full four-contrast criterion
was not met. See the [results report](RESULTS.md). Combined release checks, deterministic analysis replays and the complete 33-page manuscript review have passed.

Preserve this completed evidence tree. The commands below use temporary or
new outputs and do not perform model inference.

Run from the `<project>` project root using its existing `.venv`.
The runner uses CPU float32, four threads and batches of eight aligned prompts.
Dependencies are locked by the project environment; checkpoint/tokenizer bytes
are locally hash-bound. Model weights are not redistributed.

## Synthetic checks and frozen provenance

Synthetic operator, analysis and verification fixtures do not load an inference
model or inspect new outcome arrays:

```bash
.venv/bin/python studies/bridge-008/code/run008.py self-test
.venv/bin/python studies/bridge-008/code/analyze008.py --self-test
.venv/bin/python studies/bridge-008/code/verify008.py --self-test
```

The input freeze is dated `2026-10-04T20:09:31.428590+00:00`, covers 181 files,
and has SHA-256
`418cd9b2cade7911ad18355065c59b526e2e7561baa950fbf203bb0517277339`.
It includes the protocol/scientific code, exact Bridge 007 rows, references,
narrow observations, canonical audit and checkpoint identity. Verification and
reporting helpers are outside the scientific freeze and receive separate
release provenance. Local freezing is not public preregistration.

In particular, **do not overwrite Bridge 007's canonical verification file**;
its bytes are part of Bridge 008's input freeze. Repeat a Bridge 007 audit only
with a fresh explicit `--output` path. Do not change freezes to bypass a mismatch.

## Saved-observation analysis, audit and report

Canonical analysis, independent verification and the report are already retained.
All conditions were evaluable; no rows were dropped. The audit checked six
measurement records, layer-vector matching, computational replay and all four
primary intervals. Scientific success is decided by the intervals, separately
from the audit passing.

The preferred safe original-analysis replay is:

```bash
.venv/bin/python scripts/replay_extensions.py --only 008
```

The wrapper creates temporary child directories, reruns the frozen analyzer
with `OPENBLAS_NUM_THREADS=1`, and compares summaries except `utc` fields plus
NPZ keys, shapes, dtypes and values exactly, including aligned NaNs. It refuses
an unfinished study. This deterministic replay does not repeat inference or
add independent samples.

For fresh inspectable analysis or independent audit outputs:

```bash
bridge008_check_dir=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-008/code/analyze008.py \
  --output "$bridge008_check_dir/analysis"
OPENBLAS_NUM_THREADS=1 .venv/bin/python studies/bridge-008/code/verify008.py \
  --output "$bridge008_check_dir/verification.json"
```

The independent verifier reads the canonical analysis and independently
reconstructs its primary contrasts/intervals. It also checks frozen inputs,
operator vectors, norm matching, donor mappings, calibration, chunk aggregation
and computational replay. It does not use the temporary analysis above as its
source. Both output destinations must be new; never overwrite the retained
audit to obtain a later timestamp.

An alternate report can be generated without changing the retained report:

```bash
bridge008_report_dir=$(mktemp -d)
.venv/bin/python studies/bridge-008/code/report008.py \
  --output "$bridge008_report_dir/report"
```

The helper reads verified saved summaries, writes `RESULTS.md` and
`tables/summary.tex` in that fresh destination, and performs no inference or
statistical analysis. It writes nothing if no canonical verification exists.
There is no `--check-ready` option on this report helper. The final release
manifest binds the helper and rendered artifacts.

All extensions now have terminal audits. `make verify-extensions` repeats
independent audits into temporary outputs, while `make replay-extensions`
replays the original frozen analyses. Figure/table generation and manuscript
number checks use verified summaries. These are distinct validation steps,
not additional scientific samples.

The completed fresh-engine replay used eight fixed rows for all four conditions,
with identical behavior and maximum singleton signature error 0.0000143051
(tolerance 0.0005). Independent reconstruction of primary adjusted intervals
agreed within 6.94×10⁻¹⁸. These checks reproduce computation on existing cases.

## Interruption recovery and full re-execution

An interrupted, unfinished run can resume only after its previous process has
stopped, with unchanged inputs and verified chunk hashes. A chunk written just
before interruption can be recovered from its embedded input identity and
row indices. Keep model processes sequential and use four CPU threads.

This recovery behavior is not a safe procedure for re-executing a completed
release. Re-running the completed model pipeline can rewrite completion
timestamps even when its observations are reused, invalidating downstream
hashes. Do not rerun it in the retained evidence tree.

Full model re-execution belongs in a separate preserved copy of the complete
project, with the same environment/checkpoint inputs and new run provenance.
Inherited dependencies and stage freezes need care; no clean-rerun command has
been validated here. Do not delete freezes or reset canonical outputs to force
execution. Saved-observation replay and audit are the routine reproduction path.

## Scope of the evidence

This study reuses all 3,200 Bridge 007 configurations and narrow outcomes; it is
an outcome-informed, correlated follow-up. Neither the new interventions nor
computational replay increase the number of independent language cases.

Source arrays retain seven layer sums, not 112 independent residual injections.
Matching targets the norm of the narrow vector sum separately in each layer,
configuration and donor condition. Undefined matching is explicitly unavailable
without deleting observations or reallocating the four-quantity error budget.
New signatures have shape `[2,n,176,2]` and behavior `[n,6]`. See
[the operator documentation](code/README.md) for exact arithmetic and records.

Primary intervals use 50,000 crossed bootstrap draws over the same 20 context
blocks and 20 disjoint name triples. Mandatory cellwise heterogeneity,
behavioral controls and endpoint distances constrain interpretation. A positive
result would concern source coverage at S2 in layers 6–12 in this assay; a null
signed mean would not establish equivalence, recipient-context causation or the
absence of wording information shared by clean and corrupted source states.
