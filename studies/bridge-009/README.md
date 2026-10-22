# Bridge 009: fixed F10/F11 replication attempt in GPT-2 Small

**Completed with a verified behavioral-screen stop.** F11 reached 83.3% and
58.3% top-1 accuracy in the two orders, below the fixed 85% gate. F10 reached
100% in both orders; all four cells had 100% pairwise accuracy. The 96-row screen
therefore stopped the sole permitted F10/F11 pair at `no_screen_pair`.
Scientific fitting, selection, independent qualification and final transfer
were not measured.

The [results report](RESULTS.md) documents the complete bounded attempt. The
canonical independent audit passed on 4 October 2026 at 20:09:56 UTC, checking
635 unique files and four measurement records. Audit success verifies correct
execution and reporting of the stopping outcome. It does not mean that Small
qualified or that an unmeasured transfer prediction failed.

## Fixed scope

This attempt used the exact Bridge 007 F10/F11 pair, names and context banks,
with no Small-specific pair search. The planned final stage had 3,200
configurations from 20 context blocks crossed with 20 disjoint name triples,
the historical four-head Small intervention and 60 later measurement heads.
That final stage was not reached. There is no second qualified checkpoint.

The original candidate manifest is retained byte for byte, including F12/F13.
The planned mean bank would use all four fixed Medium families (1,024 clean
fitting rows); fitting and selection signatures would each use only 512 F10/F11
rows. These fitting populations were not measured after the screen failed.
F12/F13 cannot enter Small eligibility or transfer decisions. Same-case
comparisons across Medium and Small are correlated and do not increase the
independent language sample count. The source-set size and measured dimensions
also differ, so this attempt cannot isolate model size as a cause.

## Provenance and preservation

Read [the protocol](docs/PROTOCOL.md), [the preparation review](docs/DATA-REVIEW.md),
[the design review](docs/DESIGN-REVIEW.md) and
[reproduction instructions](REPRODUCIBILITY.md). Implementation origins are in
`provenance/code-origins.json`; tokenizer/checkpoint/input checks are in
`data/preparation-audit.json`.

The effective input freeze is dated **2026-10-04 18:52:00.654834 UTC**, covers
156 files, and has SHA-256
`703e5e1edf64432debd40f25013320dc641d2eca53f0c2f0004447e5ae62e7c4`.
The secondary plan was frozen at **18:52:21.841321 UTC**, SHA-256
`a64da08bf278bd0b8ac5d01dbd454f870948f6e5fa48c94d7540a58c8ad256b3`.
These are local prospective records, not public preregistration.

A pre-inference amendment corrected a stale secondary-plan seed label from
2026100407 to 2026100409. The numerical implementation and protocol already
used 2026100409 and were unchanged. The original source and both original
freezes remain under `provenance/amendments/seed-label/`;
`data/pre-inference-amendment.json` records their hashes and path overrides.
The effective freezes supersede those metadata records without deleting them.

The reporting helper accepts only a terminal, independently verified study.
It formats saved outcomes and records their hashes; it does not perform
inference or statistical analysis. Preserve canonical outputs and use fresh
verification destinations. Do not rerun the completed model pipeline in this
retained tree; it can change timestamps bound by later provenance.
