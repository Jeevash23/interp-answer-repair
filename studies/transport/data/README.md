# Data

Commit small prompt manifests and metadata only. Do not commit model weights, token caches, large activation tensors, or generated raw outputs.

Expected subdirectories, created by scripts when needed:

- `raw/` — generated model outputs and activation shards; ignored by Git
- `processed/` — derived datasets; ignored by Git
- `manifests/` — frozen prompt definitions suitable for version control

Experiment 001 keeps its compact authoritative files at this directory root:

- `experiment-001-manifest.jsonl` — 240 frozen paired examples
- `experiment-001-freeze.json` — seed, model revision, and manifest SHA-256
- `experiment-001-implementation-freeze.json` — pre-confirmatory source hashes
- `experiment-001-audit.md` — 100-row structured [redacted]-assisted audit
- `experiment-002-implementation-freeze.json` — Medium source, model, localization, and selection hashes
- `model-manifest.json` — cached model revisions and architecture metadata
- `experiment-003-manifest.jsonl` and associated freeze/audit files — 320 prompts from 16 new
  template families with split-exclusive entities
- `experiment-004-manifest.jsonl` — the same frozen texts retokenized for Pythia
- `experiment-004-feasibility-freeze.json` — Pythia behavior/hook gate frozen before evaluation
- `experiment-004-implementation-freeze.json` and amendments 001--002 — pre-outcome circuit code,
  hypotheses, weight hash, and memory-motivated execution changes
- `pythia-model-manifest.json` — exact Pythia revision, architecture, and weight hash

Experiment 002 deliberately reuses `experiment-001-manifest.jsonl` so model scale changes while the
prompt set remains fixed. Its generated results and verification report live under
`results/experiment-002/`.

Experiment 003 tests the frozen Medium groups on new templates. Experiment 004 independently
localizes Pythia components and replicates the qualitative causal claim; no GPT-2 activation is
inserted into Pythia.

The September 5 audit found that the batch-size amendment also changed original
mismatch donors and random-control assignments; it was not solely a numerical
execution change. Study 005 implements controls independent of batching. This
interpretation correction does not rewrite the historical amendment files.
