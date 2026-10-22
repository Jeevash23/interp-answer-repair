# Reproducing Bridge 004

Use the project’s locked Python environment. Source code is versioned under `code/`; exact model-specific prompts, semantic token positions, selections and freezes are under `data/`. Do not modify the preserved Bridges 001–003 to run this study.

## Model inference

Fresh inference requires separately obtained matching GPT-2 Medium and public TinyLlama model files at the paths recorded in `data/checkpoints.json`. The TinyLlama identity is `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, revision `59f6f375b26bde864a6ca194a9a3044570490064`. This is not Meta Llama 3.2. Weights, environments and caches are excluded from research archives.

Inference uses float32 eager attention, deterministic settings and the 6 GB NVIDIA GPU. Run one model at a time. The native TinyLlama wrapper measures 32 query-output heads per layer while respecting four grouped key/value heads. Exact inference settings and calibration errors are retained.

The completed records are immutable. To repeat model inference, use a separate copy, preserve all manifests, selections and freeze records, and relocate the completed result directories before running a stage. The stage runners validate hashes and skip completed bundles; simply rerunning them in the original workspace does not constitute a new inference replication.

The stage sequence is numerical calibration, input freeze, model-specific screen, discovery freeze, discovery signatures, pair selection, TinyLlama source localization if applicable, test freeze, held-out transfers, and Medium fixed-group clamps. Model-specific selection rules and the primary families are specified in [the protocol](docs/PROTOCOL.md). Failed gates remain outcomes. No test case may be discarded to improve qualification.

## Saved-output analysis

Run from the umbrella project or workspace with the installed package available:

```text
.venv/bin/python studies/bridge-004/code/analyze004_corrected.py --model gpt2-medium --output reproduced/medium
.venv/bin/python studies/bridge-004/code/analyze004_corrected.py --model tinyllama-1.1b-base --output reproduced/tinyllama
.venv/bin/python studies/bridge-004/code/analyze_clamps004.py --output reproduced/clamps
```

Output directories must not already exist. The source and data freezes must be present, including `analysis-correction-freeze.json`. The original `analyze004.py` is intentionally preserved as a failed implementation record: its advanced donor indexing transposed axes. Use the corrected version. The [correction note](docs/ANALYSIS-CORRECTION.md) and direct-pairing regression distinguish this technical change from the old scientific calibration failure.

Bootstrap draws are 20,000 with seed 2026090804. The paired-metric primary family has 24 quantities across both models, even when a model fails qualification. The Medium group family has 20 quantities. Other intervals are pointwise 95%. Fits and source selections are fixed; this replay does not estimate discovery-pipeline variability.

The `audit004.py` checks raw bundle hashes, an independent expression for distances, exact summary/array replay, unedited clamp identities, and the four-run decomposition. Its `reproduced` directories must not exist before a new audit. The prior research preservation audit uses the previous artifact manifest in the original workspace; an extracted archive requires that manifest as a separate preserved release record.

## Reading the evidence

Start with [RESULTS.md](RESULTS.md), then the protocol and [contribution audit](docs/CONTRIBUTION-AUDIT.md). The manuscript is under `paper/draft-v2/` in the umbrella project. The previous draft and old archives remain historical records. A successful statistical replay confirms consistency of saved measurements, not an independent rerun of every model forward pass.

## Supplemental and descriptive checks

The projected-head random controls and multi-context endpoint replay are separately documented in `docs/SUPPLEMENTAL-CONTROLS.md` and frozen in `data/supplemental-controls-freeze.json`. They were added after the initial Medium outcomes, so they are post hoc checks. The projected controls re-execute corrupted and wording behavior for all 384 configurations per model, without repeating every signature.

`baseline_geometry004.py` is an additional post hoc description comparing the edited signature with its corrupted starting state. It reports changes in distance to both clean endpoints, rather than conflating relative proximity with movement induced by the edit. It uses saved arrays and does not alter primary decisions.

`audit_supplemental004.py` verifies the supplemental freeze and raw hashes, recomputes projected-control norms and intervals, checks all repeated behavior, and independently calculates the baseline-distance changes. Its output is `results/supplemental-verification.json`. The v4 release check also executes these audits in an extracted copy with the retained source files; no model inference is repeated by that archive check.
