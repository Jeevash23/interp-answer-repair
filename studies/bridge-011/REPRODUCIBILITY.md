# Study 6: reproduction

All commands run from the AI-01 project root with the project environment (`.venv/`). Inference used the 6 GB NVIDIA GPU (CUDA, float32, deterministic algorithms, TF32 off) at 8 prompts and 8 heads per pass.

## Check the retained record (no model inference)

```bash
.venv/bin/python studies/bridge-011/code/freeze011.py verify
```

This checks the protocol, input and development freezes. `audit011.py` wrote `results/verification.json` once: freezes, every runner record and chunk hash, the stage records, and an independent plain-NumPy reconstruction of the primary results from the saved arrays and saved bootstrap indices. `analyze011.py --mode self-test` reruns the synthetic analysis checks recorded before the freeze.

## Order of the original run

| Step | Command | Writes |
|---|---|---|
| Banks (tokenisation only) | `code/prepare011.py --write` | `data/design.json`, `data/rows.json`, `data/preparation-audit.json` |
| Protocol freeze (before approval) | `code/freeze_protocol011.py` | `data/protocol-freeze.json` |
| Input freeze | `code/freeze011.py input` | analysis self-test record, bootstrap indices, `data/input-freeze.json` |
| Calibration | `code/run011.py calibration` | `results/calibration*` |
| Development | `code/run011.py development` | `results/development*` |
| Development freeze | `code/freeze011.py development` | `data/development-freeze.json` |
| Held-out inference | `code/run011.py test` | `results/test/`, `results/inference-complete.json` |
| Replays (fresh processes) | `code/run011.py replay`, `code/run011.py cpu_replay` | `results/gpu-replay.json`, `results/cpu-replay.json` |
| Analyses | `code/analyze011.py --mode development`, `code/analyze011.py` | `results/analysis-development/`, `results/analysis/` |
| Audit | `code/audit011.py` | `results/verification.json` |

Every stage refuses to overwrite its outputs. **Do not rerun these stages in this retained tree.** A fresh run needs a copy of the project with this study's `results/` removed and new freezes; its timestamps will differ.

Run logs are in `provenance/run-logs/`. `provenance/runtime-benchmark/` holds the design-stage GPU timing, and `provenance/code-smoke-test/` the code test on Study 2's calibration rows before the input freeze (with what it showed).
