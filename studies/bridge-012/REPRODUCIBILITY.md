# Study 7: reproduction

Commands run from the AI-01 project root. The Modal CLI is `~/.local/bin/modal`, client 1.6.2. The project environment (`.venv/`) is used for local checks.

## Check the retained record (no inference)

```bash
.venv/bin/python studies/bridge-012/code/verify012.py
```

This checks the protocol and input freezes, and every pulled-back stage freeze for the files present locally. Files kept only on the Modal Volume `ai01-study7` are counted as remote.
- The study README is listed in the protocol freeze and was rewritten after the run to give the final status. The verifier accepts it only through `provenance/navigation-updates.json`, and the frozen version is preserved in `provenance/README-at-protocol-freeze.md`.
- `code/freeze012.py verify`, frozen with the inputs, reports that file as changed.

## How the study ran

| Step | Command | Writes |
|---|---|---|
| Banks (tokenisation only) | `.venv/bin/python studies/bridge-012/code/prepare012.py --write` | `data/design.json`, `data/rows.json`, `data/preparation-audit.json` |
| Input freeze | `.venv/bin/python studies/bridge-012/code/freeze012.py input` | bootstrap indices, `data/input-freeze.json` |
| Code test (GPT-2 Medium, calibration bank, separate Volume) | `AI01_STUDY7_VOLUME=ai01-study7-codetest5 ~/.local/bin/modal run studies/bridge-012/code/modal_app012.py --stages all --test` | the code-test Volume, `provenance/codetest-log.jsonl` |
| Stages on GPT-2 Large | `~/.local/bin/modal run studies/bridge-012/code/modal_app012.py --stages calibration,calibration_cpu,screen,sources,fit_select` | the Volume `ai01-study7`, `provenance/stage-log.jsonl` |
| Pull back records | `~/.local/bin/modal volume get ai01-study7 <path> studies/bridge-012/results/<path>` | `results/` |

Every stage refuses to overwrite its record, and runner measurements resume from hash-checked chunks. **Do not rerun stages on the Volume `ai01-study7`.** A fresh run needs a new Volume name and new freezes.
