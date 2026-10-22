# Bridge 010 (Study 5): reproduction

From the project root, with the locked environment in `.venv`:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python studies/bridge-010/code/run010.py calibration
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python studies/bridge-010/code/run010.py inference
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python studies/bridge-010/code/analyze010.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python studies/bridge-010/code/run010.py replay
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python studies/bridge-010/code/audit010.py
```

Every stage verifies the 737-file freeze (`data/input-freeze.json`) first. Calibration, inference and replay resume from hash-checked records and refuse changed inputs; the analysis and audit refuse to overwrite existing outputs, so rerunning them requires moving the retained outputs aside and comparing the regenerated files with them. The draws, mappings and bootstrap indices in `data/draws/` were saved before the freeze and are never redrawn.
