# Answer repair and downstream dependencies in activation transfer: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
# The shared packages are importable from this folder even without installing the project.
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/studies/transport/src:$(CURDIR)/studies/routing/src
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 1
.NOTPARALLEL:

S5 := studies/bridge-010
S6 := studies/bridge-011
S7 := studies/bridge-012
REGENERATED_TABLES := paper/sections/geometric-null-main-table.tex paper/sections/geometric-null-tables.tex \
                      paper/review/study5-table-provenance.json paper/sections/study2-secondary-table.tex

.PHONY: help setup python-check models tokenisers link-models unpack check-freezes quick-check verify test \
        figures independent-check

help:
	@echo "make setup          create .venv with the pinned environment (uv sync; downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU only, a few minutes, no model and no network: freezes, Study 7's records,"
	@echo "                    the Study 2, 5, 6 and 7 tables regenerated from the saved outputs, analysis self-tests"
	@echo "make verify         quick-check plus two post hoc analyses rerun from saved arrays (several minutes)"
	@echo "make test           the 93 unit tests (need the GPT-2 Small and Pythia tokenisers: make tokenisers)"
	@echo "make figures        redraw Figures 3, 4, 5 and 7 into paper/figures/"
	@echo "make models         download GPT-2 Medium and Small (about 2.1 GB)"
	@echo "make independent-check   the plain-NumPy GPT-2 Medium re-implementation check (needs GPT-2 Medium)"

setup:
	uv sync --frozen --extra experiment --group dev

python-check:
	@$(PY) -c "import numpy, torch, transformers" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

models: python-check
	$(PY) tools/get_models.py gpt2-medium gpt2-small

tokenisers: python-check
	$(PY) tools/get_models.py --no-weights gpt2-small pythia-410m

# In the original project studies/transport/models was a link to the shared models/ folder; the historical
# transport scripts still read checkpoints through it. Set MODELS_FROM=DIR to use checkpoints already in
# DIR/<folder> instead of downloading them.
link-models:
	@if [ -n "$(MODELS_FROM)" ]; then $(PY) tools/get_models.py --link-from "$(MODELS_FROM)"; fi
	@[ -e studies/transport/models ] || ln -s ../../models studies/transport/models

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	CUDA_VISIBLE_DEVICES= $(PY) tools/check_freezes.py
	CUDA_VISIBLE_DEVICES= $(PY) $(S7)/code/verify012.py
	CUDA_VISIBLE_DEVICES= $(PY) $(S6)/code/freeze011.py verify
	CUDA_VISIBLE_DEVICES= $(PY) $(S5)/code/freeze010.py
	CUDA_VISIBLE_DEVICES= $(PY) paper/review/render_study6_tables.py --check
	CUDA_VISIBLE_DEVICES= $(PY) paper/review/render_study7_tables.py --check
	CUDA_VISIBLE_DEVICES= $(PY) paper/review/render_study5_tables.py
	CUDA_VISIBLE_DEVICES= $(PY) paper/review/render_study2_secondary.py
	$(PY) tools/compare_regenerated.py $(REGENERATED_TABLES)
	CUDA_VISIBLE_DEVICES= $(PY) studies/bridge-007/code/analyze007.py --mode self-test > /dev/null && echo "Study 2 primary analysis self-test passed"
	CUDA_VISIBLE_DEVICES= $(PY) studies/bridge-007/code/secondary007.py --mode self-test > /dev/null && echo "Study 2 secondary analysis self-test passed"

verify: quick-check
	CUDA_VISIBLE_DEVICES= $(PY) analysis/qualification_uncertainty.py
	CUDA_VISIBLE_DEVICES= $(PY) paper/review/posthoc_baseline.py
	$(PY) tools/compare_regenerated.py analysis/results/qualification-uncertainty.json \
	    paper/review/posthoc-baseline.json paper/sections/posthoc-baseline-table.tex

test: unpack link-models
	$(PY) -m pytest -q -p no:cacheprovider

figures: unpack
	$(PY) analysis/make_figures.py
	$(PY) analysis/make_extension_figures.py

independent-check: unpack link-models
	$(PY) analysis/independent_check.py --default > analysis/results/independent-check-rerun.jsonl
	$(PY) tools/compare_jsonl.py analysis/results/independent-check.jsonl analysis/results/independent-check-rerun.jsonl
