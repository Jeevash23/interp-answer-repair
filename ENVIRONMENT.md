# Environment

The studies ran in one Python environment, recorded exactly in `uv.lock` (180 locked packages, including the
project itself). `pyproject.toml` and `uv.lock` are the original project's files, unchanged: the `experiment`
extra is the full research environment, the `dev` group adds pytest and ruff, and the project builds the four
shared packages (`transport_bridge`, `activation_transfer`, `circuit_transport`, `causal_circuit_routing`) with
hatchling. The Makefile also puts them on `PYTHONPATH`, so the code runs without installing the project.

| Package | Version |
|---|---|
| Python | 3.12.14 (`uv` installs it if it is missing) |
| torch | 2.13.0 (CUDA 13 build from PyPI) |
| transformers | 5.16.1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.29.0 |

The code imports torch, transformers, transformer-lens, numpy, scipy, matplotlib, safetensors, einops,
jaxtyping and huggingface-hub (and pytest for `make test`). The Modal image for Study 7 is built from the same two
files (`uv_sync` with the `experiment` extra). `studies/transport/` and `studies/routing/` also keep the
`pyproject.toml` and `uv.lock` of the two earlier projects, as records of the environments their first runs
used; the project-level environment reproduces them.

## A fresh install

```sh
uv sync --frozen --extra experiment --group dev        # what make setup runs
```

On Linux x86-64 this downloads 176 packages (plus hatchling, to build the project), about 3.0 GB of wheels. The largest are torch (527 MB) and the
NVIDIA CUDA libraries it depends on (about 2.0 GB together: cuBLAS 423 MB, cuDNN 366 MB, cuFFT 214 MB, NCCL 206
MB, cuSOLVER 201 MB, cuSPARSELt 170 MB, cuSPARSE 146 MB and smaller ones), then triton (198 MB). The sizes come
from `uv.lock`; installing creates `.venv/` in this folder.

## A smaller CPU-only environment

The quick check, the verifiers and the number and figure scripts run on a CPU and need no CUDA. To avoid the
CUDA libraries, install the same versions of the packages the code imports, with PyTorch's CPU build:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv transformers==5.16.1 transformer-lens==3.8.0 numpy==2.5.2 scipy==1.18.1 \
    matplotlib==3.11.1 safetensors==0.8.0 einops==0.8.2 jaxtyping==0.3.11 huggingface-hub==1.29.0 pytest
```

This is not the locked environment, so use it for checking saved outputs; for model reruns that are meant to
reproduce the recorded numbers to the protocols' tolerances, use `make setup`.
