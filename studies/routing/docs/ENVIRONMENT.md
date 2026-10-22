# Environment record — 2026-09-05

## Inspected machine

- Linux on an Intel Core 5 210H, 12 logical CPUs.
- Approximately 15 GiB RAM, 15 GiB swap; about 9 GiB RAM available at inspection.
- Btrfs home filesystem; 89 GiB available before setup, approximately 87 GiB afterward.
- Prior project documented a 6 GB NVIDIA GPU with 6 GiB VRAM. GPU access is
  **unverified in this session**. No driver or operating-system settings were changed.

The Desktop and Documents research paths resolve to the same underlying directory. The new
project is a sibling of the earlier work, not a second copy of it.

## Installed project environment

One `.venv`, CPython 3.12.14, with an independent `uv.lock` and editable project package:

- PyTorch 2.13.0 (CUDA build present; pilot explicitly uses CPU).
- TransformerLens 3.8.0.
- Transformers 5.16.1.
- NumPy 2.5.2.
- Safetensors 0.8.0.
- Matplotlib 3.11.1, SciPy 1.18.1, pytest 9.1.1, Ruff 0.16.5.

The dependency stack matches the previously tested package versions, while the root package
and environment are new. `uv lock --check --offline` passed. Offline synchronization installed
177 packages from cached distributions, including the new local project.

The shared cache was copied with Btrfs copy-on-write reflinks into ignored `.uv-cache`, allowing
cache locks and metadata writes to stay within this project. This shares underlying data blocks
initially; it is not a hardlinked or symlinked Python environment. Subsequent dependency changes
cannot edit the earlier environment. No new model checkpoint was downloaded.

## Local model

GPT-2 Small, `openai-community/gpt2`, recorded revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. Weights and tokenizer are opened read-only from
`../transport-project/models/gpt2-small`.

The revision label comes from the earlier manifest. Exact current bytes are independently
SHA-256 hashed by this project's freeze, including weights, config and tokenizer assets.
The pilot compares native and compatibility-mode centered logits and checks all 72 identity
head hooks before measurement. Actual errors, runtime and peak process memory belong in
`results/pilot-001/run.json`.

## Access limitation

Sandbox `nvidia-smi` could not contact the driver. An outside-sandbox read-only GPU check was
rejected before execution by automatic approval review, whose stated error was an unavailable
review-service deployment (HTTP 404). The same service blocked browser navigation to a public
arXiv paper. A shell request to arXiv also failed DNS resolution inside the sandbox.

These observations establish a tool-access limitation, not a defective GPU or a completed
literature audit. CPU inference and local file work are the permitted alternatives used here.

## Completed study 002

The same environment and cached checkpoint supported the 256-recipient study and all 30 test
conditions. Discovery/development took 56.33 seconds. Test natural signatures, donor states and
interventions took 1470.46 seconds (24.51 minutes),
with peak process RSS 2970.63 MiB (2.90 GiB). Inference
used CPU float32, four threads and batches of 16. No GPU run or new checkpoint download occurred.
The frozen source, data, model bytes and prior-project integrity checks all passed.
