# Environment record

Initial inspection on 31 August 2026:

- OS: [distro]
- Kernel: `7.2.2-1-[distro]`
- System Python: `3.14.7`
- Memory: 15 GiB RAM and 15 GiB swap
- Home filesystem: approximately 112 GiB available
- Integrated GPU: Intel Raptor Lake-P
- Discrete GPU: [6 GB NVIDIA GPU]
- NVIDIA driver/userspace version: `610.57.04`

## GPU compute readiness

The initial sandboxed `nvidia-smi` check could not access the NVIDIA device. A subsequent
host-terminal check and an unrestricted project check both succeeded on 31 August 2026:

- GPU: [6 GB NVIDIA GPU]
- driver: `610.57.04`
- VRAM: `6141 MiB`
- CUDA compute capability: `8.9`

This confirms that the earlier failure was an execution-sandbox visibility issue, not a broken
host driver. The experiment environment may use CUDA after a PyTorch device smoke test passes.

## Verified experiment environment

Installed into the repository-local Python 3.12 `.venv` on 31 August 2026:

- PyTorch `2.13.0+cu130` with CUDA runtime `13.0`
- TransformerLens `3.8.0`
- Transformers `5.16.1`
- Hugging Face Hub `1.29.0`
- the pinned scientific-analysis and Jupyter packages in `uv.lock`

An actual CUDA matrix multiplication passed on the 6 GB NVIDIA GPU. TransformerLens v3
`TransformerBridge` inference and activation caching also passed:

- GPT-2 Small, float32: 12 cached residual hook points; approximately 2028 MiB peak VRAM
- GPT-2 Medium, bfloat16: 24 cached residual hook points; approximately 5532 MiB peak VRAM
- GPT-2 Large, bfloat16: direct GPU compatibility processing exceeded VRAM; processing weights on
  CPU first and then moving the model to CUDA passed with 36 cached residual hook points and
  approximately 1951 MiB peak VRAM for the smoke prompt

These measurements use a single short prompt and a filtered activation cache. Experiment batches
and broader cache filters must still be sized deliberately.

## Cached model snapshots

Only safetensors weights and the required configuration/tokenizer files were downloaded. Duplicate
TensorFlow, Flax, and legacy binary weights were excluded.

- GPT-2 Small: `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- GPT-2 Medium: `openai-community/gpt2-medium` at `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`
- GPT-2 Large: `openai-community/gpt2-large` at `32b71b12589c2f8d625668d2335a01cac3249519`
- Pythia-410M-deduped: `EleutherAI/pythia-410m-deduped` at
  `c4fc8d586d62df497f1f9b69d66d3ca419992d3e`; `model.safetensors` SHA-256
  `e7ae132489f63d5d86009a8178a75c7d5d195410d067fca01a3160623e370fae`

The snapshots occupy approximately 5.9 GiB under the ignored `models/` directory. The environment
occupies approximately 5.4 GiB under `.venv`. Exact metadata is tracked in
`data/model-manifest.json` and `data/pythia-model-manifest.json`.

TransformerLens 3.8's unprocessed Pythia converter expects the pre-Transformers-5 output-module
name `embed_out`; the current raw checkpoint exposes the identical tensor as `lm_head`. Experiment
004 supplies an in-memory alias and verifies converted logits against raw Hugging Face logits before
use (maximum absolute error 0.00221 under a frozen 0.005 tolerance). No installed package is patched.
Pythia localization uses an algebraically equivalent pre-output head hook to avoid materializing a
17.5 GiB expanded result tensor, and transport uses four-example batches; both changes are recorded
before scientific outcomes in the amendment chain.

## Verified writing toolchain

Installed from the configured [distro]/Arch repositories on 31 August 2026:

- Code-OSS `1.135.0`
- LaTeX Workshop `10.18.0` from Open VSX, with a valid extension signature
- TeX Live `2026.1`
- latexmk `4.87`
- uv `0.12.7`

The official TMLR research edition compiles successfully with pdfLaTeX and BibTeX. The final
13-page PDF is rendered and visually inspected before delivery; citations resolve and the final log
is checked for layout and unresolved-reference warnings.

The repository-local Python 3.12 environment occupies approximately 5.4 GiB and contains the pinned
CUDA experiment stack, analysis tools, pytest, and Ruff. Model weights remain separately stored in
the ignored `models/` directory.

## Reproducibility records

- `uv.lock` freezes the complete Python dependency graph.
- `data/model-manifest.json` freezes model repository revisions and architecture metadata.
- `scripts/cache_gpt2_models.py` reproduces the local snapshots.
- `scripts/cache_pythia_model.py` reproduces the pinned Pythia snapshot and local config cache.
- `scripts/smoke_test_environment.py` checks CUDA and the TransformerLens v3 bridge.
- Experiment-specific seeds, precision, prompt manifests, and cache filters must be recorded with
  every result.
