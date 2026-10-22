# Modal smoke test for Study 7 (17 October 2026)

**Passed.** On a Modal T4, the frozen Study 6 engine recomputed Study 6's eight fixed held-out GPT-2 Medium configurations in all nine signature conditions. Signatures and behaviour match the saved, audited arrays within 2.5 × 10⁻⁵ logits (tolerance 5 × 10⁻⁴). This is the first AI-01 computation outside the local machine.

| | |
|---|---|
| GPUs | Tesla T4 (reproduction and GPT-2 Large timing), NVIDIA L4 (GPT-2 Large timing) |
| Image | `modal.Image.debian_slim(python_version="3.12")` + `uv_sync(extras=["experiment"], frozen=True)` from AI-01's `pyproject.toml` and `uv.lock`: Python 3.12.10, torch 2.13.0+cu130 (CUDA 13.0), transformers 5.16.1, transformer-lens 3.8.0, numpy 2.5.2 |
| Weights | Downloaded inside Modal from Hugging Face at the pinned revisions (gpt2-medium `6dcaa7a…`, gpt2-large `32b71b1…`); safetensors SHA-256 and size asserted at image build |
| Largest deviation | 2.48 × 10⁻⁵ logits (n2 signatures); every condition within 2.5 × 10⁻⁵ |
| Runtime | T4 function 132 s (model load 8 s, reproduction 54 s, GPT-2 Large timing); L4 function 44 s |
| Cost | $0.10 by Modal's billing report for both attempts (T4 $0.056, L4 $0.023, CPU and memory $0.015, egress $0.008); $0.00 billed after credits |
| Data | About 1.3 MB uploaded per run (code, small inputs, `pyproject.toml`, `uv.lock`); 0.18 MB received |

**GPT-2 Large timing** (random tokens only; 20 tokens, 8 prompts per batch, all 360 heads of layers 18–35 under both references): 0.180 s per row on the T4 and 0.095 s on the L4 with 20 heads per pass, or 0.212 s and 0.117 s with three heads held in every pass. Peak GPU memory was 6.7 GB on the T4 and 4.7 GB on the L4. Per row, the L4 is both faster and cheaper: about $2.1 × 10⁻⁵ against $2.9 × 10⁻⁵.

**Attempts.** The first attempt (03:28–03:33 UTC) ran fully on Modal but failed when the local machine unpickled the result, because `torch.__version__` is a torch type and the Modal CLI's own Python has no torch. Its log is `run-attempt1.log`, and its cost is in the total. The second attempt (03:33–03:37 UTC) returned plain types. One known defect remains: `remote-record.json` keeps only one non-weight file hash per model (`tokenizer_config.json`, which matches the local copy) because of a dictionary-key bug; the weights themselves were checked at build.

Files: `modal_smoke.py` (Modal app), `compare_smoke.py` (local comparison, project environment), `records/` (`remote-record.json`, `modal-signatures.npz`, `comparison.json`, `billing-report.json`, `smoke-result.json`), `run.log` and `run-attempt1.log`. Both apps are stopped; nothing is deployed or kept warm.

Reproduce from the AI-01 root:

```bash
~/.local/bin/modal run studies/bridge-012/provenance/modal-smoke-test/modal_smoke.py
```

The script refuses to run while `records/` exists.
