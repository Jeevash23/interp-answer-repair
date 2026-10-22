# Model checkpoints

No weights are included. The code reads checkpoints from this folder, one subfolder per model:

| Subfolder | Hugging Face repository | Revision | Size | Used by |
|---|---|---|---|---|
| `gpt2-medium` | `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 1.52 GB | Studies 1–3, 5 and 6, the F4/F7 follow-up, the independent NumPy check |
| `gpt2-small` | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 0.55 GB | Study 4, historical comparisons; the unit tests read its tokeniser |
| `gpt2-large` | `openai-community/gpt2-large` | `32b71b12589c2f8d625668d2335a01cac3249519` | 3.25 GB | Study 7, which downloads it inside its Modal image instead |
| `pythia-410m-deduped` | `EleutherAI/pythia-410m-deduped` | `c4fc8d586d62df497f1f9b69d66d3ca419992d3e` | 0.91 GB | historical comparisons; the unit tests read its tokeniser |
| `tinyllama-1.1b-base` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `59f6f375b26bde864a6ca194a9a3044570490064` | 4.40 GB | historical comparisons |

`models.json` lists every file of every checkpoint with the SHA-256 recorded before the runs; the studies'
freeze records name the same digests.

```sh
python tools/get_models.py --list                          # the table above
python tools/get_models.py gpt2-medium gpt2-small          # what most studies need (make models)
python tools/get_models.py --no-weights gpt2-small pythia-410m   # tokenisers and configurations only (make tokenisers)
python tools/get_models.py --check                         # check what is here, without network access
python tools/get_models.py --link-from /path/to/models     # use checkpoints you already have, through links
```

`get_models.py` uses `huggingface_hub` (part of the environment) and stops if any file's digest differs from the
record. The historical transport scripts read checkpoints from `studies/transport/models`; `make link-models`
links that folder to this one, as in the original project. The checkpoints are distributed under their own
licences (MIT for GPT-2, Apache 2.0 for Pythia and TinyLlama).
