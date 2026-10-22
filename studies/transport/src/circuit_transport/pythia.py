"""Pinned offline loading for the Pythia cross-architecture study."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pythia-410m-deduped"
MODEL_MANIFEST = PROJECT_ROOT / "data" / "pythia-model-manifest.json"
CONFIG_CACHE = PROJECT_ROOT / "models" / "huggingface-config-cache"
MODEL_ALIAS = "pythia-410m-deduped"


def load_pythia_for_hooks() -> tuple[HookedTransformer, PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an unprocessed HookedTransformer and the raw HF model for validation.

    Transformers 5 renamed GPT-NeoX's output module from ``embed_out`` to
    ``lm_head`` before TransformerLens 3.8 updated its converter. The in-memory
    alias below restores the converter's expected public weight reference; it
    does not modify either library or change a tensor.
    """
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    revision = manifest["revision"]
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float32
    )
    hf_model.embed_out = hf_model.lm_head
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ALIAS,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device="cpu",
        dtype=torch.float32,
        move_to_device=False,
        default_prepend_bos=False,
        cache_dir=CONFIG_CACHE,
        revision=revision,
        local_files_only=True,
    )
    model.set_use_attn_result(True)
    model.eval()
    hf_model.eval()
    return model, hf_model, tokenizer
