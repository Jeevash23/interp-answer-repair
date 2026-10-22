"""Verify the pinned environment, CUDA device, and TransformerLens v3 model bridge."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("gpt2-small", "gpt2-medium", "gpt2-large")
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default="gpt2-small")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument(
        "--process-on-cpu",
        action="store_true",
        help="Process TransformerLens compatibility weights on CPU before moving to CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = PROJECT_ROOT / "models" / args.model
    dtype = DTYPES[args.dtype]
    print(f"torch={torch.__version__}")
    print(f"transformer-lens={version('transformer-lens')}")
    print(f"transformers={version('transformers')}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    if not model_path.exists():
        raise RuntimeError(f"Missing local model snapshot: {model_path}")

    device = torch.device("cuda:0")
    print(f"model={args.model}")
    print(f"dtype={args.dtype}")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"compute_capability={torch.cuda.get_device_capability(device)}")

    left = torch.randn((1024, 1024), device=device)
    right = left @ left
    torch.cuda.synchronize()
    print(f"cuda_matmul_shape={tuple(right.shape)}")
    del left, right

    load_device = torch.device("cpu") if args.process_on_cpu else device
    print(f"compatibility_processing_device={load_device}")
    bridge = TransformerBridge.boot_transformers(
        str(model_path),
        device=load_device,
        dtype=dtype,
    )
    bridge.enable_compatibility_mode()
    if args.process_on_cpu:
        bridge.to(device)
    prompt = "When Mary and John went to the store, John gave a book to"
    with torch.inference_mode():
        logits, cache = bridge.run_with_cache(
            prompt,
            names_filter=lambda name: name.endswith("hook_resid_pre"),
        )
    if not cache:
        raise RuntimeError("TransformerLens returned an empty activation cache")

    print(f"logits_shape={tuple(logits.shape)}")
    print(f"cached_hook_points={len(cache)}")
    print(f"peak_gpu_memory_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
