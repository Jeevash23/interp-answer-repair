"""Numerical validation only, using previously observed pilot prompts."""

import argparse
import json
import time

import torch

from transport_bridge.common import ROOT, now, read, write
from transport_bridge.engine import Engine

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2-medium")
p.add_argument("--device", default="cuda")
args = p.parse_args()
old = [
    json.loads(x)
    for x in (ROOT / "studies/routing/data/pilot-001.jsonl")
    .read_text()
    .splitlines()
][:2]
rows = [
    {
        "token_ids": r["clean_token_ids"],
        "io_id": r["io_token_id"],
        "subject_id": r["subject_token_id"],
        "alt_id": r["io_token_id"],
        "s2": r["positions"]["subject_second"],
    }
    for r in old
]
start = time.monotonic()
with torch.inference_mode():
    engine = Engine(args.model, read(ROOT / "configs/study.json"), args.device)
    result = engine.validate(rows)
result.update(
    model=args.model,
    utc=now(),
    seconds=time.monotonic() - start,
    device=args.device,
    gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
    max_gpu_memory=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
)
write(ROOT / f"results/{args.model}-technical-check.json", result)
print(json.dumps(result, indent=2))
