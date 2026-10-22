"""Head-node interventions with explicit END positions."""

from __future__ import annotations

import torch


def head_hook(layer, head, replacement, positions):
    calls = {"count": 0}

    def replace(value, hook):
        del hook
        if value.ndim != 4 or value.shape[2:] != (12, 64):
            raise RuntimeError(f"Unexpected GPT-2 hook_z geometry: {tuple(value.shape)}")
        output = value.clone()
        rows = torch.arange(value.shape[0], device=value.device)
        output[rows, positions.to(value.device), head, :] = replacement.to(value.device)
        calls["count"] += 1
        return output

    return (f"blocks.{layer}.attn.hook_z", replace), calls


def outcomes(logits, records):
    last = logits[:, -1, :].float()
    rows = torch.arange(len(records), device=last.device)
    io = torch.tensor([r["io_token_id"] for r in records], device=last.device)
    subject = torch.tensor([r["subject_token_id"] for r in records], device=last.device)
    probs = last.softmax(-1)
    gap = last[rows, io] - last[rows, subject]
    return {
        "logit_difference": gap.cpu(),
        "probability_difference": (probs[rows, io] - probs[rows, subject]).cpu(),
        "io_probability": probs[rows, io].cpu(),
        "subject_probability": probs[rows, subject].cpu(),
        "pairwise_correct": (gap > 0).float().cpu(),
        "top1_token_id": last.argmax(-1).cpu(),
    }
