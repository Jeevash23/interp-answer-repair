"""TransformerLens intervention primitives for Experiment 001."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

Head = tuple[int, int]

HEAD_GROUPS: dict[str, tuple[Head, ...]] = {
    "duplicate_induction": ((0, 1), (3, 0), (5, 5), (6, 9)),
    "s_inhibition": ((7, 3), (7, 9), (8, 6), (8, 10)),
    "name_mover": ((9, 9), (9, 6), (10, 0)),
}

RANDOM_HEAD_GROUPS: dict[str, tuple[Head, ...]] = {
    "duplicate_induction": ((0, 11), (3, 11), (5, 11), (6, 11)),
    "s_inhibition": ((7, 0), (7, 11), (8, 0), (8, 11)),
    "name_mover": ((9, 1), (9, 2), (10, 3)),
}

STAGE_POSITIONS = {
    "duplicate_induction": "subject_second",
    "s_inhibition": "end",
    "name_mover": "end",
}

CONTROLS = ("correct", "random_head", "random_direction", "off_target", "mismatched_roles")


def result_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_result"


def pattern_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"


def cache_filter(name: str) -> bool:
    """Cache head results everywhere and attention patterns needed for Name Movers."""
    return "hook_result" in name or (
        "hook_pattern" in name and name.startswith(("blocks.9.", "blocks.10."))
    )


def tokens_from_records(
    records: Sequence[Mapping[str, Any]],
    condition: str,
    variant: str,
    *,
    device: torch.device,
) -> Tensor:
    key = f"{variant}_token_ids"
    return torch.tensor(
        [record[condition][key] for record in records], dtype=torch.long, device=device
    )


def positions_from_records(
    records: Sequence[Mapping[str, Any]], key: str, *, device: torch.device
) -> Tensor:
    return torch.tensor([record["positions"][key] for record in records], device=device)


def final_outcomes(logits: Tensor, records: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    """Return paired IO/subject final-token outcomes on CPU."""
    device = logits.device
    rows = torch.arange(len(records), device=device)
    io_ids = torch.tensor([record["io_token_id"] for record in records], device=device)
    subject_ids = torch.tensor([record["subject_token_id"] for record in records], device=device)
    final_logits = logits[:, -1, :]
    probabilities = final_logits.softmax(dim=-1)
    io_logits = final_logits[rows, io_ids]
    subject_logits = final_logits[rows, subject_ids]
    io_probabilities = probabilities[rows, io_ids]
    subject_probabilities = probabilities[rows, subject_ids]
    return {
        "probability_difference": (io_probabilities - subject_probabilities).detach().cpu(),
        "logit_difference": (io_logits - subject_logits).detach().cpu(),
        "io_probability": io_probabilities.detach().cpu(),
        "subject_probability": subject_probabilities.detach().cpu(),
        "io_wins_pairwise": (io_logits > subject_logits).to(torch.float32).detach().cpu(),
    }


def _head_result(cache: Mapping[str, Tensor], head: Head, positions: Tensor) -> Tensor:
    layer, head_index = head
    value = cache[result_hook_name(layer)]
    rows = torch.arange(value.shape[0], device=value.device)
    return value[rows, positions, head_index, :]


def summed_head_result(
    cache: Mapping[str, Tensor], heads: Iterable[Head], positions: Tensor
) -> Tensor:
    values = [_head_result(cache, head, positions) for head in heads]
    return torch.stack(values, dim=0).sum(dim=0)


def name_mover_attention_preference(
    cache: Mapping[str, Tensor], records: Sequence[Mapping[str, Any]]
) -> Tensor:
    """Return mean Name Mover attention to IO minus attention to S1."""
    sample = cache[pattern_hook_name(9)]
    device = sample.device
    rows = torch.arange(len(records), device=device)
    end = positions_from_records(records, "end", device=device)
    io = positions_from_records(records, "io", device=device)
    subject_first = positions_from_records(records, "subject_first", device=device)
    preferences = []
    for layer, head in HEAD_GROUPS["name_mover"]:
        pattern = cache[pattern_hook_name(layer)]
        preferences.append(pattern[rows, head, end, io] - pattern[rows, head, end, subject_first])
    return torch.stack(preferences, dim=0).mean(dim=0)


def name_mover_direct_contribution(
    cache: Mapping[str, Tensor], records: Sequence[Mapping[str, Any]], unembed: Tensor
) -> Tensor:
    """Return summed Name Mover contribution in the IO-minus-subject logit direction."""
    sample = cache[result_hook_name(9)]
    device = sample.device
    end = positions_from_records(records, "end", device=device)
    result = summed_head_result(cache, HEAD_GROUPS["name_mover"], end)
    io_ids = torch.tensor([record["io_token_id"] for record in records], device=device)
    subject_ids = torch.tensor([record["subject_token_id"] for record in records], device=device)
    direction = unembed[:, io_ids].T - unembed[:, subject_ids].T
    return torch.einsum("bd,bd->b", result, direction)


def early_local_projection(
    clean_cache: Mapping[str, Tensor],
    corrupt_cache: Mapping[str, Tensor],
    patched_cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
) -> Tensor:
    """Project patched S-Inhibition change onto the target clean causal direction."""
    device = clean_cache[result_hook_name(7)].device
    end = positions_from_records(records, "end", device=device)
    clean = summed_head_result(clean_cache, HEAD_GROUPS["s_inhibition"], end)
    corrupt = summed_head_result(corrupt_cache, HEAD_GROUPS["s_inhibition"], end)
    patched = summed_head_result(patched_cache, HEAD_GROUPS["s_inhibition"], end)
    target_direction = clean - corrupt
    patched_change = patched - corrupt
    denominator = torch.einsum("bd,bd->b", target_direction, target_direction).clamp_min(1e-12)
    return torch.einsum("bd,bd->b", patched_change, target_direction) / denominator


def scalar_local_outcome(
    stage: str,
    cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    unembed: Tensor,
) -> Tensor:
    if stage == "s_inhibition":
        return name_mover_attention_preference(cache, records)
    if stage == "name_mover":
        return name_mover_direct_contribution(cache, records, unembed)
    raise ValueError(f"Stage {stage!r} does not have a scalar local outcome")


def build_patch_hooks(
    source_clean_cache: Mapping[str, Tensor],
    source_corrupt_cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    control: str,
    random_seed: int,
) -> list[tuple[str, Callable[[Tensor, Any], Tensor]]]:
    """Build additive source-difference hooks for one stage and control."""
    if stage not in HEAD_GROUPS:
        raise ValueError(f"Unknown stage: {stage}")
    if control not in CONTROLS:
        raise ValueError(f"Unknown control: {control}")

    device = source_clean_cache[result_hook_name(0)].device
    source_positions = positions_from_records(records, STAGE_POSITIONS[stage], device=device)
    target_positions = source_positions.clone()
    if control == "off_target":
        target_positions = target_positions - 1

    source_heads = RANDOM_HEAD_GROUPS[stage] if control == "random_head" else HEAD_GROUPS[stage]
    patches_by_hook: dict[str, list[tuple[int, Tensor]]] = {}
    for layer, head in source_heads:
        delta = _head_result(source_clean_cache, (layer, head), source_positions) - _head_result(
            source_corrupt_cache, (layer, head), source_positions
        )
        if control == "mismatched_roles":
            # Adjacent ABBA/BABA rows share an entity triple, so shift by two rows.
            delta = delta.roll(shifts=2, dims=0)
        elif control == "random_direction":
            generator = torch.Generator(device=device)
            generator.manual_seed(random_seed + 100 * layer + head)
            random_delta = torch.randn(
                delta.shape, generator=generator, device=device, dtype=delta.dtype
            )
            scale = delta.norm(dim=-1, keepdim=True) / random_delta.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            delta = random_delta * scale
        patches_by_hook.setdefault(result_hook_name(layer), []).append((head, delta))

    hooks: list[tuple[str, Callable[[Tensor, Any], Tensor]]] = []
    for hook_name, entries in patches_by_hook.items():

        def patch_result(
            value: Tensor,
            hook: Any,
            *,
            entries: list[tuple[int, Tensor]] = entries,
            positions: Tensor = target_positions,
        ) -> Tensor:
            del hook
            patched = value.clone()
            rows = torch.arange(patched.shape[0], device=patched.device)
            for head, delta in entries:
                patched[rows, positions, head, :] += delta
            return patched

        hooks.append((hook_name, patch_result))
    return hooks
