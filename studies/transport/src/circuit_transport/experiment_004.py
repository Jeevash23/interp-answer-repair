"""Pythia-specific transport primitives for Experiment 004."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from circuit_transport.experiment_002 import CONTROLS, SITES, Head, result_hook_name


def condition_positions(
    records: Sequence[Mapping[str, Any]], condition: str, key: str, *, device: torch.device
) -> Tensor:
    """Return condition-specific role positions for the Pythia manifest."""
    if condition not in {"canonical", "controlled"}:
        raise ValueError(f"Unknown condition: {condition}")
    return torch.tensor([record["positions"][condition][key] for record in records], device=device)


def _head_result(cache: Mapping[str, Tensor], head: Head, positions: Tensor) -> Tensor:
    layer, head_index = head
    value = cache[result_hook_name(layer)]
    rows = torch.arange(value.shape[0], device=value.device)
    return value[rows, positions, head_index, :]


def summed_head_result(
    cache: Mapping[str, Tensor], heads: Sequence[Head], positions: Tensor
) -> Tensor:
    if not heads:
        raise ValueError("heads must not be empty")
    return torch.stack([_head_result(cache, head, positions) for head in heads]).sum(dim=0)


def build_group_patch_hooks(
    source_clean_cache: Mapping[str, Tensor],
    source_corrupt_cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    source_condition: str,
    target_condition: str,
    heads: Sequence[Head],
    site: str,
    control: str,
    random_heads: Sequence[Head] | None,
    random_seed: int,
) -> tuple[list[tuple[str, Callable[[Tensor, Any], Tensor]]], tuple[Head, ...]]:
    """Build additive patches with separately resolved source and target positions."""
    if site not in SITES:
        raise ValueError(f"Unknown site: {site}")
    if control not in CONTROLS:
        raise ValueError(f"Unknown control: {control}")
    if not heads:
        raise ValueError("heads must not be empty")
    sample = source_clean_cache[result_hook_name(heads[0][0])]
    source_positions = condition_positions(
        records, source_condition, SITES[site], device=sample.device
    )
    target_positions = condition_positions(
        records, target_condition, SITES[site], device=sample.device
    )
    if control == "off_target":
        target_positions = target_positions - 1

    active_heads = tuple(random_heads) if control == "random_head" else tuple(heads)
    if control == "random_head" and (not random_heads or len(random_heads) != len(heads)):
        raise ValueError("random_head control requires a matched group")

    patches_by_hook: dict[str, list[tuple[int, Tensor]]] = {}
    for layer, head in active_heads:
        delta = _head_result(source_clean_cache, (layer, head), source_positions) - _head_result(
            source_corrupt_cache, (layer, head), source_positions
        )
        if control == "mismatched_entities":
            delta = delta.roll(shifts=2, dims=0)
        elif control == "random_direction":
            generator = torch.Generator(device=sample.device)
            generator.manual_seed(random_seed + 100 * layer + head)
            random_delta = torch.randn(
                delta.shape, generator=generator, device=sample.device, dtype=delta.dtype
            )
            scale = delta.float().norm(dim=-1, keepdim=True) / random_delta.float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            delta = (random_delta.float() * scale).to(delta.dtype)
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
    return hooks, active_heads


def s2_local_projection(
    clean_cache: Mapping[str, Tensor],
    corrupt_cache: Mapping[str, Tensor],
    patched_cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    target_condition: str,
    target_end_heads: Sequence[Head],
) -> Tensor:
    device = clean_cache[result_hook_name(target_end_heads[0][0])].device
    positions = condition_positions(records, target_condition, "end", device=device)
    clean = summed_head_result(clean_cache, target_end_heads, positions).float()
    corrupt = summed_head_result(corrupt_cache, target_end_heads, positions).float()
    patched = summed_head_result(patched_cache, target_end_heads, positions).float()
    direction = clean - corrupt
    change = patched - corrupt
    denominator = torch.einsum("bd,bd->b", direction, direction).clamp_min(1e-12)
    return torch.einsum("bd,bd->b", change, direction) / denominator


def end_direct_contribution(
    cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    target_condition: str,
    heads: Sequence[Head],
    unembed: Tensor,
) -> Tensor:
    device = cache[result_hook_name(heads[0][0])].device
    positions = condition_positions(records, target_condition, "end", device=device)
    result = summed_head_result(cache, heads, positions).float()
    io_ids = torch.tensor([record["io_token_id"] for record in records], device=device)
    subject_ids = torch.tensor([record["subject_token_id"] for record in records], device=device)
    direction = unembed[:, io_ids].T.float() - unembed[:, subject_ids].T.float()
    return torch.einsum("bd,bd->b", result, direction)
