"""Independently localized GPT-2 Medium transport primitives for Experiment 002."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.stats import rankdata
from torch import Tensor

Head = tuple[int, int]

SITES = {"s2": "subject_second", "end": "end"}
CONTROLS = ("correct", "random_head", "random_direction", "off_target", "mismatched_entities")


def result_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_result"


def result_cache_filter(name: str) -> bool:
    return name.endswith("attn.hook_result")


def positions_from_records(
    records: Sequence[Mapping[str, Any]], key: str, *, device: torch.device
) -> Tensor:
    return torch.tensor([record["positions"][key] for record in records], device=device)


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
    return torch.stack([_head_result(cache, head, positions) for head in heads], dim=0).sum(dim=0)


def make_layer_sweep_hook(delta: Tensor, positions: Tensor) -> Callable[[Tensor, Any], Tensor]:
    """Patch one different head per expanded row for an exhaustive layer sweep.

    ``delta`` has shape ``[batch, heads, d_head]``. The target batch must be expanded in
    example-major, head-minor order to ``batch * heads`` rows.
    """

    if delta.ndim != 3:
        raise ValueError("delta must have shape [batch, heads, d_head]")
    if positions.ndim != 1 or positions.shape[0] != delta.shape[0]:
        raise ValueError("positions must have one value per source example")

    def patch(value: Tensor, hook: Any) -> Tensor:
        del hook
        batch, heads, _ = delta.shape
        if value.shape[0] != batch * heads or value.shape[2] != heads:
            raise ValueError("expanded target batch does not match the localization delta")
        patched = value.clone()
        rows = torch.arange(batch * heads, device=value.device)
        example_rows = torch.div(rows, heads, rounding_mode="floor")
        head_rows = rows.remainder(heads)
        patched[rows, positions[example_rows], head_rows, :] += delta[example_rows, head_rows, :]
        return patched

    return patch


def select_top_heads(
    scores: Mapping[Head, float], *, count: int, positive_only: bool = False
) -> tuple[Head, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    candidates = [item for item in scores.items() if not positive_only or item[1] > 0]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} heads are eligible for a top-{count} selection")
    ordered = sorted(candidates, key=lambda item: (-item[1], item[0][0], item[0][1]))
    return tuple(head for head, _ in ordered[:count])


def localization_similarity(
    canonical: Mapping[Head, float], controlled: Mapping[Head, float]
) -> dict[str, float | int]:
    if set(canonical) != set(controlled):
        raise ValueError("localization score maps must contain the same heads")
    heads = sorted(canonical)
    left = np.asarray([canonical[head] for head in heads], dtype=np.float64)
    right = np.asarray([controlled[head] for head in heads], dtype=np.float64)
    left_ranks = rankdata(left, method="average")
    right_ranks = rankdata(right, method="average")
    correlation = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    output: dict[str, float | int] = {"n_heads": len(heads), "spearman": correlation}
    for count in (8, 32):
        left_top = set(select_top_heads(canonical, count=count))
        right_top = set(select_top_heads(controlled, count=count))
        intersection = len(left_top & right_top)
        union = len(left_top | right_top)
        output[f"top_{count}_intersection"] = intersection
        output[f"top_{count}_jaccard"] = intersection / union
    return output


def choose_random_heads(
    selected: Sequence[Head], *, excluded: Sequence[Head], n_heads: int
) -> tuple[Head, ...]:
    """Choose deterministic, layer-matched heads outside the selected union."""
    blocked = set(excluded)
    chosen: set[Head] = set()
    output: list[Head] = []
    for layer, head in selected:
        for offset in range(1, n_heads + 1):
            candidate = (layer, (head + offset) % n_heads)
            if candidate not in blocked and candidate not in chosen:
                chosen.add(candidate)
                output.append(candidate)
                break
        else:
            raise ValueError(f"No unused random head remains in layer {layer}")
    return tuple(output)


def build_group_patch_hooks(
    source_clean_cache: Mapping[str, Tensor],
    source_corrupt_cache: Mapping[str, Tensor],
    records: Sequence[Mapping[str, Any]],
    *,
    heads: Sequence[Head],
    site: str,
    control: str,
    random_heads: Sequence[Head] | None,
    random_seed: int,
) -> tuple[list[tuple[str, Callable[[Tensor, Any], Tensor]]], tuple[Head, ...]]:
    if site not in SITES:
        raise ValueError(f"Unknown site: {site}")
    if control not in CONTROLS:
        raise ValueError(f"Unknown control: {control}")
    if not heads:
        raise ValueError("heads must not be empty")

    sample = source_clean_cache[result_hook_name(heads[0][0])]
    device = sample.device
    source_positions = positions_from_records(records, SITES[site], device=device)
    target_positions = source_positions.clone()
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
            generator = torch.Generator(device=device)
            generator.manual_seed(random_seed + 100 * layer + head)
            random_delta = torch.randn(
                delta.shape, generator=generator, device=device, dtype=delta.dtype
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
    target_end_heads: Sequence[Head],
) -> Tensor:
    device = clean_cache[result_hook_name(target_end_heads[0][0])].device
    positions = positions_from_records(records, "end", device=device)
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
    heads: Sequence[Head],
    unembed: Tensor,
) -> Tensor:
    device = cache[result_hook_name(heads[0][0])].device
    positions = positions_from_records(records, "end", device=device)
    result = summed_head_result(cache, heads, positions).float()
    io_ids = torch.tensor([record["io_token_id"] for record in records], device=device)
    subject_ids = torch.tensor([record["subject_token_id"] for record in records], device=device)
    direction = unembed[:, io_ids].T.float() - unembed[:, subject_ids].T.float()
    return torch.einsum("bd,bd->b", result, direction)


def paired_bootstrap_contrast(
    first: ArrayLike,
    second: ArrayLike,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 1 or first_array.shape != second_array.shape or first_array.size == 0:
        raise ValueError("paired contrast inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("paired contrast inputs must be finite")
    differences = first_array - second_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    boot = differences[indices].mean(axis=1)
    low, high = np.percentile(boot, (2.5, 97.5))
    return {
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def sign_flip_pvalue(
    differences: ArrayLike, *, draws: int, seed: int, alternative: str = "greater"
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite one-dimensional array")
    if draws < 1:
        raise ValueError("draws must be positive")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, values.size))
    null = (signs * values).mean(axis=1)
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed)
    else:
        extreme = np.count_nonzero(np.abs(null) >= abs(observed))
    return float((extreme + 1) / (draws + 1))


def benjamini_hochberg(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues:
        return {}
    names = list(pvalues)
    values = np.asarray([pvalues[name] for name in names], dtype=np.float64)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.minimum.accumulate(
        (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1]
    )[::-1]
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return {name: float(adjusted[index]) for index, name in enumerate(names)}
