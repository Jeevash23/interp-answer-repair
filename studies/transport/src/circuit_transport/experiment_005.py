"""Batch-independent controls for the post-hoc robustness extension.

These utilities do not change any frozen Study 1--4 implementation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

Head = tuple[int, int]
SEED = 20260905
SITES = {"s2": "subject_second", "end": "end"}


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sampled_heads(
    reference: Sequence[Head], excluded: Sequence[Head], *, n_heads: int, seed: int
) -> tuple[Head, ...]:
    """Uniform samples without replacement within each reference layer."""
    rng = random.Random(seed)
    blocked = set(excluded)
    output = []
    for layer, count in sorted(Counter(layer for layer, _ in reference).items()):
        candidates = [h for h in range(n_heads) if (layer, h) not in blocked]
        if len(candidates) < count:
            raise ValueError(f"Insufficient eligible heads in layer {layer}")
        output.extend((layer, h) for h in sorted(rng.sample(candidates, count)))
    return tuple(output)


def donor_map(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Seeded disjoint donor matching in each full template, preserving pattern.

    All donor/recipient entity triples must be disjoint. Mapping is keyed by IDs,
    not by record order or inference minibatch boundaries.
    """
    groups: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(dict)
    for row in records:
        pair_id = str(row["example_id"]).rsplit("-", 1)[0]
        group = groups[str(row["template_id"])].setdefault(pair_id, {})
        if row["pattern"] in group:
            raise ValueError("Duplicate pair/pattern")
        group[str(row["pattern"])] = row
    result = {}
    for template, group in sorted(groups.items()):
        pair_ids = sorted(group)
        if len(pair_ids) < 2:
            raise ValueError("A mismatch pool needs at least two entity pairs")
        names = {}
        for pair_id, pair in group.items():
            if set(pair) != {"ABBA", "BABA"}:
                raise ValueError("Each entity pair must contain ABBA and BABA")
            sets = [
                {r[k] for k in ("io_name", "subject_name", "corrupt_name")} for r in pair.values()
            ]
            if sets[0] != sets[1]:
                raise ValueError("Paired patterns must share the same entity triple")
            names[pair_id] = sets[0]
        rng = random.Random(stable_seed(SEED, "donor", template))
        candidates = {}
        for recipient in pair_ids:
            eligible = [donor for donor in pair_ids if not names[recipient] & names[donor]]
            rng.shuffle(eligible)
            candidates[recipient] = eligible
        assigned: dict[str, str] = {}

        def augment(recipient, seen, *, candidates=candidates, assigned=assigned):
            for donor in candidates[recipient]:
                if donor in seen:
                    continue
                seen.add(donor)
                if donor not in assigned or augment(assigned[donor], seen):
                    assigned[donor] = recipient
                    return True
            return False

        for recipient in pair_ids:
            if not augment(recipient, set()):
                raise ValueError(f"No disjoint donor permutation for {template}")
        for donor, recipient in assigned.items():
            for pattern in ("ABBA", "BABA"):
                result[str(group[recipient][pattern]["example_id"])] = str(
                    group[donor][pattern]["example_id"]
                )
    if len(result) != len(records) or len(set(result.values())) != len(records):
        raise ValueError("Donor map must be a permutation of unique example IDs")
    return result


def positions(records: Sequence[Mapping[str, Any]], condition: str, site: str) -> torch.Tensor:
    def pos(row: Mapping[str, Any]) -> int:
        value = row["positions"]
        return int(value[condition][SITES[site]] if condition in value else value[SITES[site]])

    return torch.tensor([pos(r) for r in records], dtype=torch.long)


def random_directions(
    delta: torch.Tensor,
    example_ids: Sequence[str],
    *,
    model: str,
    site: str,
    head: Head,
    draw: int,
) -> torch.Tensor:
    """Per-ID isotropic residual-space directions with matched per-head norms."""
    output = []
    for row, example_id in zip(delta, example_ids, strict=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(SEED, model, example_id, site, head, draw))
        direction = torch.randn(row.shape, generator=generator, dtype=torch.float32)
        output.append(direction * row.float().norm() / direction.norm().clamp_min(1e-12))
    return torch.stack(output).to(delta.device)


def symmetric_scale(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Match concatenated head-result norms, not the norm after summing heads.

    Input shape [examples, heads, d_model]. A zero norm in either condition sets
    both matched interventions to zero and is explicitly recorded by the runner.
    """
    a = left.float().flatten(1).norm(dim=1)
    b = right.float().flatten(1).norm(dim=1)
    shared = (a * b).sqrt()
    return shared / a.clamp_min(1e-12), shared / b.clamp_min(1e-12), a, b, shared


def addition_hooks(
    deltas: Mapping[Head, torch.Tensor],
    target_positions: torch.Tensor,
) -> list[tuple[str, Any]]:
    """Add the sum of selected head-result deltas to each layer's attention output.

    This equals adding those vectors before the per-head sum. Later selected
    layers retain the live target computation affected by earlier injections.
    """
    layers: dict[int, list[torch.Tensor]] = defaultdict(list)
    for (layer, _), delta in deltas.items():
        layers[layer].append(delta)
    hooks = []
    for layer, values in layers.items():
        delta = torch.stack(values).sum(0)

        def add(value: torch.Tensor, hook: Any, *, delta=delta) -> torch.Tensor:
            del hook
            output = value.clone()
            rows = torch.arange(len(value), device=value.device)
            output[rows, target_positions.to(value.device)] += delta.to(value.device)
            return output

        hooks.append((f"blocks.{layer}.hook_attn_out", add))
    return hooks


def clamp_hooks(
    corrupt_z: torch.Tensor,
    heads: Sequence[Head],
    target_positions: torch.Tensor,
) -> list[tuple[str, Any]]:
    """Hold END head nodes at their own corrupt outputs, not specific edges.

    corrupt_z has shape [examples, layers, heads, d_head].
    """
    layers: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        layers[layer].append(head)
    hooks = []
    for layer, group in layers.items():

        def clamp(value: torch.Tensor, hook: Any, *, layer=layer, group=tuple(group)):
            del hook
            output = value.clone()
            rows = torch.arange(len(value), device=value.device)
            for head in group:
                output[rows, target_positions.to(value.device), head] = corrupt_z[
                    :, layer, head
                ].to(value.device)
            return output

        hooks.append((f"blocks.{layer}.attn.hook_z", clamp))
    return hooks
