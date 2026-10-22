"""Template-level inference primitives for Experiment 003."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def paired_example_id(example_id: str) -> str:
    """Return the entity-pair identifier shared by adjacent ABBA/BABA rows."""
    prefix, pattern = example_id.rsplit("-", 1)
    if pattern not in {"abba", "baba"}:
        raise ValueError(f"Unexpected example pattern suffix: {example_id}")
    return prefix


def pair_arrays_by_template(
    rows: Sequence[Mapping[str, Any]],
    value_keys: Sequence[str],
) -> dict[str, NDArray[np.float64]]:
    """Average ABBA/BABA rows into aligned pair arrays for each template."""
    if not rows or not value_keys:
        raise ValueError("rows and value_keys must not be empty")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["template_id"]), paired_example_id(str(row["example_id"])))
        grouped.setdefault(key, []).append(row)
    templates: dict[str, list[tuple[str, list[float]]]] = {}
    for (template, pair_id), pair_rows in grouped.items():
        if len(pair_rows) != 2 or {row["pattern"] for row in pair_rows} != {"ABBA", "BABA"}:
            raise ValueError(f"{template}/{pair_id} is not one complete ABBA/BABA pair")
        values = [float(np.mean([float(row[key]) for row in pair_rows])) for key in value_keys]
        if not np.isfinite(values).all():
            raise ValueError(f"{template}/{pair_id} contains non-finite values")
        templates.setdefault(template, []).append((pair_id, values))
    output = {}
    for template, pairs in templates.items():
        ordered = [values for _, values in sorted(pairs)]
        output[template] = np.asarray(ordered, dtype=np.float64)
    return output


def two_stage_bootstrap(
    values_by_template: Mapping[str, ArrayLike],
    statistic: Callable[[NDArray[np.float64]], ArrayLike],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample templates, then entity pairs within each sampled template."""
    if samples < 1 or not values_by_template:
        raise ValueError("samples and values_by_template must be non-empty")
    template_names = sorted(values_by_template)
    arrays = [np.asarray(values_by_template[name], dtype=np.float64) for name in template_names]
    if any(array.ndim != 2 or array.shape[0] < 1 for array in arrays):
        raise ValueError("each template must contain a non-empty two-dimensional array")
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError("template arrays must have the same number of columns")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("template arrays must be finite")

    point = np.atleast_1d(np.asarray(statistic(np.concatenate(arrays)), dtype=np.float64))
    rng = np.random.default_rng(seed)
    draws: list[NDArray[np.float64]] = []
    for _ in range(samples):
        sampled_template_indices = rng.integers(0, len(arrays), size=len(arrays))
        sampled_arrays = []
        for template_index in sampled_template_indices:
            array = arrays[int(template_index)]
            pair_indices = rng.integers(0, array.shape[0], size=array.shape[0])
            sampled_arrays.append(array[pair_indices])
        draw = np.atleast_1d(
            np.asarray(statistic(np.concatenate(sampled_arrays)), dtype=np.float64)
        )
        if draw.shape != point.shape or not np.isfinite(draw).all():
            raise ValueError("statistic returned an invalid bootstrap draw")
        draws.append(draw)
    draw_array = np.stack(draws)
    return {
        "point": point.tolist(),
        "ci_low": np.percentile(draw_array, 2.5, axis=0).tolist(),
        "ci_high": np.percentile(draw_array, 97.5, axis=0).tolist(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "templates": len(arrays),
        "pairs": int(sum(array.shape[0] for array in arrays)),
    }


def exact_sign_flip_pvalue(differences: ArrayLike, *, alternative: str = "greater") -> float:
    """Enumerate every template-level sign assignment for a paired mean contrast."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError(f"Unsupported alternative: {alternative}")
    observed = float(values.mean())
    null = np.asarray(
        [
            np.mean(values * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-15))
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values."""
    if not pvalues:
        return {}
    if any(not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return adjusted
