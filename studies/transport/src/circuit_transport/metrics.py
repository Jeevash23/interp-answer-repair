"""Predeclared metrics for causal-transport experiments."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _paired(first: ArrayLike, second: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first_array = _as_1d(first, name="first")
    second_array = _as_1d(second, name="second")
    if first_array.shape != second_array.shape:
        raise ValueError("paired outcomes must have identical shapes")
    return first_array, second_array


def mean_paired_effect(outcome: ArrayLike, baseline: ArrayLike) -> float:
    """Return the mean paired effect E[outcome - baseline]."""

    outcome_array, baseline_array = _paired(outcome, baseline)
    return float(np.mean(outcome_array - baseline_array))


def causal_transport_score(
    target_clean: ArrayLike,
    target_corrupt: ArrayLike,
    target_patched: ArrayLike,
    *,
    minimum_denominator: float = 1e-8,
) -> float:
    """Return cross-condition recovery divided by the target clean effect.

    This is a ratio of two mean paired effects, not a mean of per-example
    ratios. Report the numerator and denominator alongside the ratio and
    estimate uncertainty using the frozen evaluation protocol.
    """

    clean_array = _as_1d(target_clean, name="target_clean")
    corrupt_array = _as_1d(target_corrupt, name="target_corrupt")
    patched_array = _as_1d(target_patched, name="target_patched")

    if clean_array.shape != corrupt_array.shape or clean_array.shape != patched_array.shape:
        raise ValueError("clean, corrupt, and patched outcomes must have identical shapes")

    target_effect = float(np.mean(clean_array - corrupt_array))
    if abs(target_effect) < minimum_denominator:
        raise ValueError("target clean effect is too small to normalize safely")

    transport_effect = float(np.mean(patched_array - corrupt_array))
    return transport_effect / target_effect


def paired_bootstrap_transport(
    target_clean: ArrayLike,
    target_corrupt: ArrayLike,
    target_patched: ArrayLike,
    *,
    samples: int = 2_000,
    seed: int = 20_260_831,
    minimum_denominator: float = 1e-8,
) -> dict[str, float | int | None]:
    """Summarize paired transport effects with percentile bootstrap intervals."""
    clean = _as_1d(target_clean, name="target_clean")
    corrupt = _as_1d(target_corrupt, name="target_corrupt")
    patched = _as_1d(target_patched, name="target_patched")
    if clean.shape != corrupt.shape or clean.shape != patched.shape:
        raise ValueError("clean, corrupt, and patched outcomes must have identical shapes")
    if samples < 1:
        raise ValueError("samples must be positive")

    numerator_values = patched - corrupt
    denominator_values = clean - corrupt
    numerator = float(np.mean(numerator_values))
    denominator = float(np.mean(denominator_values))
    ratio = numerator / denominator if abs(denominator) >= minimum_denominator else None

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(samples, clean.size))
    boot_numerator = np.mean(numerator_values[indices], axis=1)
    boot_denominator = np.mean(denominator_values[indices], axis=1)
    valid = np.abs(boot_denominator) >= minimum_denominator
    boot_ratio = boot_numerator[valid] / boot_denominator[valid]

    def interval(values: NDArray[np.float64]) -> tuple[float | None, float | None]:
        if values.size == 0:
            return None, None
        low, high = np.percentile(values, (2.5, 97.5))
        return float(low), float(high)

    numerator_low, numerator_high = interval(boot_numerator)
    denominator_low, denominator_high = interval(boot_denominator)
    ratio_low, ratio_high = interval(boot_ratio)
    return {
        "n": int(clean.size),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "numerator": numerator,
        "numerator_ci_low": numerator_low,
        "numerator_ci_high": numerator_high,
        "denominator": denominator,
        "denominator_ci_low": denominator_low,
        "denominator_ci_high": denominator_high,
        "ratio": ratio,
        "ratio_ci_low": ratio_low,
        "ratio_ci_high": ratio_high,
        "valid_ratio_bootstrap_samples": int(valid.sum()),
    }
