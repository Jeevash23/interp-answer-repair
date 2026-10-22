import numpy as np
import pytest

from circuit_transport.metrics import (
    causal_transport_score,
    mean_paired_effect,
    paired_bootstrap_transport,
)


def test_mean_paired_effect() -> None:
    outcome = np.array([0.8, 0.6, 0.4])
    baseline = np.array([0.2, 0.1, 0.0])
    assert mean_paired_effect(outcome, baseline) == pytest.approx(0.5)


def test_transport_score_full_recovery() -> None:
    clean = np.array([0.8, 0.6, 0.4])
    corrupt = np.array([0.2, 0.1, 0.0])
    assert causal_transport_score(clean, corrupt, clean) == pytest.approx(1.0)


def test_transport_score_zero_recovery() -> None:
    clean = np.array([0.8, 0.6, 0.4])
    corrupt = np.array([0.2, 0.1, 0.0])
    assert causal_transport_score(clean, corrupt, corrupt) == pytest.approx(0.0)


def test_transport_score_rejects_small_denominator() -> None:
    values = np.array([0.2, 0.1, 0.0])
    with pytest.raises(ValueError, match="too small"):
        causal_transport_score(values, values, values)


def test_transport_score_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        causal_transport_score([0.8, 0.6], [0.2], [0.4, 0.3])


def test_paired_bootstrap_transport_is_deterministic() -> None:
    clean = np.array([0.8, 0.6, 0.4, 0.7])
    corrupt = np.array([0.2, 0.1, 0.0, 0.2])
    patched = corrupt + 0.5 * (clean - corrupt)
    first = paired_bootstrap_transport(clean, corrupt, patched, samples=100, seed=12)
    second = paired_bootstrap_transport(clean, corrupt, patched, samples=100, seed=12)
    assert first == second
    assert first["ratio"] == pytest.approx(0.5)
    assert first["numerator"] == pytest.approx(0.25)


def test_paired_bootstrap_transport_omits_unstable_ratio() -> None:
    values = np.array([0.2, 0.1, 0.0])
    summary = paired_bootstrap_transport(
        values, values, values, samples=10, minimum_denominator=0.01
    )
    assert summary["ratio"] is None
    assert summary["ratio_ci_low"] is None
