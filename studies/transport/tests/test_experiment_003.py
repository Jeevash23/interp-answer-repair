import numpy as np
import pytest

from circuit_transport.experiment_003 import (
    exact_sign_flip_pvalue,
    holm_adjust,
    pair_arrays_by_template,
    paired_example_id,
    two_stage_bootstrap,
)


def _rows() -> list[dict]:
    rows = []
    for template in ("a", "b"):
        for pair in range(2):
            for pattern, offset in (("ABBA", 0.0), ("BABA", 0.2)):
                rows.append(
                    {
                        "template_id": template,
                        "example_id": f"example-{template}-{pair}-{pattern.lower()}",
                        "pattern": pattern,
                        "first": 1.0 + pair + offset,
                        "second": 0.5,
                    }
                )
    return rows


def test_pair_arrays_preserve_templates_and_average_patterns() -> None:
    assert paired_example_id("example-a-0-abba") == "example-a-0"
    arrays = pair_arrays_by_template(_rows(), ("first", "second"))
    assert set(arrays) == {"a", "b"}
    assert arrays["a"].shape == (2, 2)
    assert arrays["a"][0].tolist() == pytest.approx([1.1, 0.5])


def test_two_stage_bootstrap_is_deterministic_and_template_level() -> None:
    arrays = pair_arrays_by_template(_rows(), ("first", "second"))

    def statistic(values: np.ndarray) -> np.ndarray:
        return values.mean(axis=0)

    first = two_stage_bootstrap(arrays, statistic, samples=200, seed=7)
    second = two_stage_bootstrap(arrays, statistic, samples=200, seed=7)
    assert first == second
    assert first["point"] == pytest.approx([1.6, 0.5])
    assert first["templates"] == 2
    assert first["pairs"] == 4


def test_exact_sign_flip_and_holm_adjustment() -> None:
    assert exact_sign_flip_pvalue(np.ones(8)) == pytest.approx(1 / 256)
    assert exact_sign_flip_pvalue(np.ones(8), alternative="two-sided") == pytest.approx(2 / 256)
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.20})
