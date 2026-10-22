"""Tests for the separate, post-hoc saved-output audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_research_20260905", PROJECT_ROOT / "scripts/audit_research_20260905.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def manifest_rows() -> list[dict]:
    return [
        {
            "template_id": "one-template",
            "example_id": f"example-{pair}-{pattern.lower()}",
            "pattern": pattern,
            "io_name": f"io-{pair}",
            "subject_name": f"subject-{pair}",
            "corrupt_name": f"corrupt-{pair}",
        }
        for pair in range(10)
        for pattern in ("ABBA", "BABA")
    ]


def patch_rows() -> list[dict]:
    rows = []
    directions = {
        ("canonical", "controlled"): (0.12, 0.10),
        ("controlled", "canonical"): (0.18, 0.40),
        ("canonical", "canonical"): (0.30, 0.40),
        ("controlled", "controlled"): (0.08, 0.10),
    }
    for template in ("a", "b"):
        for pair in range(3):
            for pattern in ("ABBA", "BABA"):
                for (source, target), (effect, denominator) in directions.items():
                    rows.append(
                        {
                            "template_id": template,
                            "example_id": f"example-{template}-{pair}-{pattern.lower()}",
                            "pattern": pattern,
                            "site": "end",
                            "control": "correct",
                            "source_condition": source,
                            "target_condition": target,
                            "target_clean_probability_difference": denominator,
                            "target_corrupt_probability_difference": 0.0,
                            "target_patched_probability_difference": effect,
                        }
                    )
    return rows


def test_historical_donors_change_for_half_of_twenty_rows() -> None:
    old = AUDIT.batch_assignments(manifest_rows(), 20)
    new = AUDIT.batch_assignments(manifest_rows(), 4)
    assert sum(old[key]["donor_id"] != new[key]["donor_id"] for key in old) == 10
    assert sum(old[key]["rng_batch_offset"] != new[key]["rng_batch_offset"] for key in old) == 16
    assert old["example-0-abba"]["donor_id"] == "example-9-abba"
    assert new["example-0-abba"]["donor_id"] == "example-1-abba"
    assert all(row["same_pattern"] for row in new.values())


def test_cyclic_null_is_deterministic_and_excludes_union() -> None:
    selected = [[1, 1], [1, 3], [2, 3]]
    excluded = {(1, 1), (1, 2), (1, 3), (2, 3)}
    expected = [[1, 4], [1, 5], [2, 4]]
    assert AUDIT.cyclic_heads(selected, excluded, 6) == expected
    assert AUDIT.cyclic_heads(selected, excluded, 6) == expected
    with pytest.raises(ValueError, match="No remaining"):
        AUDIT.cyclic_heads([[1, 1]], {(1, head) for head in range(6)}, 6)


def test_ratio_reverses_raw_direction_without_changing_data() -> None:
    arrays = AUDIT.aligned_pair_arrays(patch_rows(), "end", "probability")
    statistics = AUDIT.directional_statistics(np.concatenate(list(arrays.values())))
    assert statistics[2] == pytest.approx(-0.06)
    assert statistics[5] == pytest.approx(1.2)
    assert statistics[6] == pytest.approx(0.45)
    assert statistics[7] == pytest.approx(0.75)


def test_pair_alignment_uses_ids_not_input_order() -> None:
    original = AUDIT.aligned_pair_arrays(patch_rows(), "end", "probability")
    reversed_rows = AUDIT.aligned_pair_arrays(patch_rows()[::-1], "end", "probability")
    for template in original:
        np.testing.assert_array_equal(original[template], reversed_rows[template])


def test_pair_alignment_rejects_missing_or_duplicate_direction() -> None:
    rows = patch_rows()
    with pytest.raises(ValueError, match="different example"):
        AUDIT.aligned_pair_arrays(rows[:-1], "end", "probability")
    with pytest.raises(ValueError, match="Duplicate"):
        AUDIT.aligned_pair_arrays(rows + [rows[0]], "end", "probability")


def test_pair_alignment_rejects_incomplete_pair() -> None:
    rows = [row for row in patch_rows() if not row["example_id"].endswith("a-0-baba")]
    with pytest.raises(ValueError, match="Incomplete"):
        AUDIT.aligned_pair_arrays(rows, "end", "probability")


def test_bootstrap_is_seeded_and_preserves_direction_pairing() -> None:
    kwargs = {"site": "end", "metric": "probability", "samples": 30, "seed": 123}
    first = AUDIT.bootstrap_direction(patch_rows(), **kwargs)
    second = AUDIT.bootstrap_direction(patch_rows()[::-1], **kwargs)
    assert first == second
    assert first["templates"] == 2
    assert first["paired_entity_triples"] == 6
    interval = first["statistics"]["raw_forward_minus_reverse"]
    assert interval["ci_low"] == pytest.approx(-0.06)
    assert interval["ci_high"] == pytest.approx(-0.06)


def test_literal_verifier_read_does_not_execute_code() -> None:
    source = b"EXPECTED = {'path': 'hash'}\nraise RuntimeError('never run')\nDYNAMIC = call()"
    assert AUDIT.literal_constants(source) == {"EXPECTED": {"path": "hash"}}


def test_real_manifest_and_recorded_nulls_match_audit() -> None:
    inputs = AUDIT.Inputs(PROJECT_ROOT)
    batches = AUDIT.batch_audit(inputs)
    for split in ("development", "test"):
        assert batches[split]["rows"] == 160
        assert batches[split]["changed_donors"] == 80
        assert batches[split]["changed_rng_seed_offsets"] == 128
    groups = AUDIT.group_audit(inputs)
    for experiment in ("002", "004"):
        for site in ("s2", "end"):
            assert all(groups[experiment][site]["cyclic_rule_matches_saved_null"].values())
            assert not groups[experiment][site]["identical_selected_group"]
