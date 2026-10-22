import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_experiment_003", PROJECT_ROOT / "scripts" / "analyze_experiment_003.py"
)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
ANALYSIS.BOOTSTRAP_SAMPLES = 100


def behavior_rows() -> list[dict]:
    rows = []
    for template in range(8):
        for pair in range(2):
            for pattern in ("ABBA", "BABA"):
                for condition in ("canonical", "controlled"):
                    rows.append(
                        {
                            "condition": condition,
                            "template_id": f"test-{template}",
                            "example_id": f"example-{template}-{pair}-{pattern.lower()}",
                            "pattern": pattern,
                            "clean_io_wins_pairwise": True,
                            "clean_probability_difference": 0.2,
                        }
                    )
    return rows


def patch_rows() -> list[dict]:
    rows = []
    correct_effects = {
        ("s2", "canonical", "canonical"): 0.08,
        ("s2", "controlled", "controlled"): 0.07,
        ("s2", "canonical", "controlled"): 0.04,
        ("s2", "controlled", "canonical"): 0.02,
        ("end", "canonical", "canonical"): 0.10,
        ("end", "controlled", "controlled"): 0.09,
        ("end", "canonical", "controlled"): 0.12,
        ("end", "controlled", "canonical"): 0.02,
    }
    controls = {
        "random_head": 0.0,
        "random_direction": -0.01,
        "off_target": 0.01,
    }
    for template in range(8):
        for pair in range(2):
            for pattern in ("ABBA", "BABA"):
                base = {
                    "template_id": f"test-{template}",
                    "example_id": f"example-{template}-{pair}-{pattern.lower()}",
                    "pattern": pattern,
                    "target_clean_probability_difference": 0.2,
                    "target_corrupt_probability_difference": 0.0,
                    "local_clean": 1.0,
                    "local_corrupt": 0.0,
                }
                for (site, source, target), effect in correct_effects.items():
                    rows.append(
                        {
                            **base,
                            "site": site,
                            "source_condition": source,
                            "target_condition": target,
                            "control": "correct",
                            "target_patched_probability_difference": effect,
                            "local_patched": effect,
                        }
                    )
                for site in ("s2", "end"):
                    for control, effect in controls.items():
                        rows.append(
                            {
                                **base,
                                "site": site,
                                "source_condition": "canonical",
                                "target_condition": "controlled",
                                "control": control,
                                "target_patched_probability_difference": effect,
                                "local_patched": effect,
                            }
                        )
                    mismatch = 0.035 if site == "s2" else 0.02
                    rows.append(
                        {
                            **base,
                            "site": site,
                            "source_condition": "canonical",
                            "target_condition": "controlled",
                            "control": "mismatched_entities",
                            "target_patched_probability_difference": mismatch,
                            "local_patched": mismatch,
                        }
                    )
    return rows


def test_analysis_pipeline_recovers_predeclared_positive_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        ANALYSIS,
        "load_jsonl",
        lambda path: behavior_rows() if "behavior" in path.name else patch_rows(),
    )
    result = ANALYSIS.split_analysis("test")
    assert all(result["frozen_predictions"].values())
    assert result["directional"]["s2"]["difference"] > 0
    assert result["directional"]["end"]["difference"] > 0
    assert result["stage"]["retention_difference"]["estimate"] > 0
    assert result["stage"]["contrast_difference"]["estimate"] > 0
    assert all(item["holm_pvalue"] < 0.05 for item in result["controls"]["end"].values())
