"""Recompute the saved audit and Study 5 statistics without model weights or inference.

This checks a retained numerical record. It does not independently reproduce model
execution, certify external preregistration, or replace the full Study 5 verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import analyze_experiment_005 as study5
import audit_research_20260905 as audit

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"Retained input or output has changed: {path.name}")


def reproduce(root: Path) -> dict[str, Any]:
    root = root.resolve()
    expected_audit = load(root / "results/audit-2026-09-05/audit.json")
    fresh_audit = audit.run_audit(root, expected_audit["bootstrap_samples"], expected_audit["seed"])
    # Git checkout state and the installed NumPy version describe the reader's
    # environment. Every scientific value and every recorded input hash must agree.
    audit_fields = (
        "integrity",
        "batch_amendment",
        "selected_groups",
        "heldout_directional_reanalysis",
        "input_sha256",
    )
    for key in audit_fields:
        if fresh_audit[key] != expected_audit[key]:
            raise ValueError(f"Original-study audit reproduction differs: {key}")
    if not fresh_audit["integrity"]["all_checked_hashes_match"]:
        raise ValueError("Original-study frozen-file checks failed")

    expected = load(root / "results/experiment-005/analysis.json")
    if set(expected["models"]) != set(study5.MODELS):
        raise ValueError("The retained analysis must contain both complete Study 5 models")
    check_hash(root / "scripts/analyze_experiment_005.py", expected["analysis_script_sha256"])
    summaries = {}
    for model in study5.MODELS:
        directory = root / "results/experiment-005" / model
        config = load(directory / "run-config.json")
        verification = expected["run_verification"][model]
        if config.get("status") != "complete" or config.get("smoke") is not False:
            raise ValueError(f"{model} is not a completed full run")
        check_hash(directory / "run-config.json", verification["run_config_sha256"])
        check_hash(root / "scripts/verify_experiment_005.py", verification["verifier_sha256"])
        data_path = directory / "patches-per-example.jsonl"
        check_hash(data_path, config["output_sha256"])
        check_hash(data_path, expected["input_files"][model]["sha256"])
        checked = 0
        deferred = {}
        for relative, expected_hash in config["input_sha256"].items():
            path = (root / relative).resolve()
            shared_models = Path(__file__).resolve().parents[3] / "models"
            in_shared_models = (Path(relative).parts[0] == "models" and path.is_relative_to(shared_models))
            if Path(relative).is_absolute() or not (path.is_relative_to(root) or in_shared_models):
                raise ValueError("Retained input paths must stay inside the project")
            if relative == f"models/{model}/model.safetensors":
                deferred[relative] = expected_hash
            else:
                check_hash(path, expected_hash)
                checked += 1
        if len(deferred) != 1:
            raise ValueError("The retained model-weight identity is missing")
        rows = [json.loads(line) for line in data_path.read_bytes().splitlines() if line]
        actual = study5.analyze_rows(rows, model=model)
        if actual != expected["models"][model]:
            raise ValueError(f"Recomputed Study 5 statistics differ: {model}")
        summaries[model] = {
            "all_scientific_values_identical": True,
            "n_rows": actual["n_rows"],
            "n_templates": actual["n_templates"],
            "non_weight_input_hashes_checked": checked,
            "model_weights_not_checked": deferred,
        }
    return {
        "status": "passed saved-statistics reproduction",
        "scope": (
            "All saved audit and Study 5 scientific values reproduced exactly with the "
            "retained analysis implementations. No model execution or weight verification. "
            "Use verify_experiment_005.py with restored weights for full saved-run validation."
        ),
        "original_study_hash_comparisons": len(fresh_audit["integrity"]["checks"]),
        "original_audit_scientific_values_identical": True,
        "study005": summaries,
        "environment_fields_excluded_from_comparison": [
            "audit.numpy_version",
            "audit.provenance.local_git_commit_count",
            "analysis.input_files.*.path",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("Choose a new output directory; existing records are never overwritten")
    result = reproduce(ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Passed: original audit and all 17,920 Study 5 rows reproduce the saved statistics.")
    print("No model execution or model-weight verification was performed.")
    print(f"Wrote {destination / 'verification.json'}")


if __name__ == "__main__":
    main()
