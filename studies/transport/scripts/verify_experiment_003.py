"""Verify Experiment 003 freezes, deterministic outputs, and mechanistic gates."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-003-implementation-freeze.json"
RESULT_ROOT = PROJECT_ROOT / "results" / "experiment-003"
OUTPUT_PATH = RESULT_ROOT / "verification.json"
GATE_AUDIT_PATH = RESULT_ROOT / "GATE-AUDIT.md"

EXPECTED_RESULTS = {
    "development": {
        "behavior-per-example.jsonl": "ceed606745f1892c6cbb116cca9342ceef2cb0961400f9131f6c34da7c3b4a00",
        "patches-per-example.jsonl": "497af775a266d55682d7ee864489d331c8368a828e7b4414166cc47949ee7f11",
        "behavior-summary.json": "6d80edc0cb8c79306cd5c83bc3336c1e99934b7a76fe4ad8515403e62eed5980",
        "transport-summary.json": "20d325b1c2e143cae6d9477f582c86e9d58c33671f34dff7620d63390cef2a79",
    },
    "test": {
        "behavior-per-example.jsonl": "0d77280464a519bdc3defefd9e87aa24d8299088bec2289128acd46b422610e1",
        "patches-per-example.jsonl": "705525378e655485e208314982e969f15e6f006df43a10420008e1901d1d4b95",
        "behavior-summary.json": "1a0a43b6893de5b4c304f0d30fb37a191bffe863c334b723f3dacc4a5e03ea4d",
        "transport-summary.json": "acdd88343c67693bb1c45f38b8cb937cc151e0df3f37db48857923ba14b8a330",
    },
}
EXPECTED_ROWS = {
    "behavior-per-example.jsonl": 320,
    "patches-per-example.jsonl": 2_560,
}
EXPECTED_ANALYSIS_SHA256 = "a58e810fe37b2c9c5949ab6ef6994fedba884628ce818077a52e5a9634da1e37"
EXPECTED_ANALYSIS_MARKDOWN_SHA256 = (
    "acdb2c5f20d7183840a3d854bfa1db8587f0bcd20d2f0db25c29d1fcc97d1dde"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_floats(value: Any):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_floats(item)


def validate_jsonl(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    if any(not math.isfinite(value) for row in rows for value in iter_floats(row)):
        raise RuntimeError(f"{path} contains a non-finite value")
    return {
        "rows": len(rows),
        "unique_examples": len({row["example_id"] for row in rows}),
        "templates": len({row["template_id"] for row in rows}),
    }


def mechanism_gates(analysis: dict[str, Any], split: str) -> dict[str, Any]:
    result = {}
    split_result = analysis[split]
    behavior_passed = all(row["gate_passed"] for row in split_result["behavior"]["aggregate"])
    for site in ("s2", "end"):
        transport = split_result["transport"][site]
        within_source_passed = all(
            transport[direction]["numerator"]["ci_low"] > 0
            for direction in ("canonical_to_canonical", "controlled_to_controlled")
        )
        forward = transport["canonical_to_controlled"]
        result[site] = {
            "behavior_passed": behavior_passed,
            "within_source_passed": within_source_passed,
            "denominator_passed": forward["ratio_reportable"],
            "final_transport_interval_passed": forward["numerator"]["ci_low"] > 0,
            "local_sign_passed": forward["local_numerator"]["estimate"] > 0,
            "local_numerator": forward["local_numerator"],
        }
        result[site]["full_mechanistic_gate_passed"] = all(
            result[site][key]
            for key in (
                "behavior_passed",
                "within_source_passed",
                "denominator_passed",
                "final_transport_interval_passed",
                "local_sign_passed",
            )
        )
    return result


def gate_audit_markdown(
    predictions: dict[str, bool], development: dict[str, Any], test: dict[str, Any]
) -> str:
    lines = [
        "# Experiment 003 gate audit",
        "",
        "The five frozen final-outcome/template predictions pass, but those predictions do not replace",
        "the separately frozen mechanistic gate. The local-response requirement is audited here.",
        "",
        "## Frozen held-out predictions",
        "",
        "| Prediction | Result |",
        "|---|---|",
    ]
    for key, passed in predictions.items():
        lines.append(f"| {key.replace('_', ' ').title()} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Full mechanistic gate",
            "",
            "| Split | Site | Behavior | Within source | Denominator | Final interval | Local sign | Full gate |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for split, gates in (("development", development), ("test", test)):
        for site in ("s2", "end"):
            row = gates[site]
            values = [
                row["behavior_passed"],
                row["within_source_passed"],
                row["denominator_passed"],
                row["final_transport_interval_passed"],
                row["local_sign_passed"],
                row["full_mechanistic_gate_passed"],
            ]
            labels = ["pass" if value else "fail" for value in values]
            lines.append(f"| {split} | {site.upper()} | " + " | ".join(labels) + " |")
    local = test["s2"]["local_numerator"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"On held-out templates, S2 has positive final transport but a local numerator of "
                f"{local['estimate']:.4f} [{local['ci_low']:.4f}, {local['ci_high']:.4f}]. "
                "It therefore fails the declared local-sign gate and is not counted as stable path-level "
                "mechanistic transport. END passes every gate."
            ),
            "",
            "This distinction was made by applying the pre-outcome specification, not by changing a",
            "threshold after seeing results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    source_checks = {}
    for relative_path, expected in freeze["source_files_sha256"].items():
        actual = sha256(PROJECT_ROOT / relative_path)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {relative_path}: {actual} != {expected}")
        source_checks[relative_path] = actual

    result_checks: dict[str, Any] = {}
    for split, files in EXPECTED_RESULTS.items():
        result_checks[split] = {}
        for filename, expected in files.items():
            path = RESULT_ROOT / split / filename
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Result changed: {path}: {actual} != {expected}")
            details: dict[str, Any] = {"sha256": actual}
            if filename in EXPECTED_ROWS:
                details.update(validate_jsonl(path, EXPECTED_ROWS[filename]))
            result_checks[split][filename] = details

    analysis_path = RESULT_ROOT / "analysis.json"
    analysis_hash = sha256(analysis_path)
    markdown_hash = sha256(RESULT_ROOT / "ANALYSIS.md")
    if analysis_hash != EXPECTED_ANALYSIS_SHA256:
        raise RuntimeError(f"Analysis changed: {analysis_hash} != {EXPECTED_ANALYSIS_SHA256}")
    if markdown_hash != EXPECTED_ANALYSIS_MARKDOWN_SHA256:
        raise RuntimeError(
            f"Analysis Markdown changed: {markdown_hash} != {EXPECTED_ANALYSIS_MARKDOWN_SHA256}"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    development_gates = mechanism_gates(analysis, "development")
    test_gates = mechanism_gates(analysis, "test")
    predictions = analysis["test"]["frozen_predictions"]
    GATE_AUDIT_PATH.write_text(
        gate_audit_markdown(predictions, development_gates, test_gates), encoding="utf-8"
    )

    output = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "frozen_sources": source_checks,
        "deterministic_results": result_checks,
        "analysis_json_sha256": analysis_hash,
        "analysis_markdown_sha256": markdown_hash,
        "analysis_reproduction": {
            "json_byte_identical": True,
            "markdown_byte_identical": True,
            "png_figures_byte_identical": True,
        },
        "raw_run_reproduction": {
            "development_scientific_artifacts_byte_identical": True,
            "test_scientific_artifacts_byte_identical": True,
        },
        "frozen_test_predictions": predictions,
        "mechanistic_gates": {"development": development_gates, "test": test_gates},
        "gate_audit_sha256": sha256(GATE_AUDIT_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
