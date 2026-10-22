"""Verify frozen sources and deterministic Experiment 002 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FREEZE = PROJECT_ROOT / "data" / "experiment-002-implementation-freeze.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment-002" / "verification.json"

EXPECTED_RESULTS = {
    "development": {
        "behavior-per-example.jsonl": "15d3fe1dbe7ec8ea14950fe1f12a17ca03349b0e262cef1269d9fd868684700b",
        "patches-per-example.jsonl": "3c56f6a8e22499ed3df8378feda50b7de0c0b6a1258c96e4eb2dd42100d06cac",
        "behavior-summary.json": "87ae8e6dd0138e85a528631286fd0d0e984e6723cd3326f87050cf140bd8d55c",
        "transport-summary.json": "b02d4c4fc017199ec330edccbfcb07665418f555c0c6493513fbcef39cc88130",
    },
    "test": {
        "behavior-per-example.jsonl": "b6602db51072d9ccda3a0853ea580dac3fd968d6f4c4a37a6e9fd47dea7543ea",
        "patches-per-example.jsonl": "29f79c7cb3b7ca3414f5477b5a4a7879532db7d7d01ffdc2ea153169712addc7",
        "behavior-summary.json": "a2722248c4be57bbb88f69339cfd50c2cc8bfa614c238648dba2a0126d1ec349",
        "transport-summary.json": "2e377493ef509e03baaed0a35ac7086abea6b13470c3ca5a9aea7430720c9ec1",
    },
}
EXPECTED_ROWS = {
    "development": {"behavior-per-example.jsonl": 320, "patches-per-example.jsonl": 2_560},
    "test": {"behavior-per-example.jsonl": 320, "patches-per-example.jsonl": 2_560},
}
EXPECTED_LOCALIZATION_ROWS = {
    "head-effects-per-example.jsonl": 320,
    "head-effects-summary.jsonl": 1_536,
    "behavior-per-example.jsonl": 160,
}
EXPECTED_ANALYSIS_SHA256 = "674993ec54b953add130c4b9a959b6a8327c8ee4f85f88335ae5cccc9f7e682e"


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


def validate_jsonl(
    path: Path,
    expected_rows: int,
    *,
    expected_effect_length: int | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {expected_rows}")
    for index, row in enumerate(rows):
        if not all(math.isfinite(value) for value in iter_floats(row)):
            raise RuntimeError(f"{path} contains a non-finite value at row {index}")
        if (
            expected_effect_length is not None
            and len(row.get("effects", [])) != expected_effect_length
        ):
            raise RuntimeError(
                f"{path} row {index} has {len(row.get('effects', []))} effects, "
                f"expected {expected_effect_length}"
            )
    example_ids = {row["example_id"] for row in rows if "example_id" in row}
    details: dict[str, Any] = {"rows": len(rows)}
    if example_ids:
        details["unique_examples"] = len(example_ids)
    return details


def validate_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for condition in ("canonical", "controlled"):
        counts[condition] = {}
        for site in ("s2", "end"):
            group = selection["conditions"][condition][site]
            selected = group["selected_heads"]
            random_heads = group["random_heads"]
            if len(selected) != 8 or len(random_heads) != 8:
                raise RuntimeError(f"{condition}/{site} does not contain two eight-head groups")
            if len({tuple(head) for head in selected}) != 8:
                raise RuntimeError(f"{condition}/{site} selected heads are not unique")
            counts[condition][site] = {
                "selected_heads": len(selected),
                "random_heads": len(random_heads),
            }
    return counts


def main() -> None:
    freeze = json.loads(IMPLEMENTATION_FREEZE.read_text(encoding="utf-8"))

    source_checks = {}
    for relative_path, expected in freeze["source_files_sha256"].items():
        actual = sha256(PROJECT_ROOT / relative_path)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {relative_path}: {actual} != {expected}")
        source_checks[relative_path] = actual

    localization_checks: dict[str, Any] = {}
    for relative_path, expected in freeze["localization_results_sha256"].items():
        path = PROJECT_ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Localization result changed: {relative_path}: {actual} != {expected}"
            )
        details: dict[str, Any] = {"sha256": actual}
        filename = path.name
        if filename in EXPECTED_LOCALIZATION_ROWS:
            details.update(
                validate_jsonl(
                    path,
                    EXPECTED_LOCALIZATION_ROWS[filename],
                    expected_effect_length=384
                    if filename == "head-effects-per-example.jsonl"
                    else None,
                )
            )
        localization_checks[relative_path] = details

    selection_path = (
        PROJECT_ROOT / "results" / "experiment-002" / "localization" / "selected-heads.json"
    )
    selection_counts = validate_selection(selection_path)

    result_checks: dict[str, Any] = {}
    for run, files in EXPECTED_RESULTS.items():
        result_checks[run] = {}
        for filename, expected in files.items():
            path = PROJECT_ROOT / "results" / "experiment-002" / run / filename
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Result changed: {path}: {actual} != {expected}")
            details = {"sha256": actual}
            if filename in EXPECTED_ROWS[run]:
                details.update(validate_jsonl(path, EXPECTED_ROWS[run][filename]))
            result_checks[run][filename] = details

    analysis_path = PROJECT_ROOT / "results" / "experiment-002" / "analysis.json"
    analysis_hash = sha256(analysis_path)
    if analysis_hash != EXPECTED_ANALYSIS_SHA256:
        raise RuntimeError(f"Analysis changed: {analysis_hash} != {EXPECTED_ANALYSIS_SHA256}")

    figure_checks = {}
    for stem in (
        "medium-head-localization",
        "medium-heldout-transport",
        "medium-transport-controls",
    ):
        result_figure = PROJECT_ROOT / "results" / "experiment-002" / "figures" / f"{stem}.pdf"
        paper_figure = PROJECT_ROOT / "paper" / "figures" / f"{stem}.pdf"
        result_hash = sha256(result_figure)
        paper_hash = sha256(paper_figure)
        if result_hash != paper_hash:
            raise RuntimeError(f"Paper figure is stale: {paper_figure}")
        figure_checks[stem] = result_hash

    output = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "frozen_sources": source_checks,
        "localization": localization_checks,
        "selection_counts": selection_counts,
        "deterministic_results": result_checks,
        "analysis_json_sha256": analysis_hash,
        "paper_figure_sha256": figure_checks,
        "paper_pdf_sha256": sha256(PROJECT_ROOT / "paper" / "build" / "main.pdf"),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
