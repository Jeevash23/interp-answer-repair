"""Audit Experiment 004 freeze chains, reproductions, hashes, rows, and gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "experiment-004"
FREEZE_PATH = PROJECT_ROOT / "data" / "experiment-004-implementation-freeze.json"
AMENDMENT_001 = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-001.json"
AMENDMENT_002 = PROJECT_ROOT / "data" / "experiment-004-implementation-amendment-002.json"

EXPECTED_RESULT_HASHES = {
    "localization/behavior-per-example.jsonl": "5413dd34f827e386b68a15e05ef55267b8f4347cbd0770fd901180c872131943",
    "localization/head-effects-per-example.jsonl": "60ae4262355f99458a18d9769f09a6e4a11e7c3c97da84751fb50922295b8574",
    "localization/head-effects-summary.jsonl": "128fd74e078658ccbe803b8d7a6ee11af98a9650e1f8f61fe58a43f64e2549ff",
    "localization/selected-heads.json": "fcccb7fde7cc94cccfbcf59d404177933f6dcba77e473f4b5bc1eae685383544",
    "development/behavior-per-example.jsonl": "255a82827bff7153fd0873215a24ffb9561be54e38c14affc7c612931fcf5dc3",
    "development/patches-per-example.jsonl": "0fcb7074f920052bbe34310701ff698f13829391d68acc429eddcb4133279c70",
    "development/behavior-summary.json": "9207c2b0cf84735fbc74b4c6e183c7196ae33067c573c499b648278dd3ca8f23",
    "development/transport-summary.json": "72fbe74e73de6909b73c86320b76e22f20ba61f455f45c9a2b9e9400af5fd4b9",
    "test/behavior-per-example.jsonl": "e1ae1a6f878f847d49337f35e137b4cc620b325ef5121f73361c4077871d0639",
    "test/patches-per-example.jsonl": "130c8ba33221be102e786b2548612a49e2fc6722fa473e6c77524ee7be3e9bee",
    "test/behavior-summary.json": "19e9ba239f610cb2faf772aeab277fa3ca6428fc31d19a9ec74bcbf35bbea194",
    "test/transport-summary.json": "a0db405e6664fd46cbb8aa455c51d1d3647affe134b288d3ca9014c56cb55bee",
    "analysis.json": "cd1292923d512002f1fb08572bdf2137c8d790d8e955d74c31b16f67bf163f04",
    "ANALYSIS.md": "7f433cbde7c4dcdee0f4578cb5f05463a8760c1a946297ebf58485726c039ffe",
    "figures/pythia-end-localization.pdf": "3a2938be825f7a5e434afcdccf1fbc73cb2eb1c10ec1d3343443eaa502880126",
    "figures/pythia-end-localization.png": "8a4d402929eaa41727d0446e96310646a89837e446895573f49b7383049fa484",
    "figures/pythia-heldout-transport.pdf": "a2e77509a4b4c44fcd8645a437d0b815762700dae8f8525f68bd9e8460f7768f",
    "figures/pythia-heldout-transport.png": "42b8f56c30b7b5546db34ada975040ff2c79506b927403badef0434ead54d87f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def assert_reproduction(primary: str, reproduction: str, files: tuple[str, ...]) -> None:
    for name in files:
        if sha256(RESULT_ROOT / primary / name) != sha256(RESULT_ROOT / reproduction / name):
            raise RuntimeError(f"Reproduction mismatch: {primary}/{name}")


def gate_audit(analysis: dict[str, Any]) -> str:
    test = analysis["test"]
    end = test["transport"]["end"]["canonical_to_controlled"]
    s2 = test["transport"]["s2"]["canonical_to_controlled"]
    direction = test["directional"]
    stage = test["stage"]
    return "\n".join(
        [
            "# Experiment 004 gate audit",
            "",
            "**Verdict: full preregistered cross-architecture replication passed.**",
            "",
            (
                "All four primary held-out END requirements passed: positive endpoint transport, "
                "canonical-to-controlled directional advantage, superiority to every control after "
                "Holm correction, and a positive local direct-effect sign."
            ),
            "",
            "| Quantity | Held-out estimate [95% CI] | Gate |",
            "|---|---:|---|",
            (
                f"| S2 endpoint numerator | {s2['numerator']['estimate']:.4f} "
                f"[{s2['numerator']['ci_low']:.4f}, {s2['numerator']['ci_high']:.4f}] | pass |"
            ),
            (
                f"| S2 local numerator | {s2['local_numerator']['estimate']:.4f} "
                f"[{s2['local_numerator']['ci_low']:.4f}, {s2['local_numerator']['ci_high']:.4f}] | pass |"
            ),
            (
                f"| END endpoint numerator | {end['numerator']['estimate']:.4f} "
                f"[{end['numerator']['ci_low']:.4f}, {end['numerator']['ci_high']:.4f}] | pass |"
            ),
            (
                f"| END local numerator | {end['local_numerator']['estimate']:.4f} "
                f"[{end['local_numerator']['ci_low']:.4f}, {end['local_numerator']['ci_high']:.4f}] | pass |"
            ),
            (
                f"| S2 directional CTS difference | {direction['s2']['difference']:.3f} "
                f"[{direction['s2']['ci_low']:.3f}, {direction['s2']['ci_high']:.3f}] | pass |"
            ),
            (
                f"| END directional CTS difference | {direction['end']['difference']:.3f} "
                f"[{direction['end']['ci_low']:.3f}, {direction['end']['ci_high']:.3f}] | pass |"
            ),
            (
                f"| Mismatch-retention S2 minus END | "
                f"{stage['retention_difference']['estimate']:.3f} "
                f"[{stage['retention_difference']['ci_low']:.3f}, "
                f"{stage['retention_difference']['ci_high']:.3f}] | pass |"
            ),
            "",
            (
                "Every END control contrast had an exact template sign-flip p-value of 1/256 and "
                "Holm-adjusted p = 0.015625, with positive hierarchical-bootstrap confidence bounds."
            ),
            "",
            (
                "The S2 path gate also passed in Pythia. This is a cross-architecture extension beyond "
                "GPT-2 Medium, where S2 endpoint transport was positive but its local sign gate failed."
            ),
            "",
        ]
    )


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    amendment_001 = json.loads(AMENDMENT_001.read_text(encoding="utf-8"))
    amendment_002 = json.loads(AMENDMENT_002.read_text(encoding="utf-8"))
    if sha256(FREEZE_PATH) != amendment_001["previous_freeze_sha256"]:
        raise RuntimeError("Amendment 001 freeze chain failed")
    if sha256(AMENDMENT_001) != amendment_002["previous_amendment_sha256"]:
        raise RuntimeError("Amendment 002 freeze chain failed")
    expected_sources = dict(freeze["source_files_sha256"])
    expected_sources.update(amendment_001["amended_source_files_sha256"])
    expected_sources.update(amendment_002["amended_source_files_sha256"])
    for relative, expected in expected_sources.items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    if (
        sha256(PROJECT_ROOT / "scripts" / "amend_experiment_004_implementation.py")
        != amendment_001["amendment_script_sha256"]
    ):
        raise RuntimeError("Amendment 001 script hash mismatch")
    if (
        sha256(PROJECT_ROOT / "scripts" / "amend_experiment_004_transport.py")
        != amendment_002["amendment_script_sha256"]
    ):
        raise RuntimeError("Amendment 002 script hash mismatch")
    model_path = PROJECT_ROOT / "models" / "pythia-410m-deduped" / "model.safetensors"
    if sha256(model_path) != freeze["weight_sha256"]:
        raise RuntimeError("Pythia weight hash mismatch")

    assert_reproduction(
        "feasibility",
        "feasibility-reproduction",
        ("behavior-per-example.jsonl", "behavior-summary.json", "decision.json"),
    )
    assert_reproduction(
        "localization",
        "localization-reproduction",
        (
            "behavior-per-example.jsonl",
            "head-effects-per-example.jsonl",
            "head-effects-summary.jsonl",
            "selected-heads.json",
        ),
    )
    transport_files = (
        "behavior-per-example.jsonl",
        "patches-per-example.jsonl",
        "behavior-summary.csv",
        "behavior-summary.json",
        "transport-summary.csv",
        "transport-summary.json",
    )
    assert_reproduction("development", "development-reproduction", transport_files)
    assert_reproduction("test", "test-reproduction", transport_files)

    expected_rows = {
        "feasibility/behavior-per-example.jsonl": 640,
        "localization/behavior-per-example.jsonl": 320,
        "localization/head-effects-per-example.jsonl": 640,
        "localization/head-effects-summary.jsonl": 1536,
        "development/behavior-per-example.jsonl": 320,
        "development/patches-per-example.jsonl": 2560,
        "test/behavior-per-example.jsonl": 320,
        "test/patches-per-example.jsonl": 2560,
    }
    for relative, expected in expected_rows.items():
        if jsonl_rows(RESULT_ROOT / relative) != expected:
            raise RuntimeError(f"Unexpected row count: {relative}")
    for relative, expected in EXPECTED_RESULT_HASHES.items():
        if sha256(RESULT_ROOT / relative) != expected:
            raise RuntimeError(f"Result hash mismatch: {relative}")

    selection = json.loads((RESULT_ROOT / "localization" / "selected-heads.json").read_text())
    if selection["status"] != "complete" or selection["selection_count"] != 8:
        raise RuntimeError("Localization selection is incomplete")
    analysis = json.loads((RESULT_ROOT / "analysis.json").read_text(encoding="utf-8"))
    if not all(analysis["test"]["frozen_gates"].values()):
        raise RuntimeError("One or more held-out Experiment 004 gates failed")
    if analysis["protocol"] != {
        "bootstrap_samples": 5000,
        "bootstrap_seed": 20260904,
        "control_test": "exact template-level sign flip with Holm correction",
        "primary_replication_site": "end",
        "sampling_unit": "templates, then adjacent ABBA/BABA entity pairs",
    }:
        raise RuntimeError("Analysis protocol does not match the freeze")

    verification = {
        "status": "passed",
        "freeze_sha256": sha256(FREEZE_PATH),
        "amendment_001_sha256": sha256(AMENDMENT_001),
        "amendment_002_sha256": sha256(AMENDMENT_002),
        "model_weight_sha256": freeze["weight_sha256"],
        "scientific_reproductions_byte_identical": True,
        "result_hashes": EXPECTED_RESULT_HASHES,
        "row_counts": expected_rows,
        "heldout_gates": analysis["test"]["frozen_gates"],
        "localization_similarity": selection["similarity"],
    }
    (RESULT_ROOT / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULT_ROOT / "GATE-AUDIT.md").write_text(gate_audit(analysis), encoding="utf-8")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
