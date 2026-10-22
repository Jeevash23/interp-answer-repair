"""Verify completed Study 5 saved outputs without loading a model or changing old studies.

Hashes establish correspondence to the currently retained sources, not external
preregistration or proof that a logged inference computation occurred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_experiment_005 import (
    CONDITIONS,
    DRAWS,
    MODELS,
    SITES,
    expected_specs,
    validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260905
PROBABILITY_TOLERANCE = 0.001
LOGIT_TOLERANCE = 0.01
NORM_TOLERANCE = 1e-5
NOOP_TOLERANCE = 1e-5
NOOP_LOGIT_TOLERANCE = 1e-4


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def expected_donors(records: list[dict[str, Any]]) -> dict[str, str]:
    """Reconstruct the declared seeded matching using only frozen entity identities."""
    groups: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in records:
        pair = row["example_id"].rsplit("-", 1)[0]
        if row["pattern"] in groups[row["template_id"]][pair]:
            raise ValueError("Duplicate manifest pair/pattern")
        groups[row["template_id"]][pair][row["pattern"]] = row
    output = {}
    for template, pairs in sorted(groups.items()):
        names = {}
        for pair, patterns in pairs.items():
            if set(patterns) != {"ABBA", "BABA"}:
                raise ValueError("Incomplete manifest entity pair")
            triples = [
                {row[key] for key in ("io_name", "subject_name", "corrupt_name")}
                for row in patterns.values()
            ]
            if triples[0] != triples[1] or len(triples[0]) != 3:
                raise ValueError("Manifest pair has inconsistent or repeated entities")
            names[pair] = triples[0]
        rng = random.Random(stable_seed(SEED, "donor", template))
        candidates = {}
        for recipient in sorted(pairs):
            candidates[recipient] = [
                donor for donor in sorted(pairs) if names[recipient].isdisjoint(names[donor])
            ]
            rng.shuffle(candidates[recipient])
        owners: dict[str, str] = {}

        def assign(
            recipient: str, visited: set[str], *, candidates=candidates, owners=owners
        ) -> bool:
            for donor in candidates[recipient]:
                if donor in visited:
                    continue
                visited.add(donor)
                if donor not in owners or assign(owners[donor], visited):
                    owners[donor] = recipient
                    return True
            return False

        for recipient in sorted(pairs):
            if not assign(recipient, set()):
                raise ValueError(f"No complete disjoint donor assignment in {template}")
        for donor, recipient in owners.items():
            for pattern in ("ABBA", "BABA"):
                output[pairs[recipient][pattern]["example_id"]] = pairs[donor][pattern][
                    "example_id"
                ]
    return output


def sampled_heads(reference, excluded, seed):
    rng = random.Random(seed)
    output = []
    for layer, count in sorted(Counter(head[0] for head in reference).items()):
        eligible = [head for head in range(16) if [layer, head] not in excluded]
        output.extend([layer, head] for head in sorted(rng.sample(eligible, count)))
    return output


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def verify_run(
    directory: Path,
    *,
    model: str,
    indexed=None,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    directory, project_root = Path(directory), Path(project_root).resolve()
    config_path = directory / "run-config.json"
    data_path = directory / "patches-per-example.jsonl"
    config = json.loads(config_path.read_text())
    required_values = {
        "status": "complete",
        "smoke": False,
        "study": "005",
        "model": model,
        "dtype": "float32",
        "seed": SEED,
        "rows": 8960,
    }
    for key, value in required_values.items():
        if key not in config or config[key] != value:
            raise ValueError(f"Run configuration {key!r} must be {value!r}")
    if not config.get("started_at_utc") or not config.get("completed_at_utc"):
        raise ValueError("Run has no completed timestamp record")
    historical_tolerances = {
        "probability": PROBABILITY_TOLERANCE,
        "logit": LOGIT_TOLERANCE,
    }
    if config.get("historical_endpoint_tolerances") != historical_tolerances:
        raise ValueError(
            "Historical endpoint tolerances differ from the documented numerical amendment"
        )
    output_hash = sha(data_path)
    if output_hash != config.get("output_sha256"):
        raise ValueError("Output hash does not match completed run configuration")
    study, selection_study = (3, 2) if model == "gpt2-medium" else (4, 4)
    manifest_name = f"data/experiment-{study:03d}-manifest.jsonl"
    selection_name = f"results/experiment-{selection_study:03d}/localization/selected-heads.json"
    baseline_name = f"results/experiment-{study:03d}/test/behavior-per-example.jsonl"
    patches_name = f"results/experiment-{study:03d}/test/patches-per-example.jsonl"
    required_sources = {
        manifest_name,
        selection_name,
        baseline_name,
        patches_name,
        "docs/EXPERIMENT-005.md",
        "scripts/run_experiment_005.py",
        "src/circuit_transport/experiment_005.py",
        "src/circuit_transport/experiment.py",
        "src/circuit_transport/pythia.py",
        f"models/{model}/config.json",
        f"models/{model}/model.safetensors",
        "data/model-manifest.json" if study == 3 else "data/pythia-model-manifest.json",
    }
    source_hashes = config.get("input_sha256", {})
    if not required_sources <= source_hashes.keys():
        raise ValueError(
            f"Missing frozen input hashes: {sorted(required_sources - source_hashes.keys())}"
        )
    for relative, expected in source_hashes.items():
        path = (project_root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(project_root):
            raise ValueError("Input hash path must remain inside the research project")
        if sha(path) != expected:
            raise ValueError(f"Current source hash differs from retained run input: {relative}")
    if indexed is None:
        indexed = validate_rows(read_jsonl(data_path), model=model)
    first = next(iter(indexed.values()))
    records = [row for row in read_jsonl(project_root / manifest_name) if row["split"] == "test"]
    manifest = {row["example_id"]: row for row in records}
    if len(records) != 160 or len(manifest) != 160 or set(manifest) != set(first):
        raise ValueError("Saved rows do not match all 160 immutable test manifest IDs")
    for example_id, row in first.items():
        if any(row[key] != manifest[example_id][key] for key in ("template_id", "pattern")):
            raise ValueError(f"Metadata differs from immutable manifest: {example_id}")
    planned = expected_specs()
    specs = config.get("specs", [])
    if len(specs) != len(planned) or {spec.get("spec") for spec in specs} != set(planned):
        raise ValueError("Run configuration does not declare exactly the planned 56 specs")
    for spec in specs:
        if any(spec.get(key) != value for key, value in planned[spec["spec"]].items()):
            raise ValueError(f"Run configuration spec differs from protocol: {spec['spec']}")

    donors = config.get("donors", {})
    if set(donors) != set(manifest) or set(donors.values()) != set(manifest):
        raise ValueError("Donor map must be a one-to-one permutation of all manifest IDs")
    for recipient, donor in donors.items():
        a, b = manifest[recipient], manifest[donor]
        if a["pattern"] != b["pattern"] or a["template_id"] != b["template_id"]:
            raise ValueError("Mismatch donor changes pattern or template")
        if {a[key] for key in ("io_name", "subject_name", "corrupt_name")} & {
            b[key] for key in ("io_name", "subject_name", "corrupt_name")
        }:
            raise ValueError("Mismatch donor and recipient share an entity")
        partner = recipient.rsplit("-", 1)[0] + ("-baba" if a["pattern"] == "ABBA" else "-abba")
        if donors[partner].rsplit("-", 1)[0] != donor.rsplit("-", 1)[0]:
            raise ValueError("Paired patterns have different donor entity pairs")
    reconstructed = expected_donors(records)
    if donors != reconstructed or donors != expected_donors(records[::-1]):
        raise ValueError("Saved donor mapping differs from immutable seeded full-template matching")

    selection = json.loads((project_root / selection_name).read_text())
    groups = {
        f"{condition}_{site}": selection["conditions"][condition][site]["selected_heads"]
        for condition in CONDITIONS
        for site in SITES
    }
    if config.get("groups") != groups:
        raise ValueError("Selected head groups differ from the original frozen selections")
    unions = {
        site: [
            list(head)
            for head in sorted(
                {tuple(head) for condition in CONDITIONS for head in groups[f"{condition}_{site}"]}
            )
        ]
        for site in SITES
    }
    if config.get("unions") != unions:
        raise ValueError("Matched-direction groups must be the same complete selected-head union")
    random_groups = {
        f"{site}_{draw}": sampled_heads(
            groups[f"canonical_{site}"],
            unions[site],
            stable_seed(SEED, model, site, "heads", draw),
        )
        for site in SITES
        for draw in DRAWS
    }
    block_groups = {
        str(draw): sampled_heads(
            groups["controlled_end"], unions["end"], stable_seed(SEED, model, "block", draw)
        )
        for draw in DRAWS
    }
    if config.get("random_groups") != random_groups or config.get("block_groups") != block_groups:
        raise ValueError("Control head groups do not reproduce the seeded layer-matched samples")

    norm_error = 0.0
    audit = config.get("matched_norm_audit", {})
    if set(audit) != set(SITES):
        raise ValueError("Missing per-example union norm audit for both intervention sites")
    union_norms = {}
    zero_union_norm_rows = 0
    for site in SITES:
        site_rows = audit[site]
        site_audit = {row["example_id"]: row for row in site_rows}
        if len(site_rows) != 160 or set(site_audit) != set(first):
            raise ValueError("Union norm audit must cover every immutable example exactly once")
        union_norms[site] = site_audit
        for example_id, row in site_audit.items():
            a = _finite_nonnegative(row.get("canonical_union_norm"), "canonical union norm")
            b = _finite_nonnegative(row.get("controlled_union_norm"), "controlled union norm")
            shared = _finite_nonnegative(row.get("shared_norm"), "shared norm")
            expected = math.sqrt(a) * math.sqrt(b)
            if abs(shared - expected) / max(expected, 1e-12) > NORM_TOLERANCE:
                raise ValueError(f"Shared norm is not the geometric mean: {site} {example_id}")
            zero_union_norm_rows += int(shared == 0)

    def check_norm(actual, expected, label):
        nonlocal norm_error
        actual = _finite_nonnegative(actual, label)
        error = abs(actual - expected) / max(expected, 1e-12)
        norm_error = max(norm_error, error)
        if error > NORM_TOLERANCE:
            raise ValueError(f"Injection norm violates direction/dose/control matching: {label}")

    for site in SITES:
        forward = indexed[f"native_{site}_canonical_to_controlled"]
        for example_id in first:
            for source in CONDITIONS:
                a = indexed[f"native_{site}_{source}_to_canonical"][example_id]["injected_norm"]
                b = indexed[f"native_{site}_{source}_to_controlled"][example_id]["injected_norm"]
                check_norm(a, b, f"native {site} {source} {example_id}")
            shared = indexed[f"matched_{site}_canonical_to_controlled_alpha_1"][example_id][
                "injected_norm"
            ]
            check_norm(
                shared, union_norms[site][example_id]["shared_norm"], f"shared {site} {example_id}"
            )
            for source in CONDITIONS:
                target = "controlled" if source == "canonical" else "canonical"
                for dose in (0.5, 1, 2):
                    actual = indexed[f"matched_{site}_{source}_to_{target}_alpha_{dose}"][
                        example_id
                    ]["injected_norm"]
                    check_norm(
                        actual, shared * dose, f"matched {site} {source} {dose} {example_id}"
                    )
            native_norm = forward[example_id]["injected_norm"]
            for spec in [
                f"control_{site}_off_target",
                *[f"control_{site}_random_direction_{draw}" for draw in DRAWS],
            ]:
                check_norm(indexed[spec][example_id]["injected_norm"], native_norm, spec)
            check_norm(
                indexed[f"control_{site}_mismatch"][example_id]["injected_norm"],
                forward[donors[example_id]]["injected_norm"],
                f"mismatch {site} {example_id}",
            )
    for spec, rows in indexed.items():
        if not spec.startswith("block_"):
            continue
        for example_id, row in rows.items():
            expected = (
                indexed["native_s2_canonical_to_controlled"][example_id]["injected_norm"]
                if row["alpha"]
                else 0.0
            )
            check_norm(row["injected_norm"], expected, spec)

    historical_baselines = {
        (row["example_id"], row["condition"]): row
        for row in read_jsonl(project_root / baseline_name)
    }
    historical_patches = {
        (row["example_id"], row["site"], row["source_condition"], row["target_condition"]): row
        for row in read_jsonl(project_root / patches_name)
        if row["control"] == "correct"
    }
    errors = {
        key: 0.0
        for key in (
            "baseline_probability",
            "baseline_logit",
            "native_probability",
            "native_logit",
            "baseline_clamp_probability",
            "baseline_clamp_logit",
        )
    }
    for spec, rows in indexed.items():
        for example_id, row in rows.items():
            for metric, suffix in (("probability", ""), ("logit", "_logit")):
                old = historical_baselines[(example_id, row["target"])]
                for prefix, variant in (("clean", "clean"), ("baseline", "corrupt")):
                    key = f"baseline_{metric}"
                    errors[key] = max(
                        errors[key],
                        abs(row[prefix + suffix] - old[f"{variant}_{metric}_difference"]),
                    )
                if spec.startswith("native_"):
                    old_patch = historical_patches[
                        (example_id, row["site"], row["source"], row["target"])
                    ]
                    key = f"native_{metric}"
                    errors[key] = max(
                        errors[key],
                        abs(
                            row["patched" + suffix]
                            - old_patch[f"target_patched_{metric}_difference"]
                        ),
                    )
                if spec.startswith("block_") and row["alpha"] == 0:
                    key = f"baseline_clamp_{metric}"
                    errors[key] = max(
                        errors[key], abs(row["patched" + suffix] - row["baseline" + suffix])
                    )
    for key, error in errors.items():
        if key.startswith("baseline_clamp"):
            tolerance = NOOP_LOGIT_TOLERANCE if key.endswith("logit") else NOOP_TOLERANCE
        else:
            tolerance = LOGIT_TOLERANCE if key.endswith("logit") else PROBABILITY_TOLERANCE
        if error > tolerance:
            raise ValueError(
                f"Saved endpoint reproduction/no-op error exceeds tolerance: {key}={error}"
            )
    checks = config.get("checks", {})
    if checks.get("complete_row_coverage") is not True:
        raise ValueError("Runtime did not certify complete example/spec coverage")
    if checks.get("zero_shared_norm_rows") != zero_union_norm_rows:
        raise ValueError("Runtime zero-norm count differs from the per-example norm audit")
    for key, tolerance in (
        ("native_probability_max_error", PROBABILITY_TOLERANCE),
        ("native_logit_max_error", LOGIT_TOLERANCE),
        ("baseline_clamp_max_error", NOOP_TOLERANCE),
        ("baseline_clamp_logit_max_error", NOOP_LOGIT_TOLERANCE),
        ("zero_dose_max_error", NOOP_TOLERANCE),
        ("zero_dose_logit_max_error", NOOP_LOGIT_TOLERANCE),
        ("matched_norm_relative_max_error", NORM_TOLERANCE),
    ):
        if _finite_nonnegative(checks.get(key), f"checks.{key}") > tolerance:
            raise ValueError(f"Runtime validation check exceeds tolerance: {key}")
    for key in ("probability", "logit"):
        value = checks.get("baseline_error", {}).get(key)
        tolerance = PROBABILITY_TOLERANCE if key == "probability" else LOGIT_TOLERANCE
        if _finite_nonnegative(value, f"checks.baseline_error.{key}") > tolerance:
            raise ValueError("Runtime baseline reproduction check exceeds tolerance")
        if not math.isclose(value, errors[f"baseline_{key}"], rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError(
                "Runtime baseline error does not match the retained row-level evidence"
            )
    for recorded, recomputed in (
        ("native_probability_max_error", "native_probability"),
        ("native_logit_max_error", "native_logit"),
        ("baseline_clamp_max_error", "baseline_clamp_probability"),
        ("baseline_clamp_logit_max_error", "baseline_clamp_logit"),
    ):
        if not math.isclose(checks[recorded], errors[recomputed], rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError(
                f"Runtime check does not match retained row-level evidence: {recorded}"
            )
    return {
        "status": "passed",
        "model": model,
        "n_examples": 160,
        "n_templates": 8,
        "n_specs": 56,
        "n_rows": 8960,
        "input_hashes_checked": len(source_hashes),
        "output_sha256": output_hash,
        "run_config_sha256": sha(config_path),
        "verifier_sha256": sha(Path(__file__)),
        "donor_ids_checked": len(donors),
        "matched_and_control_norm_relative_max_error": norm_error,
        "geometric_mean_union_norm_rows_checked": 320,
        "zero_shared_norm_rows": zero_union_norm_rows,
        "recomputed_endpoint_errors": errors,
        "historical_endpoint_tolerances": historical_tolerances,
        "historical_agreement": (
            "Bounded CPU-to-historical-GPU endpoint agreement, not byte-identical reproduction. "
            "The separate hook implementation-equivalence diagnostic uses stricter checks."
        ),
        "runtime_checks": checks,
        "limits": (
            "Saved rows verify common-direction and proportional-dose norms, geometric-mean "
            "targets against the logged union norms, and control norms. Full source activation "
            "vectors and random-vector coordinates are not retained, so their construction and "
            "per-head random direction generation additionally rely on the hashed implementation "
            "and its tests. No model inference was rerun by this verifier."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/experiment-005")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {model: verify_run(args.input_root / model, model=model) for model in args.models}
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
