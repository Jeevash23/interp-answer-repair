"""Post-hoc CPU audit: no model loading, frozen edits, or original verifier main calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260905
DEFAULT_SAMPLES = 5000


class Inputs:
    """Read-only input registry, including every file used to calculate this audit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def read(self, relative: str) -> bytes:
        data = (self.root / relative).read_bytes()
        self.hashes[relative] = hashlib.sha256(data).hexdigest()
        return data

    def digest(self, relative: str) -> str:
        if relative not in self.hashes:
            self.read(relative)
        return self.hashes[relative]

    def json(self, relative: str) -> Any:
        return json.loads(self.read(relative))

    def jsonl(self, relative: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.read(relative).splitlines() if line]


def literal_constants(source: bytes) -> dict[str, Any]:
    """Extract literal assignments; do not import or execute verifier code."""
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


def integrity_audit(inputs: Inputs) -> dict[str, Any]:
    checks = []

    def check(relative: str, expected: str, kind: str) -> None:
        actual = inputs.digest(relative)
        checks.append(
            {
                "path": relative,
                "kind": kind,
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }
        )

    for experiment in ("001", "002", "003", "004"):
        base = f"data/experiment-{experiment}-implementation-freeze.json"
        freeze = inputs.json(base)
        expected_sources = dict(freeze["source_files_sha256"])
        if experiment == "004":
            previous = base
            for index in (1, 2):
                name = f"data/experiment-004-implementation-amendment-{index:03d}.json"
                amendment = inputs.json(name)
                field = "previous_freeze_sha256" if index == 1 else "previous_amendment_sha256"
                check(previous, amendment[field], "amendment_chain")
                expected_sources.update(amendment["amended_source_files_sha256"])
                script = (
                    "scripts/amend_experiment_004_implementation.py"
                    if index == 1
                    else "scripts/amend_experiment_004_transport.py"
                )
                check(script, amendment["amendment_script_sha256"], "amendment_script")
                previous = name
        for relative, expected in expected_sources.items():
            check(relative, expected, "effective_frozen_source")
        for field in ("localization_results_sha256", "feasibility_result_sha256"):
            for relative, expected in freeze.get(field, {}).items():
                check(relative, expected, field)
        constants = literal_constants(inputs.read(f"scripts/verify_experiment_{experiment}.py"))
        if experiment == "004":
            result_hashes = constants["EXPECTED_RESULT_HASHES"]
        else:
            result_hashes = {
                f"{run}/{name}": expected
                for run, files in constants["EXPECTED_RESULTS"].items()
                for name, expected in files.items()
            }
            for key, relative in (
                ("EXPECTED_ANALYSIS_SHA256", "analysis.json"),
                ("EXPECTED_ANALYSIS_MARKDOWN_SHA256", "ANALYSIS.md"),
            ):
                if key in constants:
                    result_hashes[relative] = constants[key]
        for relative, expected in result_hashes.items():
            check(f"results/experiment-{experiment}/{relative}", expected, "historical_result")

    reproductions = []
    for experiment in ("001", "002", "003", "004"):
        base = inputs.root / "results" / f"experiment-{experiment}"
        pairs = []
        for reproduction in sorted(base.glob("*-reproduction")):
            primary = base / reproduction.name.removesuffix("-reproduction")
            for path in sorted(reproduction.iterdir()):
                if (
                    path.is_file()
                    and path.name != "run-config.json"
                    and (primary / path.name).is_file()
                ):
                    left = str((primary / path.name).relative_to(inputs.root))
                    right = str(path.relative_to(inputs.root))
                    pairs.append(
                        {
                            "primary": left,
                            "reproduction": right,
                            "byte_identical": inputs.digest(left) == inputs.digest(right),
                        }
                    )
        reproductions.append(
            {
                "experiment": experiment,
                "retained_pairs": pairs,
                "status": (
                    "compared_retained_artifacts"
                    if pairs
                    else "no_separate_reproduction_directory_retained"
                ),
            }
        )
    return {
        "all_checked_hashes_match": all(row["matches"] for row in checks),
        "checks": checks,
        "reproductions": reproductions,
        "scope": "Local source/result hashes and retained scientific-file pairs only. "
        "Run-config timestamps differ by design and are excluded from byte equality. "
        "No fresh model rerun or model-weight hash check is performed.",
    }


def batch_assignments(records: list[dict[str, Any]], batch_size: int) -> dict[str, dict[str, Any]]:
    """Reconstruct Study 4's sorted-template batches and within-batch roll(+2)."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Batch size must preserve complete ABBA/BABA pairs")
    templates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        templates[record["template_id"]].append(record)
    assignments = {}
    offset = 0
    for template in sorted(templates):
        rows = templates[template]
        if len(rows) % batch_size:
            raise ValueError("Batch size must divide each template in this historical audit")
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for index, recipient in enumerate(batch):
                donor = batch[(index - 2) % len(batch)]
                fields = ("io_name", "subject_name", "corrupt_name")
                shared_names = sorted({recipient[k] for k in fields} & {donor[k] for k in fields})
                assignments[recipient["example_id"]] = {
                    "donor_id": donor["example_id"],
                    "same_pattern": donor["pattern"] == recipient["pattern"],
                    "shared_names": shared_names,
                    "rng_batch_offset": offset,
                    "rng_row_index": index,
                    "batch_rows": len(batch),
                }
            offset += len(batch)
    return assignments


def batch_audit(inputs: Inputs) -> dict[str, Any]:
    records = inputs.jsonl("data/experiment-004-manifest.jsonl")
    output = {}
    for split in ("development", "test"):
        rows = [r for r in records if r["split"] == split]
        old, new = batch_assignments(rows, 20), batch_assignments(rows, 4)
        changed = [key for key in old if old[key]["donor_id"] != new[key]["donor_id"]]
        output[split] = {
            "rows": len(rows),
            "changed_donors": len(changed),
            "fraction_changed": len(changed) / len(rows),
            "changed_rng_seed_offsets": sum(
                old[k]["rng_batch_offset"] != new[k]["rng_batch_offset"] for k in old
            ),
            "all_new_donors_preserve_pattern": all(row["same_pattern"] for row in new.values()),
            "new_donors_with_any_name_overlap": sum(
                bool(row["shared_names"]) for row in new.values()
            ),
            "assignments_batch20": old,
            "assignments_batch4": new,
        }
    output["interpretation"] = (
        "Changing 20 to 4 changes specific mismatch donors, and changes random-direction "
        "seed offsets for most records. The historical intervention code seeds per batch, "
        "not per example. Exact random vectors are not reconstructed by this CPU audit. "
        "The amendment remains documented as pre-transport-outcome; this is not evidence "
        "of outcome-based tuning, nor does it establish that the conclusions would change."
    )
    return output


def cyclic_heads(
    selected: list[list[int]], excluded: set[tuple[int, int]], n_heads: int
) -> list[list[int]]:
    output: list[list[int]] = []
    used = set(excluded)
    for layer, head in selected:
        for offset in range(1, n_heads + 1):
            candidate = (layer, (head + offset) % n_heads)
            if candidate not in used:
                used.add(candidate)
                output.append(list(candidate))
                break
        else:
            raise ValueError("No remaining layer-matched head")
    return output


def group_audit(inputs: Inputs) -> dict[str, Any]:
    output = {}
    for experiment in ("002", "004"):
        selection = inputs.json(f"results/experiment-{experiment}/localization/selected-heads.json")
        sites = {}
        for site in ("s2", "end"):
            groups = {
                condition: selection["conditions"][condition][site]
                for condition in ("canonical", "controlled")
            }
            sets = {
                condition: {tuple(head) for head in group["selected_heads"]}
                for condition, group in groups.items()
            }
            excluded = sets["canonical"] | sets["controlled"]
            sites[site] = {
                "canonical_heads": groups["canonical"]["selected_heads"],
                "controlled_heads": groups["controlled"]["selected_heads"],
                "intersection": len(sets["canonical"] & sets["controlled"]),
                "union": len(excluded),
                "identical_selected_group": sets["canonical"] == sets["controlled"],
                "cyclic_rule_matches_saved_null": {
                    condition: cyclic_heads(group["selected_heads"], excluded, 16)
                    == group["random_heads"]
                    for condition, group in groups.items()
                },
            }
        output[experiment] = sites
    output["interpretation"] = (
        "The legacy random_head label denotes one deterministic cyclic, layer-matched "
        "comparison group outside the selected union, not a sampled distribution of random "
        "head groups. Forward/reverse comparisons use different source-selected groups "
        "and source-delta magnitudes. Direction, group, signal, and target denominator are "
        "jointly changed; operational asymmetry alone does not isolate a directional mechanism."
    )
    return output


def aligned_pair_arrays(
    rows: list[dict[str, Any]], site: str, metric: str
) -> dict[str, np.ndarray]:
    directions = (
        ("canonical", "controlled"),
        ("controlled", "canonical"),
        ("canonical", "canonical"),
        ("controlled", "controlled"),
    )
    groups = []
    for source, target in directions:
        subset = [
            row
            for row in rows
            if row["site"] == site
            and row["control"] == "correct"
            and row["source_condition"] == source
            and row["target_condition"] == target
        ]
        mapping = {(r["template_id"], r["example_id"]): r for r in subset}
        if len(mapping) != len(subset):
            raise ValueError("Duplicate example in a direction")
        groups.append(mapping)
    if not groups[0] or any(set(g) != set(groups[0]) for g in groups):
        raise ValueError("Directions have different example identities")
    pairs: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    patterns: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key in sorted(groups[0]):
        template, example_id = key
        pair_id, pattern = example_id.rsplit("-", 1)
        if pattern not in {"abba", "baba"}:
            raise ValueError("Unexpected paired-example suffix")
        values = []
        for group in groups:
            row = group[key]
            if row["pattern"].lower() != pattern:
                raise ValueError("Pattern metadata does not match ID")
            clean = float(row[f"target_clean_{metric}_difference"])
            corrupt = float(row[f"target_corrupt_{metric}_difference"])
            patched = float(row[f"target_patched_{metric}_difference"])
            values.extend((patched - corrupt, clean - corrupt))
        pairs[template, pair_id].append(values)
        patterns[template, pair_id].add(pattern)
    templates: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, values in sorted(pairs.items()):
        if len(values) != 2 or patterns[key] != {"abba", "baba"}:
            raise ValueError("Incomplete ABBA/BABA pair")
        templates[key[0]].append(np.asarray(values, dtype=np.float64).mean(axis=0))
    output = {key: np.stack(values) for key, values in templates.items()}
    if any(not np.isfinite(values).all() for values in output.values()):
        raise ValueError("Nonfinite endpoint values")
    return output


def directional_statistics(values: np.ndarray) -> np.ndarray:
    forward, forward_den, reverse, reverse_den, within_c, _, within_x, _ = values.mean(axis=0)
    if min(abs(forward_den), abs(reverse_den)) < 1e-12:
        raise ValueError("Cannot evaluate near-zero target-denominator ratio")
    forward_cts, reverse_cts = forward / forward_den, reverse / reverse_den
    return np.asarray(
        (
            forward,
            reverse,
            forward - reverse,
            forward_den,
            reverse_den,
            forward_cts,
            reverse_cts,
            forward_cts - reverse_cts,
            within_c,
            within_x,
        )
    )


def bootstrap_direction(
    rows: list[dict[str, Any]], *, site: str, metric: str, samples: int, seed: int
) -> dict[str, Any]:
    by_template = aligned_pair_arrays(rows, site, metric)
    arrays = list(by_template.values())
    rng = np.random.default_rng(seed)
    point = directional_statistics(np.concatenate(arrays))
    draws = []
    for _ in range(samples):
        sampled = []
        for index in rng.integers(0, len(arrays), size=len(arrays)):
            array = arrays[index]
            sampled.append(array[rng.integers(0, len(array), size=len(array))])
        draws.append(directional_statistics(np.concatenate(sampled)))
    intervals = np.percentile(np.stack(draws), (2.5, 97.5), axis=0)
    names = (
        "forward_raw",
        "reverse_raw",
        "raw_forward_minus_reverse",
        "forward_denominator",
        "reverse_denominator",
        "forward_cts",
        "reverse_cts",
        "cts_forward_minus_reverse",
        "within_canonical_group_effect",
        "within_controlled_group_effect",
    )
    return {
        "metric": metric,
        "site": site,
        "templates": len(arrays),
        "paired_entity_triples": sum(len(array) for array in arrays),
        "statistics": {
            name: {
                "estimate": float(point[i]),
                "ci_low": float(intervals[0, i]),
                "ci_high": float(intervals[1, i]),
            }
            for i, name in enumerate(names)
        },
        "template_statistics": {
            name: dict(zip(names, directional_statistics(array).tolist(), strict=True))
            for name, array in by_template.items()
        },
        "inference": {
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "resampling": "templates, then ABBA/BABA entity pairs within sampled templates",
            "interval": "pointwise percentile 95%; exploratory, no multiple-testing correction",
            "scope": "Conditional on frozen selected heads, these datasets, models and dose.",
        },
    }


def run_audit(root: Path, samples: int, seed: int) -> dict[str, Any]:
    inputs = Inputs(root)
    for relative in (
        "scripts/audit_research_20260905.py",
        "src/circuit_transport/experiment_002.py",
        "src/circuit_transport/experiment_004.py",
        "scripts/run_experiment_004.py",
    ):
        inputs.read(relative)
    integrity = integrity_audit(inputs)
    batches, groups = batch_audit(inputs), group_audit(inputs)
    directions = {}
    for experiment in ("003", "004"):
        rows = inputs.jsonl(f"results/experiment-{experiment}/test/patches-per-example.jsonl")
        directions[experiment] = {
            site: {
                metric: bootstrap_direction(
                    rows, site=site, metric=metric, samples=samples, seed=seed
                )
                for metric in ("probability", "logit")
            }
            for site in ("s2", "end")
        }
    git = subprocess.run(
        ["git", "rev-list", "--all", "--count"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "audit": "2026-09-05 saved-output research audit",
        "status": "post-hoc exploratory",
        "execution": "CPU-only; no model loading, installation, frozen edits, or verifier main calls",
        "seed": seed,
        "bootstrap_samples": samples,
        "numpy_version": np.__version__,
        "integrity": integrity,
        "batch_amendment": batches,
        "selected_groups": groups,
        "heldout_directional_reanalysis": directions,
        "provenance": {
            "local_git_commit_count": int(git.stdout.strip()) if git.returncode == 0 else None,
            "local_freezes_are_external_preregistration": False,
            "limits": "Local hashes demonstrate consistency with stored records, not "
            "independent proof of chronology or absence of prior outcome access. "
            "Original predictions and their frozen outcomes are not reclassified.",
        },
        "input_sha256": dict(sorted(inputs.hashes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "results/audit-2026-09-05/audit.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    allowed = (ROOT / "results/audit-2026-09-05").resolve()
    if not args.output.resolve().is_relative_to(allowed):
        parser.error("Output must stay in the separate audit-2026-09-05 directory")
    output = run_audit(ROOT, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}; all frozen/hash checks match: "
        f"{output['integrity']['all_checked_hashes_match']}"
    )


if __name__ == "__main__":
    main()
