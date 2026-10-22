"""Deterministically replay completed Bridge007–009 saved-observation analyses.

This is separate from the independent study verifiers and performs no model
inference. All requested studies must have current canonical passing audits
before any analysis starts. Replays use fresh temporary child directories;
retained summaries, observations and audit timestamps are never overwritten.

    .venv/bin/python scripts/replay_extensions.py
    .venv/bin/python scripts/replay_extensions.py --only 007 009
    .venv/bin/python scripts/replay_extensions.py --self-test
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("007", "008", "009")
EARLY_STOPS = {"no_screen_pair", "no_selection_pair", "independent_qualification_failed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file(), f"Required completed record missing: {path}")
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON value {value} in {path}")
    return json.loads(path.read_text(), parse_constant=reject_constant)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_json(expected, actual, path="summary"):
    """Exact recursive comparison, omitting only dictionary fields named utc."""
    require(type(expected) is type(actual), f"JSON type differs at {path}")
    if isinstance(expected, dict):
        left, right = set(expected) - {"utc"}, set(actual) - {"utc"}
        require(left == right, f"JSON keys differ at {path}: {left ^ right}")
        for key in sorted(left):
            compare_json(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        require(len(expected) == len(actual), f"JSON list length differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_json(left, right, f"{path}/{i}")
    else:
        require(expected == actual, f"JSON value differs at {path}: {expected!r} != {actual!r}")


def compare_arrays(expected, actual, label="per-example"):
    """Exact keys, shapes, dtypes and numerical values, with aligned NaNs equal."""
    require(set(expected) == set(actual), f"Array keys differ in {label}: {set(expected) ^ set(actual)}")
    for key in sorted(expected):
        left, right = expected[key], actual[key]
        require(left.shape == right.shape, f"Array shape differs: {label}/{key}")
        require(left.dtype == right.dtype, f"Array dtype differs: {label}/{key}")
        require(left.dtype.kind in "biufc", f"Non-numeric array: {label}/{key} ({left.dtype})")
        if left.dtype.kind == "c":
            # Equal complex NaNs must not hide differences in the other part.
            equal = (np.array_equal(left.real, right.real, equal_nan=True)
                     and np.array_equal(left.imag, right.imag, equal_nan=True))
        else:
            equal = np.array_equal(left, right, equal_nan=True)
        require(equal, f"Array numerical values differ: {label}/{key}")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        require(len(values.files) == len(set(values.files)), "Duplicate keys in NPZ")
        return {key: values[key] for key in values.files}


def compare_outputs(canonical, replayed):
    compare_json(read(canonical / "summary.json"), read(replayed / "summary.json"))
    compare_arrays(load_arrays(canonical / "per-example.npz"),
                   load_arrays(replayed / "per-example.npz"), str(canonical))


def no_final_outputs(study):
    require(not list((study / "results/test").glob("*.npz")),
            f"Early-stop study has final observations: {study.name}")
    for folder in ("analysis", "secondary-analysis"):
        for filename in ("summary.json", "per-example.npz"):
            require(not (study / "results" / folder / filename).exists(),
                    f"Early-stop study has a final analysis: {study.name}/{folder}/{filename}")


def plan_study(number):
    """Read terminal metadata only; do not open scientific arrays or run analysis."""
    study = ROOT / f"studies/bridge-{number}"
    audit = read(study / "results/verification.json")
    require(audit.get("status") == "passed", f"Bridge {number} has no passing canonical verification")
    require(audit.get("helper_sha256") == digest(study / f"code/verify{number}.py"),
            f"Bridge {number} verifier changed after the canonical audit")
    if number in ("007", "009"):
        status = read(study / "results/status.json")
        require(status.get("status") in ("completed", "inference_completed"),
                f"Bridge {number} is not terminal")
        outcome = status.get("outcome")
        require(audit.get("outcome") == outcome and audit.get("stopping_branch_verified") is True,
                f"Bridge {number} verification does not match the terminal branch")
        require(datetime.fromisoformat(audit["utc"]) >= datetime.fromisoformat(status["utc"]),
                f"Bridge {number} verification predates its terminal record")
        if outcome in EARLY_STOPS:
            require(status.get("test_measured") is False, f"Bridge {number} early stop marked measured")
            no_final_outputs(study)
            return dict(number=number, outcome=outcome, skipped=True, jobs=[])
        require(outcome == "final_test_completed" and status.get("test_measured") is True,
                f"Bridge {number} has an unknown or incomplete terminal outcome")
    else:
        if audit.get("outcome") == "ineligible_base":
            status_path = study / "results/status.json"
            status = read(status_path)
            require(status.get("status") == "completed" and status.get("outcome") == "ineligible_base"
                    and status.get("test_measured") is False and audit.get("stopping_branch_verified") is True,
                    "Bridge 008 has an inconsistent base-eligibility stop")
            require(audit.get("status_sha256") == digest(status_path), "Bridge 008 stopping record changed")
            no_final_outputs(study)
            return dict(number=number, outcome="ineligible_base", skipped=True, jobs=[])
        outcome = "final_test_completed"
    completion_path = study / "results/inference-complete.json"
    require(read(completion_path).get("status") == "completed", f"Bridge {number} inference is incomplete")
    require(read(study / "results/computational-replay.json").get("passed") is True,
            f"Bridge {number} computational replay is incomplete")
    if number == "008":
        require(audit.get("inference_completion_sha256") == digest(completion_path),
                "Bridge 008 inference completion changed after verification")
    canonical = study / "results/analysis"
    require(audit.get("analysis_summary_sha256") == digest(canonical / "summary.json"),
            f"Bridge {number} primary summary changed after verification")
    specs = [("primary", f"analyze{number}.py", ["--mode", "final"] if number != "008" else [], "analysis")]
    if number in ("007", "009"):
        specs.append(("secondary", f"secondary{number}.py", ["--mode", "final"], "secondary-analysis"))
    jobs = []
    for label, script, args, folder in specs:
        target = study / "results" / folder
        source = study / "code" / script
        require(source.is_file(), f"Missing analyzer: {script}")
        freeze_name = "secondary-analysis-freeze.json" if label == "secondary" else "input-freeze.json"
        frozen = read(study / "data" / freeze_name)
        expected = frozen["files"].get(str(source.relative_to(ROOT)))
        require(expected is not None and expected == digest(source),
                f"Bridge {number} {label} analyzer differs from its effective scientific freeze")
        for name in ("summary.json", "per-example.npz"):
            require((target / name).is_file(), f"Bridge {number} missing canonical {folder}/{name}")
        jobs.append(dict(label=label, script=source, args=args, canonical=target))
    return dict(number=number, outcome=outcome, skipped=False, jobs=jobs)


def replay(numbers):
    # Complete preflight before launching any child, including later studies.
    plans = [plan_study(number) for number in numbers]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="ai01-extension-replays-") as temporary:
        for plan in plans:
            number = plan["number"]
            if plan["skipped"]:
                print(f"Bridge {number}: SKIP final analyses — verified {plan['outcome']}; no final transfer measured", flush=True)
                outcomes.append(dict(study=number, status="skipped", outcome=plan["outcome"]))
                continue
            for job in plan["jobs"]:
                target = Path(temporary) / f"bridge-{number}-{job['label']}"
                require(not target.exists(), f"Replay output is not fresh: {target}")
                subprocess.run([sys.executable, str(job["script"]), *job["args"],
                                "--output", str(target)], cwd=ROOT, env=environment, check=True)
                compare_outputs(job["canonical"], target)
                print(f"Bridge {number} {job['label']}: PASS — exact JSON except utc; exact array keys, shapes, dtypes and values", flush=True)
                outcomes.append(dict(study=number, analysis=job["label"], status="passed"))
    print(json.dumps(dict(status="passed", scope="Deterministic saved-observation replay; no model inference",
                          canonical_outputs_unchanged=True, analyses=outcomes)))


def self_test():
    """Synthetic comparator checks entirely in memory; no study records read."""
    compare_json({"utc":"old", "x":[1, {"utc":"old", "value":None}]},
                 {"utc":"new", "x":[1, {"utc":"new", "value":None}]})
    failures = [
        ({"x":1}, {"x":2}), ({"x":1}, {"x":1.0}), ({"x":True}, {"x":1}),
        ({"x":[1,2]}, {"x":[2,1]}), ({"x":None}, {}),
        ({"started":"old"}, {"started":"new"}),
    ]
    for left, right in failures:
        try: compare_json(left, right)
        except ValueError: pass
        else: raise AssertionError("JSON mismatch was accepted")
    arrays = {"float":np.array([0.,np.nan,np.inf,-np.inf], dtype=np.float64),
              "int":np.array([1,2], dtype=np.int32), "bool":np.array([True,False])}
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); buffer.seek(0)
    compare_arrays(arrays, load_arrays(buffer))
    left = {"x":np.array([1.,np.nan],dtype=np.float64)}
    variants = [{"y":left["x"]}, {"x":left["x"].reshape(1,2)},
                {"x":left["x"].astype(np.float32)}, {"x":np.array([2.,np.nan])},
                {"x":np.array([np.nan,1.])}]
    for right in variants:
        try: compare_arrays(left,right)
        except ValueError: pass
        else: raise AssertionError("Array mismatch was accepted")
    try: compare_arrays({"x":np.array([complex(np.nan,1)])}, {"x":np.array([complex(np.nan,2)])})
    except ValueError: pass
    else: raise AssertionError("Complex NaN hid a finite-component difference")
    print(json.dumps(dict(status="passed", scope="Synthetic in-memory comparator checks only",
                          checks=["recursive_utc_only_exclusion", "non_utc_json_mismatches_rejected",
                                  "numeric_types_and_list_order_exact", "in_memory_npz_round_trip",
                                  "array_keys_shapes_dtypes_values_exact", "aligned_nan_equality",
                                  "complex_nan_finite_part_checked"], model_inference=False, study_arrays_read=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=STUDIES, help="Replay these studies; default is all three")
    parser.add_argument("--self-test", action="store_true", help="Synthetic comparator checks only; no study files read")
    args = parser.parse_args()
    if args.self_test:
        require(args.only is None, "--self-test cannot be combined with --only")
        self_test()
    else:
        numbers = list(STUDIES) if args.only is None else list(dict.fromkeys(args.only))
        replay(numbers)


if __name__ == "__main__":
    main()
