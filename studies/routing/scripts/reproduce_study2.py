"""Replay the frozen study in a fresh directory using this project's existing environment."""

from __future__ import annotations

from activation_transfer.paths import recorded_path

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json
from causal_circuit_routing.study2_run import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    verify_inputs(selection=True)
    analysis_record = read_json(ROOT / "data/study-002-analysis-freeze.json")
    analyzer = ROOT / "scripts/analyze_study2.py"
    assert digest(analyzer) == analysis_record["analyzer_sha256"]
    bundle = args.output_dir.resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    work = bundle / "work/routing-project"
    work.mkdir(parents=True)
    with tarfile.open(ROOT / "data/study-002-frozen-source.tar.gz", "r:gz") as archive:
        archive.extractall(work, filter="data")
    frozen = read_json(work / "data/study-002-freeze.json")
    assert frozen == read_json(ROOT / "data/study-002-freeze.json")
    for path, sha in frozen["files"].items():
        assert digest(work / path) == sha, path
    shutil.copy2(analyzer, work / "scripts/analyze_study2.py")
    model_link = work / frozen["model"]["path"]
    model_link.parent.mkdir(parents=True)
    model_link.symlink_to(recorded_path(ROOT / frozen["model"]["path"]).resolve(), target_is_directory=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(work / "src")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(work / "scripts/run_study2_family.py")]
    write_json(
        bundle / "replay-record.json",
        {
            "prepared_utc": now(),
            "original_project": str(ROOT),
            "work_directory": str(work),
            "python": sys.executable,
            "frozen_files_verified": len(frozen["files"]),
            "model_cache_reused_readonly": str(model_link.resolve()),
            "source_archive_sha256": digest(ROOT / "data/study-002-frozen-source.tar.gz"),
            "prepared_only": args.prepare_only,
            "note": "A rerun has new runtime timestamps and a newly frozen selection; original records stay intact.",
        },
    )
    if args.prepare_only:
        print(f"Recovered and verified {len(frozen['files'])} frozen source/data files in {work}")
        print("No model inference was run. Use a fresh output directory for a complete replay.")
        return
    subprocess.run(command, cwd=work, env=env, check=True)
    write_json(
        work / "data/study-002-analysis-freeze.json",
        {
            "analyzer_sha256": digest(work / "scripts/analyze_study2.py"),
            "before_test_inference": True,
            "frozen_utc": now(),
            "selection_freeze_sha256": digest(work / "data/study-002-selection-freeze.json"),
        },
    )
    for script in ("run_study2_transfer.py", "analyze_study2.py"):
        subprocess.run(
            [sys.executable, str(work / "scripts" / script)], cwd=work, env=env, check=True
        )
    print(f"Full replay completed; results are in {work / 'results/study-002'}")


if __name__ == "__main__":
    main()
