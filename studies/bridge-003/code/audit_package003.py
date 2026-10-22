"""Statistical replay from a fresh extraction with imports isolated to that snapshot."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from common003 import ROOT, digest, now, write


def main():
    archive = ROOT / "output/combined-bridge-research-v3.zip"
    target = Path(tempfile.mkdtemp(prefix="combined-bridge-v3-audit-"))
    with ZipFile(archive) as z:
        manifest = json.loads(z.read("artifact-manifest.json"))
        assert z.testzip() is None
        for name in z.namelist():
            assert (target / name).resolve().is_relative_to(target)
        for name, sha in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == sha
        z.extractall(target)
    project = target / ROOT.name
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1")
    check = subprocess.check_output(
        [sys.executable, "-c", "from transport_bridge.common import ROOT; print(ROOT)"],
        cwd=project,
        env=env,
        text=True,
    ).strip()
    assert Path(check) == project
    jobs = [
        ("bridge-003", "medium-cancellation", "analyze_medium003.py", []),
        ("bridge-003", "gpt2-medium", "analyze_broad003.py", ["--model", "gpt2-medium"]),
        (
            "bridge-003",
            "pythia-410m-deduped",
            "analyze_broad003.py",
            ["--model", "pythia-410m-deduped"],
        ),
        ("bridge-002", "gpt2-medium", "analyze.py", ["--model", "gpt2-medium"]),
        ("bridge-002", "gpt2-small", "analyze.py", ["--model", "gpt2-small"]),
    ]
    results = {}
    for study, model, script, args in jobs:
        output = f"reproduced/package-{model}"
        subprocess.run(
            [sys.executable, f"studies/{study}/code/{script}", *args, "--output", output],
            cwd=project,
            env=env,
            check=True,
        )
        base = project / "studies" / study
        original = base / "results" / model / "analysis"
        replay = base / output
        assert digest(original / "summary.json") == digest(replay / "summary.json")
        with np.load(original / "per-example.npz") as a, np.load(replay / "per-example.npz") as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])
            results[f"{study}/{model}"] = {"summary_byte_exact": True, "exact_arrays": len(a.files)}
    diagnostics = {}
    for model in ["gpt2-medium", "pythia-410m-deduped"]:
        base = project / "studies/bridge-003"
        output = f"reproduced/paired-{model}"
        for script, destination in [
            ("paired_geometry003.py", output),
            ("calibrate_geometry003.py", output + "-calibration.json"),
        ]:
            subprocess.run(
                [
                    sys.executable,
                    "studies/bridge-003/code/" + script,
                    "--model",
                    model,
                    "--output",
                    destination,
                ],
                cwd=project,
                env=env,
                check=True,
            )
        original = base / "results" / model
        assert digest(original / "geometry-calibration.json") == digest(
            base / (output + "-calibration.json")
        )
        assert digest(original / "paired-geometry/summary.json") == digest(
            base / output / "summary.json"
        )
        with (
            np.load(original / "paired-geometry/per-example.npz") as a,
            np.load(base / output / "per-example.npz") as b,
        ):
            assert set(a.files) == set(b.files)
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])
            diagnostics[model] = {"both_summaries_byte_exact": True, "exact_arrays": len(a.files)}
    record = {
        "status": "passed",
        "utc": now(),
        "archive_sha256": digest(archive),
        "extracted_directory": str(target),
        "imported_project": str(project),
        "analyses": results,
        "post_hoc_geometry_diagnostics": diagnostics,
        "checkpoints_required": False,
        "scope": "Extracted-file statistical replay using existing locked dependencies",
    }
    write(ROOT / "output/combined-extracted-replay-v3.json", record)
    print(record)


if __name__ == "__main__":
    main()
