"""Replay saved statistics from an extracted snapshot with explicit import isolation."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from transport_bridge.common import ROOT, digest, now, write


def main():
    archive = ROOT / "output/combined-bridge-research-v2.zip"
    target = Path(tempfile.mkdtemp(prefix="combined-bridge-v2-audit-"))
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
        cwd=project, env=env, text=True,
    ).strip()
    assert Path(check) == project
    models = {}
    for model in ["gpt2-medium", "gpt2-small"]:
        subprocess.run(
            [sys.executable, "studies/bridge-002/code/analyze.py", "--model", model,
             "--output", f"reproduced/package-{model}"],
            cwd=project, env=env, check=True,
        )
        study = project / "studies/bridge-002"
        original = study / "results" / model / "analysis"
        replay = study / "reproduced" / f"package-{model}"
        assert digest(original / "summary.json") == digest(replay / "summary.json")
        with np.load(original / "per-example.npz") as a, np.load(replay / "per-example.npz") as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            models[model] = {"summary_byte_exact": True, "exact_arrays": len(a.files)}
    record = {
        "status": "passed", "utc": now(), "archive_sha256": digest(archive),
        "extracted_directory": str(target), "imported_project": str(project),
        "models": models, "checkpoints_required": False,
        "scope": "Fresh extracted-file statistical replay using the existing locked dependencies",
    }
    write(ROOT / "output/combined-extracted-replay-v2.json", record)
    print(record)


if __name__ == "__main__":
    main()
