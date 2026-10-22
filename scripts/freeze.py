"""Freeze the final validated source and tokenized data before study inference."""

import tarfile

from transport_bridge.common import ROOT, digest, now, read, write

target = ROOT / "data/source-freeze.json"
if target.exists():
    raise FileExistsError("Freeze exists; use a disclosed new study/version for scientific changes")
assert read(ROOT / "results/gpt2-medium-technical-check.json")["max_error"] <= 0.0005
files = []
for directory, pattern in [("src", "*.py"), ("scripts", "*.py"), ("configs", "*.json")]:
    files += list((ROOT / directory).rglob(pattern))
files += [
    ROOT / p
    for p in [
        "docs/PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
        "data/recipients.json",
        "data/prompts.json",
        "data/templates.json",
    ]
]
record = {
    "utc": now(),
    "local_only_not_public_preregistration": True,
    "files": {str(p.relative_to(ROOT)): digest(p) for p in files},
    "models": {},
}
for name in read(ROOT / "configs/study.json")["models"]:
    model_path = ROOT / "models" / name
    record["models"][name] = {
        str(p.relative_to(ROOT.parent)): digest(p) for p in model_path.iterdir() if p.is_file()
    }
write(target, record)
with tarfile.open(ROOT / "data/frozen-source.tar.gz", "w:gz") as archive:
    for p in files + [target]:
        archive.add(p, arcname=str(p.relative_to(ROOT)))
print("Frozen scientific files:", len(files), record["utc"])
