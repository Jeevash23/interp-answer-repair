"""Inherit exact Bridge007 cases and audit GPT-2 Small tokenization only.

This never imports a model inference engine. Refuses to change existing outputs.
Retained b007 row IDs intentionally identify the same cases across checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OLD = ROOT / "studies/bridge-007"
FAMILIES = [10, 11]
BANK_FAMILIES = [10, 11, 12, 13]
SEED = 2026100409


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    body = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        assert path.read_bytes() == body, f"Refusing to overwrite {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def main():
    assert not (HERE / "data/input-freeze.json").exists(), "Preparation is pre-freeze only"
    selected = read(OLD / "data/selection.json")
    assert selected["status"] == "selected" and selected["families"] == FAMILIES
    origin = [OLD / "data" / f for f in
              ["candidates.json", "design.json", "preparation-audit.json", "input-freeze.json",
               "selection.json", "selection-freeze.json"]]
    origin += [ROOT / "configs/study.json", ROOT / "studies/transport/data/model-manifest.json",
               ROOT / "studies/transport/results/experiment-001/exploratory/run-config.json"]
    for freeze_name in ["input-freeze.json", "selection-freeze.json"]:
        for name, sha in read(OLD / "data" / freeze_name)["files"].items():
            assert digest(ROOT / name) == sha, name
    inherited = {str(p.relative_to(ROOT)): digest(p) for p in origin}
    rows = read(OLD / "data/candidates.json")
    design = read(OLD / "data/design.json")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/gpt2-small", local_files_only=True)
    assert tokenizer.bos_token_id == 50256
    assert len(rows) == 11008 and len({r["id"] for r in rows}) == len(rows)
    counts, pair_counts, alignments = {}, {}, 0
    for r in rows:
        assert [tokenizer.bos_token_id] + tokenizer.encode(r["text"], add_special_tokens=False) == r["token_ids"]
        for name, key in [("io", "io_id"), ("subject", "subject_id"), ("alternative", "alt_id")]:
            assert tokenizer.encode(" " + r[name], add_special_tokens=False) == [r[key]]
        assert r["token_ids"][r["s2"]] == r["subject_id"]
        corrupt = list(r["token_ids"]); corrupt[r["s2"]] = r["alt_id"]
        assert corrupt == r["corrupt_tokens"]
        assert r["token_ids"][r["io_position"]] == r["io_id"]
        assert r["token_ids"][r["subject_position"]] == r["subject_id"]
        counts[r["split"]] = counts.get(r["split"], 0) + 1
        if r["family"] in FAMILIES:
            pair_counts[r["split"]] = pair_counts.get(r["split"], 0) + 1
    key = lambda r: (r["split"], r["template"], r["pair"], r["role"], r["family"], r["order"])
    lookup = {key(r): r for r in rows}
    for r in rows:
        if r["family"] != 10: continue
        k = key(r); donor = lookup[(*k[:4], 11, k[5])]
        assert len(r["token_ids"]) == len(donor["token_ids"]) and r["s2"] == donor["s2"]
        assert all(r[field] == donor[field] for field in ["io", "subject", "alternative"])
        alignments += 1
    expected = dict(calibration=32, screen=96, fit=512, selection=512, qualification=1152, test=3200)
    assert pair_counts == expected and counts == {key: 2 * value for key, value in expected.items()}
    bank_names = [set(v) for v in design["names"].values()]
    bank_contexts = [set(map(tuple, v)) for v in design["blocks"].values()]
    assert all(not a & b for i, a in enumerate(bank_names) for b in bank_names[i+1:])
    assert all(not a & b for i, a in enumerate(bank_contexts) for b in bank_contexts[i+1:])
    model_files = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "models/gpt2-small").iterdir()) if p.is_file()}
    new_design = dict(design, study="bridge-009", model="gpt2-small", seed=SEED,
        inherited_data_seed=design["seed"], family_ids=FAMILIES, bank_family_ids=BANK_FAMILIES,
        candidate_pairs=[FAMILIES], alignment_eligible_pairs=[FAMILIES],
        inherited_candidate_manifest_family_ids=BANK_FAMILIES,
        tokenizer_files={k:v for k,v in model_files.items() if not k.endswith("model.safetensors")},
        scope="Fixed F10/F11 chosen in Medium; exact Bridge007 cases reused across checkpoints. No Small-specific pair search. All four inherited fitting families contribute equally to the mean replacement bank, regardless of Small behavior.")
    audit = dict(status="tokenizer_only_checks_passed", scientific_inference_performed=False,
        fixed_pair=FAMILIES, bank_families=BANK_FAMILIES, token_ids_match_small=True,
        exact_candidate_bytes_inherited=True, retained_original_row_ids=True,
        rows=len(rows), rows_per_split=counts, selected_pair_rows_per_split=pair_counts,
        fit_mean_bank_rows=1024, fit_signature_rows=512, matched_pair_alignments=alignments,
        split_names_disjoint=True, split_context_pairs_disjoint=True,
        across_model_cases_identical=True, inherited_source_hashes=inherited,
        checkpoint_and_tokenizer_hashes=model_files,
        model_spec=read(ROOT / "configs/study.json")["models"]["gpt2-small"],
        prepare_source_sha256=digest(Path(__file__)))
    publish(HERE / "data/candidates.json", (OLD / "data/candidates.json").read_bytes())
    publish(HERE / "data/design.json", new_design)
    publish(HERE / "data/preparation-audit.json", audit)
    print(json.dumps({"status": audit["status"], "all_candidate_rows":len(rows),
                      "selected_pair_rows":pair_counts, "fit_mean_bank_rows":1024}))


if __name__ == "__main__":
    main()
