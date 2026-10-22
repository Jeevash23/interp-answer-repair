import json
from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset_004 import retokenize_manifest, validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pythia_manifest_retokenization_is_deterministic_and_role_valid() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "pythia-410m-deduped", local_files_only=True
    )
    source = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "experiment-003-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    first = retokenize_manifest(tokenizer, source)
    second = retokenize_manifest(tokenizer, source)
    assert first == second
    assert validate_manifest(first) == {
        "records": 320,
        "templates": 16,
        "records_per_template": 20,
        "canonical_lengths": [14, 15],
        "controlled_lengths": [14, 15],
    }
    for record in first:
        for condition in ("canonical", "controlled"):
            position = record["positions"][condition]
            clean = record[condition]["clean_token_ids"]
            assert position["end"] == len(clean) - 1
