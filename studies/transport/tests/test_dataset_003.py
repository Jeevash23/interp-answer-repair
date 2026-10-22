import json
from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset_003 import generate_manifest, validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_003_manifest_is_deterministic_balanced_and_novel() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "gpt2-small", local_files_only=True
    )
    old_records = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    first = generate_manifest(tokenizer, old_records)
    second = generate_manifest(tokenizer, old_records)
    assert first == second
    audit = validate_manifest(first, old_records)
    assert audit == {
        "records": 320,
        "templates": 16,
        "templates_per_split": 8,
        "records_per_template": 20,
        "unique_entity_triples": 160,
        "old_entity_triple_overlap": 0,
    }


def test_experiment_003_preserves_pair_adjacency_and_role_positions() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "gpt2-small", local_files_only=True
    )
    old_records = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "experiment-001-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    records = generate_manifest(tokenizer, old_records)
    for first, second in zip(records[::2], records[1::2], strict=True):
        assert (first["pattern"], second["pattern"]) == ("ABBA", "BABA")
        assert first["template_id"] == second["template_id"]
        assert first["io_name"] == second["io_name"]
        assert first["subject_name"] == second["subject_name"]
        assert first["corrupt_name"] == second["corrupt_name"]
    for record in records:
        assert record["positions"]["subject_second"] == 9
        assert record["positions"]["end"] == 13
        for condition in ("canonical", "controlled"):
            clean = record[condition]["clean_token_ids"]
            assert clean[record["positions"]["io"]] == record["io_token_id"]
            assert clean[record["positions"]["subject_first"]] == record["subject_token_id"]
            assert clean[record["positions"]["subject_second"]] == record["subject_token_id"]
