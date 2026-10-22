from pathlib import Path

from transformers import AutoTokenizer

from circuit_transport.dataset import generate_manifest, validate_manifest, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_deterministic_and_valid(tmp_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "gpt2-small", local_files_only=True
    )
    first = generate_manifest(tokenizer)
    second = generate_manifest(tokenizer)
    validate_manifest(first)
    assert first == second
    assert len(first) == 240
    assert write_manifest(first, tmp_path / "first.jsonl") == write_manifest(
        second, tmp_path / "second.jsonl"
    )


def test_manifest_balances_patterns_and_preserves_positions() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "gpt2-small", local_files_only=True
    )
    records = generate_manifest(tokenizer)
    for split in ("exploratory", "development", "test"):
        split_records = [record for record in records if record["split"] == split]
        assert len(split_records) == 80
        assert sum(record["pattern"] == "ABBA" for record in split_records) == 40
        assert sum(record["pattern"] == "BABA" for record in split_records) == 40
    for record in records:
        for condition in ("canonical", "controlled"):
            item = record[condition]
            assert len(item["clean_token_ids"]) == len(item["corrupt_token_ids"]) == 14
            assert item["clean_token_ids"][record["positions"]["io"]] == record["io_token_id"]
