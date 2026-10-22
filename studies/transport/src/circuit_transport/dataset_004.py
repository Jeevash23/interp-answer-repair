"""Retokenize the frozen multi-template text for Pythia."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase


def _single_name_token(tokenizer: PreTrainedTokenizerBase, name: str) -> int:
    token_ids = tokenizer.encode(f" {name}", add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Pythia does not encode {name!r} as one leading-space token")
    return int(token_ids[0])


def _tokenize_roundtrip(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    if tokenizer.encode(decoded, add_special_tokens=False) != token_ids:
        raise ValueError(f"Pythia round-trip tokenization failed for {text!r}")
    return [int(token_id) for token_id in token_ids]


def retokenize_manifest(
    tokenizer: PreTrainedTokenizerBase, source_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    for source in source_records:
        io_token_id = _single_name_token(tokenizer, source["io_name"])
        subject_token_id = _single_name_token(tokenizer, source["subject_name"])
        corrupt_token_id = _single_name_token(tokenizer, source["corrupt_name"])
        conditions = {}
        positions = {}
        for condition in ("canonical", "controlled"):
            condition_output = {}
            for variant in ("clean", "corrupt", "alternative_corrupt"):
                text = source[condition][f"{variant}_text"]
                condition_output[f"{variant}_text"] = text
                condition_output[f"{variant}_token_ids"] = _tokenize_roundtrip(tokenizer, text)
            clean = condition_output["clean_token_ids"]
            io_positions = [
                index for index, token_id in enumerate(clean) if token_id == io_token_id
            ]
            subject_positions = [
                index for index, token_id in enumerate(clean) if token_id == subject_token_id
            ]
            if len(io_positions) != 1 or len(subject_positions) != 2:
                raise ValueError(f"Could not recover roles for {source['example_id']}/{condition}")
            positions[condition] = {
                "io": io_positions[0],
                "subject_first": subject_positions[0],
                "subject_second": subject_positions[1],
                "end": len(clean) - 1,
            }
            conditions[condition] = condition_output
        records.append(
            {
                "example_id": source["example_id"],
                "split": source["split"],
                "template_id": source["template_id"],
                "pattern": source["pattern"],
                "io_name": source["io_name"],
                "subject_name": source["subject_name"],
                "corrupt_name": source["corrupt_name"],
                "io_token_id": io_token_id,
                "subject_token_id": subject_token_id,
                "corrupt_token_id": corrupt_token_id,
                "positions": positions,
                **conditions,
            }
        )
    return records


def validate_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 320:
        raise ValueError(f"Expected 320 records, got {len(records)}")
    template_counts: dict[str, int] = {}
    lengths: dict[str, set[int]] = {"canonical": set(), "controlled": set()}
    for record in records:
        template_counts[record["template_id"]] = template_counts.get(record["template_id"], 0) + 1
        for condition in ("canonical", "controlled"):
            clean = record[condition]["clean_token_ids"]
            corruption = record[condition]["corrupt_token_ids"]
            if len(clean) != len(corruption):
                raise ValueError(
                    f"Clean/corrupt length differs for {record['example_id']}/{condition}"
                )
            lengths[condition].add(len(clean))
            position = record["positions"][condition]
            if clean[position["io"]] != record["io_token_id"]:
                raise ValueError("IO position mismatch")
            if clean[position["subject_first"]] != record["subject_token_id"]:
                raise ValueError("First subject position mismatch")
            if clean[position["subject_second"]] != record["subject_token_id"]:
                raise ValueError("Second subject position mismatch")
    if set(template_counts.values()) != {20}:
        raise ValueError("Pythia templates are not balanced")
    return {
        "records": len(records),
        "templates": len(template_counts),
        "records_per_template": 20,
        "canonical_lengths": sorted(lengths["canonical"]),
        "controlled_lengths": sorted(lengths["controlled"]),
    }


def write_manifest(records: list[dict[str, Any]], path: Path) -> str:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()
