"""Frozen paired-prompt generation for Experiment 001."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

MANIFEST_SEED = 1729
SPLIT_TRIPLES = {"exploratory": 40, "development": 40, "test": 40}

NAME_CANDIDATES = (
    "Alice",
    "Amy",
    "Anna",
    "Anne",
    "Arthur",
    "Barbara",
    "Betty",
    "Bob",
    "Brian",
    "Carol",
    "Charles",
    "David",
    "Donna",
    "Edward",
    "Elizabeth",
    "Emily",
    "Frank",
    "Gary",
    "George",
    "Helen",
    "Henry",
    "Jack",
    "James",
    "Jason",
    "Jennifer",
    "Jessica",
    "John",
    "Karen",
    "Kevin",
    "Laura",
    "Linda",
    "Lisa",
    "Margaret",
    "Mark",
    "Martin",
    "Mary",
    "Michael",
    "Michelle",
    "Nancy",
    "Patricia",
    "Paul",
    "Peter",
    "Richard",
    "Robert",
    "Ruth",
    "Sandra",
    "Sarah",
    "Sharon",
    "Susan",
    "Thomas",
    "Walter",
    "William",
)


@dataclass(frozen=True)
class TemplatePair:
    template_id: str
    split: str
    canonical: tuple[str | None, ...]
    controlled: tuple[str | None, ...]


# None marks entity slots: first name, second name, and repeated subject.
TEMPLATES = (
    TemplatePair(
        "explore-store",
        "exploratory",
        (
            "When",
            None,
            " and",
            None,
            " went",
            " to",
            " the",
            " store",
            ",",
            None,
            " gave",
            " a",
            " drink",
            " to",
        ),
        (
            "Thus",
            None,
            " beside",
            None,
            " painted",
            " every",
            " silent",
            " garden",
            ";",
            None,
            " carried",
            " some",
            " token",
            " toward",
        ),
    ),
    TemplatePair(
        "explore-park",
        "exploratory",
        (
            "Then",
            None,
            " and",
            None,
            " went",
            " to",
            " the",
            " park",
            ",",
            None,
            " handed",
            " a",
            " book",
            " to",
        ),
        (
            "Today",
            None,
            " near",
            None,
            " crossed",
            " each",
            " quiet",
            " field",
            ";",
            None,
            " moved",
            " one",
            " object",
            " toward",
        ),
    ),
    TemplatePair(
        "development-office",
        "development",
        (
            "After",
            None,
            " and",
            None,
            " went",
            " to",
            " the",
            " office",
            ",",
            None,
            " sent",
            " a",
            " letter",
            " to",
        ),
        (
            "Later",
            None,
            " around",
            None,
            " shaped",
            " this",
            " green",
            " circle",
            ";",
            None,
            " pushed",
            " that",
            " signal",
            " toward",
        ),
    ),
    TemplatePair(
        "test-garden",
        "test",
        (
            "While",
            None,
            " and",
            None,
            " walked",
            " in",
            " the",
            " garden",
            ",",
            None,
            " brought",
            " a",
            " gift",
            " to",
        ),
        (
            "During",
            None,
            " beyond",
            None,
            " counted",
            " through",
            " each",
            " pattern",
            ";",
            None,
            " marked",
            " one",
            " symbol",
            " toward",
        ),
    ),
)


def _single_token(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected exactly one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def valid_names(tokenizer: PreTrainedTokenizerBase) -> tuple[str, ...]:
    """Return candidate names whose leading-space form is exactly one token."""
    return tuple(
        name
        for name in NAME_CANDIDATES
        if len(tokenizer.encode(f" {name}", add_special_tokens=False)) == 1
    )


def _template_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    slots: tuple[str | None, ...],
    *,
    first_name: str,
    second_name: str,
    repeated_subject: str,
) -> list[int]:
    entity_values = iter((first_name, second_name, repeated_subject))
    token_ids: list[int] = []
    for slot in slots:
        text = f" {next(entity_values)}" if slot is None else slot
        token_ids.append(_single_token(tokenizer, text))
    return token_ids


def _condition_record(
    tokenizer: PreTrainedTokenizerBase,
    slots: tuple[str | None, ...],
    *,
    first_name: str,
    second_name: str,
    subject: str,
    corrupt_name: str,
    subject_first_position: int,
) -> dict[str, Any]:
    clean = _template_token_ids(
        tokenizer,
        slots,
        first_name=first_name,
        second_name=second_name,
        repeated_subject=subject,
    )
    corrupt = clean.copy()
    corrupt[9] = _single_token(tokenizer, f" {corrupt_name}")
    alternative_corrupt = clean.copy()
    alternative_corrupt[subject_first_position] = _single_token(tokenizer, f" {corrupt_name}")

    records = {
        "clean_token_ids": clean,
        "corrupt_token_ids": corrupt,
        "alternative_corrupt_token_ids": alternative_corrupt,
        "clean_text": tokenizer.decode(clean, clean_up_tokenization_spaces=False),
        "corrupt_text": tokenizer.decode(corrupt, clean_up_tokenization_spaces=False),
        "alternative_corrupt_text": tokenizer.decode(
            alternative_corrupt, clean_up_tokenization_spaces=False
        ),
    }
    for key in ("clean", "corrupt", "alternative_corrupt"):
        text = records[f"{key}_text"]
        token_ids = records[f"{key}_token_ids"]
        if tokenizer.encode(text, add_special_tokens=False) != token_ids:
            raise ValueError(f"Round-trip tokenization failed for {text!r}")
    return records


def generate_manifest(tokenizer: PreTrainedTokenizerBase) -> list[dict[str, Any]]:
    """Generate the fully deterministic paired manifest."""
    names = valid_names(tokenizer)
    if len(names) < 20:
        raise ValueError(f"Only {len(names)} single-token names were available")

    for template in TEMPLATES:
        for slots in (template.canonical, template.controlled):
            if len(slots) != 14 or tuple(i for i, value in enumerate(slots) if value is None) != (
                1,
                3,
                9,
            ):
                raise ValueError(f"Invalid template layout: {template.template_id}")
            for slot in slots:
                if slot is not None:
                    _single_token(tokenizer, slot)

    rng = random.Random(MANIFEST_SEED)
    used_triples: set[frozenset[str]] = set()
    split_triples: dict[str, list[tuple[str, str, str]]] = {}
    for split, count in SPLIT_TRIPLES.items():
        triples: list[tuple[str, str, str]] = []
        while len(triples) < count:
            triple = tuple(rng.sample(names, 3))
            identity = frozenset(triple)
            if identity not in used_triples:
                used_triples.add(identity)
                triples.append(triple)
        split_triples[split] = triples

    records: list[dict[str, Any]] = []
    for split, triples in split_triples.items():
        templates = [template for template in TEMPLATES if template.split == split]
        for triple_index, (io_name, subject, corrupt_name) in enumerate(triples):
            template = templates[triple_index % len(templates)]
            for pattern in ("ABBA", "BABA"):
                if pattern == "ABBA":
                    first_name, second_name = io_name, subject
                    io_position, subject_first_position = 1, 3
                else:
                    first_name, second_name = subject, io_name
                    io_position, subject_first_position = 3, 1

                record_id = f"{split}-{triple_index:03d}-{pattern.lower()}"
                records.append(
                    {
                        "example_id": record_id,
                        "split": split,
                        "template_id": template.template_id,
                        "pattern": pattern,
                        "io_name": io_name,
                        "subject_name": subject,
                        "corrupt_name": corrupt_name,
                        "io_token_id": _single_token(tokenizer, f" {io_name}"),
                        "subject_token_id": _single_token(tokenizer, f" {subject}"),
                        "corrupt_token_id": _single_token(tokenizer, f" {corrupt_name}"),
                        "positions": {
                            "io": io_position,
                            "subject_first": subject_first_position,
                            "subject_second": 9,
                            "end": 13,
                        },
                        "canonical": _condition_record(
                            tokenizer,
                            template.canonical,
                            first_name=first_name,
                            second_name=second_name,
                            subject=subject,
                            corrupt_name=corrupt_name,
                            subject_first_position=subject_first_position,
                        ),
                        "controlled": _condition_record(
                            tokenizer,
                            template.controlled,
                            first_name=first_name,
                            second_name=second_name,
                            subject=subject,
                            corrupt_name=corrupt_name,
                            subject_first_position=subject_first_position,
                        ),
                    }
                )
    validate_manifest(records)
    return records


def validate_manifest(records: list[dict[str, Any]]) -> None:
    """Validate all frozen split, balance, and token-alignment invariants."""
    expected_examples = 2 * sum(SPLIT_TRIPLES.values())
    if len(records) != expected_examples:
        raise ValueError(f"Expected {expected_examples} examples, found {len(records)}")
    if len({record["example_id"] for record in records}) != len(records):
        raise ValueError("Example IDs are not unique")

    triples_by_split: dict[str, set[frozenset[str]]] = {}
    templates_by_split: dict[str, set[str]] = {}
    for record in records:
        split = record["split"]
        triple = frozenset((record["io_name"], record["subject_name"], record["corrupt_name"]))
        triples_by_split.setdefault(split, set()).add(triple)
        templates_by_split.setdefault(split, set()).add(record["template_id"])
        positions = record["positions"]
        for condition in ("canonical", "controlled"):
            item = record[condition]
            lengths = {
                len(item["clean_token_ids"]),
                len(item["corrupt_token_ids"]),
                len(item["alternative_corrupt_token_ids"]),
            }
            if lengths != {14}:
                raise ValueError(f"Unequal token lengths in {record['example_id']}")
            if item["clean_token_ids"][positions["io"]] != record["io_token_id"]:
                raise ValueError(f"IO position mismatch in {record['example_id']}")
            if item["clean_token_ids"][positions["subject_first"]] != record["subject_token_id"]:
                raise ValueError(f"S1 position mismatch in {record['example_id']}")
            if item["clean_token_ids"][positions["subject_second"]] != record["subject_token_id"]:
                raise ValueError(f"S2 position mismatch in {record['example_id']}")

    splits = tuple(SPLIT_TRIPLES)
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            if triples_by_split[first] & triples_by_split[second]:
                raise ValueError(f"Entity triples cross {first}/{second}")
    if templates_by_split["development"] & templates_by_split["test"]:
        raise ValueError("Development and test templates overlap")
    for split in splits:
        patterns = [record["pattern"] for record in records if record["split"] == split]
        if patterns.count("ABBA") != patterns.count("BABA"):
            raise ValueError(f"Unbalanced patterns in {split}")


def write_manifest(records: list[dict[str, Any]], path: Path) -> str:
    """Write canonical JSONL and return its SHA-256 digest."""
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def write_audit(records: list[dict[str, Any]], path: Path, *, count: int = 100) -> None:
    """Write a stable structured prompt audit sheet."""
    lines = [
        "# Experiment 001 structured prompt audit",
        "",
        f"First {count} manifest rows, generated before model outcome evaluation.",
        "",
    ]
    for index, record in enumerate(records[:count], start=1):
        lines.extend(
            [
                f"## {index:03d} — {record['example_id']}",
                "",
                f"- split/template/order: `{record['split']}` / `{record['template_id']}` / `{record['pattern']}`",
                f"- roles: IO=`{record['io_name']}`, subject=`{record['subject_name']}`, corruption=`{record['corrupt_name']}`",
                f"- canonical clean: `{record['canonical']['clean_text']}`",
                f"- canonical corrupt: `{record['canonical']['corrupt_text']}`",
                f"- controlled clean: `{record['controlled']['clean_text']}`",
                f"- controlled corrupt: `{record['controlled']['corrupt_text']}`",
                "- audit status: token and role invariants passed; independent human review not yet signed",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
