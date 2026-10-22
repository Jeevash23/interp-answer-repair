"""Frozen multi-template prompt generation for Experiment 003."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from circuit_transport.dataset import valid_names

MANIFEST_SEED = 20260903
TRIPLES_PER_TEMPLATE = 10


@dataclass(frozen=True)
class TemplatePair:
    template_id: str
    split: str
    canonical: tuple[str | None, ...]
    controlled: tuple[str | None, ...]


def _pair(
    template_id: str,
    split: str,
    canonical: tuple[str, ...],
    controlled: tuple[str, ...],
) -> TemplatePair:
    def slots(words: tuple[str, ...]) -> tuple[str | None, ...]:
        if len(words) != 11:
            raise ValueError(f"{template_id} requires 11 fixed tokens per condition")
        return (words[0], None, words[1], None, *words[2:7], None, *words[7:])

    return TemplatePair(template_id, split, slots(canonical), slots(controlled))


TEMPLATES = (
    _pair(
        "development-beach",
        "development",
        ("Before", " and", " drove", " to", " the", " beach", ",", " passed", " a", " note", " to"),
        (
            "Beyond",
            " beside",
            " traced",
            " every",
            " purple",
            " square",
            ";",
            " moved",
            " one",
            " marker",
            " toward",
        ),
    ),
    _pair(
        "development-market",
        "development",
        (
            "Yesterday",
            " and",
            " walked",
            " through",
            " the",
            " market",
            ",",
            " tossed",
            " a",
            " ball",
            " to",
        ),
        (
            "Perhaps",
            " near",
            " counted",
            " each",
            " narrow",
            " column",
            ";",
            " pushed",
            " that",
            " symbol",
            " toward",
        ),
    ),
    _pair(
        "development-river",
        "development",
        (
            "Soon",
            " and",
            " traveled",
            " past",
            " the",
            " river",
            ",",
            " showed",
            " a",
            " photo",
            " to",
        ),
        (
            "Always",
            " around",
            " crossed",
            " some",
            " empty",
            " circle",
            ";",
            " carried",
            " this",
            " object",
            " toward",
        ),
    ),
    _pair(
        "development-hotel",
        "development",
        (
            "Once",
            " and",
            " arrived",
            " at",
            " the",
            " hotel",
            ",",
            " offered",
            " a",
            " meal",
            " to",
        ),
        (
            "Meanwhile",
            " behind",
            " painted",
            " every",
            " bright",
            " pattern",
            ";",
            " placed",
            " one",
            " token",
            " toward",
        ),
    ),
    _pair(
        "development-school",
        "development",
        (
            "Earlier",
            " and",
            " went",
            " into",
            " the",
            " school",
            ",",
            " sent",
            " a",
            " card",
            " to",
        ),
        (
            "Inside",
            " beside",
            " mapped",
            " each",
            " quiet",
            " sector",
            ";",
            " shifted",
            " one",
            " signal",
            " toward",
        ),
    ),
    _pair(
        "development-theater",
        "development",
        (
            "Later",
            " and",
            " stopped",
            " by",
            " the",
            " theater",
            ",",
            " gave",
            " a",
            " ticket",
            " to",
        ),
        (
            "Above",
            " near",
            " formed",
            " every",
            " silver",
            " border",
            ";",
            " carried",
            " that",
            " token",
            " toward",
        ),
    ),
    _pair(
        "development-library",
        "development",
        (
            "Today",
            " and",
            " waited",
            " near",
            " the",
            " library",
            ",",
            " handed",
            " a",
            " map",
            " to",
        ),
        (
            "Across",
            " around",
            " measured",
            " some",
            " hidden",
            " angle",
            ";",
            " moved",
            " this",
            " marker",
            " toward",
        ),
    ),
    _pair(
        "development-cafe",
        "development",
        ("Outside", " and", " sat", " at", " the", " cafe", ",", " brought", " a", " snack", " to"),
        (
            "Below",
            " behind",
            " crossed",
            " each",
            " gentle",
            " pattern",
            ";",
            " pushed",
            " one",
            " object",
            " toward",
        ),
    ),
    _pair(
        "test-museum",
        "test",
        (
            "Recently",
            " and",
            " visited",
            " the",
            " old",
            " museum",
            ",",
            " gave",
            " a",
            " ticket",
            " to",
        ),
        (
            "Under",
            " beside",
            " sorted",
            " every",
            " orange",
            " figure",
            ";",
            " placed",
            " that",
            " marker",
            " toward",
        ),
    ),
    _pair(
        "test-station",
        "test",
        (
            "Suddenly",
            " and",
            " returned",
            " from",
            " the",
            " station",
            ",",
            " sent",
            " a",
            " package",
            " to",
        ),
        (
            "Near",
            " around",
            " traced",
            " some",
            " quiet",
            " channel",
            ";",
            " moved",
            " one",
            " symbol",
            " toward",
        ),
    ),
    _pair(
        "test-forest",
        "test",
        (
            "Elsewhere",
            " and",
            " walked",
            " through",
            " the",
            " forest",
            ",",
            " offered",
            " a",
            " flower",
            " to",
        ),
        (
            "Perhaps",
            " behind",
            " counted",
            " each",
            " silver",
            " object",
            ";",
            " pushed",
            " this",
            " signal",
            " toward",
        ),
    ),
    _pair(
        "test-church",
        "test",
        (
            "Eventually",
            " and",
            " arrived",
            " at",
            " the",
            " church",
            ",",
            " showed",
            " a",
            " letter",
            " to",
        ),
        (
            "Within",
            " beside",
            " shaped",
            " every",
            " narrow",
            " pattern",
            ";",
            " carried",
            " one",
            " token",
            " toward",
        ),
    ),
    _pair(
        "test-kitchen",
        "test",
        (
            "Tonight",
            " and",
            " met",
            " in",
            " the",
            " kitchen",
            ",",
            " passed",
            " a",
            " plate",
            " to",
        ),
        (
            "Across",
            " near",
            " marked",
            " some",
            " bright",
            " circle",
            ";",
            " shifted",
            " that",
            " object",
            " toward",
        ),
    ),
    _pair(
        "test-bridge",
        "test",
        (
            "Morning",
            " and",
            " stopped",
            " by",
            " the",
            " bridge",
            ",",
            " handed",
            " a",
            " key",
            " to",
        ),
        (
            "Above",
            " around",
            " formed",
            " each",
            " hidden",
            " border",
            ";",
            " placed",
            " this",
            " marker",
            " toward",
        ),
    ),
    _pair(
        "test-village",
        "test",
        (
            "Together",
            " and",
            " traveled",
            " through",
            " the",
            " village",
            ",",
            " brought",
            " a",
            " gift",
            " to",
        ),
        (
            "Beyond",
            " behind",
            " mapped",
            " every",
            " gentle",
            " sector",
            ";",
            " moved",
            " that",
            " signal",
            " toward",
        ),
    ),
    _pair(
        "test-airport",
        "test",
        (
            "Finally",
            " and",
            " waited",
            " at",
            " the",
            " airport",
            ",",
            " mailed",
            " a",
            " photo",
            " to",
        ),
        (
            "Below",
            " beside",
            " measured",
            " some",
            " purple",
            " square",
            ";",
            " pushed",
            " one",
            " token",
            " toward",
        ),
    ),
)


def _single_token(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def _template_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    slots: tuple[str | None, ...],
    *,
    first_name: str,
    second_name: str,
    repeated_subject: str,
) -> list[int]:
    entities = iter((first_name, second_name, repeated_subject))
    return [
        _single_token(tokenizer, f" {next(entities)}" if slot is None else slot) for slot in slots
    ]


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
    output: dict[str, Any] = {}
    for variant, token_ids in (
        ("clean", clean),
        ("corrupt", corrupt),
        ("alternative_corrupt", alternative_corrupt),
    ):
        text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != token_ids:
            raise ValueError(f"Round-trip tokenization failed for {text!r}")
        output[f"{variant}_token_ids"] = token_ids
        output[f"{variant}_text"] = text
    return output


def generate_manifest(
    tokenizer: PreTrainedTokenizerBase,
    excluded_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate new template-balanced records with no old entity-triple overlap."""
    names = valid_names(tokenizer)
    excluded = {
        frozenset((row["io_name"], row["subject_name"], row["corrupt_name"]))
        for row in excluded_records
    }
    used = set(excluded)
    rng = random.Random(MANIFEST_SEED)
    records: list[dict[str, Any]] = []

    for template_index, template in enumerate(TEMPLATES):
        for slots in (template.canonical, template.controlled):
            if len(slots) != 14 or tuple(i for i, value in enumerate(slots) if value is None) != (
                1,
                3,
                9,
            ):
                raise ValueError(f"Invalid layout for {template.template_id}")
            for slot in slots:
                if slot is not None:
                    _single_token(tokenizer, slot)

        triples: list[tuple[str, str, str]] = []
        while len(triples) < TRIPLES_PER_TEMPLATE:
            triple = tuple(rng.sample(names, 3))
            identity = frozenset(triple)
            if identity not in used:
                used.add(identity)
                triples.append(triple)

        for triple_index, (io_name, subject, corrupt_name) in enumerate(triples):
            for pattern in ("ABBA", "BABA"):
                if pattern == "ABBA":
                    first_name, second_name = io_name, subject
                    io_position, subject_first_position = 1, 3
                else:
                    first_name, second_name = subject, io_name
                    io_position, subject_first_position = 3, 1
                example_id = (
                    f"experiment-003-{template.split}-{template_index:02d}-"
                    f"{triple_index:02d}-{pattern.lower()}"
                )
                records.append(
                    {
                        "example_id": example_id,
                        "split": template.split,
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
    return records


def validate_manifest(
    records: list[dict[str, Any]],
    excluded_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = len(TEMPLATES) * TRIPLES_PER_TEMPLATE * 2
    if len(records) != expected:
        raise ValueError(f"Expected {expected} records, got {len(records)}")
    old_triples = {
        frozenset((row["io_name"], row["subject_name"], row["corrupt_name"]))
        for row in excluded_records
    }
    new_triples: set[frozenset[str]] = set()
    template_counts: dict[str, int] = {}
    for record in records:
        identity = frozenset((record["io_name"], record["subject_name"], record["corrupt_name"]))
        if identity in old_triples:
            raise ValueError(f"Old entity triple reused by {record['example_id']}")
        new_triples.add(identity)
        template_counts[record["template_id"]] = template_counts.get(record["template_id"], 0) + 1
        for condition in ("canonical", "controlled"):
            for variant in ("clean", "corrupt", "alternative_corrupt"):
                if len(record[condition][f"{variant}_token_ids"]) != 14:
                    raise ValueError(f"Unexpected token length for {record['example_id']}")
    if len(new_triples) != len(TEMPLATES) * TRIPLES_PER_TEMPLATE:
        raise ValueError("New entity triples are not unique across templates")
    if set(template_counts.values()) != {TRIPLES_PER_TEMPLATE * 2}:
        raise ValueError("Templates are not balanced")
    for split in ("development", "test"):
        split_rows = [row for row in records if row["split"] == split]
        if len(split_rows) != expected // 2:
            raise ValueError(f"{split} is not balanced")
    return {
        "records": len(records),
        "templates": len(template_counts),
        "templates_per_split": len(template_counts) // 2,
        "records_per_template": TRIPLES_PER_TEMPLATE * 2,
        "unique_entity_triples": len(new_triples),
        "old_entity_triple_overlap": 0,
    }


def write_manifest(records: list[dict[str, Any]], path: Path) -> str:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()
