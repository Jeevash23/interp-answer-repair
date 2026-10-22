"""Counterbalanced IOI prompts; family is crossed with answer identity."""

from __future__ import annotations

TEMPLATES = {
    "discovery": [
        ("d-museum", "Then {first} and {second} visited the museum. {subject} gave a map to"),
        ("d-cafe", "After {first} and {second} arrived at the cafe, {subject} offered a cake to"),
    ],
    "heldout": [
        (
            "h-station",
            "While {first} and {second} waited at the station, {subject} handed a ticket to",
        ),
        (
            "h-studio",
            "Because {first} and {second} worked at the studio, {subject} brought a coffee to",
        ),
    ],
}

NAME_PAIRS = {
    "discovery": [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "James"), ("Anna", "David")],
    "heldout": [
        ("Laura", "Paul"),
        ("Lisa", "Mark"),
        ("Helen", "Robert"),
        ("Susan", "Michael"),
    ],
}


def make_records(tokenizer):
    def token(name):
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Name is not one leading-space token: {name}")
        return ids[0]

    records = []
    for split, templates in TEMPLATES.items():
        pairs = NAME_PAIRS[split]
        for template_id, template in templates:
            for pair_index, pair in enumerate(pairs):
                for role_assignment in range(2):
                    io, subject = pair[role_assignment], pair[1 - role_assignment]
                    corrupt_name = pairs[(pair_index + 1) % len(pairs)][role_assignment]
                    assert len({io, subject, corrupt_name}) == 3
                    for family in ("ABBA", "BABA"):
                        first, second = (io, subject) if family == "ABBA" else (subject, io)
                        text = template.format(first=first, second=second, subject=subject)
                        corrupt = template.format(first=first, second=second, subject=corrupt_name)
                        clean_ids = [tokenizer.bos_token_id] + tokenizer.encode(
                            text, add_special_tokens=False
                        )
                        corrupt_ids = [tokenizer.bos_token_id] + tokenizer.encode(
                            corrupt, add_special_tokens=False
                        )
                        assert len(clean_ids) == len(corrupt_ids)
                        io_positions = [i for i, t in enumerate(clean_ids) if t == token(io)]
                        s_positions = [i for i, t in enumerate(clean_ids) if t == token(subject)]
                        assert len(io_positions) == 1 and len(s_positions) == 2
                        changed = [
                            i
                            for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids, strict=True))
                            if a != b
                        ]
                        assert changed == [s_positions[1]]
                        name_pair_id = f"{split}-n{pair_index}"
                        pair_id = f"{template_id}-{name_pair_id}-r{role_assignment}"
                        records.append(
                            {
                                "id": f"{pair_id}-{family}",
                                "pair_id": pair_id,
                                "name_pair_id": name_pair_id,
                                "split": split,
                                "template_id": template_id,
                                "family": family,
                                "role_assignment": role_assignment,
                                "io": io,
                                "subject": subject,
                                "corrupt_name": corrupt_name,
                                "io_token_id": token(io),
                                "subject_token_id": token(subject),
                                "corrupt_token_id": token(corrupt_name),
                                "text": text,
                                "corrupt_text": corrupt,
                                "clean_token_ids": clean_ids,
                                "corrupt_token_ids": corrupt_ids,
                                "positions": {
                                    "io": io_positions[0],
                                    "subject_first": s_positions[0],
                                    "subject_second": s_positions[1],
                                    "end": len(clean_ids) - 1,
                                },
                            }
                        )
    assert len(records) == 64 and len({r["id"] for r in records}) == 64
    return records
