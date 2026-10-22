"""Disjoint discovery/development/test prompts and prespecified four-donor cells."""

from __future__ import annotations

import hashlib

TEMPLATES = {
    "discovery": [
        "When {first} and {second} went to the market, {subject} gave a bag to",
        "After {first} and {second} visited the library, {subject} handed a book to",
        "Then {first} and {second} walked to the bakery, {subject} gave a cake to",
        "While {first} and {second} were at the office, {subject} handed a note to",
    ],
    "development": [
        "When {first} and {second} went to the restaurant, {subject} gave a drink to",
        "After {first} and {second} visited the garden, {subject} handed a flower to",
        "Then {first} and {second} arrived at the airport, {subject} gave a ticket to",
        "While {first} and {second} were at the school, {subject} handed a letter to",
    ],
    "test": [
        "When {first} and {second} went to the zoo, {subject} gave a snack to",
        "After {first} and {second} visited the theater, {subject} handed a ticket to",
        "Before {first} and {second} left the hotel, {subject} gave a key to",
        "While {first} and {second} waited at the bus stop, {subject} handed a map to",
        "Then {first} and {second} entered the shop, {subject} gave a gift to",
        "Yesterday, {first} and {second} met at the park. {subject} handed a bottle to",
        "At the party, {first} and {second} talked. {subject} gave a present to",
        "During lunch, {first} and {second} sat together. {subject} handed a cup to",
    ],
}
NAMES = {
    "discovery": [("Emma", "Daniel"), ("Grace", "Henry"), ("Lucy", "Peter"), ("Julia", "Kevin")],
    "development": [
        ("Rachel", "George"),
        ("Emily", "Thomas"),
        ("Jessica", "Charles"),
        ("Linda", "Richard"),
    ],
    "test": [("Amy", "Jack"), ("Anne", "Brian"), ("Karen", "Edward"), ("Nancy", "William")],
}


def build(tokenizer):
    library, recipients = {}, []

    def token(name):
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Not a one-token name: {name}")
        return ids[0]

    def prompt(split, ti, family, io, subject, s2=None):
        first, second = (io, subject) if family == "ABBA" else (subject, io)
        text = TEMPLATES[split][ti].format(first=first, second=second, subject=s2 or subject)
        ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
        io_pos = [p for p, value in enumerate(ids) if value == token(io)]
        subject_pos = [p for p, value in enumerate(ids) if value == token(subject)]
        assert len(io_pos) == 1 and len(subject_pos) == (1 if s2 else 2)
        second_pos = (
            next(p for p, value in enumerate(ids) if value == token(s2)) if s2 else subject_pos[1]
        )
        pid = hashlib.sha256(text.encode()).hexdigest()[:20]
        row = {
            "id": pid,
            "text": text,
            "token_ids": ids,
            "split": split,
            "template_id": f"{split}-t{ti}",
            "family": family,
            "io": io,
            "subject": subject,
            "io_token_id": token(io),
            "subject_token_id": token(subject),
            "positions": {
                "end": len(ids) - 1,
                "subject_second": second_pos,
                "subject_first": subject_pos[0],
                "io": io_pos[0],
            },
        }
        if pid in library and library[pid] != row:
            raise ValueError("Prompt identity collision")
        library[pid] = row
        return pid

    for split, templates in TEMPLATES.items():
        for ti in range(len(templates)):
            for pi, pair in enumerate(NAMES[split]):
                for role in range(2):
                    io, subject = pair[role], pair[1 - role]
                    alternative = NAMES[split][(pi + 1) % len(NAMES[split])][role]
                    assert len({io, subject, alternative}) == 3
                    for family in ("ABBA", "BABA"):
                        rid = f"{split}-t{ti}-n{pi}-r{role}-{family}"
                        pid = prompt(split, ti, family, io, subject)
                        cid = prompt(split, ti, family, io, subject, alternative)
                        donor_ids = {}
                        for wording in ("same", "other"):
                            donor_t = ti if wording == "same" else (ti + 1) % len(templates)
                            for fam in ("same", "cross"):
                                donor_f = (
                                    family
                                    if fam == "same"
                                    else ("BABA" if family == "ABBA" else "ABBA")
                                )
                                for answer in ("same", "different"):
                                    donor_io = io if answer == "same" else alternative
                                    key = f"{wording}_{fam}_{answer}"
                                    donor_ids[key] = prompt(
                                        split, donor_t, donor_f, donor_io, subject
                                    )
                        r = {
                            **library[pid],
                            "id": rid,
                            "prompt_id": pid,
                            "corrupt_id": cid,
                            "name_pair_id": f"{split}-n{pi}",
                            "role_assignment": role,
                            "alternative_answer": alternative,
                            "alternative_answer_token_id": token(alternative),
                            "donors": donor_ids,
                        }
                        assert library[pid]["token_ids"] != library[cid]["token_ids"]
                        assert len(library[pid]["token_ids"]) == len(library[cid]["token_ids"])
                        assert donor_ids["same_same_same"] == pid
                        recipients.append(r)
    assert len(recipients) == 256
    vocab = [{name for pair in pairs for name in pair} for pairs in NAMES.values()]
    assert not any(vocab[i] & vocab[j] for i in range(3) for j in range(i))
    return recipients, library
