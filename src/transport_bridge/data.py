"""New fixed tokens and full donor maps; no inference-based filtering."""

from transformers import AutoTokenizer

from .common import ROOT, read, write

NAMES = {
    "discovery": ["Aaron", "Adam", "Adrian", "Alan", "Albert", "Alex", "Alfred", "Andrew"],
    "development": [
        "Anthony",
        "Austin",
        "Barry",
        "Benjamin",
        "Bernard",
        "Billy",
        "Bobby",
        "Brandon",
    ],
    "test": ["Bruce", "Bryan", "Calvin", "Carl", "Chad", "Chris", "Christian", "Christopher"],
}
LOCATIONS = [
    "museum",
    "station",
    "church",
    "stadium",
    "castle",
    "hospital",
    "factory",
    "studio",
    "village",
    "palace",
    "cafe",
    "hall",
    "cinema",
    "garage",
    "harbor",
    "temple",
]
OBJECTS = [
    "painting",
    "newspaper",
    "candle",
    "flag",
    "coin",
    "blanket",
    "tool",
    "photo",
    "basket",
    "ring",
    "sandwich",
    "card",
    "poster",
    "rope",
    "box",
    "bowl",
]
PREFIXES = ["When", "After", "Then", "Once", "Before", "Later", "Soon", "Yesterday"] * 2
COLORS = ["silver", "golden", "orange", "green", "yellow", "black", "white", "blue"] * 2
SHAPES = ["triangle", "circle", "square", "column", "pattern", "line", "symbol", "figure"] * 2


def build():
    tok = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt2-medium", local_files_only=True
    )
    rows, library, templates = [], {}, []

    def one(s):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token: {s!r}: {ids}")
        return ids[0]

    def prompt(ti, wording, order, io, subject, alt, variant):
        first, second = (io, subject) if order == 0 else (subject, io)
        repeated = alt if variant == "corrupt" else subject
        if wording == 0:
            fixed = [
                PREFIXES[ti],
                " and",
                " stopped",
                " inside",
                " the",
                " " + LOCATIONS[ti],
                ",",
                " offered",
                " a",
                " " + OBJECTS[ti],
                " to",
            ]
        else:
            fixed = [
                PREFIXES[ti],
                " beside",
                " counted",
                " every",
                " " + COLORS[ti],
                " " + SHAPES[ti],
                ";",
                " moved",
                " one",
                " marker",
                " toward",
            ]
        chunks = [
            fixed[0],
            " " + first,
            fixed[1],
            " " + second,
            *fixed[2:7],
            " " + repeated,
            *fixed[7:],
        ]
        ids = [tok.bos_token_id] + [one(s) for s in chunks]
        text = "".join(chunks)
        assert ids[1:] == tok.encode(text, add_special_tokens=False)
        key = f"t{ti}-w{wording}-o{order}-{io}-{subject}-{variant}-{alt}"
        library[key] = {
            "id": key,
            "token_ids": ids,
            "text": text,
            "io_id": one(" " + io),
            "subject_id": one(" " + subject),
            "alt_id": one(" " + alt),
            "s2": 10,
        }
        return key

    ti = 0
    for split, count in [("discovery", 4), ("development", 4), ("test", 8)]:
        for _ in range(count):
            templates.append(
                {
                    "id": ti,
                    "split": split,
                    "location": LOCATIONS[ti],
                    "object": OBJECTS[ti],
                    "color": COLORS[ti],
                    "shape": SHAPES[ti],
                }
            )
            names = NAMES[split]
            for pair in range(4):
                for role in range(2):
                    io, subject = names[2 * pair + role], names[2 * pair + 1 - role]
                    alt = names[(2 * ((pair + 1) % 4)) + role]
                    for wording in range(2):
                        for order in range(2):
                            clean = prompt(ti, wording, order, io, subject, alt, "clean")
                            corrupt = prompt(ti, wording, order, io, subject, alt, "corrupt")
                            changed = prompt(ti, wording, order, alt, subject, io, "clean")
                            rid = f"{split}-t{ti}-p{pair}-r{role}-w{wording}-o{order}"
                            rows.append(
                                {
                                    **library[clean],
                                    "id": rid,
                                    "split": split,
                                    "template": ti,
                                    "pair": pair,
                                    "role": role,
                                    "wording": wording,
                                    "order": order,
                                    "io": io,
                                    "subject": subject,
                                    "alternative": alt,
                                    "clean": clean,
                                    "corrupt": corrupt,
                                    "changed": changed,
                                }
                            )
            ti += 1
    index = {(r["template"], r["pair"], r["role"], r["wording"], r["order"]): r for r in rows}
    for r in rows:
        r["donors"] = {}
        for factor in ["wording", "order"]:
            w = 1 - r["wording"] if factor == "wording" else r["wording"]
            o = 1 - r["order"] if factor == "order" else r["order"]
            donor = index[(r["template"], r["pair"], r["role"], w, o)]
            r["donors"][factor] = donor["id"]
            r["donors"][factor + "_changed"] = donor["changed"]
    # Verify lexical disjointness against retained earlier data, without model outcomes.
    previous = set()
    for p in (ROOT / "studies/transport/data").glob("*manifest.jsonl"):
        import json

        for line in p.read_text().splitlines():
            old = json.loads(line)
            previous.update(old[k] for k in ["io_name", "subject_name", "corrupt_name"] if k in old)
    for r in read(ROOT / "studies/routing/data/study-002-recipients.json"):
        previous.update([r["io"], r["subject"], r["alternative_answer"]])
    assert not previous.intersection(n for group in NAMES.values() for n in group)
    assert len({r["id"] for r in rows}) == 512
    assert {len(p["token_ids"]) for p in library.values()} == {15}
    write(ROOT / "data/recipients.json", rows)
    write(ROOT / "data/prompts.json", library)
    write(ROOT / "data/templates.json", templates)
    return rows, library
