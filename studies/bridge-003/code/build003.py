"""Tokenization-only construction and screening-input freeze, before inference."""

import string

import numpy as np
from common003 import HERE, MODELS, OLD, ROOT, SEED, config, digest, freeze, read, write
from transformers import AutoTokenizer

FAMILIES = [
    "After {a} and {b} walked into the {place}, {s} handed a {obj} to",
    "When {a} and {b} went to the {place}, {s} gave a {obj} to",
    "When {a} and {b} walked to the {place}, {s} passed a {obj} to",
    "After {a} and {b} visited the {place}, {s} gave the {obj} to",
    "After {a} and {b} met at the {place}, {s} lent a {obj} to",
    "While {a} and {b} waited in the {place}, {s} brought a {obj} to",
    "At the {place}, {a} and {b} met. {s} gave a {obj} to",
    "While {a} and {b} were at the {place}, {s} showed a {obj} to",
]
ARM_A = [FAMILIES[0], "When {a} and {b} arrived at the {place}, {s} handed a {obj} to"]
BLOCKS = {
    "medium": [
        ("zoo", "coin"),
        ("castle", "cup"),
        ("farm", "hat"),
        ("garage", "rope"),
        ("harbor", "box"),
        ("temple", "ring"),
        ("tower", "flag"),
        ("cinema", "watch"),
    ],
    "screen": [("courthouse", "map"), ("aquarium", "card")],
    "discovery": [
        ("warehouse", "package"),
        ("stadium", "scarf"),
        ("cafe", "pen"),
        ("laboratory", "sample"),
    ],
    "test": [
        ("campsite", "compass"),
        ("bakery", "cake"),
        ("workshop", "hammer"),
        ("palace", "medal"),
        ("quarry", "stone"),
        ("bookstore", "poster"),
        ("observatory", "chart"),
        ("greenhouse", "plant"),
    ],
}
NAMES = [
    "Alice",
    "Allan",
    "Amanda",
    "Amy",
    "Angela",
    "Ann",
    "Anna",
    "Anne",
    "Arthur",
    "Ashley",
    "Barbara",
    "Betty",
    "Beverly",
    "Bill",
    "Bob",
    "Brenda",
    "Brian",
    "Carol",
    "Caroline",
    "Catherine",
    "Charles",
    "Charlotte",
    "Christina",
    "Christine",
    "Claire",
    "Clara",
    "Craig",
    "Dale",
    "Dan",
    "Danny",
    "Deborah",
    "Dennis",
    "Diane",
    "Donald",
    "Douglas",
    "Dylan",
    "Earl",
    "Elaine",
    "Ellen",
    "Eric",
    "Ernest",
    "Ethan",
    "Eugene",
    "Florence",
    "Frances",
    "Francis",
    "Gary",
    "Gerald",
    "Gilbert",
    "Gloria",
    "Gordon",
    "Gregory",
    "Harold",
    "Harry",
    "Heather",
    "Howard",
    "Jacob",
    "Jane",
    "Janet",
    "Jean",
    "Jeffrey",
    "Jeremy",
    "Jerry",
    "Jesse",
    "Jim",
    "Jimmy",
    "Joan",
    "Joe",
    "Joel",
    "John",
    "Jonathan",
    "Jordan",
    "Joseph",
    "Joyce",
    "Juan",
    "Judy",
    "Judith",
    "Keith",
    "Kenneth",
    "Kim",
    "Larry",
    "Lawrence",
    "Lee",
    "Leo",
    "Leonard",
    "Leslie",
    "Lewis",
    "Linda",
    "Lisa",
    "Logan",
    "Louis",
    "Louise",
    "Lucy",
    "Luke",
    "Lynn",
    "Margaret",
    "Maria",
    "Marie",
    "Martha",
    "Mary",
    "Matthew",
    "Megan",
    "Michelle",
    "Monica",
    "Nathan",
    "Nicholas",
    "Nina",
    "Noah",
    "Norman",
    "Olivia",
    "Oscar",
    "Patricia",
    "Peter",
    "Philip",
    "Phillip",
    "Ralph",
    "Ray",
    "Raymond",
    "Rita",
    "Ronald",
    "Roy",
    "Russell",
    "Sally",
    "Samuel",
    "Sandra",
    "Sara",
    "Sarah",
    "Sharon",
    "Shirley",
    "Stanley",
    "Stephanie",
    "Steve",
    "Steven",
    "Teresa",
    "Terry",
    "Theresa",
    "Tina",
    "Todd",
    "Tom",
    "Tommy",
    "Tony",
    "Travis",
    "Troy",
    "Tyler",
    "Vincent",
    "Virginia",
    "Walter",
    "Warren",
    "Wayne",
    "Wendy",
    "Wesley",
    "Will",
    "Willie",
]


def tokenize(template, values, tok):
    text, spans = "", {}
    for literal, key, _, _ in string.Formatter().parse(template):
        text += literal
        if key:
            start = len(text)
            text += values[key]
            spans[key] = (start, len(text))
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tok.bos_token_id] + encoded["input_ids"]
    positions = {}
    for key in ["a", "b", "s"]:
        start, end = spans[key]
        hit = [
            i
            for i, (left, right) in enumerate(encoded["offset_mapping"])
            if right > start and left < end
        ]
        assert len(hit) == 1, (key, text, hit)
        positions[key] = hit[0] + 1
        assert ids[hit[0] + 1] == tok.encode(" " + values[key], add_special_tokens=False)[0]
    return text, ids, positions


def build():
    assert not (HERE / "data/design.json").exists()
    tokenizers = {
        m: AutoTokenizer.from_pretrained(
            ROOT / "models" / m, local_files_only=True
        )
        for m in MODELS
    }
    exclude = set()
    for path in [
        ROOT / "data/recipients.json",
        OLD / "data/candidates.json",
        OLD / "data/candidates-v2.json",
    ]:
        for r in read(path):
            exclude.update(r[k] for k in ["io", "subject", "alternative"])
    pool = [
        n
        for n in NAMES
        if n not in exclude
        and all(len(t.encode(" " + n, add_special_tokens=False)) == 1 for t in tokenizers.values())
    ]
    assert len(pool) >= 60 and len(pool) == len(set(pool))
    np.random.default_rng(SEED).shuffle(pool)
    banks = {
        "medium": pool[:18],
        "screen": pool[18:30],
        "discovery": pool[30:42],
        "test": pool[42:60],
    }
    design = {
        "families": FAMILIES,
        "medium_families": ARM_A,
        "blocks": BLOCKS,
        "names": banks,
        "excluded_prior_names": sorted(exclude),
        "selection": "Tokenization-only deterministic name sampling",
    }
    write(HERE / "data/design.json", design)
    for model, tok in tokenizers.items():
        rows = []
        for split, names in banks.items():
            if split == "medium" and model != "gpt2-medium":
                continue
            families = ARM_A if split == "medium" else FAMILIES
            for t, (place, obj) in enumerate(BLOCKS[split]):
                for pair in range(len(names) // 3):
                    triple = names[3 * pair : 3 * pair + 3]
                    for role in range(2):
                        io, subject, alt = triple[role], triple[1 - role], triple[2]
                        for f, template in enumerate(families):
                            for order in range(2):
                                a, b = (io, subject) if order == 0 else (subject, io)
                                text, tokens, pos = tokenize(
                                    template,
                                    {"a": a, "b": b, "s": subject, "place": place, "obj": obj},
                                    tok,
                                )
                                corrupt = list(tokens)
                                alt_id = tok.encode(" " + alt, add_special_tokens=False)[0]
                                corrupt[pos["s"]] = alt_id
                                rows.append(
                                    {
                                        "id": f"{split}-t{t}-p{pair}-r{role}-f{f}-o{order}",
                                        "split": split,
                                        "template": t,
                                        "pair": pair,
                                        "role": role,
                                        "family": f,
                                        "wording": f,
                                        "order": order,
                                        "io": io,
                                        "subject": subject,
                                        "alternative": alt,
                                        "text": text,
                                        "token_ids": tokens,
                                        "corrupt_tokens": corrupt,
                                        "s2": pos["s"],
                                        "io_position": pos["a" if order == 0 else "b"],
                                        "subject_position": pos["b" if order == 0 else "a"],
                                        "io_id": tok.encode(" " + io, add_special_tokens=False)[0],
                                        "subject_id": tok.encode(
                                            " " + subject, add_special_tokens=False
                                        )[0],
                                        "alt_id": alt_id,
                                    }
                                )
        write(HERE / "data" / (model + "-candidates.json"), rows)
    write(HERE / "data/config.json", config())
    prior = read(ROOT / "output/combined-artifact-manifest-v2.json")["files"]
    for path in (ROOT / "output").glob("*"):
        if path.is_file():
            prior[str(path.relative_to(ROOT.parent))] = digest(path)
    write(HERE / "data/prior-files.json", prior)
    checkpoints = {}
    for model in MODELS:
        for path in (ROOT / "models" / model).glob("*"):
            if path.is_file():
                checkpoints[str(path.relative_to(ROOT.parent))] = digest(path)
    write(HERE / "data/checkpoints.json", checkpoints)
    freeze(
        "input-freeze.json",
        [HERE / "code/build003.py"]
        + list((HERE / "docs").glob("*.md"))
        + list((HERE / "data").glob("*.json")),
    )
    print("Input design frozen; name banks", banks, flush=True)


if __name__ == "__main__":
    build()
