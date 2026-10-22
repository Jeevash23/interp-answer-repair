"""Fixed grammatical candidates and disjoint corruption triples."""

from transformers import AutoTokenizer

from transport_bridge.common import ROOT, digest, now, read, write

HERE = ROOT / "studies/bridge-002"
SEED = 2026090802
FAMILIES = [
    ["When", " went", " to", " gave"],
    ["After", " walked", " into", " handed"],
    ["When", " walked", " to", " passed"],
    ["After", " went", " to", " offered"],
]
TRIPLES = {
    "screen": [
        ("Daniel", "Emily", "Frank"),
        ("George", "Hannah", "Jack"),
        ("James", "Julia", "Karen"),
        ("Kevin", "Laura", "Mark"),
    ],
    "discovery": [
        ("Michael", "Nancy", "Oliver"),
        ("Patrick", "Rachel", "Richard"),
        ("Robert", "Rose", "Ruth"),
        ("Ryan", "Samantha", "Scott"),
    ],
    "test": [
        ("Sean", "Sophie", "Stephen"),
        ("Thomas", "Victoria", "William"),
        ("David", "Diana", "Dorothy"),
        ("Edward", "Emma", "Elizabeth"),
        ("Henry", "Helen", "Jennifer"),
        ("Jason", "Jessica", "Julie"),
    ],
}
BLOCKS = {
    "screen": [("store", "book"), ("mall", "ball"), ("park", "drink"), ("school", "letter")],
    "discovery": [("market", "gift"), ("office", "file"), ("hotel", "key"), ("restaurant", "menu")],
    "test": [
        ("bank", "ticket"),
        ("garden", "toy"),
        ("library", "note"),
        ("beach", "bottle"),
        ("airport", "bag"),
        ("theater", "flower"),
        ("shop", "apple"),
        ("museum", "camera"),
    ],
}


def build():
    path = HERE / "data/candidates.json"
    if path.exists():
        raise FileExistsError(path)
    tok = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt2-medium", local_files_only=True
    )

    def one(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        return ids[0]

    rows = []
    for split, blocks in BLOCKS.items():
        for t, (place, obj) in enumerate(blocks):
            for pair, names in enumerate(TRIPLES[split]):
                for role in range(2):
                    io, subject, alt = names[role], names[1 - role], names[2]
                    for f, (prefix, motion, prep, verb) in enumerate(FAMILIES):
                        for order in range(2):
                            first, second = (io, subject) if order == 0 else (subject, io)
                            chunks = [
                                prefix,
                                " " + first,
                                " and",
                                " " + second,
                                motion,
                                prep,
                                " the",
                                " " + place,
                                ",",
                                " " + subject,
                                verb,
                                " a",
                                " " + obj,
                                " to",
                            ]
                            tokens = [tok.bos_token_id] + [one(c) for c in chunks]
                            assert tokens[1:] == tok.encode("".join(chunks))
                            corrupted = list(tokens)
                            corrupted[10] = one(" " + alt)
                            rows.append(
                                {
                                    "id": f"{split}-t{t}-p{pair}-r{role}-f{f}-o{order}",
                                    "split": split,
                                    "template": t,
                                    "pair": pair,
                                    "role": role,
                                    "family": f,
                                    "order": order,
                                    "io": io,
                                    "subject": subject,
                                    "alternative": alt,
                                    "io_id": one(" " + io),
                                    "subject_id": one(" " + subject),
                                    "alt_id": one(" " + alt),
                                    "s2": 10,
                                    "io_position": 2 if order == 0 else 4,
                                    "subject_position": 4 if order == 0 else 2,
                                    "text": "".join(chunks),
                                    "token_ids": tokens,
                                    "corrupt_tokens": corrupted,
                                }
                            )
    old_names = {
        r[k] for r in read(ROOT / "data/recipients.json") for k in ["io", "subject", "alternative"]
    }
    new_names = {n for triples in TRIPLES.values() for triple in triples for n in triple}
    assert not new_names & old_names
    assert sum(len(v) * 3 for v in TRIPLES.values()) == len(new_names)
    # Only clean task validity is allowed to inspect screening candidates.
    write(path, rows)
    previous = dict(read(ROOT / "data/previous-project-hashes.json"))
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if (
            not p.is_file()
            or rel.parts[0] in {"studies", ".venv", ".uv-cache", ".pytest_cache", ".ruff_cache"}
            or "__pycache__" in rel.parts
            or p.suffix == ".pyc"
        ):
            continue
        previous[str(p.relative_to(ROOT.parent))] = digest(p)
    write(HERE / "data/prior-files.json", previous)
    freeze = {
        "utc": now(),
        "files": {
            str(p.relative_to(HERE)): digest(p)
            for p in [
                path,
                HERE / "code/design.py",
                HERE / "code/screen.py",
                HERE / "docs/PROTOCOL.md",
            ]
        },
    }
    write(HERE / "data/screen-freeze.json", freeze)
    print(f"Saved {len(rows)} candidate cells before any screening outcome")


def selected_rows():
    selected = read(HERE / "data/family-selection.json")["families"]
    rows = [
        dict(r, wording=selected.index(r["family"]))
        for r in read(HERE / "data/candidates.json")
        if r["family"] in selected
    ]
    index = {
        (r["split"], r["template"], r["pair"], r["role"], r["wording"], r["order"]): r for r in rows
    }
    for r in rows:
        r["donors"] = {}
        for label, dw, do, dr in [
            ("self", 0, 0, 0),
            ("role", 0, 0, 1),
            ("wording", 1, 0, 0),
            ("wording_role", 1, 0, 1),
            ("order", 0, 1, 0),
            ("order_role", 0, 1, 1),
        ]:
            d = index[
                (
                    r["split"],
                    r["template"],
                    r["pair"],
                    r["role"] ^ dr,
                    r["wording"] ^ dw,
                    r["order"] ^ do,
                )
            ]
            r["donors"][label] = d["id"]
    return rows


def specs():
    out = [
        {"name": "head_" + f, "kind": "head", "donor": f, "dose": 1.0}
        for f in ["self", "role", "wording", "wording_role", "order", "order_role"]
    ]
    out += [
        {"name": "head_" + f + "_half", "kind": "head", "donor": f, "dose": 0.5}
        for f in ["wording", "order"]
    ]
    out += [
        {"name": f"random_{i}", "kind": "random", "donor": "wording", "draw": i, "dose": 1.0}
        for i in range(3)
    ]
    out += [
        {"name": "full_" + f, "kind": "full", "donor": f, "dose": 1.0} for f in ["wording", "order"]
    ]
    return out


if __name__ == "__main__":
    build()
