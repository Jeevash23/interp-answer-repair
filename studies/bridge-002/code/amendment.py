"""Second behavior-only screen; original failed screen remains immutable."""

from transformers import AutoTokenizer

from transport_bridge.common import ROOT, digest, now, write
from design import BLOCKS, HERE, TRIPLES


def build():
    path = HERE / "data/candidates-v2.json"
    if path.exists():
        raise FileExistsError(path)
    families = {
        1: ["After", " walked", " into", " handed"],
        4: ["When", " arrived", " at", " handed"],
        5: ["While", " waited", " inside", " handed"],
        6: ["When", " walked", " through", " handed"],
    }
    triples = dict(
        TRIPLES,
        screen=[
            ("Fred", "Grace", "Justin"),
            ("Martin", "Melissa", "Nicole"),
            ("Paul", "Rebecca", "Roger"),
            ("Tim", "Susan", "Victor"),
        ],
    )
    blocks = dict(
        BLOCKS,
        screen=[
            ("station", "newspaper"),
            ("church", "candle"),
            ("hospital", "blanket"),
            ("studio", "photo"),
        ],
    )
    tok = AutoTokenizer.from_pretrained(
        ROOT / "models/gpt2-medium", local_files_only=True
    )

    def one(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        return ids[0]

    rows = []
    for split in blocks:
        for t, (place, obj) in enumerate(blocks[split]):
            for pair, names in enumerate(triples[split]):
                for role in range(2):
                    io, sub, alt = names[role], names[1 - role], names[2]
                    for f, (prefix, motion, prep, verb) in families.items():
                        for order in range(2):
                            a, b = (io, sub) if order == 0 else (sub, io)
                            chunks = [
                                prefix,
                                " " + a,
                                " and",
                                " " + b,
                                motion,
                                prep,
                                " the",
                                " " + place,
                                ",",
                                " " + sub,
                                verb,
                                " a",
                                " " + obj,
                                " to",
                            ]
                            tokens = [tok.bos_token_id] + [one(c) for c in chunks]
                            assert tokens[1:] == tok.encode("".join(chunks))
                            corrupt = list(tokens)
                            corrupt[10] = one(" " + alt)
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
                                    "subject": sub,
                                    "alternative": alt,
                                    "io_id": one(" " + io),
                                    "subject_id": one(" " + sub),
                                    "alt_id": one(" " + alt),
                                    "s2": 10,
                                    "io_position": 2 if order == 0 else 4,
                                    "subject_position": 4 if order == 0 else 2,
                                    "text": "".join(chunks),
                                    "token_ids": tokens,
                                    "corrupt_tokens": corrupt,
                                }
                            )
    assert len({n for bank in triples.values() for triple in bank for n in triple}) == 42
    write(path, rows)
    write(
        HERE / "data/screen-freeze-v2.json",
        {
            "utc": now(),
            "files": {
                str(p.relative_to(HERE)): digest(p)
                for p in [
                    path,
                    HERE / "code/amendment.py",
                    HERE / "code/screen_v2.py",
                    HERE / "docs/SCREENING-AMENDMENT-001.md",
                ]
            },
        },
    )
    print("Frozen second screening stage:", len(rows), "candidate cells")


if __name__ == "__main__":
    build()
