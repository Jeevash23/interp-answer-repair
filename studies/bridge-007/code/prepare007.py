"""Build Bridge 007's fixed fresh banks using tokenization and prior inputs only.

No model weights or scientific outputs are loaded. Existing study data are read
only to exclude earlier names, context pairs and exact prompt strings. Running
without --write reports feasibility; --write refuses to replace existing data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
import string
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SEED = 2026100407
MODEL = "gpt2-medium"
FAMILIES = {
    10: "In the {place}, {a} and {b} talked. {s} handed {article} {obj} to",
    11: "While {a} and {b} stood in the {place}, {s} passed {article} {obj} to",
    12: "Near the {place}, {a} and {b} waited. {s} gave {article} {obj} to",
    13: "When {a} and {b} stopped at the {place}, {s} showed {article} {obj} to",
}
PAIRS = [[10, 11], [10, 13], [11, 12], [12, 13]]
NAME_COUNTS = dict(calibration=6, screen=12, fit=24, selection=24,
                   qualification=36, test=60)
BLOCK_COUNTS = dict(calibration=2, screen=3, fit=8, selection=8,
                    qualification=12, test=20)
# All candidate blocks are declared before model inference. Repeated individual
# words are allowed; exact location/object pairs must be unused in all studies.
CONTEXT_POOL = [
    ("museum", "brochure"), ("library", "bookmark"), ("market", "pear"),
    ("station", "wallet"), ("school", "ruler"), ("theater", "flower"),
    ("hotel", "envelope"), ("airport", "magazine"), ("beach", "shell"),
    ("clinic", "mask"), ("gallery", "catalog"), ("office", "stapler"),
    ("park", "kite"), ("restaurant", "menu"), ("shop", "receipt"),
    ("zoo", "ticket"), ("castle", "shield"), ("farm", "apple"),
    ("garage", "brush"), ("harbor", "rope"), ("temple", "candle"),
    ("tower", "map"), ("cinema", "cup"), ("courthouse", "folder"),
    ("aquarium", "postcard"), ("warehouse", "box"), ("stadium", "cap"),
    ("cafe", "cookie"), ("laboratory", "notebook"), ("campsite", "lantern"),
    ("bakery", "roll"), ("workshop", "screw"), ("palace", "ring"),
    ("quarry", "hammer"), ("bookstore", "bag"), ("observatory", "lens"),
    ("greenhouse", "glove"), ("abbey", "book"), ("attic", "doll"),
    ("balcony", "drink"), ("barracks", "letter"), ("bistro", "plate"),
    ("boathouse", "bottle"), ("boutique", "scarf"), ("brewery", "mug"),
    ("bunker", "radio"), ("cabin", "blanket"), ("cathedral", "card"),
    ("cellar", "basket"), ("chapel", "medal"), ("chateau", "key"),
    ("clubhouse", "towel"), ("conservatory", "seed"), ("cottage", "pencil"),
    ("courtyard", "ball"), ("deli", "sandwich"), ("depot", "parcel"),
    ("dormitory", "lamp"), ("embassy", "pen"), ("fortress", "coin"),
    ("gazebo", "fan"), ("gymnasium", "whistle"), ("hangar", "glove"),
    ("hostel", "passport"), ("lighthouse", "compass"), ("lodge", "watch"),
    ("mansion", "photo"), ("monastery", "ribbon"), ("orchard", "basket"),
    ("pavilion", "hat"), ("plaza", "booklet"), ("tavern", "jug"),
]
NAME_KEYS = {"io", "subject", "alternative", "io_name", "subject_name",
             "corrupt_name", "name_a", "name_b", "name_c"}
TEXT_KEYS = {"text", "prompt", "clean_text", "corrupt_text",
             "alternative_corrupt_text"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def name_literal(path, symbol):
    """Read the previously declared name lists without importing any engine."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == symbol for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                assert node.value.func.attr == "split" and not node.value.args
                return ast.literal_eval(node.value.func.value).split()
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing declared name source {path}:{symbol}")


def prior_inputs():
    excluded, contexts, texts, sources = set(), set(), set(), {}

    def visit(value):
        if isinstance(value, list):
            for element in value:
                visit(element)
        elif isinstance(value, dict):
            location = value.get("location", value.get("place"))
            obj = value.get("object", value.get("obj"))
            if isinstance(location, str) and isinstance(obj, str):
                contexts.add((location, obj))
            for key, element in value.items():
                if key in NAME_KEYS and isinstance(element, str):
                    excluded.add(element)
                elif key in TEXT_KEYS and isinstance(element, str):
                    texts.add(element)
                elif key in {"excluded_names", "excluded_prior_names"}:
                    excluded.update(element)
                elif key == "names" and isinstance(element, dict):
                    for bank in element.values():
                        excluded.update(bank)
                elif key == "blocks" and isinstance(element, dict):
                    for bank in element.values():
                        for block in bank:
                            if isinstance(block, list) and len(block) == 2:
                                contexts.add(tuple(block))
                else:
                    visit(element)

    directories = [ROOT / "data"] + [
        path / "data" for path in sorted((ROOT / "studies").iterdir())
        if path.is_dir() and path != HERE and (path / "data").is_dir()
    ]
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".json", ".jsonl"}:
                continue
            sources[str(path.relative_to(ROOT))] = digest(path)
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip():
                        visit(json.loads(line))
            else:
                visit(read(path))

    # Early transport/routing manifests store contexts only inside prompt text.
    # Conservatively reject a proposed pair if both words occur in any old text.
    for block in CONTEXT_POOL:
        location_re = re.compile(r"\b" + re.escape(block[0]) + r"\b")
        object_re = re.compile(r"\b" + re.escape(block[1]) + r"\b")
        if any(location_re.search(text) and object_re.search(text) for text in texts):
            contexts.add(block)
    return excluded, contexts, texts, sources


def tokenize(template, values, tokenizer):
    text, spans = "", {}
    for literal, key, _, _ in string.Formatter().parse(template):
        text += literal
        if key:
            start = len(text)
            text += values[key]
            spans[key] = (start, len(text))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [tokenizer.bos_token_id] + encoded["input_ids"]
    positions = {}
    for key in ("a", "b", "s"):
        start, end = spans[key]
        hits = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                if right > start and left < end]
        assert len(hits) == 1, (key, text, hits)
        positions[key] = hits[0] + 1
        assert ids[positions[key]] == tokenizer.encode(
            " " + values[key], add_special_tokens=False)[0]
    return text, ids, positions


def build():
    excluded, old_contexts, old_texts, sources = prior_inputs()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / MODEL,
                                              local_files_only=True)
    name_sources = [
        (ROOT / "studies/bridge-003/code/build003.py", "NAMES"),
        (ROOT / "studies/bridge-004/code/build004.py", "EXTRA"),
        (ROOT / "studies/bridge-005/code/build005.py", "EXTRA_NEW"),
    ]
    declared_names = set(itertools.chain.from_iterable(
        name_literal(path, symbol) for path, symbol in name_sources))
    pool = sorted(name for name in declared_names - excluded
                  if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1)
    assert len(pool) >= sum(NAME_COUNTS.values()), len(pool)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pool)
    names, start = {}, 0
    for split, count in NAME_COUNTS.items():
        names[split] = pool[start:start + count]
        start += count

    context_pool = [block for block in CONTEXT_POOL if block not in old_contexts]
    assert len(context_pool) >= sum(BLOCK_COUNTS.values()), len(context_pool)
    assert len({place for place, obj in CONTEXT_POOL}) == len(CONTEXT_POOL)
    # Separate deterministic random stream keeps allocation independent of the
    # number of tokenizer-eligible names in the declared name pool.
    context_rng = np.random.default_rng(SEED + 1)
    context_order = context_rng.permutation(len(context_pool))
    shuffled_contexts = [context_pool[i] for i in context_order]
    blocks, start = {}, 0
    for split, count in BLOCK_COUNTS.items():
        blocks[split] = shuffled_contexts[start:start + count]
        start += count

    all_names = list(itertools.chain.from_iterable(names.values()))
    all_blocks = list(itertools.chain.from_iterable(blocks.values()))
    assert len(set(all_names)) == sum(NAME_COUNTS.values())
    assert not set(all_names) & excluded
    assert len(set(all_blocks)) == sum(BLOCK_COUNTS.values())
    assert not set(all_blocks) & old_contexts

    rows, new_prompts = [], set()
    for split, bank in names.items():
        for ti, (place, obj) in enumerate(blocks[split]):
            for pi in range(len(bank) // 3):
                triple = bank[pi * 3:pi * 3 + 3]
                for role in range(2):
                    io, subject, alternative = triple[role], triple[1-role], triple[2]
                    for family, template in FAMILIES.items():
                        for order in range(2):
                            a, b = (io, subject) if order == 0 else (subject, io)
                            article = "an" if obj[0].lower() in "aeiou" else "a"
                            text, tokens, positions = tokenize(template, dict(
                                a=a, b=b, s=subject, place=place, obj=obj,
                                article=article), tokenizer)
                            corrupt_tokens = tokens.copy()
                            alt_id = tokenizer.encode(" " + alternative,
                                                      add_special_tokens=False)[0]
                            corrupt_tokens[positions["s"]] = alt_id
                            corrupt_text = tokenizer.decode(corrupt_tokens[1:])
                            assert text not in old_texts and corrupt_text not in old_texts
                            assert text not in new_prompts
                            new_prompts.add(text)
                            assert sum(x != y for x, y in zip(tokens, corrupt_tokens)) == 1
                            rows.append(dict(
                                id=f"b007-{split}-t{ti}-p{pi}-r{role}-f{family}-o{order}",
                                split=split, template=ti, pair=pi, role=role,
                                family=family, wording=family, order=order, io=io,
                                subject=subject, alternative=alternative, text=text,
                                token_ids=tokens, corrupt_tokens=corrupt_tokens,
                                s2=positions["s"], io_position=positions["a" if order == 0 else "b"],
                                subject_position=positions["b" if order == 0 else "a"],
                                io_id=tokenizer.encode(" " + io, add_special_tokens=False)[0],
                                subject_id=tokenizer.encode(" " + subject, add_special_tokens=False)[0],
                                alt_id=alt_id,
                            ))

    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("split", "template", "pair", "role", "order"))
        groups.setdefault(key, {})[row["family"]] = row
    alignment = []
    for fa, fb in PAIRS:
        assert all(len(g[fa]["token_ids"]) == len(g[fb]["token_ids"])
                   and g[fa]["s2"] == g[fb]["s2"] for g in groups.values()), (fa, fb)
        alignment.append([fa, fb])

    design = dict(
        study="bridge-007", seed=SEED, model=MODEL,
        names=names, blocks=blocks, name_counts=NAME_COUNTS, block_counts=BLOCK_COUNTS,
        family_ids=list(FAMILIES), families=FAMILIES, candidate_pairs=PAIRS,
        article_rule="an before a vowel-initial object; a otherwise; fixed context pool has no ambiguous initials",
        alignment_eligible_pairs=alignment, excluded_names=sorted(excluded),
        tokenization_pool=len(pool), unused_eligible_names=len(pool)-len(all_names),
        context_pool=CONTEXT_POOL, context_pool_eligible=len(context_pool),
        context_pool_rejected=[b for b in CONTEXT_POOL if b in old_contexts],
        prior_input_sources=sources,
        name_pool_sources={str(p.relative_to(ROOT)): digest(p) for p, _ in name_sources},
        tokenizer_files={str(p.relative_to(ROOT)): digest(p)
                         for p in sorted((ROOT / "models" / MODEL).iterdir())
                         if p.name in {"tokenizer.json", "tokenizer_config.json",
                                       "special_tokens_map.json", "vocab.json", "merges.txt"}},
        scope="Four fixed new wordings and four fixed cross-structure pairs; names and exact context pairs unused in prior project studies; individual context words can recur. No scientific model inference or behavior-based filtering.",
    )
    audit = dict(
        status="passed", utc=datetime.now(timezone.utc).isoformat(),
        scientific_inference_performed=False, rows=len(rows),
        prior_json_sources=len(sources), prior_names=len(excluded),
        prior_prompt_strings=len(old_texts), prior_context_pairs=len(old_contexts),
        names_disjoint_from_prior_and_between_banks=True,
        context_pairs_disjoint_from_prior_and_between_banks=True,
        selected_location_words_disjoint_between_banks=True,
        clean_and_corrupted_prompts_disjoint_from_prior=True,
        all_clean_prompts_unique=True, single_token_names=True,
        one_token_corruption=True, semantic_positions_valid=True,
        alignment_eligible_pairs=alignment,
        rows_per_split={split: sum(r["split"] == split for r in rows)
                        for split in NAME_COUNTS},
        rows_per_selected_pair={split: BLOCK_COUNTS[split] * (NAME_COUNTS[split]//3) * 8
                                for split in NAME_COUNTS},
        prepare_source_sha256=digest(Path(__file__)),
    )
    return design, rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Write the approved fixed design, candidates and audit once.")
    args = parser.parse_args()
    outputs = [HERE / "data" / name for name in
               ("design.json", "candidates.json", "preparation-audit.json")]
    if args.write:
        assert not any(path.exists() for path in outputs), "Refusing to overwrite prepared data"
    design, rows, audit = build()
    if args.write:
        for path, value in zip(outputs, (design, rows, audit)):
            write_new(path, value)
    print(json.dumps(dict(wrote=args.write, eligible_names=design["tokenization_pool"],
                          eligible_contexts=design["context_pool_eligible"], **audit), indent=2))


if __name__ == "__main__":
    main()
