"""Build Study 7's banks by tokenisation only (no model weights or outputs are loaded).

Names: the declared lists of Bridges 003-005 plus NEW_NAMES below; single GPT-2 tokens; names
in any earlier study run on GPT-2 Large (the transport experiments) are excluded. The final-test
bank takes its 60 names from those used in no earlier AI-01 study. Contexts: declared pairs whose
exact pairing occurs in no earlier study. Running without --write reports feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'studies/bridge-007/code'))
import prepare007 as p7  # noqa: E402

p7.HERE = HERE  # every study except this one counts as prior input
NAME_SEED, CONTEXT_SEED = 2026101701, 2026101702
TOKENIZER = 'gpt2-large'
FAMILIES = {f: p7.FAMILIES[f] for f in (10, 11, 12, 13)}
PAIRS = [[10, 11], [10, 13], [11, 12], [12, 13]]
SPLITS = ('calibration', 'screen', 'development', 'fit', 'selection', 'qualification', 'test')
TRIPLES = dict(calibration=2, screen=4, development=6, fit=8, selection=8, qualification=12, test=20)
BLOCKS = dict(calibration=2, screen=3, development=6, fit=8, selection=8, qualification=12, test=20)
NEW_NAMES = """Aaron Abigail Adam Adrian Agnes Alan Albert Alex Alexander Alexis Alfred Alice Alicia Allen Alma Amanda Amber Amelia Amy Andrea
Andrew Angela Anita Ann Anna Anne Annie Anthony Antonio April Arthur Ashley Audrey Austin Barbara Barry Beatrice Ben Benjamin Bernard
Beth Betty Beverly Bill Billy Bob Bobby Bonnie Brad Bradley Brandon Brenda Brian Bruce Bryan Caleb Calvin Cameron Carl Carla Carlos
Carmen Carol Caroline Carolyn Carrie Catherine Cecil Charles Charlie Charlotte Chelsea Cheryl Chloe Chris Christina Christine Christopher
Cindy Claire Clara Clarence Claude Clifford Clyde Colin Connie Craig Crystal Curtis Cynthia Daisy Dale Dan Dana Daniel Danny Darren David
Dawn Dean Deborah Debra Denise Dennis Derek Diana Diane Dolores Don Donald Donna Dora Doris Dorothy Douglas Duncan Dylan Earl Eddie Edgar
Edith Edward Edwin Eileen Elaine Eleanor Elena Eli Elijah Elizabeth Ella Ellen Elliot Eloise Elsie Emily Emma Eric Erica Erin Ernest Esther
Ethan Eugene Eva Evan Evelyn Felix Fiona Florence Floyd Frances Francis Frank Fred Frederick Gabriel Gail Gary Gavin Gene George Georgia
Gerald Gilbert Gina Gladys Glen Glenn Gloria Gordon Grace Graham Grant Greg Gregory Gwen Hannah Harold Harriet Harry Harvey Hazel Heather
Hector Helen Henry Herbert Holly Howard Hugh Hugo Ian Ida Irene Iris Isaac Isabel Ivan Jack Jackie Jacob Jake James Jamie Jane Janet Janice
Jared Jason Jean Jeff Jeffrey Jennifer Jenny Jeremy Jerry Jesse Jessica Jill Jim Jimmy Joan Joanna Joe Joel John Johnny Jon Jonathan Jordan
Joseph Josh Joshua Joy Joyce Juan Judith Judy Julia Julian Julie June Justin Karen Karl Kate Katherine Kathleen Kathy Katie Keith Kelly Ken
Kenneth Kevin Kim Kirk Kyle Larry Laura Lauren Lawrence Leah Lee Leo Leon Leonard Leslie Lewis Lillian Lily Linda Lisa Lloyd Logan Lois
Lorraine Louis Louise Lucas Lucy Luke Lynn Mabel Madison Maggie Malcolm Marcus Margaret Maria Marian Marie Marilyn Mario Marion Mark Marsha
Martha Martin Marvin Mary Mason Matt Matthew Maureen Max Megan Melanie Melissa Melvin Meredith Michael Michelle Mike Mildred Miles Milton
Miranda Molly Monica Morgan Morris Nancy Naomi Natalie Nathan Neil Nelson Nicholas Nick Nicole Nina Noah Norma Norman Olive Oliver Olivia
Oscar Owen Pamela Patricia Patrick Paul Paula Pauline Pearl Peggy Penny Percy Peter Philip Phyllis Priscilla Rachel Ralph Randy Ray Raymond
Rebecca Regina Rex Rhonda Richard Rick Rita Rob Robert Roberta Robin Rodney Roger Ron Ronald Rose Ross Roy Ruby Russell Ruth Ryan Sally
Sam Samantha Samuel Sandra Sara Sarah Scott Sean Seth Shane Sharon Shawn Sheila Shirley Sidney Simon Sophia Sophie Stanley Stella Stephanie
Stephen Steve Steven Stuart Sue Susan Suzanne Sylvia Tammy Tanya Ted Teresa Terry Thelma Theodore Thomas Tim Timothy Tina Todd Tom Tony
Tracy Travis Trevor Troy Tyler Valerie Vanessa Vera Vernon Victor Victoria Vincent Viola Violet Virginia Vivian Wade Walter Wanda Warren
Wayne Wendy Wesley Wilbur Will William Willie Wilma Winston Yvonne Zach Zoe""".split()
CONTEXTS = [
    ('arcade', 'token'), ('armory', 'shield'), ('attic', 'trunk'), ('auditorium', 'program'), ('bakery', 'tray'),
    ('ballroom', 'glove'), ('bandstand', 'drum'), ('barbershop', 'razor'), ('basement', 'lamp'), ('boardroom', 'folder'),
    ('boathouse', 'oar'), ('bookshop', 'novel'), ('brewery', 'barrel'), ('bungalow', 'kettle'), ('carnival', 'balloon'),
    ('cavern', 'torch'), ('chapel', 'candle'), ('circus', 'hoop'), ('classroom', 'pencil'), ('cloister', 'bell'),
    ('college', 'diploma'), ('corridor', 'key'), ('cottage', 'quilt'), ('dairy', 'jug'), ('den', 'puzzle'),
    ('dock', 'crate'), ('dojo', 'belt'), ('estate', 'deed'), ('factory', 'badge'), ('field', 'shovel'),
    ('forest', 'axe'), ('foyer', 'coat'), ('gallery', 'frame'), ('garden', 'rake'), ('gatehouse', 'lantern'),
    ('greenhouse', 'pot'), ('hall', 'trophy'), ('harbor', 'anchor'), ('hospital', 'chart'), ('inn', 'blanket'),
    ('jail', 'cup'), ('kitchen', 'spoon'), ('laundry', 'basket'), ('library', 'journal'), ('lounge', 'magazine'),
    ('mall', 'bag'), ('mill', 'sack'), ('mine', 'pickaxe'), ('museum', 'painting'), ('nursery', 'rattle'),
    ('office', 'calculator'), ('palace', 'crown'), ('parlor', 'teacup'), ('pantry', 'jar'), ('pier', 'net'),
    ('playground', 'frisbee'), ('porch', 'broom'), ('prison', 'letter'), ('pub', 'coaster'), ('quarry', 'chisel'),
    ('ranch', 'saddle'), ('rectory', 'hymnal'), ('restaurant', 'fork'), ('school', 'eraser'), ('shed', 'ladder'),
    ('shelter', 'flashlight'), ('shop', 'purse'), ('square', 'flag'), ('stable', 'bridle'), ('studio', 'easel'),
    ('study', 'ledger'), ('suite', 'robe'), ('theater', 'script'), ('tent', 'mat'), ('terrace', 'vase'),
    ('tunnel', 'rope'), ('villa', 'mirror'), ('workshop', 'drill'), ('yard', 'hose'), ('zoo', 'camera'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def large_seen_names():
    """Names in any earlier study run on GPT-2 Large: the transport experiments (data or prompt text)."""
    found, texts = set(), []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in p7.NAME_KEYS and isinstance(item, str):
                    found.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for path in sorted((ROOT / 'studies/transport').rglob('*')):
        if path.suffix == '.json':
            try:
                visit(json.loads(path.read_text()))
            except ValueError:
                pass
        if path.suffix in ('.json', '.jsonl'):
            texts.append(path.read_text(errors='ignore'))
    return found, '\n'.join(texts)


def build():
    excluded, old_contexts, old_texts, sources = p7.prior_inputs()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models' / TOKENIZER, local_files_only=True)
    name_sources = [(ROOT / 'studies/bridge-003/code/build003.py', 'NAMES'),
                    (ROOT / 'studies/bridge-004/code/build004.py', 'EXTRA'),
                    (ROOT / 'studies/bridge-005/code/build005.py', 'EXTRA_NEW')]
    declared = set(NEW_NAMES) | set(itertools.chain.from_iterable(p7.name_literal(a, b) for a, b in name_sources))
    single = sorted(n for n in declared if len(tokenizer.encode(' ' + n, add_special_tokens=False)) == 1)
    seen, blob = large_seen_names()
    words = set(re.findall(r'\w+', blob))  # \bName\b matches exactly when Name is a whole \w+ token
    large_seen = sorted(n for n in single if n in seen or n in words)
    eligible = [n for n in single if n not in large_seen]
    never = [n for n in eligible if n not in excluded]
    assert len(eligible) == 567 and len(never) == 70, (len(eligible), len(never))  # counts stated in the protocol
    rng = np.random.default_rng(NAME_SEED)
    never_shuffled = list(rng.permutation(never))
    test_names = never_shuffled[:3 * TRIPLES['test']]
    rest = [n for n in eligible if n not in test_names]
    rest = list(np.random.default_rng(NAME_SEED + 100).permutation(rest))
    names, start = {}, 0
    for split in SPLITS[:-1]:
        names[split] = [str(x) for x in rest[start:start + 3 * TRIPLES[split]]]
        start += 3 * TRIPLES[split]
    names['test'] = [str(x) for x in test_names]

    valid, rejected = [], []
    for place, obj in CONTEXTS:
        both = any(re.search(r'\b' + place + r'\b', t) and re.search(r'\b' + obj + r'\b', t) for t in old_texts)
        (rejected if (place, obj) in old_contexts or both else valid).append((place, obj))
    assert len({p for p, _ in CONTEXTS}) == len(CONTEXTS)
    need = sum(BLOCKS.values())
    assert len(valid) >= need, (len(valid), need)
    order = np.random.default_rng(CONTEXT_SEED).permutation(len(valid))
    shuffled = [valid[i] for i in order]
    blocks, start = {}, 0
    for split in SPLITS:
        blocks[split] = shuffled[start:start + BLOCKS[split]]
        start += BLOCKS[split]

    rows, prompts = [], set()
    for split in SPLITS:
        bank = names[split]
        for ti, (place, obj) in enumerate(blocks[split]):
            for pi in range(TRIPLES[split]):
                triple = bank[pi * 3:pi * 3 + 3]
                for role in range(2):
                    io, subject, alternative = triple[role], triple[1 - role], triple[2]
                    for family, template in FAMILIES.items():
                        for order_ in range(2):
                            a, b = (io, subject) if order_ == 0 else (subject, io)
                            article = 'an' if obj[0].lower() in 'aeiou' else 'a'
                            text, tokens, positions = p7.tokenize(template, dict(
                                a=a, b=b, s=subject, place=place, obj=obj, article=article), tokenizer)
                            corrupt = tokens.copy()
                            alt_id = tokenizer.encode(' ' + alternative, add_special_tokens=False)[0]
                            corrupt[positions['s']] = alt_id
                            corrupt_text = tokenizer.decode(corrupt[1:])
                            assert text not in old_texts and corrupt_text not in old_texts and text not in prompts
                            prompts.add(text)
                            rows.append(dict(
                                id=f'b012-{split}-t{ti}-p{pi}-r{role}-f{family}-o{order_}', split=split,
                                template=ti, pair=pi, role=role, family=family, wording=family, order=order_,
                                io=io, subject=subject, alternative=alternative, place=place, obj=obj, text=text,
                                token_ids=tokens, corrupt_tokens=corrupt, s2=positions['s'],
                                io_position=positions['a' if order_ == 0 else 'b'],
                                subject_position=positions['b' if order_ == 0 else 'a'],
                                io_id=tokenizer.encode(' ' + io, add_special_tokens=False)[0],
                                subject_id=tokenizer.encode(' ' + subject, add_special_tokens=False)[0], alt_id=alt_id))
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ('split', 'template', 'pair', 'role', 'order')), {})[row['family']] = row
    for fa, fb in PAIRS:
        assert all(len(g[fa]['token_ids']) == len(g[fb]['token_ids']) and g[fa]['s2'] == g[fb]['s2']
                   for g in groups.values()), (fa, fb)
    for key, g in groups.items():
        other = groups[key[:4] + (1 - key[4],)]
        assert all(len(g[f]['token_ids']) == len(other[f]['token_ids']) and g[f]['s2'] == other[f]['s2'] for f in FAMILIES)
    all_names = list(itertools.chain.from_iterable(names.values()))
    assert len(set(all_names)) == len(all_names) and not set(all_names) & set(large_seen)
    assert not set(names['test']) & excluded
    design = dict(
        study='bridge-012 (Study 7)', model='gpt2-large', tokenizer=TOKENIZER, name_seed=NAME_SEED,
        context_seed=CONTEXT_SEED, families=FAMILIES, candidate_pairs=PAIRS, names=names, blocks=blocks,
        triples=TRIPLES, block_counts=BLOCKS, eligible_names=len(eligible), never_used_names=len(never),
        large_seen_names_excluded=large_seen, context_candidates=CONTEXTS, context_rejected=rejected,
        singleton_rule='Final-test rows of the selected pair with block = triple = 4*wording + 2*order + role (wording 0/1 in pair order)',
        prior_input_sources=sources,
        name_pool_sources={str(p.relative_to(ROOT)): digest(p) for p, _ in name_sources},
        tokenizer_files={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'models' / TOKENIZER).iterdir())
                         if p.name in {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}},
        scope='Tokenisation only; no scientific inference or behaviour-based filtering.')
    audit = dict(status='passed', utc=datetime.now(timezone.utc).isoformat(), scientific_inference_performed=False,
                 rows=len(rows), rows_per_split={s: sum(r['split'] == s for r in rows) for s in SPLITS},
                 final_test_names_never_used_in_ai01=True, names_disjoint_between_banks=True,
                 no_name_seen_by_gpt2_large=True, exact_context_pairs_unused=True,
                 clean_and_corrupted_prompts_unused=True, candidate_pairs_aligned=True, order_donors_aligned=True,
                 prepare_source_sha256=digest(Path(__file__)))
    return design, rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    outputs = [HERE / 'data' / n for n in ('design.json', 'rows.json', 'preparation-audit.json')]
    if args.write:
        assert not any(p.exists() for p in outputs), 'Refusing to overwrite prepared data'
    design, rows, audit = build()
    if args.write:
        for path, value in zip(outputs, (design, rows, audit)):
            write_new(path, value)
    print(json.dumps(dict(wrote=args.write, contexts_valid=len(design['context_candidates']) - len(design['context_rejected']),
                          rejected=design['context_rejected'], **audit), indent=1))


if __name__ == '__main__':
    main()
