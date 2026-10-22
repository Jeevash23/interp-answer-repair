"""Build Study 6's fresh calibration, development and held-out banks by tokenisation only.

No model weights or outputs are loaded. Earlier study data are read only to exclude
names, context pairs and exact prompt strings, with Study 2's exclusion code. Running
without --write reports feasibility; --write refuses to replace existing data.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'studies/bridge-007/code'))
import prepare007 as p7

p7.HERE = HERE  # every study except this one counts as prior input
NAME_SEED = 2026101601
CONTEXT_SEED = 2026101602
MODEL = 'gpt2-medium'
FAMILIES = {10: p7.FAMILIES[10], 11: p7.FAMILIES[11]}
SPLITS = ('calibration', 'development', 'test')
NAME_COUNTS = dict(calibration=6, development=18, test=60)
BLOCK_COUNTS = dict(calibration=2, development=6, test=20)
POOL_UNUSED = [('hotel', 'envelope'), ('garage', 'brush'), ('temple', 'candle'), ('tower', 'map'),
               ('warehouse', 'box'), ('cafe', 'cookie'), ('observatory', 'lens'), ('abbey', 'book'),
               ('cabin', 'blanket'), ('cellar', 'basket'), ('clubhouse', 'towel'), ('fortress', 'coin'),
               ('gazebo', 'fan'), ('gymnasium', 'whistle'), ('hangar', 'glove'), ('orchard', 'basket')]
NEW_CONTEXTS = [('barn', 'bucket'), ('canteen', 'spoon'), ('chalet', 'sweater'), ('diner', 'napkin'),
                ('lobby', 'badge'), ('marina', 'anchor'), ('meadow', 'daisy'), ('pharmacy', 'bandage'),
                ('salon', 'comb'), ('terminal', 'suitcase'), ('vineyard', 'grape'), ('arena', 'banner')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def build():
    excluded, old_contexts, old_texts, sources = p7.prior_inputs()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models' / MODEL, local_files_only=True)
    name_sources = [(ROOT / 'studies/bridge-003/code/build003.py', 'NAMES'),
                    (ROOT / 'studies/bridge-004/code/build004.py', 'EXTRA'),
                    (ROOT / 'studies/bridge-005/code/build005.py', 'EXTRA_NEW')]
    declared = set(itertools.chain.from_iterable(p7.name_literal(a, b) for a, b in name_sources))
    pool = sorted(n for n in declared - excluded
                  if len(tokenizer.encode(' ' + n, add_special_tokens=False)) == 1)
    assert len(pool) == 120, len(pool)  # the count stated in the frozen protocol
    rng = np.random.default_rng(NAME_SEED)
    rng.shuffle(pool)
    names, start = {}, 0
    for split in SPLITS:
        names[split] = pool[start:start + NAME_COUNTS[split]]
        start += NAME_COUNTS[split]

    # The protocol lists the unused pool pairs in Study 2's declared order.
    unused = [b for b in p7.CONTEXT_POOL if b not in old_contexts]
    assert unused == POOL_UNUSED, unused
    for block in NEW_CONTEXTS:
        assert block not in old_contexts, block
        assert not any(p7.re.search(r'\b' + block[0] + r'\b', t) for t in old_texts), block
    contexts = POOL_UNUSED + NEW_CONTEXTS
    assert len({place for place, _ in contexts}) == len(contexts) == 28
    order = np.random.default_rng(CONTEXT_SEED).permutation(len(contexts))
    shuffled = [contexts[i] for i in order]
    blocks, start = {}, 0
    for split in SPLITS:
        blocks[split] = shuffled[start:start + BLOCK_COUNTS[split]]
        start += BLOCK_COUNTS[split]

    rows, prompts = [], set()
    for split in SPLITS:
        bank = names[split]
        for ti, (place, obj) in enumerate(blocks[split]):
            for pi in range(len(bank) // 3):
                triple = bank[pi * 3:pi * 3 + 3]
                for role in range(2):
                    io, subject, alternative = triple[role], triple[1 - role], triple[2]
                    for wording, family in enumerate(FAMILIES):
                        for order_ in range(2):
                            a, b = (io, subject) if order_ == 0 else (subject, io)
                            article = 'an' if obj[0].lower() in 'aeiou' else 'a'
                            text, tokens, positions = p7.tokenize(FAMILIES[family], dict(
                                a=a, b=b, s=subject, place=place, obj=obj, article=article), tokenizer)
                            corrupt = tokens.copy()
                            alt_id = tokenizer.encode(' ' + alternative, add_special_tokens=False)[0]
                            corrupt[positions['s']] = alt_id
                            corrupt_text = tokenizer.decode(corrupt[1:])
                            assert text not in old_texts and corrupt_text not in old_texts
                            assert text not in prompts
                            prompts.add(text)
                            assert sum(x != y for x, y in zip(tokens, corrupt)) == 1
                            rows.append(dict(
                                id=f'b011-{split}-t{ti}-p{pi}-r{role}-f{family}-o{order_}',
                                split=split, template=ti, pair=pi, role=role, family=family,
                                wording=wording, order=order_, io=io, subject=subject,
                                alternative=alternative, place=place, obj=obj, text=text,
                                token_ids=tokens, corrupt_tokens=corrupt, s2=positions['s'],
                                io_position=positions['a' if order_ == 0 else 'b'],
                                subject_position=positions['b' if order_ == 0 else 'a'],
                                io_id=tokenizer.encode(' ' + io, add_special_tokens=False)[0],
                                subject_id=tokenizer.encode(' ' + subject, add_special_tokens=False)[0],
                                alt_id=alt_id))
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ('split', 'template', 'pair', 'role', 'order')), []).append(row)
    assert all(len(g) == 2 and len(g[0]['token_ids']) == len(g[1]['token_ids']) and g[0]['s2'] == g[1]['s2']
               for g in groups.values()), 'F10/F11 rows must align'
    # Order donors share token length and S2 position with their recipients.
    for key, g in groups.items():
        other = groups[key[:4] + (1 - key[4],)]
        assert all(len(x['token_ids']) == len(y['token_ids']) and x['s2'] == y['s2'] for x, y in zip(g, other))
    test = [r for r in rows if r['split'] == 'test']
    index = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(test)}
    singletons = [index[(c, c, role, w, o)] for w in range(2) for o in range(2) for role in range(2)
                  for c in [4 * w + 2 * o + role]]
    design = dict(
        study='bridge-011 (Study 6)', model=MODEL, name_seed=NAME_SEED, context_seed=CONTEXT_SEED,
        families=FAMILIES, names=names, blocks=blocks, name_counts=NAME_COUNTS, block_counts=BLOCK_COUNTS,
        context_list=contexts, excluded_names=sorted(excluded), tokenization_pool=len(pool),
        unused_eligible_names_after=len(pool) - sum(NAME_COUNTS.values()),
        singleton_ids=[test[i]['id'] for i in singletons],
        singleton_rule='Held-out row with block = triple = 4*wording + 2*order + role, one per wording x order x role',
        prior_input_sources=sources,
        name_pool_sources={str(p.relative_to(ROOT)): digest(p) for p, _ in name_sources},
        tokenizer_files={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'models' / MODEL).iterdir())
                         if p.name in {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}},
        scope='Tokenisation only; no scientific inference or behaviour-based filtering.')
    audit = dict(
        status='passed', utc=datetime.now(timezone.utc).isoformat(), scientific_inference_performed=False,
        rows=len(rows), rows_per_split={s: sum(r['split'] == s for r in rows) for s in SPLITS},
        prior_json_sources=len(sources), prior_names=len(excluded), prior_prompt_strings=len(old_texts),
        prior_context_pairs=len(old_contexts), names_disjoint_from_prior_and_between_banks=True,
        context_pairs_disjoint_from_prior_and_between_banks=True, new_location_words_absent_from_prior_prompts=True,
        clean_and_corrupted_prompts_disjoint_from_prior=True, all_clean_prompts_unique=True,
        single_token_names=True, one_token_corruption=True, wording_and_order_donors_aligned=True,
        prepare_source_sha256=digest(Path(__file__)))
    all_names = list(itertools.chain.from_iterable(names.values()))
    assert len(set(all_names)) == len(all_names) and not set(all_names) & excluded
    all_blocks = list(itertools.chain.from_iterable(blocks.values()))
    assert len(set(all_blocks)) == len(all_blocks) and not set(all_blocks) & old_contexts
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
    print(json.dumps(dict(wrote=args.write, names={k: len(v) for k, v in design['names'].items()},
                          blocks=design['blocks'], singleton_ids=design['singleton_ids'], **audit), indent=1))


if __name__ == '__main__':
    main()
