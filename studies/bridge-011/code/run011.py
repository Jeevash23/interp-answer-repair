"""Study 6 calibration, development, held-out inference and replays.

Stages run in this order, each refusing to start before the previous record exists:
calibration -> development -> (freeze011.py development) -> test -> replay -> cpu_replay.
A failed calibration stops the study for the author's review.
"""
import argparse
import time
import traceback

import numpy as np
import torch

from common011 import (HERE, BASE, TOL, KAPPAS, TECH_HELD, FREEZE, DEV_FREEZE, SIGNATURES, Engine011,
                       rows_for, fit_bank, head_index, read, write, load, digest, now, verify_freeze,
                       save_arrays, load_arrays, delta, hooks, batches, measure_signature,
                       measure_behaviour, measure_attribution)
from runner007 import measure_natural
from analyze007 import validate_rows
from common005 import donor_indices

OUT = HERE / 'results'


def save_fixed(path, **arrays):
    if path.exists():
        previous = load_arrays(path)
        assert previous.keys() == arrays.keys(), path
        for key in arrays:
            np.testing.assert_array_equal(previous[key], arrays[key])
    else:
        save_arrays(path, **arrays)


def measure_rows(e, rows, condition, bank, prefix=True, ids=None):
    """One signature condition over the given rows, in batches of equal-length prompts."""
    recipient, d, held, donors = condition
    ids = np.arange(len(rows)) if ids is None else np.asarray(ids)
    sub = [rows[i] for i in ids]
    raw = np.empty((2, len(ids), 176, 2), np.float32)
    behavior = np.empty((len(ids), 6), np.float32)
    last = None
    for part in batches(sub, 8):
        index = ids[part]
        rr = [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if recipient == 'corrupt' else rows[i] for i in index]
        edits = () if d is None else hooks(e, d[index], rr)
        boundary = None
        if donors is not None:
            _, cache = e.forward([rows[donors[i]] for i in index])
            boundary = cache[f'blocks.{e.site}.hook_resid_pre']
        raw[:, part], behavior[part], last = e.measure(rr, bank, edits, boundary, held, prefix=prefix)
        last = (last, index)
    return raw, behavior, last


def specs(rows, nat, held, kappa):
    """Every signature condition: recipient tokens, S2 edit, held END heads, replay donors."""
    d_self = delta(nat, rows, 'self')
    out = dict(clean=('clean', None, (), None), corrupt=('corrupt', None, (), None),
               wording=('corrupt', delta(nat, rows, 'wording'), (), None),
               self=('corrupt', d_self, (), None),
               order=('corrupt', delta(nat, rows, 'order'), (), None),
               replay=('corrupt', None, (), donor_indices(rows, 'wording')))
    if held:
        out.update(self_kappa=('corrupt', d_self * kappa, (), None), n1=('clean', None, held, None),
                   n2=('corrupt', d_self, held, None))
    return out


def natural(e, rows, folder, freeze):
    a = measure_natural(e, rows, folder / 'natural-clean.npz', freeze)
    b = measure_natural(e, [dict(r, token_ids=r['corrupt_tokens']) for r in rows],
                        folder / 'natural-corrupt.npz', freeze)
    return {'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}


@torch.inference_mode()
def calibration(e):
    target = OUT / 'calibration.json'
    if target.exists():
        assert read(target)['passed'], 'Calibration failed earlier: author review required'
        return
    record = dict(started=now(), freeze_sha256=digest(FREEZE), passed=False, tolerance=TOL, checks={},
                  purpose='Technical only; no decision uses calibration outcomes')
    try:
        rows = rows_for('calibration')
        validate_rows(rows)
        folder = OUT / 'calibration'
        nat = natural(e, rows, folder, FREEZE)
        bank = torch.tensor(fit_bank())
        cond = specs(rows, nat, TECH_HELD, 0.0)
        cond['n_technical'] = cond.pop('n1')
        for name in ('self_kappa', 'n2'):
            cond.pop(name)
        cond['self_zero'] = ('corrupt', delta(nat, rows, 'self') * 0.0, (), None)
        results = {}
        for name, condition in cond.items():
            held = condition[2]
            fast, fb, (cache, _) = measure_rows(e, rows, condition, bank, prefix=True)
            slow, sb, _ = measure_rows(e, rows, condition, bank, prefix=False)
            results[name] = dict(raw=fast, behavior=fb)
            save_fixed(folder / f'{name}.npz', prefix_raw=fast, full_raw=slow, prefix_behavior=fb, full_behavior=sb)
            error = max(float(np.abs(fast - slow).max()), float(np.abs(fb - sb).max()))
            record['checks'][f'prefix_vs_full_{name}'] = error
            assert error <= TOL, (name, error)
            if held:
                z = np.stack([cache[f'blocks.{l}.attn.hook_z'][:, -1, h].cpu().numpy() for l, h in held], 1)
                expected = np.stack([bank[l - 13, h].numpy() for l, h in held])
                record['checks']['held_heads_equal_bank_max_error'] = float(np.abs(z - expected[None]).max())
                assert record['checks']['held_heads_equal_bank_max_error'] == 0.0
        donors = donor_indices(rows, 'wording')
        replay_error = max(float(np.abs(results['replay']['raw'] - results['clean']['raw'][:, donors]).max()),
                           float(np.abs(results['replay']['behavior'] - results['clean']['behavior'][donors]).max()))
        record['checks']['replay_equals_donor'] = replay_error
        assert replay_error <= TOL, replay_error
        zero_error = max(float(np.abs(results['self_zero']['raw'] - results['corrupt']['raw']).max()),
                         float(np.abs(results['self_zero']['behavior'] - results['corrupt']['behavior']).max()))
        record['checks']['self_at_zero_dose_equals_corrupt'] = zero_error
        assert zero_error == 0.0, zero_error
        # Cross-device agreement on eight fixed rows (one per wording x order x role).
        eight = [next(i for i, r in enumerate(rows) if (r['wording'], r['order'], r['role']) == (w, o, ro))
                 for w in range(2) for o in range(2) for ro in range(2)]
        cpu = Engine011('cpu')
        for name in ('clean', 'wording', 'n_technical'):
            raw, b, _ = measure_rows(cpu, rows, cond[name], bank, ids=eight)
            error = max(float(np.abs(raw - results[name]['raw'][:, eight]).max()),
                        float(np.abs(b - results[name]['behavior'][eight]).max()))
            record['checks'][f'gpu_vs_cpu_{name}'] = error
            assert error <= TOL, (name, error)
        del cpu
        # Frozen-scale attribution: the whole final residual reproduces the logit gap.
        worst_gap, worst = 0.0, 0.0
        for part in batches(rows, 8):
            rr = [rows[i] for i in part]
            behavior, cache = e.forward(rr)
            att = e.attribution(rr, cache)
            worst_gap = max(worst_gap, float(np.abs(att['linearised_gap'] - behavior[:, 0].cpu().numpy()).max()))
        record['checks']['linearised_gap_error'] = worst_gap
        assert worst_gap <= TOL
        # Per-head outputs z W_O sum to each layer's attention output, up to a shared bias.
        rr = [rows[i] for i in next(iter(batches(rows, 8)))]
        tokens = torch.tensor([r['token_ids'] for r in rr], device=e.device)
        _, full = e.model.run_with_cache(tokens, names_filter=[f'blocks.{l}.hook_attn_out' for l in e.layers]
                                         + [f'blocks.{l}.attn.hook_z' for l in e.layers])
        for l in e.layers:
            summed = torch.einsum('nhk,hkd->nd', full[f'blocks.{l}.attn.hook_z'][:, -1], e.weights[l])
            actual = full[f'blocks.{l}.hook_attn_out'][:, -1]
            diff = (summed - summed[:1]) - (actual - actual[:1])
            diff = diff - diff.mean(-1, keepdim=True)
            worst = max(worst, float(diff.abs().max()))
        record['checks']['head_outputs_sum_to_attention_output'] = worst
        assert worst <= TOL, worst
        record.update(passed=True, n=len(rows), ids=[r['id'] for r in rows], technical_held=TECH_HELD)
    except Exception as error:
        record.update(error=str(error), traceback=traceback.format_exc(), finished=now())
        write(target, record)
        write(OUT / 'status.json', dict(status='calibration_failed', author_review_required=True,
                                        reason=str(error), utc=now()))
        raise
    record['finished'] = now()
    write(target, record)
    print(now(), 'ALL CALIBRATION CHECKS PASSED', record['checks'], flush=True)


def summary(b, rows):
    return dict(gap=float(b[:, 0].mean()), pairwise=float(b[:, 2].mean()), top1=float(b[:, 3].mean()))


@torch.inference_mode()
def development(e):
    assert read(OUT / 'calibration.json')['passed']
    target = OUT / 'development.json'
    if target.exists():
        return read(target)
    rows = rows_for('development')
    validate_rows(rows)
    folder = OUT / 'development'
    nat = natural(e, rows, folder, FREEZE)
    d_self = delta(nat, rows, 'self')
    clean = measure_attribution(e, rows, folder / 'attribution-clean.npz', FREEZE, 'clean')
    corrupt_b = measure_behaviour(e, rows, folder / 'behaviour-corrupt.npz', FREEZE, 'corrupt')
    self_b = measure_behaviour(e, rows, folder / 'behaviour-self.npz', FREEZE, 'corrupt', d_self)
    gap = float(clean['behavior'][:, 0].mean())
    dla, attention = clean['dla'].mean(0), clean['attention'].mean(0)
    heads = [(l, h) for l in e.layers for h in range(e.heads)]
    candidates = [heads[i] for i in range(len(heads))
                  if dla[i] > 0 and attention[i, 0] > attention[i, 1] and attention[i, 0] > attention[i, 2]]
    candidates.sort(key=lambda lh: (-float(dla[head_index(*lh)]), lh[0], lh[1]))
    self_recovery = float((self_b[:, 0] - corrupt_b[:, 0]).mean())
    tried, chosen = [], None
    for k in (3, 2, 1):
        if len(candidates) < k:
            tried.append(dict(k=k, passed=False, reason='fewer candidates than k'))
            continue
        held = candidates[:k]
        tag = '-'.join(f'L{l}H{h}' for l, h in held)
        n1 = measure_behaviour(e, rows, folder / f'behaviour-n1-{tag}.npz', FREEZE, 'clean', held=held)
        n2 = measure_behaviour(e, rows, folder / f'behaviour-n2-{tag}.npz', FREEZE, 'corrupt', d_self, held)
        share = float(sum(dla[head_index(l, h)] for l, h in held)) / gap
        n2_recovery = float((n2[:, 0] - corrupt_b[:, 0]).mean())
        item = dict(k=k, heads=[list(x) for x in held], dla_share_of_clean_gap=share,
                    n1=summary(n1, rows), n1_gap_fraction=float(n1[:, 0].mean()) / gap,
                    n2=summary(n2, rows), n2_recovery=n2_recovery, n2_recovery_fraction=n2_recovery / self_recovery,
                    conditions=dict(a_share=share >= 0.20,
                                    b_n1=float(n1[:, 2].mean()) >= 0.90 and float(n1[:, 0].mean()) >= 0.5 * gap,
                                    c_n2=n2_recovery >= 0.5 * self_recovery and float(n2[:, 2].mean()) >= 0.75))
        item['passed'] = all(item['conditions'].values())
        tried.append(item)
        if item['passed']:
            chosen = item
            break
    kappa, grid = None, []
    if chosen:
        for kappa_value in KAPPAS:
            b = measure_behaviour(e, rows, folder / f'behaviour-self-kappa{kappa_value:.2f}.npz', FREEZE,
                                  'corrupt', d_self * kappa_value)
            grid.append(dict(kappa=kappa_value, recovery=float((b[:, 0] - corrupt_b[:, 0]).mean()), **summary(b, rows)))
        best = min(abs(g['recovery'] - chosen['n2_recovery']) for g in grid)
        kappa = max(g['kappa'] for g in grid if abs(g['recovery'] - chosen['n2_recovery']) == best)
    held = [tuple(x) for x in chosen['heads']] if chosen else []
    signatures = {}
    for name, (recipient, d, h, donors) in specs(rows, nat, held, kappa).items():
        measure_signature(e, rows, folder / f'{name}.npz', FREEZE, recipient, d, h, donors)
        signatures[name] = digest(folder / f'{name}.npz')
    measure_attribution(e, rows, folder / 'attribution-self.npz', FREEZE, 'corrupt', d_self)
    if held:
        measure_attribution(e, rows, folder / 'attribution-n1.npz', FREEZE, 'clean', held=held)
        measure_attribution(e, rows, folder / 'attribution-n2.npz', FREEZE, 'corrupt', d_self, held)
    checks = {}
    if held:
        bank = torch.tensor(fit_bank())
        eight = [next(i for i, r in enumerate(rows) if (r['wording'], r['order'], r['role']) == (w, o, ro))
                 for w in range(2) for o in range(2) for ro in range(2)]
        conditions = specs(rows, nat, held, kappa)
        for name in ('n1', 'n2'):
            fast, fb, _ = measure_rows(e, rows, conditions[name], bank, prefix=True, ids=eight)
            slow, sb, _ = measure_rows(e, rows, conditions[name], bank, prefix=False, ids=eight)
            saved = load_arrays(folder / f'{name}.npz')
            checks[f'prefix_vs_full_{name}'] = max(float(np.abs(fast - slow).max()), float(np.abs(fb - sb).max()))
            checks[f'saved_vs_recomputed_{name}'] = float(np.abs(fast - saved['raw'][:, eight]).max())
            assert max(checks[f'prefix_vs_full_{name}'], checks[f'saved_vs_recomputed_{name}']) <= TOL, checks
    value = dict(utc=now(), n=len(rows), clean=summary(clean['behavior'], rows),
                 corrupt=summary(corrupt_b, rows), self=summary(self_b, rows), self_recovery=self_recovery,
                 clean_linearised_gap_error=float(np.abs(clean['linearised_gap'] - clean['behavior'][:, 0]).max()),
                 candidates=[dict(head=list(lh), mean_dla=float(dla[head_index(*lh)]),
                                  attention_io_s1_s2=attention[head_index(*lh)].tolist()) for lh in candidates],
                 trials=tried, established=chosen is not None, held=[list(x) for x in held],
                 k=len(held), kappa=kappa, kappa_grid=grid, signature_sha256=signatures, checks=checks,
                 scope='Development data fix H, k and kappa only (protocol); every value here is development evidence')
    write(target, value)
    print(now(), 'DEVELOPMENT', dict(held=value['held'], kappa=kappa, trials=tried), flush=True)
    return value


@torch.inference_mode()
def test(e):
    verify_freeze(DEV_FREEZE)
    dev = read(OUT / 'development.json')
    rows = rows_for('test')
    validate_rows(rows)
    folder = OUT / 'test'
    held, kappa = [tuple(x) for x in dev['held']], dev['kappa']
    start = time.perf_counter()
    nat = natural(e, rows, folder, DEV_FREEZE)
    outputs = {}
    for name, (recipient, d, h, donors) in specs(rows, nat, held, kappa).items():
        write(OUT / 'status.json', dict(status='inference', condition=name, utc=now()))
        measure_signature(e, rows, folder / f'{name}.npz', DEV_FREEZE, recipient, d, h, donors)
        outputs[name] = digest(folder / f'{name}.npz')
    d_self = delta(nat, rows, 'self')
    measure_attribution(e, rows, folder / 'attribution-clean.npz', DEV_FREEZE, 'clean')
    measure_attribution(e, rows, folder / 'attribution-self.npz', DEV_FREEZE, 'corrupt', d_self)
    if held:
        measure_attribution(e, rows, folder / 'attribution-n1.npz', DEV_FREEZE, 'clean', held=held)
        measure_attribution(e, rows, folder / 'attribution-n2.npz', DEV_FREEZE, 'corrupt', d_self, held)
    for p in sorted(folder.glob('attribution-*.npz')) + sorted(folder.glob('natural-*.npz')):
        outputs[p.stem] = digest(p)
    target = OUT / 'inference-complete.json'
    value = dict(passed=True, utc=now(), n=len(rows), held=dev['held'], kappa=kappa, outputs=outputs,
                 development_freeze_sha256=digest(DEV_FREEZE), invocation_seconds=time.perf_counter() - start)
    if target.exists():
        assert read(target)['outputs'] == outputs
    else:
        write(target, value)
    write(OUT / 'status.json', dict(status='inference_completed', utc=now()))
    print(now(), 'INFERENCE COMPLETE', round(time.perf_counter() - start, 1), 's', flush=True)


def replay_rows():
    rows = rows_for('test')
    ids = read(HERE / 'data/design.json')['singleton_ids']
    index = {r['id']: i for i, r in enumerate(rows)}
    return rows, [index[x] for x in ids]


@torch.inference_mode()
def replay(e, device_label):
    """Fresh process and engine; one prompt at a time; one head variant per ablated pass."""
    assert read(OUT / 'inference-complete.json')['passed']
    path = OUT / f'{device_label}-replay.json'
    if path.exists():
        assert read(path)['passed']
        return
    e.config['head_chunk'] = 1
    dev = read(OUT / 'development.json')
    held, kappa = [tuple(x) for x in dev['held']], dev['kappa']
    rows, ii = replay_rows()
    folder = OUT / 'test'
    nat = {'clean_head_results': load(folder / 'natural-clean.npz')['head_results'],
           'corrupt_head_results': load(folder / 'natural-corrupt.npz')['head_results']}
    names = SIGNATURES if device_label == 'gpu' else ('corrupt', 'n1', 'order')
    conditions = specs(rows, nat, held, kappa)
    bank = torch.tensor(fit_bank())
    checks = {}
    for name in names:
        if name not in conditions:
            continue
        recipient, d, h, donors = conditions[name]
        raw = np.empty((2, len(ii), 176, 2), np.float32)
        behavior = np.empty((len(ii), 6), np.float32)
        for k, i in enumerate(ii):
            rr = [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if recipient == 'corrupt' else rows[i]]
            edits = () if d is None else hooks(e, d[[i]], rr)
            boundary = None
            if donors is not None:
                _, cache = e.forward([rows[donors[i]]])
                boundary = cache[f'blocks.{e.site}.hook_resid_pre']
            x, y, _ = e.measure(rr, bank, edits, boundary, h)
            raw[:, k], behavior[k] = x[:, 0], y[0]
        saved = load(folder / f'{name}.npz')
        checks[name] = dict(raw=float(np.abs(raw - saved['raw'][:, ii]).max()),
                            behavior=float(np.abs(behavior - saved['behavior'][ii]).max()))
        save_fixed(OUT / f'replay-{device_label}/{name}.npz', raw=raw, behavior=behavior, indices=np.asarray(ii))
        print(now(), device_label, 'replay', name, checks[name], flush=True)
    passed = all(max(v.values()) <= TOL for v in checks.values())
    write(path, dict(passed=passed, checks=checks, tolerance=TOL, utc=now(), n=len(ii), device=device_label,
                     fresh_engine=True, head_chunk=1, prompt_batch=1, ids=[rows[i]['id'] for i in ii]))
    assert passed, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['calibration', 'development', 'test', 'replay', 'cpu_replay'])
    args = parser.parse_args()
    verify_freeze(FREEZE)
    if args.stage == 'cpu_replay':
        replay(Engine011('cpu'), 'cpu')
        return
    e = Engine011('cuda')
    if args.stage == 'replay':
        replay(e, 'gpu')
    else:
        globals()[args.stage](e)


if __name__ == '__main__':
    main()
