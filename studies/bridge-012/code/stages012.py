"""Study 7 stages, run inside Modal containers with the results Volume mounted at results/.

Order: calibration -> calibration_cpu -> screen -> sources -> fit_select -> development ->
qualification -> test -> replay_gpu -> replay_cpu. Each stage refuses to start before the previous
stage's freeze exists, writes its record once and freezes it. A failed gate writes status.json and
stops the study. In code-test mode (GPT-2 Medium, calibration bank for every stage, a separate
Volume) gates are recorded but never stop.
"""
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from common012 import (HERE, ROOT, TOL, SPECS, PAIRS, KAPPAS, Engine012, read, write, sha, now, rows_for,
                       load_arrays, save_arrays, batches, delta, hooks, donor_indices, freeze, outcomes,
                       result_hooks, measure_natural, measure_signature, measure_behaviour, measure_attribution,
                       random_delta)
from analyze007 import qualification_summary, validate_rows
from common005 import behavior_cells, behavior_pass, separation, min_separation, fit_reference

OUT = HERE / 'results'
FZ = OUT / 'freezes'


def verify(path):
    """A stage starts only after the previous stage's freeze exists and still matches."""
    assert Path(path).exists(), f'Missing freeze {path}: run the previous stage first'
    return freeze(path, [], 'verify only')


class Ctx:
    def __init__(self, test=False):
        self.test = test
        self.commit = lambda: None  # replaced by the Volume's commit in a Modal container
        self.model = 'gpt2-medium' if test else 'gpt2-large'
        self.spec = SPECS[self.model]

    def split(self, name):
        return 'calibration' if self.test else name

    def input_freeze(self):
        if not self.test:
            path = HERE / 'data/input-freeze.json'
            for name, expected in read(path)['files'].items():
                assert sha(ROOT / name) == expected, name
            return path
        path = FZ / 'codetest-input-freeze.json'
        freeze(path, [HERE / 'data/rows.json'] + list((HERE / 'code').glob('*.py')), 'Code test only (GPT-2 Medium); not a scientific freeze')
        return path

    def stop(self, stage, reason, record):
        write(OUT / 'status.json', dict(status='stopped', stage=stage, reason=reason, utc=now(), code_test=self.test))
        return dict(record, stopped=not self.test, stop_reason=reason)


def summary(b):
    return dict(gap=float(b[:, 0].mean()), pairwise=float(b[:, 2].mean()), top1=float(b[:, 3].mean()))


def measure_rows(e, rows, condition, bank, prefix=True, ids=None):
    """One signature condition over rows (no records; technical checks), in equal-length batches."""
    recipient, d, held, donors = condition
    ids = np.arange(len(rows)) if ids is None else np.asarray(ids)
    sub = [rows[i] for i in ids]
    raw = np.empty((2, len(ids), len(e.layers) * e.heads, 2), np.float32)
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
        raw[:, part], behavior[part], cache = e.measure(rr, bank, edits, boundary, held, prefix=prefix)
        last = (cache, index)
    return raw, behavior, last


def specs(rows, nat, held, kappa):
    d_self = delta(nat, rows, 'self')
    out = dict(clean=('clean', None, (), None), corrupt=('corrupt', None, (), None),
               wording=('corrupt', delta(nat, rows, 'wording'), (), None), self=('corrupt', d_self, (), None),
               order=('corrupt', delta(nat, rows, 'order'), (), None),
               replay=('corrupt', None, (), donor_indices(rows, 'wording')))
    if held:
        out.update(self_kappa=('corrupt', d_self * kappa, (), None), n1=('clean', None, held, None),
                   n2=('corrupt', d_self, held, None))
    return out


def eight(rows):
    return [next(i for i, r in enumerate(rows) if (r['wording'], r['order'], r['role']) == (w, o, ro))
            for w in range(2) for o in range(2) for ro in range(2)]


@torch.inference_mode()
def calibration(ctx):
    target = OUT / 'calibration.json'
    if target.exists():
        return read(target)
    record = dict(started=now(), model=ctx.model, passed=False, tolerance=TOL, checks={},
                  purpose='Technical only; technical sources, held heads and bank; no decision uses these outcomes')
    try:
        freeze_in = ctx.input_freeze()
        rows = rows_for('calibration', [10, 11])
        validate_rows(rows)
        e = Engine012(ctx.model, ctx.spec['technical_sources'])
        folder = OUT / 'calibration'
        a = measure_natural(e, rows, folder / 'natural-clean.npz', freeze_in, 'clean', store_z=True)
        b = measure_natural(e, rows, folder / 'natural-corrupt.npz', freeze_in, 'corrupt')
        nat = {'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}
        bank = torch.tensor(a['z'].mean(0))
        save_arrays(folder / 'technical-bank.npz', bank=bank.numpy())
        cond = specs(rows, nat, ctx.spec['technical_held'], 0.0)
        cond['n_technical'] = cond.pop('n1')
        cond.pop('n2'), cond.pop('self_kappa')
        cond['self_zero'] = ('corrupt', delta(nat, rows, 'self') * 0.0, (), None)
        results = {}
        for name, c in cond.items():
            fast, fb, (cache, _) = measure_rows(e, rows, c, bank, True)
            slow, sb, _ = measure_rows(e, rows, c, bank, False)
            results[name] = dict(raw=fast, behavior=fb)
            save_arrays(folder / f'{name}.npz', prefix_raw=fast, full_raw=slow, prefix_behavior=fb, full_behavior=sb)
            record['checks'][f'prefix_vs_full_{name}'] = max(float(np.abs(fast - slow).max()), float(np.abs(fb - sb).max()))
            assert record['checks'][f'prefix_vs_full_{name}'] <= TOL, name
            if c[2]:
                z = np.stack([cache[f'blocks.{l}.attn.hook_z'][:, -1, h].cpu().numpy() for l, h in c[2]], 1)
                expected = np.stack([bank[e.layers.index(l), h].numpy() for l, h in c[2]])
                record['checks']['held_heads_equal_bank_max_error'] = float(np.abs(z - expected[None]).max())
                assert record['checks']['held_heads_equal_bank_max_error'] == 0.0
        donors = donor_indices(rows, 'wording')
        record['checks']['replay_equals_donor'] = max(
            float(np.abs(results['replay']['raw'] - results['clean']['raw'][:, donors]).max()),
            float(np.abs(results['replay']['behavior'] - results['clean']['behavior'][donors]).max()))
        assert record['checks']['replay_equals_donor'] <= TOL
        record['checks']['self_at_zero_dose_equals_corrupt'] = max(
            float(np.abs(results['self_zero']['raw'] - results['corrupt']['raw']).max()),
            float(np.abs(results['self_zero']['behavior'] - results['corrupt']['behavior']).max()))
        assert record['checks']['self_at_zero_dose_equals_corrupt'] == 0.0
        worst_gap, worst = 0.0, 0.0
        for part in batches(rows, 8):
            rr = [rows[i] for i in part]
            behavior, cache = e.forward(rr)
            att = e.attribution(rr, cache)
            worst_gap = max(worst_gap, float(np.abs(att['linearised_gap'] - behavior[:, 0].cpu().numpy()).max()))
        record['checks']['linearised_gap_error'] = worst_gap
        assert worst_gap <= TOL
        # Per-head outputs z W_O sum to each layer's attention output up to a shared bias (cancelled by
        # differencing rows). The discrepancy is expressed in logits through the run's frozen final
        # LayerNorm scale and the answer's unembedding direction, which is what direct attribution uses.
        rr = [rows[i] for i in next(iter(batches(rows, 8)))]
        tokens = torch.tensor([r['token_ids'] for r in rr], device=e.device)
        _, full = e.model.run_with_cache(tokens, names_filter=[f'blocks.{l}.hook_attn_out' for l in e.layers]
                                         + [f'blocks.{l}.attn.hook_z' for l in e.layers] + [e.final_resid])
        model = e.model.original_model
        x = full[e.final_resid][:, -1].double()
        sigma = (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(model.transformer.ln_f.eps).sqrt()
        gamma = model.transformer.ln_f.weight.double()
        unembed = model.lm_head.weight.double()
        direction = unembed[torch.tensor([r['io_id'] for r in rr], device=e.device)] - \
            unembed[torch.tensor([r['subject_id'] for r in rr], device=e.device)]
        worst_resid = 0.0
        for l in e.layers:
            summed = torch.einsum('nhk,hkd->nd', full[f'blocks.{l}.attn.hook_z'][:, -1].double(), e.weights[l].double())
            actual = full[f'blocks.{l}.hook_attn_out'][:, -1].double()
            diff = (summed - summed[:1]) - (actual - actual[:1])
            diff = diff - diff.mean(-1, keepdim=True)
            worst_resid = max(worst_resid, float(diff.abs().max()))
            worst = max(worst, float(((diff / sigma * gamma) * direction).sum(-1).abs().max()))
        record['checks']['head_outputs_sum_to_attention_output_logits'] = worst
        record['checks']['head_outputs_sum_to_attention_output_residual_units'] = worst_resid
        assert worst <= TOL
        record.update(passed=True, n=len(rows), cpu_check='pending (calibration_cpu)')
    except Exception as error:
        record.update(error=str(error), traceback=traceback.format_exc())
        write(target, record)
        return ctx.stop('calibration', str(error), record)
    record['finished'] = now()
    write(target, record)
    return record


@torch.inference_mode()
def calibration_cpu(ctx):
    target = OUT / 'calibration-cpu.json'
    if target.exists():
        return read(target)
    gpu = read(OUT / 'calibration.json')
    assert gpu['passed']
    rows = rows_for('calibration', [10, 11])
    folder = OUT / 'calibration'
    nat = {'clean_head_results': load_arrays(folder / 'natural-clean.npz')['head_results'],
           'corrupt_head_results': load_arrays(folder / 'natural-corrupt.npz')['head_results']}
    bank = torch.tensor(load_arrays(folder / 'technical-bank.npz')['bank'])
    e = Engine012(ctx.model, ctx.spec['technical_sources'], 'cpu')
    cond = specs(rows, nat, ctx.spec['technical_held'], 0.0)
    cond['n_technical'] = cond.pop('n1')
    ids = eight(rows)
    checks = {}
    for name in ('clean', 'wording', 'n_technical'):
        raw, b, _ = measure_rows(e, rows, cond[name], bank, True, ids)
        saved = load_arrays(folder / f'{name}.npz')
        checks[f'gpu_vs_cpu_{name}'] = max(float(np.abs(raw - saved['prefix_raw'][:, ids]).max()),
                                           float(np.abs(b - saved['prefix_behavior'][ids]).max()))
    value = dict(passed=max(checks.values()) <= TOL, checks=checks, tolerance=TOL, n=len(ids), utc=now())
    write(target, value)
    if value['passed']:
        freeze(FZ / 'calibration-freeze.json', [ctx.input_freeze(), OUT / 'calibration.json', target]
               + list(folder.glob('*.npz')), 'Calibration passed on GPU and CPU')
    else:
        ctx.stop('calibration_cpu', 'GPU and CPU signatures disagree', value)
    return value


@torch.inference_mode()
def screen(ctx):
    target = OUT / 'screen.json'
    if target.exists():
        return read(target)
    fz = FZ / 'calibration-freeze.json'
    verify(fz)
    rows = rows_for(ctx.split('screen'))
    e = Engine012(ctx.model, ctx.spec['technical_sources'])
    b = measure_behaviour(e, rows, None, OUT / 'screen/behaviour.npz', fz, 'clean')
    cells = behavior_cells(b, rows)
    eligible = [int(f) for f, c in cells.items() if behavior_pass({f: c})]
    pairs = [p for p in PAIRS if all(f in eligible for f in p)]
    value = dict(utc=now(), cells=cells, eligible=eligible, pairs=pairs, n=len(rows),
                 gate='pairwise >= 0.95 and top-1 >= 0.85 in both orders, per family')
    write(target, value)
    freeze(FZ / 'screen-freeze.json', [fz, target, OUT / 'screen/behaviour.npz'], 'Screen complete')
    if not pairs:
        return ctx.stop('screen', 'No candidate pair has both families passing the behavioural screen', value)
    return value


@torch.inference_mode()
def sources(ctx):
    target = OUT / 'sources.json'
    if target.exists():
        return read(target)
    fz = FZ / 'screen-freeze.json'
    verify(fz)
    scr = read(OUT / 'screen.json')
    families = scr['eligible'] if scr['pairs'] else [10, 11, 12, 13]
    rows = rows_for(ctx.split('development'), families)
    candidates = [(l, h) for l in ctx.spec['source_layers'] for h in range(ctx.spec['heads'])]
    e = Engine012(ctx.model, candidates)
    n = len(rows)
    gaps = np.empty((n, len(candidates)), np.float32)
    clean_b, corrupt_b = np.empty((n, 6), np.float32), np.empty((n, 6), np.float32)
    started = time.perf_counter()
    for part in batches(rows, 8):
        rr = [rows[i] for i in part]
        cr = [dict(r, token_ids=r['corrupt_tokens']) for r in rr]
        bc, cache = e.forward(rr)
        hc = e.extract(rr, cache)[2]
        bx, cache = e.forward(cr)
        hx = e.extract(cr, cache)[2]
        clean_b[part], corrupt_b[part] = bc.cpu().numpy(), bx.cpu().numpy()
        d = hc - hx
        for layer in ctx.spec['source_layers']:
            idx = [k for k, (l, _) in enumerate(candidates) if l == layer]
            seqs = [cr[i] for _ in idx for i in range(len(cr))]
            add_d = torch.cat([d[:, k] for k in idx], 0).to(e.device)
            pos = torch.tensor([r['s2'] for r in seqs], device=e.device)

            def add(value, hook, *, add_d=add_d, pos=pos):
                del hook
                v = value.clone()
                v[torch.arange(len(v), device=v.device), pos] += add_d
                return v
            tokens = torch.tensor([r['token_ids'] for r in seqs], device=e.device)
            logits = e.model.run_with_hooks(tokens, fwd_hooks=[(f'blocks.{layer}.hook_attn_out', add)])
            gap = outcomes(logits, seqs)[:, 0].reshape(len(idx), len(cr)).T.cpu().numpy()
            gaps[np.ix_(part, idx)] = gap
    recovery = (gaps - corrupt_b[:, :1]).mean(0)
    order = sorted(range(len(candidates)), key=lambda k: (-float(recovery[k]), candidates[k]))
    positive = [k for k in order if recovery[k] > 0]
    clean_gap = float((clean_b[:, 0] - corrupt_b[:, 0]).mean())
    trials, chosen = [], None
    for count in (12, 24):
        sel = positive[:count]
        heads = [candidates[k] for k in sel]
        joint = np.empty((n, 6), np.float32)
        for part in batches(rows, 8):
            rr = [rows[i] for i in part]
            cr = [dict(r, token_ids=r['corrupt_tokens']) for r in rr]
            _, cache = e.forward(rr)
            hc = e.extract(rr, cache)[2][:, sel]
            _, cache = e.forward(cr)
            hx = e.extract(cr, cache)[2][:, sel]
            pos = torch.tensor([r['s2'] for r in cr], device=e.device)
            joint[part] = e.forward(cr, result_hooks(heads, (hc - hx).to(e.device), pos))[0].cpu().numpy()
        rec = float((joint[:, 0] - corrupt_b[:, 0]).mean())
        item = dict(count=count, heads=[list(h) for h in heads], recovery=rec, fraction_of_clean_gap=rec / clean_gap,
                    pairwise=float(joint[:, 2].mean()), passed=bool(len(sel) == count and rec >= 0.25 * clean_gap
                                                                     and float(joint[:, 2].mean()) >= 0.75))
        trials.append(item)
        if item['passed']:
            chosen = item
            break
    save_arrays(OUT / 'development/source-scan.npz', gaps=gaps, clean_behavior=clean_b, corrupt_behavior=corrupt_b,
                candidates=np.array(candidates))
    value = dict(utc=now(), n=n, families=families, clean_gap=clean_gap, seconds=time.perf_counter() - started,
                 ranking=[dict(head=list(candidates[k]), mean_recovery=float(recovery[k])) for k in order[:40]],
                 trials=trials, passed=chosen is not None,
                 sources=chosen['heads'] if chosen else (trials[0]['heads'] if ctx.test else None),
                 rule='Single-head self transfer at S2; top 12 by positive mean recovery (ties: layer, head); joint gate >= 25% of the clean gap and pairwise >= 0.75; top 24 once')
    write(target, value)
    freeze(FZ / 'sources-freeze.json', [fz, target, OUT / 'development/source-scan.npz'], 'Source heads fixed')
    if chosen is None:
        return ctx.stop('sources', 'Joint self transfer through the selected sources does not repair the answer', value)
    return value


@torch.inference_mode()
def fit_select(ctx):
    target = OUT / 'selection.json'
    if target.exists():
        return read(target)
    fz = FZ / 'sources-freeze.json'
    verify(fz)
    scr, src = read(OUT / 'screen.json'), read(OUT / 'sources.json')
    families = scr['eligible'] if scr['pairs'] else [10, 11, 12, 13]
    pairs = scr['pairs'] or PAIRS
    e = Engine012(ctx.model, [tuple(h) for h in src['sources']])
    fit, sel = rows_for(ctx.split('fit'), families), rows_for(ctx.split('selection'), families)
    nat = measure_natural(e, fit, OUT / 'fit/natural.npz', fz, 'clean', store_z=True)
    bank = nat['z'].mean(0)
    save_arrays(OUT / 'fit/bank.npz', bank=bank)
    ctx.commit()
    a = measure_signature(e, fit, bank, OUT / 'fit/clean.npz', fz, 'clean')
    ctx.commit()
    b = measure_signature(e, sel, bank, OUT / 'selection/clean.npz', fz, 'clean')
    ctx.commit()
    scores = {}
    for pair in pairs:
        ii = np.array([i for i, r in enumerate(fit) if r['family'] in pair])
        jj = np.array([i for i, r in enumerate(sel) if r['family'] in pair])
        rr = [dict(fit[i], wording=pair.index(fit[i]['family'])) for i in ii]
        vv = [dict(sel[i], wording=pair.index(sel[i]['family'])) for i in jj]
        ref = fit_reference(a['raw'][:, ii], rr)
        ts, vs = separation(a['raw'][:, ii], rr, ref), separation(b['raw'][:, jj], vv, ref)
        tc, vc = behavior_cells(a['behavior'][ii], rr), behavior_cells(b['behavior'][jj], vv)
        worst = min_separation(vs)
        scores[','.join(map(str, pair))] = dict(
            families=pair, reference=ref, fit_behavior=tc, selection_behavior=vc, fit_separation=ts,
            selection_separation=vs, worst=worst, average=float(np.mean(list(vs.values()))),
            eligible=bool(behavior_pass(tc) and behavior_pass(vc) and min_separation(ts) >= .8 and worst >= .9))
    good = sorted([v for v in scores.values() if v['eligible']], key=lambda v: (-v['worst'], -v['average'], v['families']))
    selected = good[0] if good else (next(iter(scores.values())) if ctx.test else None)
    value = dict(utc=now(), status='selected' if good else 'no_qualified_pair', scores=scores,
                 families=selected['families'] if selected else None,
                 reference=selected['reference'] if selected else None,
                 source_heads=src['sources'], bank_sha256=sha(OUT / 'fit/bank.npz'), code_test_override=ctx.test and not good)
    write(target, value)
    freeze(FZ / 'selection-freeze.json', [fz, target, OUT / 'fit/bank.npz', OUT / 'fit/natural-record.json',
                                          OUT / 'fit/clean-record.json', OUT / 'selection/clean-record.json'],
           'Pair selected; reference and bank fixed')
    if not good:
        return ctx.stop('fit_select', 'No candidate pair passed Study 2 selection', value)
    return value


@torch.inference_mode()
def development(ctx):
    target = OUT / 'development.json'
    if target.exists():
        return read(target)
    fz = FZ / 'selection-freeze.json'
    verify(fz)
    sel = read(OUT / 'selection.json')
    rows = rows_for(ctx.split('development'), sel['families'])
    validate_rows(rows)
    e = Engine012(ctx.model, [tuple(h) for h in sel['source_heads']])
    bank = load_arrays(OUT / 'fit/bank.npz')['bank']
    folder = OUT / 'development'
    a = measure_natural(e, rows, folder / 'natural-clean.npz', fz, 'clean')
    b = measure_natural(e, rows, folder / 'natural-corrupt.npz', fz, 'corrupt')
    nat = {'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}
    d_self = delta(nat, rows, 'self')
    clean = measure_attribution(e, rows, bank, folder / 'attribution-clean.npz', fz, 'clean')
    corrupt_b = b['behavior']
    self_b = measure_behaviour(e, rows, bank, folder / 'behaviour-self.npz', fz, 'corrupt', d_self)
    gap = float(clean['behavior'][:, 0].mean())
    dla, attention = clean['dla'].mean(0), clean['attention'].mean(0)
    heads = [(l, h) for l in e.layers for h in range(e.heads)]
    candidates = [heads[i] for i in range(len(heads))
                  if dla[i] > 0 and attention[i, 0] > attention[i, 1] and attention[i, 0] > attention[i, 2]]
    candidates.sort(key=lambda lh: (-float(dla[e.head_index(*lh)]), lh[0], lh[1]))
    self_recovery = float((self_b[:, 0] - corrupt_b[:, 0]).mean())
    tried, chosen = [], None
    for k in (3, 2, 1):
        if len(candidates) < k:
            tried.append(dict(k=k, passed=False, reason='fewer candidates than k'))
            continue
        held = candidates[:k]
        tag = '-'.join(f'L{l}H{h}' for l, h in held)
        n1 = measure_behaviour(e, rows, bank, folder / f'behaviour-n1-{tag}.npz', fz, 'clean', held=held)
        n2 = measure_behaviour(e, rows, bank, folder / f'behaviour-n2-{tag}.npz', fz, 'corrupt', d_self, held)
        share = float(sum(dla[e.head_index(l, h)] for l, h in held)) / gap
        n2_recovery = float((n2[:, 0] - corrupt_b[:, 0]).mean())
        item = dict(k=k, heads=[list(x) for x in held], dla_share_of_clean_gap=share, n1=summary(n1),
                    n1_gap_fraction=float(n1[:, 0].mean()) / gap, n2=summary(n2), n2_recovery=n2_recovery,
                    n2_recovery_fraction=n2_recovery / self_recovery,
                    conditions=dict(a_share=share >= 0.20,
                                    b_n1=float(n1[:, 2].mean()) >= 0.90 and float(n1[:, 0].mean()) >= 0.5 * gap,
                                    c_n2=n2_recovery >= 0.5 * self_recovery and float(n2[:, 2].mean()) >= 0.75))
        item['passed'] = all(item['conditions'].values())
        tried.append(item)
        if item['passed']:
            chosen = item
            break
    if chosen is None and ctx.test and tried and 'heads' in tried[0]:
        chosen = tried[0]
    kappa, grid = None, []
    if chosen:
        for value_k in KAPPAS:
            kb = measure_behaviour(e, rows, bank, folder / f'behaviour-self-kappa{value_k:.2f}.npz', fz, 'corrupt', d_self * value_k)
            grid.append(dict(kappa=value_k, recovery=float((kb[:, 0] - corrupt_b[:, 0]).mean()), **summary(kb)))
        best = min(abs(g['recovery'] - chosen['n2_recovery']) for g in grid)
        kappa = max(g['kappa'] for g in grid if abs(g['recovery'] - chosen['n2_recovery']) == best)
    held = [tuple(x) for x in chosen['heads']] if chosen else []
    signatures = {}
    for name, (recipient, d, h, donors) in specs(rows, nat, held, kappa).items():
        measure_signature(e, rows, bank, folder / f'{name}.npz', fz, recipient, d, h, donors)
        signatures[name] = sha(folder / f'{name}.npz')
        ctx.commit()
    measure_attribution(e, rows, bank, folder / 'attribution-self.npz', fz, 'corrupt', d_self)
    checks = {}
    if held:
        measure_attribution(e, rows, bank, folder / 'attribution-n1.npz', fz, 'clean', held=held)
        measure_attribution(e, rows, bank, folder / 'attribution-n2.npz', fz, 'corrupt', d_self, held)
        conditions = specs(rows, nat, held, kappa)
        ids = eight(rows)
        for name in ('n1', 'n2'):
            fast, fb, _ = measure_rows(e, rows, conditions[name], bank, True, ids)
            slow, sb, _ = measure_rows(e, rows, conditions[name], bank, False, ids)
            saved = load_arrays(folder / f'{name}.npz')
            checks[f'prefix_vs_full_{name}'] = max(float(np.abs(fast - slow).max()), float(np.abs(fb - sb).max()))
            checks[f'saved_vs_recomputed_{name}'] = float(np.abs(fast - saved['raw'][:, ids]).max())
            assert max(checks[f'prefix_vs_full_{name}'], checks[f'saved_vs_recomputed_{name}']) <= TOL, checks
    value = dict(utc=now(), n=len(rows), clean=summary(clean['behavior']), corrupt=summary(corrupt_b), self=summary(self_b),
                 self_recovery=self_recovery,
                 clean_linearised_gap_error=float(np.abs(clean['linearised_gap'] - clean['behavior'][:, 0]).max()),
                 candidates=[dict(head=list(lh), mean_dla=float(dla[e.head_index(*lh)]),
                                  attention_io_s1_s2=attention[e.head_index(*lh)].tolist()) for lh in candidates[:10]],
                 trials=tried, established=chosen is not None and all(chosen.get('conditions', {}).values()),
                 held=[list(x) for x in held], k=len(held), kappa=kappa, kappa_grid=grid,
                 signature_sha256=signatures, checks=checks,
                 scope='Development data fix H, k and kappa only; every value here is development evidence')
    write(target, value)
    freeze(FZ / 'development-freeze.json', [fz, target] + list(folder.glob('*-record.json')), 'Held heads and dose fixed')
    return value


@torch.inference_mode()
def qualification(ctx):
    target = OUT / 'qualification-summary.json'
    if target.exists():
        return read(target)
    fz = FZ / 'development-freeze.json'
    verify(fz)
    sel = read(OUT / 'selection.json')
    rows = rows_for(ctx.split('qualification'), sel['families'])
    e = Engine012(ctx.model, [tuple(h) for h in sel['source_heads']])
    bank = load_arrays(OUT / 'fit/bank.npz')['bank']
    clean = measure_signature(e, rows, bank, OUT / 'qualification/clean.npz', fz, 'clean')
    value = qualification_summary(clean, rows, sel['reference'])
    value.update(utc=now(), families=sel['families'], n=len(rows))
    # Selected-pair boundary replay on the calibration bank, both directions.
    cal = rows_for('calibration', sel['families'])
    donors = donor_indices(cal, 'wording')
    errors = dict(signature=0.0, behavior=0.0)
    with torch.inference_mode():
        for part in batches(cal, 8):
            dr = [cal[donors[i]] for i in part]
            cr = [dict(cal[i], token_ids=cal[i]['corrupt_tokens']) for i in part]
            _, cache = e.forward(dr)
            x, xb, _ = e.measure(dr, bank)
            y, yb, _ = e.measure(cr, bank, boundary=cache[f'blocks.{e.site}.hook_resid_pre'])
            errors['signature'] = max(errors['signature'], float(np.abs(x - y).max()))
            errors['behavior'] = max(errors['behavior'], float(np.abs(xb - yb).max()))
    replay = dict(passed=max(errors.values()) <= TOL, errors=errors, n=len(cal), tolerance=TOL, utc=now())
    write(OUT / 'selected-pair-replay.json', replay)
    write(target, value)
    ok = value['qualified'] and replay['passed']
    if ok or ctx.test:
        freeze(FZ / 'test-freeze.json', [fz, target, OUT / 'selected-pair-replay.json', OUT / 'qualification/clean-record.json'],
               'Independent qualification passed; final inputs fixed')
    if not ok:
        return ctx.stop('qualification', 'Independent qualification or the selected-pair replay failed', value)
    return value


@torch.inference_mode()
def test(ctx):
    done = OUT / 'inference-complete.json'
    if done.exists():
        return read(done)
    fz = FZ / 'test-freeze.json'
    verify(fz)
    sel, dev = read(OUT / 'selection.json'), read(OUT / 'development.json')
    rows = rows_for(ctx.split('test'), sel['families'])
    validate_rows(rows)
    e = Engine012(ctx.model, [tuple(h) for h in sel['source_heads']])
    bank = load_arrays(OUT / 'fit/bank.npz')['bank']
    held, kappa = [tuple(x) for x in dev['held']], dev['kappa']
    folder = OUT / 'test'
    started = time.perf_counter()
    a = measure_natural(e, rows, folder / 'natural-clean.npz', fz, 'clean')
    b = measure_natural(e, rows, folder / 'natural-corrupt.npz', fz, 'corrupt')
    nat = {'clean_head_results': a['head_results'], 'corrupt_head_results': b['head_results']}
    outputs = {}
    for name, (recipient, d, h, donors) in specs(rows, nat, held, kappa).items():
        write(OUT / 'status.json', dict(status='inference', condition=name, utc=now()))
        measure_signature(e, rows, bank, folder / f'{name}.npz', fz, recipient, d, h, donors)
        outputs[name] = sha(folder / f'{name}.npz')
        ctx.commit()  # a timed-out stage resumes from committed chunks
    dw = delta(nat, rows, 'wording')
    controls = dict(zero=dw * 0, role=delta(nat, rows, 'role'), order_role=delta(nat, rows, 'order_role'))
    for kind in ('residual', 'projected'):
        for draw in range(3):
            dd = random_delta(e, dw, rows, kind, draw)
            assert float((dd.norm(dim=-1) - dw.norm(dim=-1)).abs().max()) < 1e-4
            controls[f'{kind}{draw}'] = dd
    for name, d in controls.items():
        measure_behaviour(e, rows, bank, folder / f'{name}.npz', fz, 'corrupt', d)
        outputs[name] = sha(folder / f'{name}.npz')
    ctx.commit()
    d_self = delta(nat, rows, 'self')
    measure_attribution(e, rows, bank, folder / 'attribution-clean.npz', fz, 'clean')
    measure_attribution(e, rows, bank, folder / 'attribution-self.npz', fz, 'corrupt', d_self)
    if held:
        measure_attribution(e, rows, bank, folder / 'attribution-n1.npz', fz, 'clean', held=held)
        measure_attribution(e, rows, bank, folder / 'attribution-n2.npz', fz, 'corrupt', d_self, held)
    for p in sorted(folder.glob('attribution-*.npz')) + sorted(folder.glob('natural-*.npz')):
        outputs[p.stem] = sha(p)
    value = dict(passed=True, status='completed', utc=now(), n=len(rows), held=dev['held'], kappa=kappa, outputs=outputs,
                 test_freeze_sha256=sha(fz), invocation_seconds=time.perf_counter() - started)
    write(done, value)
    write(OUT / 'status.json', dict(status='inference_completed', utc=now()))
    return value


def replay_rows(ctx, rows):
    if ctx.test:
        return eight(rows)
    index = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(rows)}
    return [index[(c, c, role, w, o)] for w in range(2) for o in range(2) for role in range(2) for c in [4 * w + 2 * o + role]]


@torch.inference_mode()
def replay(ctx, device):
    target = OUT / f'{device}-replay.json'
    if target.exists():
        return read(target)
    assert read(OUT / 'inference-complete.json')['passed']
    sel, dev = read(OUT / 'selection.json'), read(OUT / 'development.json')
    rows = rows_for(ctx.split('test'), sel['families'])
    e = Engine012(ctx.model, [tuple(h) for h in sel['source_heads']], 'cuda' if device == 'gpu' else 'cpu')
    e.config['head_chunk'] = 1
    bank = torch.tensor(load_arrays(OUT / 'fit/bank.npz')['bank'])
    held, kappa = [tuple(x) for x in dev['held']], dev['kappa']
    folder = OUT / 'test'
    nat = {'clean_head_results': load_arrays(folder / 'natural-clean.npz')['head_results'],
           'corrupt_head_results': load_arrays(folder / 'natural-corrupt.npz')['head_results']}
    conditions = specs(rows, nat, held, kappa)
    names = list(conditions) if device == 'gpu' else ['corrupt', 'n1' if held else 'wording', 'order']
    ii = replay_rows(ctx, rows)
    checks = {}
    for name in names:
        recipient, d, h, donors = conditions[name]
        raw = np.empty((2, len(ii), len(e.layers) * e.heads, 2), np.float32)
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
        saved = load_arrays(folder / f'{name}.npz')
        checks[name] = dict(raw=float(np.abs(raw - saved['raw'][:, ii]).max()),
                            behavior=float(np.abs(behavior - saved['behavior'][ii]).max()))
        save_arrays(OUT / f'replay-{device}/{name}.npz', raw=raw, behavior=behavior, indices=np.asarray(ii))
    value = dict(passed=all(max(v.values()) <= TOL for v in checks.values()), checks=checks, tolerance=TOL, utc=now(),
                 n=len(ii), device=device, fresh_container=True, head_chunk=1, prompt_batch=1,
                 ids=[rows[i]['id'] for i in ii])
    write(target, value)
    return value


STAGES = dict(calibration=calibration, screen=screen, sources=sources, fit_select=fit_select, development=development,
              qualification=qualification, test=test, replay_gpu=lambda c: replay(c, 'gpu'))
CPU_STAGES = dict(calibration_cpu=calibration_cpu, replay_cpu=lambda c: replay(c, 'cpu'))
