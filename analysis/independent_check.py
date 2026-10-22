"""Independent re-implementation of the Bridge 005 edit and signature assay in plain NumPy.

Reads the GPT-2 Medium weights directly from models/gpt2-medium/model.safetensors
(no torch, no TransformerLens) and recomputes, for selected final-test rows, the
behavioural outcomes and all 176 x 2 signature coordinates in the clean, corrupted,
wording-transfer and self-transfer conditions, comparing them with the saved arrays.

Usage (from the project root):
    python3 analysis/independent_check.py 196,504,732          # row indices
    python3 analysis/independent_check.py --default            # the 12 rows reported in the paper
Each row takes about 20 s on two CPU cores. One JSON line per row is printed.

Definition note: the original pipeline uses TransformerLens in compatibility mode,
where LayerNorm and value biases are folded. Its "zero" replacement of a head's z
therefore equals setting the raw Hugging Face z to c_h = ln_1.bias . W_V^h + b_V^h,
and its mean bank is stored in the same folded coordinates. Both are reproduced here.
"""
import json, struct, sys, os, time, numpy as np
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
B = P + '/studies/bridge-005'
MP = P + '/models/gpt2-medium/model.safetensors'
f = open(MP, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; H = json.loads(f.read(n)); base = 8 + n
mm = np.memmap(MP, dtype=np.uint8, mode='r')
def T(k):
    s, e = H[k]['data_offsets']; return np.frombuffer(mm[base + s: base + e], dtype=np.float32).reshape(H[k]['shape'])
wte, wpe = T('wte.weight'), T('wpe.weight')
Ls = [{k: T(f'h.{l}.{k}') for k in ['ln_1.weight','ln_1.bias','attn.c_attn.weight','attn.c_attn.bias','attn.c_proj.weight','attn.c_proj.bias','ln_2.weight','ln_2.bias','mlp.c_fc.weight','mlp.c_fc.bias','mlp.c_proj.weight','mlp.c_proj.bias']} for l in range(24)]
lnf_w, lnf_b = T('ln_f.weight'), T('ln_f.bias')
NH, DH, D = 16, 64, 1024
MEAS = list(range(13, 24))
def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def mlp(x, p): return gelu(ln(x, p['ln_2.weight'], p['ln_2.bias']) @ p['mlp.c_fc.weight'] + p['mlp.c_fc.bias']) @ p['mlp.c_proj.weight'] + p['mlp.c_proj.bias']

def full(tokens, s2, add=None, src=()):
    """Full forward of one sequence. add: {layer: vec[1024]} added to attn_out at s2.
    Returns final END logits, per-layer caches for measured layers, and source head results at s2."""
    t = np.array(tokens); Tn = len(t); x = (wte[t] + wpe[:Tn]).astype(np.float32)
    mask = np.triu(np.full((Tn, Tn), -np.inf, np.float32), 1)
    cache = {}; res = {}
    for l in range(24):
        p = Ls[l]; h = ln(x, p['ln_1.weight'], p['ln_1.bias'])
        qkv = h @ p['attn.c_attn.weight'] + p['attn.c_attn.bias']
        q, k, v = [a.reshape(Tn, NH, DH).transpose(1, 0, 2) for a in np.split(qkv, 3, -1)]
        sc = q @ k.transpose(0, 2, 1) / 8.0 + mask; sc -= sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        z = (a @ v).transpose(1, 0, 2)  # [T, NH, DH]
        if l in MEAS: cache[l] = dict(x=x[-1].copy(), k=k.copy(), v=v.copy(), z_end=z[-1].copy())
        for (sl, sh) in src:
            if sl == l: res[(sl, sh)] = z[s2, sh] @ p['attn.c_proj.weight'][sh * DH:(sh + 1) * DH]
        out = z.reshape(Tn, D) @ p['attn.c_proj.weight'] + p['attn.c_proj.bias']
        if add is not None and l in add: out[s2] += add[l]
        x = x + out; x = x + mlp(x, p)
    logits = ln(x[-1], lnf_w, lnf_b) @ wte.T
    return logits, cache, res

def outcome(logits, io, sub):
    lg = logits; m = lg.max(-1, keepdims=True); p = np.exp(lg - m); p /= p.sum(-1, keepdims=True)
    return np.stack([lg[..., io] - lg[..., sub], p[..., io] - p[..., sub], (lg[..., io] > lg[..., sub]).astype(float), (lg.argmax(-1) == io).astype(float)], -1)

def signature(cache, base_out, bank, io, sub):
    """END-only ablation of each measured head (zero and mean replacement)."""
    xs = np.zeros((0, D), np.float32); tags = []
    for l in MEAS:
        p = Ls[l]; c = cache[l]
        # continuing variants: full END step at layer l with cached non-END K/V
        if len(xs):
            h = ln(xs, p['ln_1.weight'], p['ln_1.bias']); qkv = h @ p['attn.c_attn.weight'] + p['attn.c_attn.bias']
            q, k, v = [a.reshape(-1, NH, DH) for a in np.split(qkv, 3, -1)]
            K = c['k'][:, :-1]; V = c['v'][:, :-1]  # [NH, T-1, DH]
            s_prev = np.einsum('nhd,htd->nht', q, K) / 8.0; s_self = (q * k).sum(-1, keepdims=True) / 8.0
            s = np.concatenate([s_prev, s_self], -1); s -= s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
            z = np.einsum('nht,htd->nhd', a[..., :-1], V) + a[..., -1:] * v
            xs = xs + z.reshape(-1, D) @ p['attn.c_proj.weight'] + p['attn.c_proj.bias']
        # new variants starting at layer l
        zn = np.repeat(c['z_end'][None], 2 * NH, 0)
        for bi in range(2):
            for hh in range(NH):
                zn[bi * NH + hh, hh] = (0.0 if bi == 0 else bank[l - 13, hh]) if not isinstance(bank, tuple) else bank[bi][l - 13, hh]
                tags.append((bi, l, hh))
        xn = c['x'][None] + zn.reshape(-1, D) @ p['attn.c_proj.weight'] + p['attn.c_proj.bias']
        xs = np.concatenate([xs, xn.astype(np.float32)]); xs = xs + mlp(xs, p)
    logits = ln(xs, lnf_w, lnf_b) @ wte.T
    o = outcome(logits, io, sub)
    raw = np.zeros((2, 11 * NH, 2))
    for (bi, l, hh), oo in zip(tags, o): raw[bi, (l - 13) * NH + hh] = base_out[:2] - oo[:2]
    return raw

if __name__ == '__main__':
    DEFAULT = '196,504,732,53,461,733,342,474,626,95,427,1003'
    idx = [int(a) for a in (DEFAULT if sys.argv[1] == '--default' else sys.argv[1]).split(',')]
    sel = json.load(open(B + '/data/selection.json')); fams = sel['families']; src = [tuple(s) for s in sel['source_heads']]
    rows = [dict(r, wording=fams.index(r['family'])) for r in json.load(open(B + '/data/candidates.json')) if r['split'] == 'test' and r['family'] in fams]
    key = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(rows)}
    bank = np.load(B + '/results/fit/bank.npz')['bank']
    # TransformerLens (compatibility mode) z = raw z - c, c = ln_1.bias @ W_V + b_V (LayerNorm and value biases folded).
    cv = np.stack([(Ls[l]['ln_1.bias'] @ Ls[l]['attn.c_attn.weight'][:, 2048:] + Ls[l]['attn.c_attn.bias'][2048:]).reshape(NH, DH) for l in MEAS])
    rep = (cv, bank + cv)
    saved = {k: np.load(f'{B}/results/test/{k}.npz') for k in ['clean', 'corrupt', 'wording', 'self']}
    for i in idx:
        t0 = time.time(); r = rows[i]
        d = rows[key[(r['template'], r['pair'], r['role'], 1 - r['wording'], r['order'])]]
        assert d['s2'] == r['s2'] and len(d['token_ids']) == len(r['token_ids'])
        io, sub = r['io_id'], r['subject_id']
        def deltas(row):
            _, _, rc = full(row['token_ids'], row['s2'], src=src); _, _, rk = full(row['corrupt_tokens'], row['s2'], src=src)
            add = {}
            for hd in src: add[hd[0]] = add.get(hd[0], 0) + (rc[hd] - rk[hd])
            return add
        out = {}
        for name, toks, add in [('clean', r['token_ids'], None), ('corrupt', r['corrupt_tokens'], None),
                                ('wording', r['corrupt_tokens'], deltas(d)), ('self', r['corrupt_tokens'], deltas(r))]:
            lg, cache, _ = full(toks, r['s2'], add=add)
            bo = outcome(lg, io, sub)
            raw = signature(cache, bo, rep, io, sub)
            sb = saved[name]['behavior'][i]; sr = saved[name]['raw'][:, i]
            out[name] = dict(beh_err=float(np.abs(bo[:2] - sb[:2]).max()), top1_match=bool(bo[3] == sb[3]),
                             sig_err_logit=float(np.abs(raw[..., 0] - sr[..., 0]).max()), sig_err_prob=float(np.abs(raw[..., 1] - sr[..., 1]).max()),
                             sig_scale=float(np.abs(sr[..., 0]).max()), logit_gap=float(bo[0]))
        print(json.dumps(dict(i=i, id=r['id'], secs=round(time.time() - t0, 1), **out)), flush=True)
