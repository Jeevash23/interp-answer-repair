"""Study 6 design-stage runtime benchmark (technical only).

Times the inherited causal-prefix signature engine on the GPU, using Study 2's 32
F10/F11 calibration rows, which earlier served only technical checks. It writes
timings, peak memory and the maximum GPU-versus-CPU signature difference on eight
rows. No signature value, distance or behavioural outcome is printed or kept.
"""
import json, platform, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'studies/bridge-006/code')]
from cpu_engine import CPUEngine, Engine003
from common005 import delta, hooks, batches

class GPUEngine(CPUEngine):
    def __init__(self):
        Engine003.__init__(self, 'gpt2-medium', 'cuda')

def main():
    target = HERE / 'runtime-benchmark.json'
    assert not target.exists(), target
    rows = [dict(r, wording=[10, 11].index(r['family'])) for r in json.loads((ROOT / 'studies/bridge-007/data/candidates.json').read_text())
            if r['split'] == 'calibration' and r['family'] in (10, 11)]
    bank_np = np.load(ROOT / 'studies/bridge-007/results/fit/bank.npz')['bank']
    record = dict(purpose='Runtime and cross-device agreement only; no scientific outcome inspected or retained',
                  rows='Study 2 calibration rows, families 10/11', n=len(rows),
                  torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                  python=platform.python_version(), runs=[])
    keep = {}
    with torch.inference_mode():
        for device, batch, chunk, subset in (('cuda', 8, 8, rows), ('cuda', 16, 8, rows), ('cuda', 16, 16, rows), ('cpu', 8, 8, rows[:8])):
            e = GPUEngine() if device == 'cuda' else CPUEngine()
            e.config = dict(e.config, head_chunk=chunk)
            bank = torch.tensor(bank_np, device=e.device)
            hc, hx = [], []
            for ids in batches(subset, batch):
                rr = [subset[i] for i in ids]
                _, cache = e.forward(rr); hc.append(e.extract(rr, cache)[2])
                cr = [dict(r, token_ids=r['corrupt_tokens']) for r in rr]
                _, cache = e.forward(cr); hx.append(e.extract(cr, cache)[2])
            order = np.concatenate([ids for ids in batches(subset, batch)])
            nat = {'clean_head_results': torch.cat(hc).numpy()[np.argsort(order)],
                   'corrupt_head_results': torch.cat(hx).numpy()[np.argsort(order)]}
            dd = delta(nat, subset, 'wording')
            for kind in ('clean', 'wording'):
                for repeat in range(2):
                    if device == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter(); out = np.empty((2, len(subset), 176, 2), np.float32)
                    for ids in batches(subset, batch):
                        rr = [subset[i] for i in ids]
                        if kind == 'wording':
                            rr = [dict(r, token_ids=r['corrupt_tokens']) for r in rr]
                            raw, _ = e.signatures(rr, bank, hooks(e, dd[ids], rr))
                        else:
                            raw, _ = e.signatures(rr, bank)
                        out[:, ids] = raw
                    if device == 'cuda': torch.cuda.synchronize()
                    seconds = time.perf_counter() - t0
                    record['runs'].append(dict(device=device, batch=batch, head_chunk=chunk, condition=kind, repeat=repeat,
                                               rows=len(subset), seconds=seconds, seconds_per_row=seconds / len(subset),
                                               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else None))
                keep[(device, batch, chunk, kind)] = out
            del e
            if device == 'cuda': torch.cuda.empty_cache()
    diffs = {}
    for kind in ('clean', 'wording'):
        cpu = keep[('cpu', 8, 8, kind)]
        diffs[f'{kind}_gpu_b8c8_vs_cpu_8rows'] = float(np.abs(keep[('cuda', 8, 8, kind)][:, :8] - cpu).max())
        diffs[f'{kind}_gpu_b16c16_vs_b8c8'] = float(np.abs(keep[('cuda', 16, 16, kind)] - keep[('cuda', 8, 8, kind)]).max())
    record['max_abs_signature_difference_logits'] = diffs
    record['tolerance'] = 5e-4
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(diffs=diffs, per_row={f"{r['device']}-b{r['batch']}-c{r['head_chunk']}-{r['condition']}-{r['repeat']}": round(r['seconds_per_row'], 4) for r in record['runs']}), indent=1))

if __name__ == '__main__':
    main()
