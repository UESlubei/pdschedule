"""Fit a context-aware work proxy from historical fixed-cap runs only."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import nnls
from apb.common import ROOT, dump, digest

OLD = Path('/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320')

def main():
    train, valid, sources = [], [], []
    for seed in [17, 29, 43, 71, 101]:
        for cap in [512, 1024, 2048, 4096]:
            dest = OLD / 'raw/formal' / f'default-fixed{cap}-seed{seed}'
            windows = [json.loads(p.read_text()) for p in dest.glob('*/metrics.json')]
            path = dest / 'scheduler.jsonl'
            sources.append({'path': str(path), 'sha256': digest(path)})
            steps = {}
            for line in path.open():
                row = json.loads(line)
                if row['type'] == 'step':
                    steps[row['step']] = row
                    continue
                s = steps.pop(row['step'])
                if not any(w['started_ns'] <= row['completed_ns'] <= w['ended_ns'] for w in windows):
                    continue
                if not s['prompt'] or not s['decode_tokens']:
                    continue
                # Only existing Decode token intervals, not first-token events.
                gaps = [dt for rid, n, dt in row['events']
                        if dt is not None and n == 1 and s['allocations'].get(rid, [0, 0, 0])[2] > 0]
                if not gaps:
                    continue
                attention = sum((p + replay) * (pos + (p + replay + 1) / 2)
                                for p, replay, d, pos, n in s['allocations'].values())
                decode_context = sum(pos * d for p, replay, d, pos, n in s['allocations'].values())
                x = [1, s['prompt'] / 1024, attention / 1e6, s['decode_tokens'], decode_context / 1e4]
                item = (x, float(np.median(gaps)))
                (train if seed in [17, 29, 43] else valid).append(item)
    x, y = np.array([r[0] for r in train]), np.array([r[1] for r in train])
    coef, _ = nnls(x, y)
    # A single robust refit removes extreme host stalls; validation remains unfiltered.
    residual = y - x @ coef
    mask = abs(residual - np.median(residual)) <= max(5., 4 * np.median(abs(residual - np.median(residual))))
    coef, _ = nnls(x[mask], y[mask])
    vx, vy = np.array([r[0] for r in valid]), np.array([r[1] for r in valid])
    prediction = vx @ coef
    result = dict(features=['intercept', 'prefill/1024', 'attention_pairs/1e6', 'decode_count', 'decode_context/1e4'],
                  coefficients=coef.tolist(), train_steps=len(train), retained_steps=int(mask.sum()), valid_steps=len(valid),
                  validation_mae_ms=float(np.mean(abs(vy - prediction))),
                  validation_absolute_error_p95_ms=float(np.percentile(abs(vy - prediction), 95)),
                  context_weight=coef[2] / 1e6 / (coef[1] / 1024),
                  train_seeds=[17, 29, 43], validation_seeds=[71, 101], sources=sources,
                  note='Host completed Decode interval model, not GPU kernel timing. Equalization uses only the prefill/context ratio.')
    dump(ROOT / 'reports/optimization_v2/cost_fit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))

if __name__ == '__main__':
    main()
