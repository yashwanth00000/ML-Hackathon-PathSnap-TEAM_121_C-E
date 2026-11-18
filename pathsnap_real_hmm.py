"""Development-only directed Viterbi evaluation on the existing PathSnap feature cache.
No evaluator data is accessed. Pointwise targets are nearest-reference proxies.
"""
from pathlib import Path
from collections import deque, Counter
from functools import lru_cache
from difflib import SequenceMatcher
import json
import time
import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
FEATURES = 7
MAX_HOPS = 4
BETA = 0.5
GAP_SECONDS = 60.0
DISCONNECTED_PENALTY = 12.0


def decode(ids, probs, gaps, arcs):
    """Viterbi over directed candidate arcs; hop-based transition potentials."""
    adj = {}
    for u, v in arcs:
        adj.setdefault(int(u), []).append(int(v))

    @lru_cache(maxsize=100000)
    def reachable(start):
        d = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            if d[u] >= MAX_HOPS - 1:
                continue
            for v in adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    @lru_cache(maxsize=200000)
    def transition(a, b):
        if a == b:
            return 0.0
        hops = reachable(int(arcs[a, 1])).get(int(arcs[b, 0]))
        return -BETA * (hops + 1) if hops is not None else -DISCONNECTED_PENALTY

    T, K = ids.shape
    if T == 0:
        return np.empty(0, dtype=int)
    assert np.isfinite(probs).all() and ids.min() >= 0 and ids.max() < len(arcs)
    p = np.clip(probs, 1e-12, 1.0)
    emit = np.log(p / p.sum(axis=1, keepdims=True))
    starts = [0] + (np.flatnonzero((gaps[1:] > GAP_SECONDS) | (gaps[1:] <= 0)) + 1).tolist() + [T]
    out = np.empty(T, dtype=int)
    back = np.zeros((T, K), dtype=np.int32)
    for start, end in zip(starts[:-1], starts[1:]):
        delta = emit[start].copy()
        for t in range(start + 1, end):
            trans = np.array([[transition(int(a), int(b)) for b in ids[t]] for a in ids[t-1]])
            scores = delta[:, None] + trans
            back[t] = scores.argmax(axis=0)
            delta = emit[t] + scores.max(axis=0)
        j = int(delta.argmax())
        for t in range(end - 1, start - 1, -1):
            out[t] = ids[t, j]
            if t > start:
                j = int(back[t, j])
    return out


def collapse(path):
    return [int(a) for i, a in enumerate(path) if i == 0 or a != path[i-1]]


def metrics(pred, ref, route, arcs):
    valid = (pred[:-1] == pred[1:]) | (arcs[pred[:-1], 1] == arcs[pred[1:], 0])
    a, b = collapse(pred), collapse(route)
    # This measures arc coverage, not chronological accuracy.
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    return dict(proxy_correct=int((pred == ref).sum()), observations=len(ref),
                direct_transitions=int(valid.sum()), transitions=len(valid),
                sequence_overlap_f1=2*overlap/(len(a)+len(b)) if a or b else 1.0,
                ordered_sequence_similarity=SequenceMatcher(None, a, b, autojunk=False).ratio())


def main():
    out = ROOT / 'results/official_hmm'
    out.mkdir(parents=True, exist_ok=True)
    art = ROOT / 'artifacts/official_tuning'
    cfg = json.loads((art / 'configuration.json').read_text())
    expected = ['distance_m','bearing_difference_deg','speed_mps','arc_length_m','projection_fraction','time_gap_s','candidate_rank']
    assert cfg['features'] == expected
    model = joblib.load(art / 'tuned_model.joblib')
    with np.load(ROOT / 'data/processed/student_test.npz') as cache:
        X, ids, group, ref = (cache[k] for k in ('X','ids','group','ref'))
    assert X.shape[1:] == (cfg['k'], FEATURES) and len(np.unique(group)) == 10
    folders = {int(p.name): p for p in (ROOT / 'data/student_test').iterdir()
               if p.is_dir() and p.name.isdigit()}
    if set(map(int, np.unique(group))) != set(folders):
        raise ValueError("Student-test folders do not match feature cache")
    rows, predictions = [], []
    for trip in np.unique(group):
        idx = np.flatnonzero(group == trip)
        folder = folders[int(trip)]
        arcs = np.loadtxt(folder / (folder.name + '.arcs'), dtype=int, ndmin=2)
        x, candidates = X[idx], ids[idx]
        start = time.perf_counter()
        p = model.predict_proba(x.reshape(-1, FEATURES))
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        p = p[:, positive].reshape(candidates.shape)
        baseline = candidates[np.arange(len(idx)), p.argmax(axis=1)]
        classifier_seconds = time.perf_counter() - start
        start = time.perf_counter()
        decoded = decode(candidates, p, x[:, 0, 5], arcs)
        decoder_seconds = time.perf_counter() - start
        # Reference is accessed only after both predictions have been produced.
        route = np.loadtxt(folder / (folder.name + '.route'), dtype=int, ndmin=1)
        for method, pred, seconds in [('classifier', baseline, classifier_seconds), ('hmm_viterbi', decoded, classifier_seconds + decoder_seconds)]:
            m = metrics(pred, ref[idx], route, arcs)
            rows.append(dict(trip_id=int(trip), method=method, inference_seconds=seconds, **m))
            print(f'{trip} {method}: proxy accuracy={m["proxy_correct"]/m["observations"]:.4f}, direct continuity={m["direct_transitions"]/max(1,m["transitions"]):.4f}', flush=True)
        predictions.append(pd.DataFrame({'trip_id':int(trip),'observation_index':np.arange(len(idx)), 'classifier_arc':baseline,'hmm_arc':decoded}))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'per_trip_metrics.csv', index=False)
    # Prediction-only IDs; do not publish this file without checking data-sharing rules.
    pd.concat(predictions, ignore_index=True).to_csv(out / 'student_predictions.csv', index=False)
    summary = []
    for name, f in frame.groupby('method'):
        summary.append(dict(method=name, top1_proxy_accuracy=f.proxy_correct.sum()/f.observations.sum(), direct_route_continuity=f.direct_transitions.sum()/f.transitions.sum(), mean_sequence_overlap_f1=f.sequence_overlap_f1.mean(), mean_ordered_sequence_similarity=f.ordered_sequence_similarity.mean(), total_inference_seconds=f.inference_seconds.sum()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'hmm_comparison.csv', index=False)
    (out / 'decoder_config.json').write_text(json.dumps(dict(beta=BETA, max_hops=MAX_HOPS, disconnected_penalty=DISCONNECTED_PENALTY, gap_seconds=GAP_SECONDS, parameter_selection='initial fixed configuration; not optimized', label_method='nearest-reference proxy', sequence_metric='collapsed arc multiset F1 and difflib ordered matching similarity; skipped intermediate arcs not inserted'), indent=2))
    print('\nDEVELOPMENT HMM COMPARISON (PROXY LABELS)', flush=True)
    print(summary.round(4).to_string(index=False))
    print('\nSaved:', out)


if __name__ == '__main__':
    main()
