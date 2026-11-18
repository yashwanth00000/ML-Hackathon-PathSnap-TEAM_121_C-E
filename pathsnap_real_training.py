"""PathSnap development-only baseline: prepare official trips and compare five models.
Reference .route files are ordered traversed arcs, not per-fix labels. This baseline
uses nearest reference segment as a provisional per-fix target; it is NOT verified
pointwise ground truth and does not enforce reference sequence order.
Never reads evaluator_test. Features are independent of route labels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree
from pyproj import Transformer
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix
import joblib

FEATURES = ['distance_m', 'bearing_difference_deg', 'speed_mps', 'arc_length_m', 'projection_fraction', 'time_gap_s', 'candidate_rank']
K = 5
SEED = 42


def read_table(path, ncols, dtype=float):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if a.shape[1] != ncols or not len(a):
        raise ValueError(f'Unexpected shape {a.shape}: {path}')
    return a


def candidate_features(track, nodes, arcs, k=K):
    """Return candidate features and arc IDs; no reference labels are accepted."""
    if len(arcs) < k:
        raise ValueError('Fewer road segments than requested candidates')
    lon, lat = np.median(track[:, :2], axis=0)
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    crs = (32600 if lat >= 0 else 32700) + zone
    tf = Transformer.from_crs(4326, crs, always_xy=True)
    gps = np.column_stack(tf.transform(track[:, 0], track[:, 1]))
    xy = np.column_stack(tf.transform(nodes[:, 0], nodes[:, 1]))
    ends = xy[arcs]
    lines = shapely.linestrings(ends)
    tree = STRtree(lines)
    points = shapely.points(gps)
    ids = np.empty((len(track), k), dtype=np.int32)
    # Find exact nearest k segments in the projected CRS by expanding the radius
    # only for observations with fewer than k segments inside the current radius.
    for start in range(0, len(track), 512):
        pending = np.arange(start, min(start + 512, len(track)))
        radius = 25.0
        while len(pending):
            pair = tree.query(points[pending], predicate='dwithin', distance=radius)
            if pair.size:
                q, a = pair
                dist = shapely.distance(points[pending[q]], lines[a])
                order = np.lexsort((a, dist, q))
                q, a = q[order], a[order]
                count = np.bincount(q, minlength=len(pending))
                offsets = np.cumsum(count) - count
                good = np.flatnonzero(count >= k)
                if len(good):
                    ids[pending[good]] = a[offsets[good, None] + np.arange(k)]
                pending = pending[count < k]
            radius *= 2
            if radius > 100000 and len(pending):
                raise ValueError('Candidate search exceeded 100 km; inspect coordinates')
    vector = ends[:, 1] - ends[:, 0]
    length = np.hypot(vector[:, 0], vector[:, 1])
    delta = np.diff(gps, axis=0, prepend=gps[:1])
    dt = np.diff(track[:, 2], prepend=track[:1, 2])
    speed = np.divide(np.hypot(delta[:, 0], delta[:, 1]), dt,
                      out=np.zeros(len(dt)), where=dt > 0)
    heading = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360
    arc_heading = np.degrees(np.arctan2(vector[:, 0], vector[:, 1])) % 360
    angle = np.abs((heading[:, None] - arc_heading[ids] + 180) % 360 - 180)
    angle[(dt <= 0) | (speed < 0.5)] = 90.0
    v = vector[ids]
    relative = gps[:, None, :] - ends[ids, 0]
    denom = np.sum(v * v, axis=2)
    ratio = np.clip(np.divide(np.sum(relative * v, axis=2), denom,
                     out=np.zeros_like(denom), where=denom > 0), 0, 1)
    diff = relative - ratio[:, :, None] * v
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    x = np.stack([dist, angle, np.broadcast_to(speed[:, None], ids.shape),
                  length[ids], ratio, np.broadcast_to(dt[:, None], ids.shape),
                  np.broadcast_to(np.arange(k), ids.shape)], axis=2).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite candidate features')
    return x, ids, lines, points


def prepare_trip(folder):
    stem = folder.name
    track = read_table(folder / (stem + '.track'), 3)
    nodes = read_table(folder / (stem + '.nodes'), 2)
    arcs = read_table(folder / (stem + '.arcs'), 2, int)
    route = read_table(folder / (stem + '.route'), 1, int).ravel()
    raw_count = len(track)
    track = track[np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)]
    track = track[np.argsort(track[:, 2], kind='stable')]
    if not len(track) or not np.isfinite(nodes).all():
        raise ValueError(f'Invalid track or network: {folder}')
    if arcs.min() < 0 or arcs.max() >= len(nodes) or route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f'Invalid node or arc index: {folder}')
    X, ids, lines, points = candidate_features(track, nodes, arcs)
    # The reference is consulted only after candidate construction.
    route_ids = np.array(list(dict.fromkeys(route.tolist())), dtype=np.int64)
    query = STRtree(lines[route_ids]).query_nearest(points, all_matches=True)
    q, a = query
    order = np.lexsort((a, q))
    q, a = q[order], a[order]
    first = np.r_[True, q[1:] != q[:-1]]
    ref = np.empty(len(track), dtype=np.int64)
    ref[q[first]] = route_ids[a[first]]
    y = (ids == ref[:, None]).astype(np.uint8)
    inventory = dict(trip_id=stem, raw_gps=raw_count, valid_gps=len(track),
                     nodes=len(nodes), arcs=len(arcs), reference_sequence_length=len(route),
                     candidate_miss_rate=float(np.mean(~y.any(axis=1))))
    return dict(X=X, y=y, ids=ids, from_node=arcs[ids, 0], to_node=arcs[ids, 1], ref=ref), inventory


def prepare(root):
    cache = root / 'data' / 'processed'; cache.mkdir(parents=True, exist_ok=True)
    out = root / 'results' / 'official_baseline'; out.mkdir(parents=True, exist_ok=True)
    inventory = []
    seen = set()
    for split, expected in [('train', 80), ('student_test', 10)]:
        folders = sorted(p for p in (root / 'data' / split).iterdir() if p.is_dir() and (p / (p.name + '.track')).is_file())
        if len(folders) != expected:
            raise ValueError(f'{split}: expected {expected} trips, found {len(folders)}')
        chunks = []
        for i, folder in enumerate(folders, 1):
            if folder.name in seen:
                raise ValueError('Trip overlap across development partitions')
            seen.add(folder.name)
            data, inv = prepare_trip(folder)
            data['group'] = np.full(len(data['X']), int(folder.name), dtype=np.int32)
            chunks.append(data)
            inventory.append(dict(partition=split, **inv))
            print(f'{split} {i}/{expected}: {folder.name}, {inv["valid_gps"]} fixes, miss={inv["candidate_miss_rate"]:.3f}', flush=True)
        np.savez_compressed(cache / (split + '.npz'), **{key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]})
    pd.DataFrame(inventory).to_csv(out / 'dataset_inventory.csv', index=False)
    print('Feature extraction complete. Targets are nearest-reference proxies, not verified per-fix labels.', flush=True)


def models():
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=500, random_state=SEED)),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=10, n_jobs=2, random_state=SEED),
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=31, early_stopping=False, random_state=SEED),
    }


def train(root):
    cache = root / 'data' / 'processed'
    out = root / 'results' / 'official_baseline'; out.mkdir(parents=True, exist_ok=True)
    artifacts = root / 'artifacts' / 'official_baseline'; artifacts.mkdir(parents=True, exist_ok=True)
    with np.load(cache / 'train.npz') as a:
        X, y, group = a['X'], a['y'], a['group']
    with np.load(cache / 'student_test.npz') as a:
        tx, ty, tg, tf, tt, ti = (a[key] for key in ('X', 'y', 'group', 'from_node', 'to_node', 'ids'))
    if X.shape[1:] != (K, len(FEATURES)) or tx.shape[1:] != (K, len(FEATURES)):
        raise ValueError('Unexpected feature cache schema')
    if set(group) & set(tg):
        raise ValueError('Group leakage')
    xf, yf = X.reshape(-1, len(FEATURES)), y.ravel()
    xt, yt = tx.reshape(-1, len(FEATURES)), ty.ravel()
    folds = list(GroupKFold(n_splits=5).split(X, groups=group))
    rows, fold_rows = [], []
    best = -1.0
    best_name = None
    configurations = {}
    for name, model in models().items():
        print(f'\nTraining {name}', flush=True)
        configurations[name] = {k: str(v) for k, v in model.get_params().items()}
        scores = []
        for f, (tr, va) in enumerate(folds, 1):
            m = clone(model).fit(X[tr].reshape(-1, len(FEATURES)), y[tr].ravel())
            pred = m.predict(X[va].reshape(-1, len(FEATURES)))
            score = f1_score(y[va].ravel(), pred, average='macro', zero_division=0)
            scores.append(score)
            fold_rows.append({'model': name, 'fold': f, 'macro_f1': score})
            print(f'  Fold {f}/5: Macro F1 {score:.4f}', flush=True)
        tick = time.perf_counter()
        fitted = clone(model).fit(xf, yf)
        train_sec = time.perf_counter() - tick
        tick = time.perf_counter()
        prob = fitted.predict_proba(xt)
        inference_sec = time.perf_counter() - tick
        if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
            raise ValueError(f'Invalid probabilities for {name}')
        pos = int(np.flatnonzero(fitted.classes_ == 1)[0])
        raw = fitted.classes_[np.argmax(prob, axis=1)]
        top = prob[:, pos].reshape(-1, K).argmax(axis=1)
        ix = np.arange(len(tx))
        chosen = ti[ix, top]
        chosen_from, chosen_to = tf[ix, top], tt[ix, top]
        within = tg[1:] == tg[:-1]
        contiguous = (chosen[1:] == chosen[:-1]) | (chosen_from[1:] == chosen_to[:-1])
        row = dict(model=name, cv_macro_f1=np.mean(scores), cv_f1_std=np.std(scores),
                   candidate_accuracy=accuracy_score(yt, raw), student_macro_f1=f1_score(yt, raw, average='macro', zero_division=0),
                   precision=precision_score(yt, raw, zero_division=0), recall=recall_score(yt, raw, zero_division=0),
                   top1_proxy_accuracy=np.mean(ty[ix, top]), candidate_miss_rate=np.mean(~ty.any(axis=1)),
                   candidate_log_loss=log_loss(yt, prob, labels=fitted.classes_),
                   direct_route_continuity=np.mean(contiguous[within]), training_seconds=train_sec, inference_seconds=inference_sec)
        rows.append(row)
        slug = name.lower().replace(' ', '_')
        joblib.dump(fitted, artifacts / (slug + '.joblib'))
        np.savetxt(out / (slug + '_confusion.csv'), confusion_matrix(yt, raw, labels=[0, 1]), fmt='%d', delimiter=',')
        if row['cv_macro_f1'] > best:
            best, best_name = row['cv_macro_f1'], name
            joblib.dump(fitted, artifacts / 'best_untuned.joblib')
        pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(out / 'cv_folds.csv', index=False)
        print(row, flush=True)
    (artifacts / 'configuration.json').write_text(json.dumps({'features': FEATURES, 'k': K, 'seed': SEED,
        'best_by_cv_macro_f1': best_name, 'label_method': 'nearest reference segment proxy; no sequence alignment', 'models': configurations}, indent=2))
    df = pd.DataFrame(rows).sort_values('cv_macro_f1', ascending=False)
    print('\nDEVELOPMENT BASELINE: PROXY LABELS', flush=True)
    print(df.round(4).to_string(index=False), flush=True)
    print('Best by training-trip CV:', best_name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('cv_macro_f1', 'student_macro_f1', 'top1_proxy_accuracy', 'candidate_log_loss', 'direct_route_continuity'):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(df.model, df[metric]); ax.invert_yaxis(); ax.set_xlabel(metric); ax.set_title('Development baseline — proxy labels'); fig.tight_layout()
        fig.savefig(out / (metric + '.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    args = p.parse_args()
    (prepare if args.stage == 'prepare' else train)(args.root.resolve())
