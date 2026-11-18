"""PathSnap final evaluator deployment using the frozen model."""
import json
import time
from functools import lru_cache
from pathlib import Path

import folium
import joblib
import numpy as np
from shapely.strtree import STRtree
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pathsnap_real_training import FEATURES, K, candidate_features, read_table
from pathsnap_real_hmm import decode, metrics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/evaluator_test'
ART = ROOT / 'artifacts/official_tuning'
app = FastAPI(title='PathSnap Final Evaluator API')
# A separately hosted UI on Laptop 2 can use its own localhost:8083.
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:8083', 'http://127.0.0.1:8083'], allow_methods=['GET'], allow_headers=['*'])


def folders():
    try:
        entries = sorted(DATA.iterdir())
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(503, 'Reserved evaluator dataset is missing. Add it to data/evaluator_test.') from exc
    return {p.name: p for p in entries if p.is_dir() and p.name.isdigit() and (p / (p.name + '.track')).is_file()}


@lru_cache(maxsize=1)
def model():
    cfg = json.loads((ART / 'configuration.json').read_text())
    if cfg['features'] != FEATURES or cfg['k'] != K:
        raise ValueError('Trained model feature schema does not match inference')
    m = joblib.load(ART / 'tuned_model.joblib')
    if m.n_features_in_ != len(FEATURES) or set(m.classes_) != {0, 1}:
        raise ValueError('Unexpected trained classifier')
    return m


def infer(trip_id):
    folder = folders().get(trip_id)
    if folder is None:
        raise HTTPException(404, 'Unknown evaluator trip')
    start = time.perf_counter()
    track = read_table(folder / (trip_id + '.track'), 3)
    nodes = read_table(folder / (trip_id + '.nodes'), 2)
    arcs = read_table(folder / (trip_id + '.arcs'), 2, int)
    track = track[np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)]
    track = track[np.argsort(track[:, 2], kind='stable')]
    if not len(track) or not np.isfinite(nodes).all() or arcs.min() < 0 or arcs.max() >= len(nodes):
        raise HTTPException(422, 'Invalid track or network')
    x, ids, lines, points = candidate_features(track, nodes, arcs)
    m = model()
    prob = m.predict_proba(x.reshape(-1, len(FEATURES)))
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
        raise HTTPException(500, 'Invalid classifier probabilities')
    p = prob[:, int(np.flatnonzero(m.classes_ == 1)[0])].reshape(ids.shape)
    baseline = ids[np.arange(len(ids)), p.argmax(axis=1)]
    decoded = decode(ids, p, x[:, 0, 5], arcs)
    return folder, track, nodes, arcs, ids, lines, points, baseline, decoded, time.perf_counter() - start


def summarize(pred, ref, route, arcs):
    m = metrics(pred, ref, route, arcs)
    return dict(proxy_accuracy=m['proxy_correct']/m['observations'],
                continuity=m['direct_transitions']/m['transitions'] if m['transitions'] else None,
                sequence_overlap_f1=float(m['sequence_overlap_f1']),
                ordered_sequence_similarity=float(m['ordered_sequence_similarity']))


@app.get('/')
def health():
    return {'status':'running', 'mode':'final evaluation', 'model':'Tuned Histogram Gradient Boosting + directed Viterbi'}


@app.get('/trips')
def trips():
    return {'trips': list(folders())}


@app.get('/predict/{trip_id}')
def predict(trip_id: str):
    folder, track, nodes, arcs, ids, lines, points, baseline, decoded, elapsed = infer(trip_id)
    # Read reference labels ONLY after inference. Never return them or use them as features.
    route = read_table(folder / (trip_id + '.route'), 1, int).ravel()
    if route.min() < 0 or route.max() >= len(arcs):
        raise HTTPException(422, 'Invalid reference arc')
    route_ids = np.array(list(dict.fromkeys(route.tolist())), dtype=np.int64)
    q, a = STRtree(lines[route_ids]).query_nearest(points, all_matches=True)
    order = np.lexsort((a, q)); q, a = q[order], a[order]
    first = np.r_[True, q[1:] != q[:-1]]
    if len(q[first]) != len(track):
        raise HTTPException(500, 'Reference alignment failed')
    ref = np.empty(len(track), dtype=np.int64); ref[q[first]] = route_ids[a[first]]
    return {'trip_id':trip_id, 'mode':'final evaluation', 'gps_observations':len(track),
            'model':'Tuned Histogram Gradient Boosting + directed Viterbi',
            'label_method':'nearest-reference proxy; not verified per-fix ground truth',
            'classifier':summarize(baseline, ref, route, arcs), 'hmm':summarize(decoded, ref, route, arcs),
            'candidate_miss_rate':float(np.mean(~(ids == ref[:, None]).any(axis=1))),
            'prediction_seconds':elapsed, 'predicted_route':decoded.tolist()}


@app.get('/map/{trip_id}', response_class=HTMLResponse)
def route_map(trip_id: str):
    _, track, nodes, arcs, ids, _, _, baseline, decoded, _ = infer(trip_id)
    m = folium.Map(location=track[:, :2].mean(axis=0)[::-1].tolist(), zoom_start=14)
    def draw(name, path, color, weight):
        layer = folium.FeatureGroup(name=name)
        for arc in dict.fromkeys(map(int, path)):
            a,b = arcs[arc]
            folium.PolyLine(nodes[[a,b], :2][:, ::-1].tolist(), color=color, weight=weight, opacity=0.85).add_to(layer)
        layer.add_to(m)
    draw('Nearby roads', np.unique(ids), '#777777', 2)
    draw('Classifier only', baseline, '#dc2626', 6)
    draw('HMM Viterbi', decoded, '#16a34a', 4)
    gps = folium.FeatureGroup(name='GPS observations')
    positions = track[:, :2][:, ::-1].tolist()
    folium.PolyLine(positions, color='#2563eb', weight=2, opacity=0.5).add_to(gps)
    # Limit point markers for browser performance; full GPS trace remains visible.
    step = max(1, (len(track)+999)//1000)
    for point in positions[::step]:
        folium.CircleMarker(point, radius=2, color='#2563eb', fill=True, fill_opacity=0.7).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(positions, padding=(24,24))
    return m.get_root().render()


@app.get('/ui', response_class=HTMLResponse)
def ui():
    return (ROOT / 'pathsnap_evaluator_ui.html').read_text(encoding='utf-8')
