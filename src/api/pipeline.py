"""
pipeline.py
===========
Loads the trained PathSnap prediction pipeline and runs end-to-end inference
on a single trip dict.

Public API
----------
load_pipeline(model_path, scaler_path, hmm_config_path) -> dict
predict_trip(trip_dict, pipeline)                        -> dict
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running directly
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[2]  # …/pathsnap/src/..  → pathsnap root? No
# _HERE = src/api/pipeline.py → parents[0]=api, parents[1]=src, parents[2]=pathsnap
_PATHSNAP_ROOT = Path(__file__).resolve().parents[2]
if str(_PATHSNAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PATHSNAP_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports from sibling src packages
# ---------------------------------------------------------------------------

def _import_src_modules():
    """Import src.* modules; called once after sys.path is patched."""
    from src.preprocessing.preprocessor import GPSPreprocessor          # noqa: F401
    from src.features.feature_engineering import engineer_features       # noqa: F401
    from src.hmm import HMMDecoder, evaluate_full                        # noqa: F401
    return GPSPreprocessor, engineer_features, HMMDecoder, evaluate_full


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(
    model_path: str | Path,
    scaler_path: str | Path,
    hmm_config_path: str | Path,
) -> Dict[str, Any]:
    """Load all artefacts needed for inference.

    Parameters
    ----------
    model_path      : path to a joblib-serialised sklearn classifier
    scaler_path     : path to a joblib-serialised StandardScaler
    hmm_config_path : path to a joblib-serialised HMMDecoder (or config dict)

    Returns
    -------
    dict with keys:
        ``classifier``  : fitted sklearn model
        ``scaler``      : fitted StandardScaler
        ``hmm``         : HMMDecoder instance
        ``model_name``  : str (stem of model_path)
        ``loaded``      : bool
    """
    model_path      = Path(model_path)
    scaler_path     = Path(scaler_path)
    hmm_config_path = Path(hmm_config_path)

    pipeline: Dict[str, Any] = {
        "classifier": None,
        "scaler":     None,
        "hmm":        None,
        "model_name": model_path.stem,
        "loaded":     False,
    }

    try:
        pipeline["classifier"] = joblib.load(model_path)
        logger.info("Classifier loaded from %s", model_path)
    except Exception as exc:
        logger.error("Failed to load classifier: %s", exc)
        return pipeline

    try:
        pipeline["scaler"] = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)
    except Exception as exc:
        logger.error("Failed to load scaler: %s", exc)
        return pipeline

    try:
        pipeline["hmm"] = joblib.load(hmm_config_path)
        logger.info("HMM decoder loaded from %s", hmm_config_path)
    except Exception as exc:
        logger.error("Failed to load HMM decoder: %s", exc)
        return pipeline

    pipeline["loaded"] = True
    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_trip(
    trip_dict: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full prediction pipeline on one trip.

    Parameters
    ----------
    trip_dict : dict as returned by ``src.preprocessing.data_loader.load_trip``
                Keys: trip_id, track (DataFrame), nodes (DataFrame),
                      arcs (DataFrame), route (list[int] | None)
    pipeline  : dict as returned by :func:`load_pipeline`

    Returns
    -------
    dict with keys:
        predicted_route   : list[int]
        gps_points        : list[[lon, lat]]
        arc_geometries    : list[dict]
        metrics           : dict
        inference_time_ms : float
    """
    GPSPreprocessor, engineer_features, HMMDecoder, evaluate_full = _import_src_modules()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------
    track_raw   = trip_dict["track"]
    nodes_df    = trip_dict["nodes"]
    arcs_df     = trip_dict["arcs"]
    ground_truth: Optional[List[int]] = trip_dict.get("route")

    # Apply GPS preprocessor (drop invalids, smooth timestamps)
    preprocessor = GPSPreprocessor()
    track_clean = preprocessor.transform(track_raw)

    # ------------------------------------------------------------------
    # 2. Feature engineering — produce (X, candidate_arc_ids, obs_indices)
    # ------------------------------------------------------------------
    # engineer_features returns (X, y) for training.
    # For inference we call it without route (y will be None / dummy).
    # We also need to know *which candidate arcs* correspond to each row.
    result = engineer_features(
        trip_dict={**trip_dict, "track": track_clean, "route": None},
        k=5,
        return_meta=True,   # extended signature — returns (X, None, meta_df)
    )

    if isinstance(result, tuple) and len(result) == 3:
        X_raw, _, meta_df = result
    else:
        # Fallback: older signature returning (X, y)
        X_raw, _ = result
        meta_df  = None

    # ------------------------------------------------------------------
    # 3. Scale features
    # ------------------------------------------------------------------
    scaler = pipeline["scaler"]
    X_scaled = scaler.transform(X_raw)

    # ------------------------------------------------------------------
    # 4. Classifier — emit emission probabilities
    # ------------------------------------------------------------------
    clf = pipeline["classifier"]
    proba = clf.predict_proba(X_scaled)  # shape (N_rows, 2)
    # Column 1 = P(arc is correct)
    emission_probs = proba[:, 1]

    # ------------------------------------------------------------------
    # 5. HMM + Viterbi decoding
    # ------------------------------------------------------------------
    hmm: HMMDecoder = pipeline["hmm"]

    if meta_df is not None:
        obs_arc_ids = meta_df["arc_id"].tolist()
        obs_indices = meta_df["obs_idx"].tolist()
    else:
        # Heuristic fallback: assume rows are (obs_i, candidate_j) in order
        n_gps  = len(track_clean)
        k      = len(X_raw) // n_gps if n_gps > 0 else 5
        obs_indices  = [i // k for i in range(len(X_raw))]
        obs_arc_ids  = [int(arcs_df.index[i % len(arcs_df)]) for i in range(len(X_raw))]

    predicted_route = hmm.decode(
        obs_arc_ids=obs_arc_ids,
        obs_indices=obs_indices,
        emission_probs=emission_probs,
        arcs_df=arcs_df,
        nodes_df=nodes_df,
    )

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # ------------------------------------------------------------------
    # 6. Metrics (only if ground truth is available)
    # ------------------------------------------------------------------
    if ground_truth is not None:
        metrics_raw = evaluate_full(predicted_route, ground_truth)
        metrics = {
            "pointwise_accuracy": float(metrics_raw.get("pointwise_accuracy", 0.0)),
            "route_continuity":   float(metrics_raw.get("route_continuity",   0.0)),
            "macro_f1":           float(metrics_raw.get("macro_f1",           0.0)),
        }
    else:
        # No ground truth available — cannot compute accuracy
        metrics = {
            "pointwise_accuracy": -1.0,
            "route_continuity":   -1.0,
            "macro_f1":           -1.0,
        }

    # ------------------------------------------------------------------
    # 7. Build arc_geometries for map rendering
    # ------------------------------------------------------------------
    arc_geometries = []
    unique_arcs = list(dict.fromkeys(predicted_route))  # preserves order, dedup
    for arc_id in unique_arcs:
        if arc_id not in arcs_df.index:
            continue
        row = arcs_df.loc[arc_id]
        fn_id = int(row["from_node_id"])
        to_id = int(row["to_node_id"])
        if fn_id not in nodes_df.index or to_id not in nodes_df.index:
            continue
        fn = nodes_df.loc[fn_id]
        to = nodes_df.loc[to_id]
        arc_geometries.append({
            "arc_id":   int(arc_id),
            "from_lon": float(fn["lon"]),
            "from_lat": float(fn["lat"]),
            "to_lon":   float(to["lon"]),
            "to_lat":   float(to["lat"]),
        })

    # GPS points as [[lon, lat], ...]
    gps_points = track_clean[["lon", "lat"]].values.tolist()

    return {
        "predicted_route":   [int(a) for a in predicted_route],
        "gps_points":        gps_points,
        "arc_geometries":    arc_geometries,
        "metrics":           metrics,
        "inference_time_ms": t_elapsed_ms,
    }
