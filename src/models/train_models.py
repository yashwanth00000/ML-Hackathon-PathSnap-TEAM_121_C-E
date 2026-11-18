"""
PathSnap — Model Training Module
=================================
Trains and evaluates 7 classifier families for the GPS arc-matching task.

Feature columns (in order):
    0: perp_dist_m        – perpendicular distance GPS→arc (m)
    1: proj_ratio         – along-arc projection ratio [0,1]
    2: bearing_diff_deg   – |GPS heading − arc direction| (°)
    3: speed_mps          – GPS speed (m/s)
    4: time_gap_s         – seconds since last GPS fix
    5: from_node_lon
    6: from_node_lat
    7: to_node_lon
    8: to_node_lat
    9: arc_length_m       – arc length (m)

y: binary  1 = correct arc, 0 = wrong arc
"""

import os
import time
import warnings
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PATHSNAP_ROOT = _HERE.parents[2]          # …/pathsnap/
ARTIFACTS_DIR  = _PATHSNAP_ROOT / "artifacts"
SCALERS_DIR    = ARTIFACTS_DIR / "scalers"
MODELS_DIR     = ARTIFACTS_DIR / "models"
LOGS_DIR       = ARTIFACTS_DIR / "experiment_logs"
PLOTS_DIR      = _PATHSNAP_ROOT / "results" / "plots"

for _d in [SCALERS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "perp_dist_m",
    "proj_ratio",
    "bearing_diff_deg",
    "speed_mps",
    "time_gap_s",
    "from_node_lon",
    "from_node_lat",
    "to_node_lon",
    "to_node_lat",
    "arc_length_m",
]

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS: Dict[str, object] = {
    "logistic_regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    ),
    "naive_bayes": GaussianNB(
        var_smoothing=1e-9,
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        n_jobs=-1,
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_imbalance(y: np.ndarray, threshold: float = 5.0) -> bool:
    """Return True if the minority:majority ratio exceeds `threshold`."""
    counts = np.bincount(y.astype(int))
    if len(counts) < 2:
        return False
    ratio = counts.max() / max(counts.min(), 1)
    return ratio > threshold


def _apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling; fall back to class_weight if unavailable."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info("SMOTE applied: %d → %d samples", len(y), len(y_res))
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed; skipping SMOTE.")
        return X, y


def build_scaler_and_scale(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    save: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Fit StandardScaler on X_train, transform both splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test) if X_test is not None else None
    if save:
        path = SCALERS_DIR / "scaler.joblib"
        joblib.dump(scaler, path)
        logger.info("Scaler saved → %s", path)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a fitted model on held-out data.

    Returns
    -------
    dict with keys: accuracy, macro_f1, log_loss_val, inference_time_s
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        # fallback: one-hot from hard predictions
        y_prob = np.eye(2)[y_pred.astype(int)]
    elapsed = time.perf_counter() - t0

    return {
        "accuracy":        accuracy_score(y_test, y_pred),
        "macro_f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "log_loss_val":    log_loss(y_test, y_prob),
        "inference_time_s": elapsed,
    }


def _cv_model(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, float]:
    """Run stratified k-fold CV and return aggregated metrics."""
    logger.info("CV (%d-fold) → %s", cv, name)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["f1_macro", "accuracy", "neg_log_loss"],
        n_jobs=1,
        return_train_score=False,
    )

    return {
        "model":            name,
        "cv_macro_f1_mean": scores["test_f1_macro"].mean(),
        "cv_macro_f1_std":  scores["test_f1_macro"].std(),
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std":  scores["test_accuracy"].std(),
        "cv_logloss_mean":  -scores["test_neg_log_loss"].mean(),
        "cv_logloss_std":    scores["test_neg_log_loss"].std(),
        "cv_fit_time_mean": scores["fit_time"].mean(),
    }


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Optional[str] = None,
    cv: int = 5,
    handle_imbalance: bool = True,
) -> pd.DataFrame:
    """
    Train + cross-validate every model in MODELS.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_test,  y_test  : held-out test split
    results_dir      : directory to save model_comparison.csv
                       (defaults to artifacts/experiment_logs/)
    cv               : number of CV folds
    handle_imbalance : apply SMOTE if class imbalance > 5:1

    Returns
    -------
    pd.DataFrame  — one row per model with all metrics
    """
    results_dir = Path(results_dir) if results_dir else LOGS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- scale ---
    X_train_s, X_test_s, _ = build_scaler_and_scale(X_train, X_test)

    # --- class imbalance ---
    y_train_cv = y_train
    X_train_cv = X_train_s
    if handle_imbalance and _check_imbalance(y_train):
        logger.info("Class imbalance detected (>5:1). Applying SMOTE …")
        X_train_cv, y_train_cv = _apply_smote(X_train_s, y_train)

    all_rows = []
    fitted_models: Dict[str, object] = {}

    for name, model in MODELS.items():
        logger.info("=== %s ===", name)
        cv_row = _cv_model(name, model, X_train_cv, y_train_cv, cv=cv)

        # Refit on full training set, evaluate on held-out test
        t0 = time.perf_counter()
        model.fit(X_train_cv, y_train_cv)
        fit_time = time.perf_counter() - t0
        fitted_models[name] = model

        test_metrics = evaluate_model(model, X_test_s, y_test)
        cv_row.update({
            "test_accuracy":        test_metrics["accuracy"],
            "test_macro_f1":        test_metrics["macro_f1"],
            "test_log_loss":        test_metrics["log_loss_val"],
            "inference_time_s":     test_metrics["inference_time_s"],
            "train_fit_time_s":     fit_time,
        })
        all_rows.append(cv_row)
        logger.info(
            "  CV macro-F1: %.4f ± %.4f | test macro-F1: %.4f",
            cv_row["cv_macro_f1_mean"], cv_row["cv_macro_f1_std"],
            cv_row["test_macro_f1"],
        )

    df = pd.DataFrame(all_rows).sort_values("cv_macro_f1_mean", ascending=False)
    csv_path = results_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Comparison saved → %s", csv_path)

    # Save every fitted model
    for name, m in fitted_models.items():
        mpath = MODELS_DIR / f"{name}.joblib"
        joblib.dump(m, mpath)

    # Print summary
    print("\n=== Model Comparison (sorted by CV macro-F1) ===")
    print(df[["model", "cv_macro_f1_mean", "cv_macro_f1_std",
              "test_macro_f1", "cv_logloss_mean"]].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PathSnap model training")
    parser.add_argument("--X_train", required=True, help="Path to X_train .npy")
    parser.add_argument("--y_train", required=True, help="Path to y_train .npy")
    parser.add_argument("--X_test",  required=True, help="Path to X_test .npy")
    parser.add_argument("--y_test",  required=True, help="Path to y_test .npy")
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()

    X_tr = np.load(args.X_train)
    y_tr = np.load(args.y_train)
    X_te = np.load(args.X_test)
    y_te = np.load(args.y_test)

    results = train_all(X_tr, y_tr, X_te, y_te, results_dir=args.results_dir)
    print("\nBest model:", results.iloc[0]["model"])
