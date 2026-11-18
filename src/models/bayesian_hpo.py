"""
PathSnap — Bayesian Hyperparameter Optimisation
================================================
Uses scikit-optimize BayesSearchCV to tune the best-performing model
identified during the model-comparison phase.

Usage
-----
    from src.models.bayesian_hpo import run_bayesian_hpo
    best_params, best_score, results_df = run_bayesian_hpo(
        X, y,
        model_name="random_forest",
        n_iter=50,
    )
"""

import os
import logging
import warnings
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file → …/pathsnap/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]

ARTIFACTS_DIR = _ROOT / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
LOGS_DIR      = ARTIFACTS_DIR / "experiment_logs"
PLOTS_DIR     = _ROOT / "results" / "plots"

for _d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Search-space definitions (skopt format)
# ---------------------------------------------------------------------------
try:
    from skopt.space import Real, Integer, Categorical
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not found; Bayesian HPO will be unavailable.")

SEARCH_SPACES: Dict[str, Any] = {}

if _SKOPT_AVAILABLE:
    SEARCH_SPACES = {
        "random_forest": {
            "n_estimators":    Integer(50, 500),
            "max_depth":       Integer(3, 30),
            "min_samples_leaf": Integer(1, 20),
            "max_features":    Categorical(["sqrt", "log2", None]),
            "min_samples_split": Integer(2, 20),
        },
        "gradient_boosting": {
            "n_estimators":  Integer(50, 400),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth":     Integer(2, 10),
            "subsample":     Real(0.5, 1.0),
            "min_samples_leaf": Integer(1, 20),
        },
        "logistic_regression": {
            "C":        Real(1e-4, 100.0, prior="log-uniform"),
            "max_iter": Integer(200, 2000),
        },
        "knn": {
            "n_neighbors": Integer(3, 31),
            "leaf_size":   Integer(10, 60),
            "p":           Integer(1, 2),
        },
        "decision_tree": {
            "max_depth":        Integer(2, 25),
            "min_samples_leaf": Integer(1, 20),
            "min_samples_split": Integer(2, 20),
        },
        "mlp": {
            "alpha":        Real(1e-6, 1e-2, prior="log-uniform"),
            "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        },
    }

# Base model instances to clone during HPO
_BASE_MODELS: Dict[str, Any] = {
    "random_forest": RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "logistic_regression": LogisticRegression(
        class_weight="balanced", solver="lbfgs", random_state=42
    ),
    "knn": KNeighborsClassifier(weights="distance", n_jobs=-1),
    "decision_tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        early_stopping=True,
        random_state=42,
        max_iter=300,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_bayesian_hpo(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "random_forest",
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = "f1_macro",
    save_dir: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    """
    Run Bayesian hyperparameter optimisation with BayesSearchCV.

    Parameters
    ----------
    X, y        : feature matrix and binary labels (pre-scaled recommended)
    model_name  : key into SEARCH_SPACES / _BASE_MODELS
    n_iter      : number of Bayesian optimisation iterations
    cv          : number of CV folds
    scoring     : sklearn scoring string (default "f1_macro")
    save_dir    : override directory for CSV / model artefacts
    verbose     : verbosity level passed to BayesSearchCV

    Returns
    -------
    best_params : dict
    best_score  : float (CV macro-F1 of best config)
    results_df  : pd.DataFrame with all trial results
    """
    if not _SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize is required for Bayesian HPO.\n"
            "Install it with:  pip install scikit-optimize"
        )

    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for '{model_name}'. "
            f"Available: {list(SEARCH_SPACES)}"
        )

    from skopt import BayesSearchCV

    _save_dir = Path(save_dir) if save_dir else LOGS_DIR
    _save_dir.mkdir(parents=True, exist_ok=True)

    base_model = _BASE_MODELS[model_name]
    search_space = SEARCH_SPACES[model_name]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    logger.info(
        "Starting BayesSearchCV for '%s'  (%d iters, %d-fold CV, scoring=%s)",
        model_name, n_iter, cv, scoring,
    )
    t0 = time.perf_counter()

    opt = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=skf,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
        verbose=verbose,
        random_state=42,
        return_train_score=True,
    )
    opt.fit(X, y)

    elapsed = time.perf_counter() - t0
    logger.info("HPO completed in %.1f s", elapsed)
    logger.info("Best score (CV %s): %.4f", scoring, opt.best_score_)
    logger.info("Best params: %s", opt.best_params_)

    # --- Build results DataFrame ---
    cv_results = pd.DataFrame(opt.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score").reset_index(drop=True)

    # Persist
    csv_path = _save_dir / "bayes_hpo_results.csv"
    cv_results.to_csv(csv_path, index=False)
    logger.info("HPO trial results saved → %s", csv_path)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(opt.best_estimator_, model_path)
    logger.info("Best model saved → %s", model_path)

    # Save best-params separately for quick reference
    best_params_path = _save_dir / "best_params.json"
    import json
    with open(best_params_path, "w") as fh:
        json.dump(dict(opt.best_params_), fh, indent=2, default=str)
    logger.info("Best params saved → %s", best_params_path)

    return dict(opt.best_params_), float(opt.best_score_), cv_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PathSnap Bayesian HPO")
    parser.add_argument("--X",          required=True, help=".npy feature matrix")
    parser.add_argument("--y",          required=True, help=".npy label vector")
    parser.add_argument("--model",      default="random_forest",
                        choices=list(_BASE_MODELS))
    parser.add_argument("--n_iter",     type=int, default=50)
    parser.add_argument("--cv",         type=int, default=5)
    parser.add_argument("--save_dir",   default=None)
    args = parser.parse_args()

    X = np.load(args.X)
    y = np.load(args.y)

    # Scale before HPO
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    best_params, best_score, results = run_bayesian_hpo(
        X_s, y,
        model_name=args.model,
        n_iter=args.n_iter,
        cv=args.cv,
        save_dir=args.save_dir,
    )
    print(f"\nBest score : {best_score:.4f}")
    print(f"Best params: {best_params}")
