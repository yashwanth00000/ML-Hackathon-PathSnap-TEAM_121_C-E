"""Development-only Bayesian tuning of PathSnap HistGradientBoosting.
Uses nearest-reference proxy labels, never evaluator data.
Run from the project root after pathsnap_real_training.py prepare and train.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix
from pathsnap_real_training import FEATURES, K, SEED

ROOT = Path.cwd()
CACHE = ROOT / 'data/processed'
OUT = ROOT / 'results/official_tuning'
ART = ROOT / 'artifacts/official_tuning'
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
with np.load(CACHE / 'train.npz') as a:
    X, y, g = (a[k] for k in ('X','y','group'))
with np.load(CACHE / 'student_test.npz') as a:
    tx, ty, tg, ids, fn, tn = (a[k] for k in ('X','y','group','ids','from_node','to_node'))
assert len(set(g)) == 80 and len(set(tg)) == 10 and not (set(g) & set(tg))
assert X.shape[1:] == tx.shape[1:] == (K, len(FEATURES))
folds = list(GroupKFold(n_splits=5).split(X, groups=g))

# No student-test observations enter the search objective.
def build(params):
    return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=SEED)

def objective(trial):
    params = dict(max_iter=trial.suggest_int('max_iter', 60, 160, step=20),
                  learning_rate=trial.suggest_float('learning_rate', 0.025, 0.2, log=True),
                  max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 63, step=8),
                  min_samples_leaf=trial.suggest_int('min_samples_leaf', 20, 100, step=20),
                  l2_regularization=trial.suggest_float('l2_regularization', 1e-5, 5.0, log=True))
    scores = []
    for tr, va in folds:
        m = build(params).fit(X[tr].reshape(-1,len(FEATURES)), y[tr].ravel())
        pred = m.predict(X[va].reshape(-1,len(FEATURES)))
        scores.append(float(f1_score(y[va].ravel(), pred, average='macro', zero_division=0)))
    trial.set_user_attr('fold_scores', scores)
    return float(np.mean(scores))

def persist(study, trial):
    study.trials_dataframe().to_csv(OUT / 'bayesian_trials.csv', index=False)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5))
start = time.perf_counter()
study.optimize(objective, n_trials=15, callbacks=[persist])
search_seconds = time.perf_counter() - start
m = build(study.best_params)
start = time.perf_counter()
m.fit(X.reshape(-1,len(FEATURES)), y.ravel())
fit_seconds = time.perf_counter() - start
start = time.perf_counter()
p = m.predict_proba(tx.reshape(-1,len(FEATURES)))
inference_seconds = time.perf_counter() - start
assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
raw = m.classes_[p.argmax(axis=1)]
pos = int(np.flatnonzero(m.classes_ == 1)[0])
ix = np.arange(len(tx))
top = p[:,pos].reshape(-1,K).argmax(axis=1)
chosen = ids[ix,top]
cf, ct = fn[ix,top], tn[ix,top]
within = tg[1:] == tg[:-1]
cont = (chosen[1:] == chosen[:-1]) | (cf[1:] == ct[:-1])
row = dict(model='Histogram Gradient Boosting (tuned)',cv_macro_f1=study.best_value,cv_f1_std=np.std(study.best_trial.user_attrs['fold_scores']),candidate_accuracy=accuracy_score(ty.ravel(),raw),student_macro_f1=f1_score(ty.ravel(),raw,average='macro',zero_division=0),precision=precision_score(ty.ravel(),raw,zero_division=0),recall=recall_score(ty.ravel(),raw,zero_division=0),top1_proxy_accuracy=np.mean(ty[ix,top]),candidate_miss_rate=np.mean(~ty.any(axis=1)),candidate_log_loss=log_loss(ty.ravel(),p,labels=m.classes_),direct_route_continuity=np.mean(cont[within]),training_seconds=fit_seconds,inference_seconds=inference_seconds)
base = pd.read_csv(ROOT / 'results/official_baseline/model_comparison.csv')
base = base.loc[base.model == 'Histogram Gradient Boosting']
assert len(base) == 1
comparison = pd.concat([base,pd.DataFrame([row])],ignore_index=True)
comparison.to_csv(OUT / 'tuned_comparison.csv',index=False)
np.savetxt(OUT / 'tuned_confusion.csv',confusion_matrix(ty.ravel(),raw,labels=[0,1]),fmt='%d',delimiter=',')
joblib.dump(m, ART / 'tuned_model.joblib')
(ART / 'configuration.json').write_text(json.dumps(dict(features=FEATURES,k=K,seed=SEED,best_params=study.best_params,best_trial=study.best_trial.number,cv_macro_f1=study.best_value,search_seconds=search_seconds,trials=len(study.trials),label_method='nearest reference segment proxy; not verified per-fix labels'),indent=2))
print('\nBest parameters:',study.best_params)
print('Search seconds:',search_seconds)
print(comparison.round(4).to_string(index=False))
print('\nDEVELOPMENT BAYESIAN OPTIMISATION COMPLETE — PROXY LABELS')
