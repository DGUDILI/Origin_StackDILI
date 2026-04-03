"""
compute_stackdili_cv_baseline.py

StackDILI를 10-Fold CV로 평가.
GA로 선택된 피처는 고정 (Feature/Feature.csv 기준).
각 fold에서 base models + stacking meta-learner를 재학습.

환경변수:
  USE_CLEAN_DATA=1  → Code/Dataset_feature_clean.csv 사용

출력: DGUDILI_2026/outputs_cv[_clean]/stackdili_baseline.json
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, matthews_corrcoef, f1_score,
        accuracy_score, precision_score, recall_score, confusion_matrix,
    )
except ImportError:
    print("[ERROR] pandas / scikit-learn not installed.")
    sys.exit(1)

ROOT       = os.path.dirname(os.path.abspath(__file__))
_USE_CLEAN = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix    = "_clean" if _USE_CLEAN else ""

FEATURE_CSV = os.path.join(ROOT, "Feature", "Feature.csv")
FEAT_DATA   = os.path.join(ROOT, "Code", f"Dataset_feature{_suffix}.csv")
MODEL_DIR   = os.path.join(ROOT, "Code", "Model")
OUT_DIR     = os.path.join(ROOT, "DGUDILI_2026", f"outputs_cv{_suffix}")

for p in [FEATURE_CSV, FEAT_DATA]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}")
        sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 10
SEED    = 42

print("=" * 60)
print(f"compute_stackdili_cv_baseline  ({'CLEAN' if _USE_CLEAN else 'ORIGINAL'} data)")
print("=" * 60)

# GA-selected feature columns
df_ref  = pd.read_csv(FEATURE_CSV, nrows=0)
ga_cols = [c for c in df_ref.columns if c not in ("SMILES", "Label", "ref")]
print(f"GA-selected features: {len(ga_cols)}")

df = pd.read_csv(FEAT_DATA)
missing = [c for c in ga_cols if c not in df.columns]
if missing:
    print(f"[ERROR] {len(missing)} GA features not found in dataset.")
    sys.exit(1)

X = df[ga_cols].values.astype(np.float32)
y = df["Label"].values
print(f"Samples: {len(y)}")

base_names = ["RF", "ET", "HistGB", "XGBoost"]

def load_model(name):
    path = os.path.join(MODEL_DIR, f"best_model_{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_stacking():
    with open(os.path.join(MODEL_DIR, "best_model_stacking.pkl"), "rb") as f:
        return pickle.load(f)

outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
inner_skf = StratifiedKFold(n_splits=5,       shuffle=True, random_state=SEED)
cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(outer_skf.split(X, y), 1):
    print(f"\n── Fold {fold}/{N_FOLDS}  train={len(train_idx)}, test={len(test_idx)} ──")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV: OOF predictions on train fold → train stacking meta-learner
    oof_probs = np.zeros((len(train_idx), len(base_names)))
    for i, name in enumerate(base_names):
        for inner_tr, inner_val in inner_skf.split(X_train, y_train):
            m = load_model(name)
            m.fit(X_train[inner_tr], y_train[inner_tr])
            oof_probs[inner_val, i] = m.predict_proba(X_train[inner_val])[:, 1]

    # Retrain base models on full outer train fold → get test predictions
    test_probs = np.zeros((len(test_idx), len(base_names)))
    for i, name in enumerate(base_names):
        m = load_model(name)
        m.fit(X_train, y_train)
        test_probs[:, i] = m.predict_proba(X_test)[:, 1]
        print(f"  Trained: {name}")

    # Retrain stacking meta-learner on OOF
    stacking = load_stacking()
    stacking.fit(oof_probs, y_train)
    print("  Trained: Stacking")

    proba = stacking.predict_proba(test_probs)[:, 1]
    pred  = stacking.predict(test_probs)
    tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

    m_dict = {
        "AUC":         float(roc_auc_score(y_test, proba)),
        "MCC":         float(matthews_corrcoef(y_test, pred)),
        "F1":          float(f1_score(y_test, pred)),
        "ACC":         float(accuracy_score(y_test, pred)),
        "Precision":   float(precision_score(y_test, pred)),
        "Sensitivity": float(recall_score(y_test, pred)),
        "Specificity": float(tn / (tn + fp_)),
    }
    fold_metrics.append(m_dict)
    print(f"  AUC={m_dict['AUC']:.4f}  MCC={m_dict['MCC']:.4f}  F1={m_dict['F1']:.4f}")

avg = {c: float(np.mean([m[c] for m in fold_metrics])) for c in cols}
print("\n" + "=" * 60)
print("StackDILI 10-Fold CV Average:")
print("  " + "  ".join(f"{c}: {avg[c]:.4f}" for c in cols))

out_path = os.path.join(OUT_DIR, "stackdili_baseline.json")
with open(out_path, "w") as f:
    json.dump(avg, f, indent=2)
print(f"\nSaved: {out_path}")
print("compute_stackdili_cv_baseline OK")
