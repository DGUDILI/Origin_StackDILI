"""
compute_stackdili_clean_baseline.py

StackDILI를 clean train 데이터로 재학습하고 test(DILIrank) 성능을 측정.
GA로 선택된 피처 셋은 그대로 유지 (Feature/Feature.csv 기준).

출력: DGUDILI_2026/outputs_clean/stackdili_baseline.json
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

try:
    import pandas as pd
    from sklearn.metrics import (
        roc_auc_score, matthews_corrcoef, f1_score,
        accuracy_score, precision_score, recall_score, confusion_matrix,
    )
except ImportError:
    print("[ERROR] pandas / scikit-learn not installed.")
    sys.exit(1)

ROOT        = os.path.dirname(os.path.abspath(__file__))
FEATURE_CSV = os.path.join(ROOT, "Feature", "Feature.csv")
FEAT_CLEAN  = os.path.join(ROOT, "Code",    "Dataset_feature_clean.csv")
MODEL_DIR   = os.path.join(ROOT, "Code",    "Model")
OUT_DIR     = os.path.join(ROOT, "DGUDILI_2026", "outputs_clean")

for p in [FEATURE_CSV, FEAT_CLEAN]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}")
        sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("compute_stackdili_clean_baseline")
print("=" * 60)

# ── GA 선택 피처 컬럼 목록 추출 ────────────────────────────────────────────
df_feat_ref = pd.read_csv(FEATURE_CSV, nrows=0)
ga_cols = [c for c in df_feat_ref.columns if c not in ("SMILES", "Label", "ref")]
print(f"GA-selected features: {len(ga_cols)}")

# ── Clean 데이터에서 동일 피처 추출 ───────────────────────────────────────
df_clean = pd.read_csv(FEAT_CLEAN)

missing = [c for c in ga_cols if c not in df_clean.columns]
if missing:
    print(f"[ERROR] {len(missing)} GA features not found in clean dataset: {missing[:5]}...")
    sys.exit(1)

train_mask = df_clean["ref"] != "DILIrank"
test_mask  = df_clean["ref"] == "DILIrank"

X_train = df_clean.loc[train_mask, ga_cols].values.astype(np.float32)
X_test  = df_clean.loc[test_mask,  ga_cols].values.astype(np.float32)
y_train = df_clean.loc[train_mask, "Label"].values
y_test  = df_clean.loc[test_mask,  "Label"].values

print(f"Clean train: {len(y_train)}  |  Test (DILIrank): {len(y_test)}")

# ── 베이스 모델 재학습 ────────────────────────────────────────────────────
base_names = ["RF", "ET", "HistGB", "XGBoost"]
base_probs_train = []
base_probs_test  = []

for name in base_names:
    path = os.path.join(MODEL_DIR, f"best_model_{name}.pkl")
    if not os.path.exists(path):
        print(f"[ERROR] Missing model: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        model = pickle.load(f)
    model.fit(X_train, y_train)
    base_probs_train.append(model.predict_proba(X_train)[:, 1])
    base_probs_test.append(model.predict_proba(X_test)[:, 1])
    print(f"  Retrained: {name}")

X_stack_train = np.column_stack(base_probs_train)
X_stack_test  = np.column_stack(base_probs_test)

# ── 스태킹 모델 재학습 ────────────────────────────────────────────────────
stack_path = os.path.join(MODEL_DIR, "best_model_stacking.pkl")
if not os.path.exists(stack_path):
    print(f"[ERROR] Missing model: {stack_path}")
    sys.exit(1)
with open(stack_path, "rb") as f:
    stacking = pickle.load(f)
stacking.fit(X_stack_train, y_train)
print("  Retrained: Stacking")

# ── 평가 ──────────────────────────────────────────────────────────────────
proba = stacking.predict_proba(X_stack_test)[:, 1]
pred  = stacking.predict(X_stack_test)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

metrics = {
    "AUC":         float(roc_auc_score(y_test, proba)),
    "MCC":         float(matthews_corrcoef(y_test, pred)),
    "F1":          float(f1_score(y_test, pred)),
    "ACC":         float(accuracy_score(y_test, pred)),
    "Precision":   float(precision_score(y_test, pred)),
    "Sensitivity": float(recall_score(y_test, pred)),
    "Specificity": float(tn / (tn + fp_)),
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 60)
print("StackDILI (clean train) — Test set: DILIrank")
print("=" * 60)
print("  " + "  ".join(f"{c}: {metrics[c]:.4f}" for c in cols))

out_path = os.path.join(OUT_DIR, "stackdili_baseline.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {out_path}")
print("compute_stackdili_clean_baseline OK")
