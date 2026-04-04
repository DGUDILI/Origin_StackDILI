"""
compute_stackdili_clean_baseline.py

StackDILI를 clean train 데이터로 재학습하고 test(DILIrank) 성능을 측정.
GA로 선택된 피처 셋은 그대로 유지 (Feature/Feature.csv 기준).

[수정] 원본 코드의 test set 기반 모델 선택(data leakage) 문제를 수정:
  - train → 80% train_sub / 20% val (stratified) 분할
  - base 모델 5회, stacking 10회 random seed 시도
  - val AUC 기준으로 best 선택 → test 1회 최종 평가

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
    from sklearn.model_selection import StratifiedShuffleSplit
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

N_TRIALS_BASE  = 5   # base 모델 random seed 시도 횟수 (원본과 동일)
N_TRIALS_STACK = 10  # stacking random seed 시도 횟수 (원본과 동일)
VAL_RATIO      = 0.2
SEED           = 42
np.random.seed(SEED)

print("=" * 60)
print("compute_stackdili_clean_baseline  (val-based selection)")
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

X_train_all = df_clean.loc[train_mask, ga_cols].values.astype(np.float32)
X_test      = df_clean.loc[test_mask,  ga_cols].values.astype(np.float32)
y_train_all = df_clean.loc[train_mask, "Label"].values
y_test      = df_clean.loc[test_mask,  "Label"].values

print(f"Clean train: {len(y_train_all)}  |  Test (DILIrank): {len(y_test)}")

# ── Train → train_sub (80%) / val (20%) 분할 ─────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
tr_idx, val_idx = next(sss.split(X_train_all, y_train_all))
X_tr,  X_val  = X_train_all[tr_idx],  X_train_all[val_idx]
y_tr,  y_val  = y_train_all[tr_idx],  y_train_all[val_idx]

print(f"  train_sub: {len(y_tr)}  |  val: {len(y_val)}")

# ── 베이스 모델 재학습 (val AUC 기준 best seed 선택) ──────────────────────
base_names        = ["RF", "ET", "HistGB", "XGBoost"]
best_base_models  = {}

for name in base_names:
    path = os.path.join(MODEL_DIR, f"best_model_{name}.pkl")
    if not os.path.exists(path):
        print(f"[ERROR] Missing model: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        template = pickle.load(f)

    best_val_auc = -np.inf
    best_model   = None

    for i in range(N_TRIALS_BASE):
        seed_i = np.random.randint(0, 10000)
        m = pickle.loads(pickle.dumps(template))
        m.set_params(random_state=seed_i)
        m.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model   = m

    best_base_models[name] = best_model
    print(f"  {name}: best val AUC = {best_val_auc:.4f}")

# ── base 모델 예측: train_sub / val / test ─────────────────────────────────
X_tr_stack  = np.column_stack([best_base_models[n].predict_proba(X_tr)[:, 1]  for n in base_names])
X_val_stack = np.column_stack([best_base_models[n].predict_proba(X_val)[:, 1] for n in base_names])
X_te_stack  = np.column_stack([best_base_models[n].predict_proba(X_test)[:, 1] for n in base_names])

# ── 스태킹 모델 재학습 (val AUC 기준 best seed 선택) ──────────────────────
stack_path = os.path.join(MODEL_DIR, "best_model_stacking.pkl")
if not os.path.exists(stack_path):
    print(f"[ERROR] Missing model: {stack_path}")
    sys.exit(1)
with open(stack_path, "rb") as f:
    stack_template = pickle.load(f)

best_st_val_auc = -np.inf
best_stacking   = None

for i in range(N_TRIALS_STACK):
    seed_i = np.random.randint(0, 10000)
    st = pickle.loads(pickle.dumps(stack_template))
    st.set_params(random_state=seed_i)
    st.fit(X_tr_stack, y_tr)
    val_auc = roc_auc_score(y_val, st.predict_proba(X_val_stack)[:, 1])
    if val_auc > best_st_val_auc:
        best_st_val_auc = val_auc
        best_stacking   = st

print(f"  Stacking: best val AUC = {best_st_val_auc:.4f}")

# ── 최종 평가 (test 1회) ──────────────────────────────────────────────────
proba = best_stacking.predict_proba(X_te_stack)[:, 1]
pred  = best_stacking.predict(X_te_stack)
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
print("StackDILI (clean, val-based) — Test set: DILIrank")
print("=" * 60)
print("  " + "  ".join(f"{c}: {metrics[c]:.4f}" for c in cols))

out_path = os.path.join(OUT_DIR, "stackdili_baseline.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {out_path}")
print("compute_stackdili_clean_baseline OK")
