import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import FTV6StyleEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K, D_V = 16, 32, 4
SEED   = 42

print("=" * 60)
print("Step 4: ft_v6-style — 16-dim extract + Stacking OOF (Stage 2)")
print("=" * 60)
print(f"Pooling: {POOLING}")

enc_path = os.path.join(OUT_DIR, f"pretrained_encoder_ftv6_{POOLING}.pt")
assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step3 first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_t(name):
    return torch.tensor(
        np.load(os.path.join(DATA_DIR, name)),
        dtype=torch.float32
    ).to(device)

fp_train   = load_t("fp_full_train.npy")
fp_test    = load_t("fp_full_test.npy")
cham_train = load_t(f"cham_full_train_{POOLING}.npy")
cham_test  = load_t(f"cham_full_test_{POOLING}.npy")
y_train    = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))

fp_in_dim   = fp_train.shape[1]
chem_in_dim = cham_train.shape[1]

encoder = FTV6StyleEncoder(
    fp_in_dim=fp_in_dim, chem_in_dim=chem_in_dim, k=K, d_k=D_K, d_v=D_V
).to(device)
encoder.load_state_dict(torch.load(enc_path, map_location=device))
encoder.eval()
print(f"Encoder loaded ({device}) | fp_in_dim={fp_in_dim}, chem_in_dim={chem_in_dim}")

# ── 16-dim Feature 추출 ──────────────────────────────────────────────────────
with torch.no_grad():
    X_train_feat = encoder.encode(cham_train, fp_train).cpu().numpy()
    X_test_feat  = encoder.encode(cham_test,  fp_test).cpu().numpy()

print(f"Feature Space: train={X_train_feat.shape}, test={X_test_feat.shape}")

# ── Base Models ──────────────────────────────────────────────────────────────
BASE_MODELS = [
    ("RF",     RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
    ("ET",     ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
    ("HistGB", HistGradientBoostingClassifier(max_iter=300, random_state=SEED)),
    ("XGB",    XGBClassifier(
                   n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", random_state=SEED)),
]

kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
n_base = len(BASE_MODELS)

oof_probs  = np.zeros((len(X_train_feat), n_base))
test_probs = np.zeros((len(X_test_feat),  n_base))

# ── 5-Fold OOF Stacking ──────────────────────────────────────────────────────
print(f"\n[Stacking] 5-Fold OOF training ({n_base} base models)...")
for i, (name, clf_template) in enumerate(BASE_MODELS):
    fold_test_preds = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_feat, y_train)):
        clf_fold = clone(clf_template)
        clf_fold.fit(X_train_feat[tr_idx], y_train[tr_idx])
        oof_probs[val_idx, i] = clf_fold.predict_proba(X_train_feat[val_idx])[:, 1]
        fold_test_preds.append(clf_fold.predict_proba(X_test_feat)[:, 1])
    test_probs[:, i] = np.mean(fold_test_preds, axis=0)
    oof_auc = roc_auc_score(y_train, oof_probs[:, i])
    print(f"  {name:8s}: OOF AUC={oof_auc:.4f}")

# ── LR Meta-model ────────────────────────────────────────────────────────────
print("\n[Meta-model] LogisticRegression on OOF probs...")
meta = LogisticRegression(max_iter=1000, random_state=SEED)
meta.fit(oof_probs, y_train)

final_proba = meta.predict_proba(test_probs)[:, 1]
final_pred  = (final_proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, final_pred).ravel()

# ── 평가 결과 ────────────────────────────────────────────────────────────────
dgudili = {
    "AUC":         roc_auc_score(y_test, final_proba),
    "MCC":         matthews_corrcoef(y_test, final_pred),
    "F1":          f1_score(y_test, final_pred),
    "ACC":         accuracy_score(y_test, final_pred),
    "Precision":   precision_score(y_test, final_pred),
    "Sensitivity": recall_score(y_test, final_pred),
    "Specificity": tn / (tn + fp_),
}
baseline = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}
prev_dgudili = {
    "AUC": 0.8013, "MCC": 0.4484, "F1": 0.6950, "ACC": 0.7146,
    "Precision": 0.6151, "Sensitivity": 0.7989, "Specificity": 0.6567,
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 75)
print(f"ft_v6-style DGUDILI ({POOLING}) vs StackDILI  |  DILIrank N=452")
print("=" * 75)
print(f"{'':25s}" + "".join(f"{c:>10s}" for c in cols))
print("-" * 75)
print(f"{'StackDILI (목표)':25s}" + "".join(f"{baseline[c]:>10.4f}" for c in cols))
print(f"{'이전 DGUDILI (GroupedCA)':25s}" + "".join(f"{prev_dgudili[c]:>10.4f}" for c in cols))
print(f"{'DGUDILI_ftv6':25s}" + "".join(f"{dgudili[c]:>10.4f}" for c in cols))
print("-" * 75)
print(f"{'vs StackDILI':25s}" + "".join(f"{dgudili[c]-baseline[c]:>+10.4f}" for c in cols))
print(f"{'vs 이전 버전':25s}" + "".join(f"{dgudili[c]-prev_dgudili[c]:>+10.4f}" for c in cols))
print("=" * 75)
print(f"\nPipeline: FP {fp_in_dim}-dim → Linear({fp_in_dim},{K}) [learnable] → CrossAttn → {K}-dim")
print(f"          Stacking: RF/ET/HistGB/XGB (5-Fold OOF) → LR meta-model")

results_df = pd.DataFrame(
    [baseline, prev_dgudili, dgudili],
    index=["StackDILI", "DGUDILI_GroupedCA", f"DGUDILI_ftv6_{POOLING}"]
)
csv_path = os.path.join(OUT_DIR, f"results_ftv6_{POOLING}.csv")
results_df.to_csv(csv_path)
print(f"Saved: {csv_path}")
print("Step 4 OK")