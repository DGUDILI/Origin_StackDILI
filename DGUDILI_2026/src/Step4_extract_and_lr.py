import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix    = "_clean" if _USE_CLEAN else ""
DATA_DIR   = os.path.join(ROOT, f"data{_suffix}")
OUT_DIR    = os.path.join(ROOT, f"outputs{_suffix}")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K = 16, 32
SEED   = 42

# ── 재현성 시드 설정 ──────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("Step 4: 16-dim Extraction + Multi-Classifier Comparison (Stage 2)")
print("=" * 60)
print(f"Pooling: {POOLING}")

enc_path = os.path.join(OUT_DIR, f"pretrained_encoder_tune_{POOLING}.pt")
assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step3 first."

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = CrossAttentionEncoder(k=K, d_k=D_K).to(device)
encoder.load_state_dict(torch.load(enc_path, map_location=device))
encoder.eval()
print(f"Encoder loaded ({device})")

def load_t(name):
    return torch.tensor(
        np.load(os.path.join(DATA_DIR, name)),
        dtype=torch.float32
    ).to(device)

fp_train   = load_t("fp_k16_train.npy")
fp_test    = load_t("fp_k16_test.npy")
cham_train = load_t(f"cham_full_train_{POOLING}.npy")
cham_test  = load_t(f"cham_full_test_{POOLING}.npy")
y_train    = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with torch.no_grad():
    X_train_16 = encoder.encode(cham_train, fp_train).cpu().numpy()
    X_test_16  = encoder.encode(cham_test,  fp_test).cpu().numpy()

print(f"Feature Space: train={X_train_16.shape}, test={X_test_16.shape}")

classifiers = {
    "LR":  LogisticRegression(max_iter=1000, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                         random_state=SEED, early_stopping=True),
    "RF":  RandomForestClassifier(n_estimators=200, random_state=SEED),
    "XGB": XGBClassifier(n_estimators=200, learning_rate=0.05,
                         max_depth=4, subsample=0.8,
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=SEED),
}

_baseline_defaults = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}
_baseline_json = os.path.join(OUT_DIR, "stackdili_baseline.json")
if os.path.exists(_baseline_json):
    with open(_baseline_json) as f:
        baseline = json.load(f)
    print(f"StackDILI baseline loaded from: {_baseline_json}")
elif _USE_CLEAN:
    print("[WARN] outputs_clean/stackdili_baseline.json not found.")
    print("       Run compute_stackdili_clean_baseline.py first for a fair comparison.")
    print("       Falling back to original StackDILI metrics (NOT comparable).")
    baseline = _baseline_defaults
else:
    baseline = _baseline_defaults

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
all_results = {"StackDILI": baseline}

for clf_name, clf in classifiers.items():
    clf.fit(X_train_16, y_train)
    print(f"{clf_name} trained")
    proba = clf.predict_proba(X_test_16)[:, 1]
    pred  = clf.predict(X_test_16)
    tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()
    all_results[f"DGUDILI_{clf_name}_{POOLING}"] = {
        "AUC":         roc_auc_score(y_test, proba),
        "MCC":         matthews_corrcoef(y_test, pred),
        "F1":          f1_score(y_test, pred),
        "ACC":         accuracy_score(y_test, pred),
        "Precision":   precision_score(y_test, pred),
        "Sensitivity": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp_),
    }

print("\n" + "=" * 88)
print(f"DGUDILI 2026 ({POOLING}) Classifier Comparison  |  Test set: DILIrank (N=452)")
print("=" * 88)
print(f"{'Model':28s}" + "".join(f"{c:>9s}" for c in cols))
print("-" * 88)
for row_name, metrics in all_results.items():
    print(f"{row_name:28s}" + "".join(f"{metrics[c]:>9.4f}" for c in cols))
print("-" * 88)
best_key = max(
    (k for k in all_results if k != "StackDILI"),
    key=lambda k: all_results[k]["AUC"]
)
best = all_results[best_key]
print(f"{'Delta(best vs StackDILI)':28s}" +
      "".join(f"{best[c]-baseline[c]:>+9.4f}" for c in cols))
print(f"  Best classifier: {best_key}")
print("=" * 88)
print(f"\nFeature count: StackDILI ~209 (GA)  ->  DGUDILI 16 (Cross-Attention)")

results_df = pd.DataFrame(all_results).T
csv_path = os.path.join(OUT_DIR, f"results_comparison_tune_{POOLING}.csv")
results_df.to_csv(csv_path)
print(f"Saved: {csv_path}")
print("Step 4 OK")