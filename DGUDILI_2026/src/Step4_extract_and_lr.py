import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K = 16, 32

print("=" * 60)
print("Step 4: 16-dim Extraction + Logistic Regression (Stage 2)")
print("=" * 60)

enc_path = os.path.join(OUT_DIR, "pretrained_encoder.pt")
assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step3 first."

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = CrossAttentionEncoder(k=K, d_k=D_K).to(device)
encoder.load_state_dict(torch.load(enc_path, map_location=device))
encoder.eval()
print(f"Encoder loaded ({device})")

def load_t(name):
    return torch.tensor(np.load(os.path.join(DATA_DIR, name)),
                        dtype=torch.float32).to(device)

fp_train   = load_t("fp_k16_train.npy")
fp_test    = load_t("fp_k16_test.npy")
cham_train = load_t("cham_k16_train.npy")
cham_test  = load_t("cham_k16_test.npy")
y_train    = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test     = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with torch.no_grad():
    X_train_16 = encoder.encode(cham_train, fp_train).cpu().numpy()
    X_test_16  = encoder.encode(cham_test,  fp_test).cpu().numpy()

print(f"Feature Space: train={X_train_16.shape}, test={X_test_16.shape}")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_16, y_train)
print("LogisticRegression trained")

proba = lr_model.predict_proba(X_test_16)[:, 1]
pred  = lr_model.predict(X_test_16)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

dgudili = {
    "AUC":         roc_auc_score(y_test, proba),
    "MCC":         matthews_corrcoef(y_test, pred),
    "F1":          f1_score(y_test, pred),
    "ACC":         accuracy_score(y_test, pred),
    "Precision":   precision_score(y_test, pred),
    "Sensitivity": recall_score(y_test, pred),
    "Specificity": tn / (tn + fp_),
}
baseline = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 70)
print("DGUDILI 2026 vs StackDILI  |  Test set: DILIrank (N=452)")
print("=" * 70)
print(f"{'':22s}" + "".join(f"{c:>11s}" for c in cols))
print("-" * 70)
print(f"{'StackDILI':22s}" + "".join(f"{baseline[c]:>11.4f}" for c in cols))
print(f"{'DGUDILI_2026':22s}" + "".join(f"{dgudili[c]:>11.4f}" for c in cols))
print("-" * 70)
print(f"{'Delta(+up)':22s}" + "".join(f"{dgudili[c]-baseline[c]:>+11.4f}" for c in cols))
print("=" * 70)
print(f"\nFeature count: StackDILI ~209 (GA)  ->  DGUDILI 16 (Cross-Attention)")

results_df = pd.DataFrame([baseline, dgudili], index=["StackDILI", "DGUDILI_2026"])
csv_path = os.path.join(OUT_DIR, "results_comparison.csv")
results_df.to_csv(csv_path)
print(f"Saved: {csv_path}")
print("Step 4 OK")
