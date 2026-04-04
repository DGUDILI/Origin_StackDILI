import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix    = "_clean" if _USE_CLEAN else ""
DATA_DIR   = os.path.join(ROOT, f"data{_suffix}")
OUT_DIR    = os.path.join(ROOT, f"outputs_cv{_suffix}")
SRC_DIR    = os.path.join(ROOT, "src")
_STACKDILI_ROOT = os.environ.get("STACKDILI_ROOT", os.path.dirname(ROOT))

sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K, D_V = 16, 32, 16
LR_RATE     = 1e-3
EPOCHS      = 200
BATCH_SIZE  = 32
PATIENCE    = 20
VAL_RATIO   = 0.2
N_FOLDS     = 10
SEED        = 42

FEAT_PATH = os.path.join(_STACKDILI_ROOT, "Code", f"Dataset_feature{_suffix}.csv")
DATA_PATH = os.path.join(_STACKDILI_ROOT, "Data", f"Dataset{_suffix}.csv")

for p in [FEAT_PATH, DATA_PATH]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}")
        sys.exit(1)

emb_path = os.path.join(DATA_DIR, f"chemberta_embeddings_{POOLING}.npy")
if not os.path.exists(emb_path):
    print(f"[ERROR] Missing: {emb_path}\nRun Step1 first.")
    sys.exit(1)

print("=" * 60)
print(f"Step CV: DGUDILI 10-Fold CV  ({'CLEAN' if _USE_CLEAN else 'ORIGINAL'} data)")
print("=" * 60)

df_meta = pd.read_csv(DATA_PATH)
df_feat = pd.read_csv(FEAT_PATH)
assert list(df_meta["SMILES"]) == list(df_feat["SMILES"]), \
    "SMILES order mismatch between Dataset.csv and Dataset_feature.csv"

feat_cols = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all  = df_feat[feat_cols].values.astype(np.float32)
y_all     = df_feat["Label"].values.astype(np.float32)
embeddings = np.load(emb_path).astype(np.float32)

print(f"Samples: {len(y_all)}  |  FP features: {len(feat_cols)}  |  ChemBERTa: {embeddings.shape[1]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_fp_all, y_all), 1):
    print(f"\n── Fold {fold}/{N_FOLDS}  train={len(train_idx)}, test={len(test_idx)} ──")

    X_fp_tr, X_fp_te = X_fp_all[train_idx], X_fp_all[test_idx]
    X_ch_tr, X_ch_te = embeddings[train_idx], embeddings[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    # FP: SelectKBest per fold (fit on train only)
    np.random.seed(SEED)
    sel_fp = SelectKBest(mutual_info_classif, k=K)
    sel_fp.fit(X_fp_tr, y_tr)
    fp_idx = sel_fp.get_support(indices=True)
    X_fp_tr, X_fp_te = X_fp_tr[:, fp_idx], X_fp_te[:, fp_idx]

    sc_fp = RobustScaler()
    X_fp_tr = sc_fp.fit_transform(X_fp_tr).astype(np.float32)
    X_fp_te = sc_fp.transform(X_fp_te).astype(np.float32)

    # ChemBERTa: full embedding + StandardScaler (chem_proj inside encoder)
    sc_ch = StandardScaler()
    X_ch_tr = sc_ch.fit_transform(X_ch_tr).astype(np.float32)
    X_ch_te = sc_ch.transform(X_ch_te).astype(np.float32)

    # CrossAttentionEncoder (chem_proj 내장, full 768-dim 입력)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    encoder   = CrossAttentionEncoder(k=K, d_k=D_K, d_v=D_V).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR_RATE)

    fp_tr_t = torch.tensor(X_fp_tr).to(device)
    ch_tr_t = torch.tensor(X_ch_tr).to(device)
    y_tr_t  = torch.tensor(y_tr).to(device)
    fp_te_t = torch.tensor(X_fp_te).to(device)
    ch_te_t = torch.tensor(X_ch_te).to(device)

    # train/val split (stratified) for early stopping
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    tr_idx, val_idx = next(sss.split(np.zeros(len(y_tr)), y_tr))

    full_ds  = TensorDataset(ch_tr_t, fp_tr_t, y_tr_t)
    train_ds = Subset(full_ds, tr_idx)
    val_ds   = Subset(full_ds, val_idx)

    dl     = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    best_val_auc, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        for x_c, x_f, y_b in dl:
            x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(encoder(x_c, x_f).squeeze(1), y_b)
            loss.backward()
            optimizer.step()

        encoder.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for x_c, x_f, y_b in val_dl:
                x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)
                val_logits.append(torch.sigmoid(encoder(x_c, x_f).squeeze(1)).cpu())
                val_labels.append(y_b.cpu())
        auc_val = roc_auc_score(
            torch.cat(val_labels).numpy(),
            torch.cat(val_logits).numpy()
        )

        if auc_val > best_val_auc:
            best_val_auc = auc_val
            best_state   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    encoder.load_state_dict(best_state)
    encoder.eval()

    with torch.no_grad():
        X_tr_16 = encoder.encode(ch_tr_t, fp_tr_t).cpu().numpy()
        X_te_16 = encoder.encode(ch_te_t, fp_te_t).cpu().numpy()

    lr_model = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_model.fit(X_tr_16, y_tr)

    proba = lr_model.predict_proba(X_te_16)[:, 1]
    pred  = lr_model.predict(X_te_16)
    tn, fp_, fn, tp = confusion_matrix(y_te, pred).ravel()

    m = {
        "AUC":         float(roc_auc_score(y_te, proba)),
        "MCC":         float(matthews_corrcoef(y_te, pred)),
        "F1":          float(f1_score(y_te, pred)),
        "ACC":         float(accuracy_score(y_te, pred)),
        "Precision":   float(precision_score(y_te, pred)),
        "Sensitivity": float(recall_score(y_te, pred)),
        "Specificity": float(tn / (tn + fp_)),
    }
    fold_metrics.append(m)
    print(f"  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}  F1={m['F1']:.4f}  (best encoder AUC={best_val_auc:.4f})")

# Average metrics
avg = {c: float(np.mean([m[c] for m in fold_metrics])) for c in cols}
std = {c: float(np.std([m[c] for m in fold_metrics]))  for c in cols}

# Load StackDILI CV baseline
baseline_path = os.path.join(OUT_DIR, "stackdili_baseline.json")
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"\nStackDILI CV baseline: {baseline_path}")
else:
    baseline = None
    print(f"\n[WARN] StackDILI CV baseline not found: {baseline_path}")
    print("       Run compute_stackdili_cv_baseline.py first.")

print("\n" + "=" * 80)
print(f"DGUDILI 2026 (10-Fold CV) vs StackDILI  |  {'CLEAN' if _USE_CLEAN else 'ORIGINAL'} data")
print("=" * 80)
print(f"{'':22s}" + "".join(f"{c:>11s}" for c in cols))
print("-" * 80)
if baseline:
    print(f"{'StackDILI (CV)':22s}" + "".join(f"{baseline[c]:>11.4f}" for c in cols))
print(f"{'DGUDILI_2026 (CV)':22s}" + "".join(f"{avg[c]:>11.4f}" for c in cols))
print(f"{'  (±std)':22s}" + "".join(f"{std[c]:>11.4f}" for c in cols))
if baseline:
    print("-" * 80)
    print(f"{'Delta(+up)':22s}" + "".join(f"{avg[c]-baseline[c]:>+11.4f}" for c in cols))
print("=" * 80)

rows = []
if baseline:
    rows.append({"model": "StackDILI", **baseline})
rows.append({"model": "DGUDILI_2026", **avg})
rows.append({"model": "DGUDILI_2026_std", **std})

pd.DataFrame(rows).set_index("model").to_csv(os.path.join(OUT_DIR, "results_cv.csv"))
print(f"\nSaved: {os.path.join(OUT_DIR, 'results_cv.csv')}")
print("Step CV OK")
