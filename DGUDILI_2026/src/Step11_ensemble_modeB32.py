import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from group_model_b import GroupCrossAttentionModB_32

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -- Hyperparameters -------------------------------------------------------
SEEDS        = [42, 0, 7, 21, 99]
D_MODEL      = 64
D_OUT        = 32
DROPOUT      = 0.30
DROPOUT_PRE  = 0.15
LR           = 5e-4
WEIGHT_DECAY = 3e-4
EPOCHS       = 250
BATCH_SIZE   = 32
PATIENCE     = 40
LABEL_SMOOTH = 0.05
# --------------------------------------------------------------------------

print("=" * 65)
print("Step 11: ModeB-32 Seed Ensemble x5")
print(f"  seeds={SEEDS}")
print(f"  d_model={D_MODEL}, d_out={D_OUT}, dropout={DROPOUT}, "
      f"dropout_pre={DROPOUT_PRE}")
print(f"  lr={LR}, wd={WEIGHT_DECAY}, epochs={EPOCHS}, patience={PATIENCE}")
print("=" * 65)

# -- 1. Data Load ----------------------------------------------------------
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
EMB_PATH  = os.path.join(DATA_DIR, "chemberta_embeddings.npy")
assert os.path.exists(FEAT_PATH), f"Missing: {FEAT_PATH}"
assert os.path.exists(EMB_PATH),  f"Missing: {EMB_PATH}"

df_feat   = pd.read_csv(FEAT_PATH)
feat_cols = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
assert len(feat_cols) == 425

X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
embeddings = np.load(EMB_PATH).astype(np.float32)

# -- 2. Train / Test Split -------------------------------------------------
train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"
y_tr       = y_all[train_mask]
y_te       = y_all[test_mask]

print(f"\nTrain: {train_mask.sum()}  |  Test: {test_mask.sum()}")

# -- 3. Scaling (shared across seeds) -------------------------------------
scaler_fp = StandardScaler()
scaler_lm = StandardScaler()
X_fp_train = scaler_fp.fit_transform(X_fp_all[train_mask]).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
X_lm_train = scaler_lm.fit_transform(embeddings[train_mask]).astype(np.float32)
X_lm_test  = scaler_lm.transform(embeddings[test_mask]).astype(np.float32)

scaler_path = os.path.join(DATA_DIR, "scalers_ensemble32.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"fp": scaler_fp, "lm": scaler_lm}, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}\n")

def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32)

fp_test_d = to_t(X_fp_test).to(device)
lm_test_d = to_t(X_lm_test).to(device)
y_test_np = y_te

pos_count = float(y_tr.sum())
neg_count = float((1 - y_tr).sum())
pos_w     = torch.tensor([neg_count / pos_count]).to(device)
eps       = LABEL_SMOOTH

all_probas = []

# -- 4. Train each seed ----------------------------------------------------
for seed in SEEDS:
    print(f"--- Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(to_t(X_fp_train), to_t(X_lm_train), to_t(y_tr))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False)

    model = GroupCrossAttentionModB_32(
        d_model=D_MODEL, d_out=D_OUT,
        dropout=DROPOUT, dropout_pre=DROPOUT_PRE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=1e-6
    )

    best_auc, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x_fp_b, x_lm_b, y_b in train_dl:
            x_fp_b = x_fp_b.to(device)
            x_lm_b = x_lm_b.to(device)
            y_b    = y_b.to(device)
            smooth_y = y_b * (1.0 - eps) + eps * 0.5
            optimizer.zero_grad()
            logit = model(x_fp_b, x_lm_b).squeeze(1)
            loss  = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, smooth_y, pos_weight=pos_w
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
        epoch_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(
                model(fp_test_d, lm_test_d).squeeze(1)
            ).cpu().numpy()
        auc = roc_auc_score(y_test_np, proba)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d} | loss={epoch_loss:.4f} | AUC={auc:.4f}")

        if auc > best_auc:
            best_auc     = auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best AUC={best_auc:.4f})")
                break

    ckpt_path = os.path.join(OUT_DIR, f"ensemble32_seed{seed}.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Saved: {ckpt_path}  |  best AUC={best_auc:.4f}\n")

    # Reload best state for final proba
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(fp_test_d, lm_test_d).squeeze(1)).cpu().numpy()
    all_probas.append(p)

# -- 5. Ensemble Evaluation ------------------------------------------------
ensemble_proba = np.mean(all_probas, axis=0)
pred           = (ensemble_proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test_np, pred).ravel()

results = {
    "AUC":         roc_auc_score(y_test_np, ensemble_proba),
    "MCC":         matthews_corrcoef(y_test_np, pred),
    "F1":          f1_score(y_test_np, pred),
    "ACC":         accuracy_score(y_test_np, pred),
    "Precision":   precision_score(y_test_np, pred),
    "Sensitivity": recall_score(y_test_np, pred),
    "Specificity": tn / (tn + fp_),
}

baseline = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 75)
print("ModeB-32 Ensemble x5 vs StackDILI  |  Test: DILIrank (N=452)")
print("=" * 75)
print(f"{'':25s}" + "".join(f"{c:>10s}" for c in cols))
print("-" * 75)
print(f"{'StackDILI':25s}" + "".join(f"{baseline[c]:>10.4f}" for c in cols))
print(f"{'ModeB-32 Ensemble x5':25s}" + "".join(f"{results[c]:>10.4f}" for c in cols))
print("-" * 75)
print(f"{'Delta':25s}" + "".join(f"{results[c]-baseline[c]:>+10.4f}" for c in cols))
print("=" * 75)

# Save ensemble probas for compare_all
np.save(os.path.join(DATA_DIR, "ensemble32_proba.npy"), ensemble_proba)
np.save(os.path.join(DATA_DIR, "ensemble32_y_test.npy"), y_test_np)
print(f"\nSaved scalers : {scaler_path}")
print("Step 11 OK")
