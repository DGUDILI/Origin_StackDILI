import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
OUT_DIR    = os.path.join(ROOT, "outputs")
SRC_DIR    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import FTV6StyleEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K, D_V    = 16, 32, 4
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
EPOCHS         = 300
BATCH_SIZE     = 32
PATIENCE       = 40
SCHED_PATIENCE = 10
SCHED_FACTOR   = 0.5
SCHED_MIN_LR   = 1e-5
SEED           = 42

print("=" * 60)
print("Step 3: FTV6StyleEncoder Pre-training (Stage 1)")
print(f"  pooling={POOLING}, k={K}, d_k={D_K}, d_v={D_V}, lr={LR}, wd={WEIGHT_DECAY}")
print(f"  epochs={EPOCHS}, patience={PATIENCE}, early_stop=val_loss")
print("=" * 60)

# Fix-2: 완전한 재현성 보장
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load(name):
    path = os.path.join(DATA_DIR, name)
    assert os.path.exists(path), f"Missing: {path}\nRun Step2 first."
    return torch.tensor(np.load(path), dtype=torch.float32)

fp_train   = load("fp_full_train.npy")      # (1398, 425) 전체 FP
fp_test    = load("fp_full_test.npy")        # (452,  425)
cham_train = load(f"cham_full_train_{POOLING}.npy")
cham_test  = load(f"cham_full_test_{POOLING}.npy")
y_train    = load("y_train.npy")
y_test     = load("y_test.npy")

fp_in_dim   = fp_train.shape[1]
chem_in_dim = cham_train.shape[1]
print(f"FP: {fp_in_dim}-dim (full) | ChemBERTa: {chem_in_dim}-dim")

# Fix-1: Train → Val 분리 (stratified 15%, early stopping은 Val 기준)
idx = np.arange(len(y_train))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_train.numpy().astype(int), random_state=SEED
)

cham_tr, fp_tr, y_tr    = cham_train[idx_tr], fp_train[idx_tr], y_train[idx_tr]
cham_val, fp_val, y_val  = cham_train[idx_val], fp_train[idx_val], y_train[idx_val]

print(f"Train: {fp_train.shape[0]} (tr={len(idx_tr)}, val={len(idx_val)})  |  Test: {fp_test.shape[0]}")

train_ds = TensorDataset(cham_tr, fp_tr, y_tr)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = FTV6StyleEncoder(
    fp_in_dim=fp_in_dim, chem_in_dim=chem_in_dim,
    k=K, d_k=D_K, d_v=D_V, dropout=0.3
).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
)

n_params = sum(p.numel() for p in encoder.parameters())
print(f"Parameters: {n_params:,}  |  Device: {device}")

fp_val_d    = fp_val.to(device)
cham_val_d  = cham_val.to(device)
y_val_d     = y_val.to(device)
y_val_np    = y_val.numpy()

fp_test_d   = fp_test.to(device)
cham_test_d = cham_test.to(device)
y_test_np   = y_test.numpy()

best_val_loss, best_state, patience_cnt = float('inf'), None, 0

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    epoch_loss = 0.0

    for x_c, x_f, y_b in train_dl:
        x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)

        optimizer.zero_grad()
        loss = criterion(encoder(x_c, x_f).squeeze(1), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * len(y_b)

    epoch_loss /= len(train_ds)

    encoder.eval()
    with torch.no_grad():
        val_logits  = encoder(cham_val_d, fp_val_d).squeeze(1)
        val_loss    = criterion(val_logits, y_val_d).item()
        val_proba   = torch.sigmoid(val_logits).cpu().numpy()
        test_proba  = torch.sigmoid(encoder(cham_test_d, fp_test_d)).squeeze(1).cpu().numpy()
    val_auc  = roc_auc_score(y_val_np, val_proba)
    test_auc = roc_auc_score(y_test_np, test_proba)

    scheduler.step(val_loss)

    if epoch % 20 == 0 or epoch == 1:
        cur_lr    = optimizer.param_groups[0]['lr']
        best_mark = " * best" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f} | test_AUC={test_auc:.4f} | lr={cur_lr:.2e}{best_mark}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt  = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

save_path = os.path.join(OUT_DIR, f"pretrained_encoder_ftv6_{POOLING}.pt")
torch.save(best_state, save_path)

print(f"\nBest val loss: {best_val_loss:.4f}")
print(f"Saved: {save_path}")
print("Step 3 OK")