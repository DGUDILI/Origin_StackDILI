import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import (
    K, D_MODEL, NUM_HEADS, DROPOUT, MODEL_NAME,
    BATCH_SIZE, MAX_LENGTH, SEED, EPOCHS, PATIENCE,
    LR_CHEM, LR_OTHER, WEIGHT_DECAY,
    SCHED_PATIENCE, SCHED_FACTOR, SCHED_MIN_LR,
    DATA_DIR, OUT_DIR,
)
from utils import set_seed, SMILESDataset
from model import E2E_MHAResidualEncoder

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("Step 2: E2E_MHAResidualEncoder Pre-training (Stage 1)")
print(f"  ChemBERTa last-layer lr={LR_CHEM}  |  MHA/Proj lr={LR_OTHER}")
print(f"  k={K}, d_model={D_MODEL}, num_heads={NUM_HEADS}  |  feature_dim={K}")
print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, early_stop=val_AUC")
print("=" * 65)

set_seed(SEED)

# ── 데이터 로드 (Step1 저장 npy 사용) ────────────────────────────────────────────
fp_train_path = os.path.join(DATA_DIR, "fp_full_train.npy")
assert os.path.exists(fp_train_path), f"Missing: {fp_train_path}\nRun Step1 first."

fp_train     = np.load(fp_train_path)
y_train      = np.load(os.path.join(DATA_DIR, "y_train.npy"))
smiles_train = np.load(os.path.join(DATA_DIR, "smiles_train.npy"), allow_pickle=True)
fp_in_dim    = fp_train.shape[1]
n_pos_tr = int(y_train.sum()); n_neg_tr = len(y_train) - n_pos_tr
print(f"Train: {len(y_train)}  |  FP: {fp_in_dim}-dim  |  pos={n_pos_tr}, neg={n_neg_tr}")

# ── Train / Val 분리 ─────────────────────────────────────────────────────────────
idx = np.arange(len(y_train))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_train.astype(int), random_state=SEED
)
print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

# ── Tokenizer & Dataset ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_cache = os.path.join(DATA_DIR, f"train_tokens_{SEED}.pt")
val_cache   = os.path.join(DATA_DIR, f"val_tokens_{SEED}.pt")

train_ds = SMILESDataset(
    smiles_train[idx_tr], fp_train[idx_tr], tokenizer, MAX_LENGTH,
    labels=y_train[idx_tr], cache_path=train_cache,
)
val_ds = SMILESDataset(
    smiles_train[idx_val], fp_train[idx_val], tokenizer, MAX_LENGTH,
    labels=y_train[idx_val], cache_path=val_cache,
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 모델 ─────────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = E2E_MHAResidualEncoder(
    fp_in_dim=fp_in_dim, k=K, d_model=D_MODEL, num_heads=NUM_HEADS,
    dropout=DROPOUT, model_name=MODEL_NAME,
).to(device)

n_layers = len(encoder.chemberta.encoder.layer)
print(f"ChemBERTa layers: {n_layers}  (frozen: 0~{n_layers-2}, unfrozen: {n_layers-1})")
trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total     = sum(p.numel() for p in encoder.parameters())
print(f"Parameters: trainable={trainable:,} / total={total:,}  |  Device: {device}")

# ── 차등 학습률 ──────────────────────────────────────────────────────────────────
last_layer_ids   = {id(p) for p in encoder.chemberta.encoder.layer[n_layers - 1].parameters()}
second_layer_ids = {id(p) for p in encoder.chemberta.encoder.layer[n_layers - 2].parameters()}
chem_ids = last_layer_ids | second_layer_ids
optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in encoder.parameters() if id(p) in last_layer_ids],          "lr": LR_CHEM},        # 1e-4
        {"params": [p for p in encoder.parameters() if id(p) in second_layer_ids],        "lr": LR_CHEM * 0.3},  # 3e-5
        {"params": [p for p in encoder.parameters() if id(p) not in chem_ids],            "lr": LR_OTHER},       # 3e-4
    ],
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
)
# mild pos_weight: train 내부 비율만 보정 (0.56보다 완만하게)
criterion = nn.BCEWithLogitsLoss()


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────────
def run_inference(model, dl):
    model.eval()
    all_logits, all_proba = [], []
    with torch.no_grad():
        for ids, mask, fp, _ in dl:
            ids, mask, fp = ids.to(device), mask.to(device), fp.to(device)
            logits = model(ids, mask, fp).squeeze(1)
            all_logits.append(logits.cpu())
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
    return torch.cat(all_logits), np.concatenate(all_proba)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────────
best_val_auc, best_state, patience_cnt = 0.0, None, 0
y_val_np = y_train[idx_val]
y_val_t  = torch.tensor(y_val_np, dtype=torch.float32)

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    epoch_loss = 0.0

    for ids, mask, fp, y_b in train_dl:
        ids, mask, fp, y_b = (
            ids.to(device), mask.to(device), fp.to(device), y_b.to(device)
        )
        optimizer.zero_grad()
        loss = criterion(encoder(ids, mask, fp).squeeze(1), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(y_b)

    epoch_loss /= len(train_ds)

    val_logits, val_proba = run_inference(encoder, val_dl)
    val_loss = criterion(val_logits, y_val_t).item()
    val_auc  = roc_auc_score(y_val_np, val_proba)

    scheduler.step(val_auc)

    if epoch % 10 == 0 or epoch == 1:
        cur_lr    = optimizer.param_groups[1]["lr"]
        best_mark = " * best" if val_auc > best_val_auc else ""
        print(
            f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f}"
            f" | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
            f" | lr={cur_lr:.2e}{best_mark}"
        )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (best val_AUC={best_val_auc:.4f})")
            break

save_path = os.path.join(OUT_DIR, "pretrained_encoder.pt")
torch.save(best_state, save_path)

print(f"\nBest val AUC: {best_val_auc:.4f}")
print(f"Saved: {save_path}")
print("Step 2 OK")
