import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import E2E_FTV6StyleEncoder

DATA_PATH = r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv"
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"

os.makedirs(OUT_DIR, exist_ok=True)

K             = 16
D_K           = 32
D_V           = 4
LR_CHEM       = 1e-4   # ChemBERTa 마지막 레이어
LR_OTHER      = 3e-4   # CrossAttn + Projection
WEIGHT_DECAY  = 1e-4
EPOCHS        = 200
BATCH_SIZE    = 16     # ChemBERTa forward로 메모리 사용량 증가 → 보수적 설정
PATIENCE      = 30
SCHED_PATIENCE = 8
SCHED_FACTOR  = 0.5
SCHED_MIN_LR  = 1e-5
MAX_LENGTH    = 256
SEED          = 42
MODEL_NAME    = "DeepChem/ChemBERTa-77M-MLM"

print("=" * 65)
print("Step 3 E2E: E2E_FTV6StyleEncoder Pre-training (Stage 1)")
print(f"  ChemBERTa last-layer fine-tune lr={LR_CHEM}")
print(f"  CrossAttn/Proj lr={LR_OTHER}  |  k={K}, d_k={D_K}, d_v={D_V}")
print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, early_stop=val_loss")
print("=" * 65)

# 재현성
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
df_meta = pd.read_csv(DATA_PATH)
df_feat = pd.read_csv(FEAT_PATH)
assert list(df_meta["SMILES"]) == list(df_feat["SMILES"]), "SMILES order mismatch"

feat_cols = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all  = df_feat[feat_cols].values.astype(np.float32)
y_all     = df_feat["Label"].values.astype(np.float32)
ref_all   = df_feat["ref"].values
smiles_all = df_meta["SMILES"].values

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

smiles_train = smiles_all[train_mask]
smiles_test  = smiles_all[test_mask]
fp_train_raw = X_fp_all[train_mask]
fp_test_raw  = X_fp_all[test_mask]
y_train      = y_all[train_mask]
y_test       = y_all[test_mask]

# FP scaler 로드 (Step2에서 저장된 scaler 재사용)
scaler_path = os.path.join(DATA_DIR, "scalers_ftv6_cls.pkl")
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step2 first."
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
scaler_fp = scalers["fp"]

fp_train_scaled = scaler_fp.transform(fp_train_raw).astype(np.float32)
fp_test_scaled  = scaler_fp.transform(fp_test_raw).astype(np.float32)

fp_in_dim = fp_train_scaled.shape[1]
print(f"Train: {len(y_train)}  |  Test: {len(y_test)}  |  FP: {fp_in_dim}-dim")

# ── Train / Val 분리 ─────────────────────────────────────────────────────────
idx = np.arange(len(y_train))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_train.astype(int), random_state=SEED
)
print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, fp_array, labels):
        self.smiles  = smiles_list
        self.fp      = fp_array
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.smiles[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.fp[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


train_ds = SMILESDataset(smiles_train[idx_tr], fp_train_scaled[idx_tr], y_train[idx_tr])
val_ds   = SMILESDataset(smiles_train[idx_val], fp_train_scaled[idx_val], y_train[idx_val])
test_ds  = SMILESDataset(smiles_test, fp_test_scaled, y_test)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 모델 ─────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = E2E_FTV6StyleEncoder(
    fp_in_dim=fp_in_dim, k=K, d_k=D_K, d_v=D_V, dropout=0.3,
    model_name=MODEL_NAME,
).to(device)

n_layers = len(encoder.chemberta.encoder.layer)
print(f"ChemBERTa layers: {n_layers}  (frozen: 0~{n_layers-2}, unfrozen: {n_layers-1})")

trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total     = sum(p.numel() for p in encoder.parameters())
print(f"Parameters: trainable={trainable:,} / total={total:,}  |  Device: {device}")

# ── 차등 학습률 (Differential LR) ────────────────────────────────────────────
last_layer_param_ids = {
    id(p) for p in encoder.chemberta.encoder.layer[n_layers - 1].parameters()
}
last_layer_params = [
    p for p in encoder.parameters() if id(p) in last_layer_param_ids
]
other_params = [
    p for p in encoder.parameters() if id(p) not in last_layer_param_ids
]

optimizer = torch.optim.AdamW(
    [
        {"params": last_layer_params, "lr": LR_CHEM},
        {"params": other_params,      "lr": LR_OTHER},
    ],
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
)
criterion = nn.BCEWithLogitsLoss()


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────
def run_inference(model, dl):
    """returns (logits_np, proba_np)"""
    model.eval()
    all_logits, all_proba = [], []
    with torch.no_grad():
        for ids, mask, fp, _ in dl:
            ids, mask, fp = ids.to(device), mask.to(device), fp.to(device)
            logits = model(ids, mask, fp).squeeze(1)
            all_logits.append(logits.cpu())
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
    return torch.cat(all_logits), np.concatenate(all_proba)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
best_val_loss, best_state, patience_cnt = float('inf'), None, 0
y_val_np  = y_train[idx_val]
y_val_t   = torch.tensor(y_val_np, dtype=torch.float32)
y_test_np = y_test

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

    val_logits, val_proba   = run_inference(encoder, val_dl)
    _,          test_proba  = run_inference(encoder, test_dl)
    val_loss  = criterion(val_logits, y_val_t).item()
    val_auc   = roc_auc_score(y_val_np, val_proba)
    test_auc  = roc_auc_score(y_test_np, test_proba)

    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        cur_lr    = optimizer.param_groups[1]["lr"]
        best_mark = " * best" if val_loss < best_val_loss else ""
        print(
            f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f}"
            f" | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
            f" | test_AUC={test_auc:.4f} | lr={cur_lr:.2e}{best_mark}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt  = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

save_path = os.path.join(OUT_DIR, "pretrained_encoder_e2e_cls.pt")
torch.save(best_state, save_path)

print(f"\nBest val loss: {best_val_loss:.4f}")
print(f"Saved: {save_path}")
print("Step 3 E2E OK")
