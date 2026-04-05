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
D_V           = 1
LR_CHEM       = 1e-4   # ChemBERTa 마지막 레이어
LR_OTHER      = 3e-4   # CrossAttn + Projection
WEIGHT_DECAY  = 1e-4
EPOCHS        = 200
BATCH_SIZE    = 16
PATIENCE      = 30
SCHED_PATIENCE = 8
SCHED_FACTOR  = 0.5
SCHED_MIN_LR  = 1e-5
MAX_LENGTH    = 256
SEED          = 42
MODEL_NAME    = "DeepChem/ChemBERTa-77M-MLM"

print("=" * 65)
print("Step 2: E2E_FTV6StyleEncoder Pre-training (Stage 1)")
print(f"  ChemBERTa last-layer lr={LR_CHEM}  |  CrossAttn/Proj lr={LR_OTHER}")
print(f"  k={K}, d_k={D_K}, d_v={D_V}  |  feature_dim={K*D_V}")
print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, early_stop=val_AUC")
print("=" * 65)

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

feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
smiles_all = df_meta["SMILES"].values

train_mask = ref_all != "DILIrank"

smiles_train = smiles_all[train_mask]
y_train      = y_all[train_mask]

scaler_path = os.path.join(DATA_DIR, "scalers.pkl")
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step1 first."
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
scaler_fp = scalers["fp"]

fp_train  = scaler_fp.transform(X_fp_all[train_mask]).astype(np.float32)
fp_in_dim = fp_train.shape[1]
print(f"Train: {len(y_train)}  |  FP: {fp_in_dim}-dim")

# ── Train / Val 분리 ─────────────────────────────────────────────────────────
idx = np.arange(len(y_train))
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=y_train.astype(int), random_state=SEED
)
print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

# ── Tokenizer & Dataset ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, fp_array, labels, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"  [Cache] 토큰 로딩 중: {cache_path}")
            enc = torch.load(cache_path, weights_only=False)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
        else:
            print(f"  [Tokenize] {len(smiles_list)}개 데이터 토크나이징 중... (잠시만 기다려주세요)")
            enc = tokenizer(
                list(smiles_list),
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            if cache_path:
                torch.save({"input_ids": self.input_ids, "attention_mask": self.attention_mask}, cache_path)
                print(f"  [Cache] 토큰 저장 완료: {cache_path}")

        self.fp     = torch.tensor(fp_array, dtype=torch.float32)
        self.labels = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.fp[idx],
            self.labels[idx],
        )


train_cache = os.path.join(DATA_DIR, f"train_tokens_{SEED}.pt")
val_cache   = os.path.join(DATA_DIR, f"val_tokens_{SEED}.pt")

train_ds = SMILESDataset(smiles_train[idx_tr], fp_train[idx_tr], y_train[idx_tr], cache_path=train_cache)
val_ds   = SMILESDataset(smiles_train[idx_val], fp_train[idx_val], y_train[idx_val], cache_path=val_cache)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

# ── 차등 학습률 ──────────────────────────────────────────────────────────────
last_layer_ids = {id(p) for p in encoder.chemberta.encoder.layer[n_layers - 1].parameters()}
optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in encoder.parameters() if id(p) in last_layer_ids],     "lr": LR_CHEM},
        {"params": [p for p in encoder.parameters() if id(p) not in last_layer_ids], "lr": LR_OTHER},
    ],
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
)
criterion = nn.BCEWithLogitsLoss()


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────
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


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
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
