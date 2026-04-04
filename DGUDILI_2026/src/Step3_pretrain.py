import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix    = "_clean" if _USE_CLEAN else ""
DATA_DIR   = os.path.join(ROOT, f"data{_suffix}")
OUT_DIR    = os.path.join(ROOT, f"outputs{_suffix}")
SRC_DIR    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K, D_V  = 16, 32, 16
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 200
BATCH_SIZE   = 32
PATIENCE     = 20
VAL_RATIO    = 0.2        # 훈련 데이터의 20%를 validation으로 분리
SEED         = 42

print("=" * 60)
print("Step 3: Cross-Attention Pre-training (Stage 1)")
print(f"  pooling={POOLING}, k={K}, d_k={D_K}, d_v={D_V}, lr={LR}, wd={WEIGHT_DECAY}, epochs={EPOCHS}, patience={PATIENCE}")
print(f"  early-stop criterion: val AUC  |  val_ratio={VAL_RATIO}")
print("=" * 60)

# ── 재현성 시드 설정 ──────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

def load(name):
    path = os.path.join(DATA_DIR, name)
    assert os.path.exists(path), f"Missing: {path}\nRun Step2 first."
    return torch.tensor(np.load(path), dtype=torch.float32)

fp_train   = load("fp_k16_train.npy")
cham_train = load(f"cham_full_train_{POOLING}.npy")
y_train    = load("y_train.npy")

print(f"Train (full): {fp_train.shape[0]}")

# ── train / val 분할 (stratified) ────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(y_train)), y_train.numpy()))

full_ds  = TensorDataset(cham_train, fp_train, y_train)
train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      worker_init_fn=seed_worker, generator=g)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train split: {len(train_idx)}  |  Val split: {len(val_idx)}")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder   = CrossAttentionEncoder(k=K, d_k=D_K, d_v=D_V).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

print(f"Parameters: {sum(p.numel() for p in encoder.parameters())}  |  Device: {device}")

best_val_auc, best_state, patience_cnt = -1.0, None, 0

for epoch in range(1, EPOCHS + 1):
    # ── train ────────────────────────────────────────────────────────
    encoder.train()
    epoch_loss = 0.0

    for x_c, x_f, y_b in train_dl:
        x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)

        optimizer.zero_grad()
        loss = criterion(encoder(x_c, x_f).squeeze(1), y_b)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y_b)

    epoch_loss /= len(train_idx)

    # ── val ──────────────────────────────────────────────────────────
    encoder.eval()
    val_logits, val_labels, val_loss_sum = [], [], 0.0
    with torch.no_grad():
        for x_c, x_f, y_b in val_dl:
            x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)
            logits = encoder(x_c, x_f).squeeze(1)
            val_loss_sum += criterion(logits, y_b).item() * len(y_b)
            val_logits.append(torch.sigmoid(logits).cpu())
            val_labels.append(y_b.cpu())
    val_loss  = val_loss_sum / len(val_idx)
    val_proba = torch.cat(val_logits).numpy()
    val_np    = torch.cat(val_labels).numpy()
    val_auc   = roc_auc_score(val_np, val_proba)

    if epoch % 20 == 0 or epoch == 1:
        best_mark = " * best" if val_auc > best_val_auc else ""
        print(f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}{best_mark}")

    # ── early stopping (val AUC 기준, 데이터 누수 없음) ───────────────
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

save_path = os.path.join(OUT_DIR, f"pretrained_encoder_tune_{POOLING}.pt")
torch.save(best_state, save_path)

print(f"\nBest val AUC: {best_val_auc:.4f}")
print(f"Saved: {save_path}")
print("Step 3 OK")