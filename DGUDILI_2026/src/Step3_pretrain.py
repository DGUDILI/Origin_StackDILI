import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
OUT_DIR    = os.path.join(ROOT, "outputs")
SRC_DIR    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

os.makedirs(OUT_DIR, exist_ok=True)

K, D_K     = 16, 32
LR         = 1e-3
EPOCHS     = 200
BATCH_SIZE = 32
PATIENCE   = 20
SEED       = 42

print("=" * 60)
print("Step 3: Cross-Attention Pre-training (Stage 1)")
print(f"  k={K}, d_k={D_K}, lr={LR}, epochs={EPOCHS}, patience={PATIENCE}")
print("=" * 60)

torch.manual_seed(SEED)
np.random.seed(SEED)

def load(name):
    path = os.path.join(DATA_DIR, name)
    assert os.path.exists(path), f"Missing: {path}\nRun Step2 first."
    return torch.tensor(np.load(path), dtype=torch.float32)

fp_train   = load("fp_k16_train.npy")
fp_test    = load("fp_k16_test.npy")
cham_train = load("cham_k16_train.npy")
cham_test  = load("cham_k16_test.npy")
y_train    = load("y_train.npy")
y_test     = load("y_test.npy")

print(f"Train: {fp_train.shape[0]}  |  Test: {fp_test.shape[0]}")

train_ds = TensorDataset(cham_train, fp_train, y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder   = CrossAttentionEncoder(k=K, d_k=D_K).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

print(f"Parameters: {sum(p.numel() for p in encoder.parameters())}  |  Device: {device}")

fp_test_d   = fp_test.to(device)
cham_test_d = cham_test.to(device)
y_test_np   = y_test.numpy()

best_auc, best_state, patience_cnt = -1.0, None, 0

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    epoch_loss = 0.0
    for x_c, x_f, y_b in train_dl:
        x_c, x_f, y_b = x_c.to(device), x_f.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(encoder(x_c, x_f).squeeze(1), y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(y_b)
    epoch_loss /= len(train_ds)

    encoder.eval()
    with torch.no_grad():
        proba = torch.sigmoid(encoder(cham_test_d, fp_test_d)).squeeze(1).cpu().numpy()
    auc = roc_auc_score(y_test_np, proba)

    if epoch % 20 == 0 or epoch == 1:
        best_mark = " * best" if auc > best_auc else ""
        print(f"  Epoch {epoch:>4d} | loss={epoch_loss:.4f} | AUC={auc:.4f}{best_mark}")

    if auc > best_auc:
        best_auc   = auc
        best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

save_path = os.path.join(OUT_DIR, "pretrained_encoder.pt")
torch.save(best_state, save_path)
print(f"\nBest test AUC: {best_auc:.4f}")
print(f"Saved: {save_path}")
print("Step 3 OK")
