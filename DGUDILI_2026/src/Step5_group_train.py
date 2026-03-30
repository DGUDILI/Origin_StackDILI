import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from group_model import GroupCrossAttentionDILI

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 하이퍼파라미터 ─────────────────────────────────────────────────
D          = 16       # 토큰 투영 차원
DROPOUT    = 0.3
LR         = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS     = 300
BATCH_SIZE = 32
PATIENCE   = 40
SEED       = 42
# ──────────────────────────────────────────────────────────────────

print("=" * 65)
print("Step 5: Group Cross-Attention DILI - End-to-End Training")
print(f"  d={D}, dropout={DROPOUT}, lr={LR}, epochs={EPOCHS}, patience={PATIENCE}")
print("=" * 65)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. 원본 데이터 로드 ────────────────────────────────────────────
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
DATA_PATH = r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv"
EMB_PATH  = os.path.join(DATA_DIR, "chemberta_embeddings.npy")

assert os.path.exists(FEAT_PATH), f"Missing: {FEAT_PATH}"
assert os.path.exists(EMB_PATH),  f"Missing: {EMB_PATH}\nRun Step1 first."

df_feat = pd.read_csv(FEAT_PATH)
df_meta = pd.read_csv(DATA_PATH)

feat_cols = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
assert len(feat_cols) == 425, f"Expected 425 FP cols, got {len(feat_cols)}"

X_fp_all = df_feat[feat_cols].values.astype(np.float32)   # (1850, 425)
y_all    = df_feat["Label"].values.astype(np.float32)
ref_all  = df_feat["ref"].values

embeddings = np.load(EMB_PATH).astype(np.float32)          # (1850, 768)
assert embeddings.shape == (len(df_feat), 768), \
    f"ChemBERTa shape mismatch: {embeddings.shape}"

# ── 2. Train / Test 분리 ──────────────────────────────────────────
train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

print(f"\nFP total features  : {len(feat_cols)}")
print(f"ChemBERTa dim      : 768")
print(f"Train / Test split : {train_mask.sum()} / {test_mask.sum()}")
y_tr = y_all[train_mask]
y_te = y_all[test_mask]
print(f"Train labels  pos={int(y_tr.sum())}, neg={int((1-y_tr).sum())}")
print(f"Test  labels  pos={int(y_te.sum())}, neg={int((1-y_te).sum())}")

X_fp_train = X_fp_all[train_mask]
X_fp_test  = X_fp_all[test_mask]
X_lm_train = embeddings[train_mask]
X_lm_test  = embeddings[test_mask]

# ── 3. 스케일링 (train 기준 fit) ───────────────────────────────────
scaler_fp  = StandardScaler()
scaler_lm  = StandardScaler()

X_fp_train = scaler_fp.fit_transform(X_fp_train).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_test).astype(np.float32)
X_lm_train = scaler_lm.fit_transform(X_lm_train).astype(np.float32)
X_lm_test  = scaler_lm.transform(X_lm_test).astype(np.float32)

# 스케일러 저장 (Step6에서 재사용)
scaler_path = os.path.join(DATA_DIR, "scalers_group.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"fp": scaler_fp, "lm": scaler_lm}, f)

# ── 4. 데이터셋 / 데이터로더 ──────────────────────────────────────
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

fp_train_t  = to_tensor(X_fp_train)
fp_test_t   = to_tensor(X_fp_test)
lm_train_t  = to_tensor(X_lm_train)
lm_test_t   = to_tensor(X_lm_test)
y_train_t   = to_tensor(y_tr)
y_test_np   = y_te

train_ds = TensorDataset(fp_train_t, lm_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ── 5. 모델 / 옵티마이저 설정 ─────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = GroupCrossAttentionDILI(d=D, dropout=DROPOUT).to(device)
n_param = sum(p.numel() for p in model.parameters())
print(f"\nDevice     : {device}")
print(f"Parameters : {n_param:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-6
)

# pos_weight: 클래스 불균형 보정
pos_w     = torch.tensor([float((1 - y_tr).sum()) / float(y_tr.sum())]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

fp_test_d  = fp_test_t.to(device)
lm_test_d  = lm_test_t.to(device)

print(f"pos_weight : {pos_w.item():.4f}  (neg/pos ratio in train)")
print()

# ── 6. 학습 루프 ──────────────────────────────────────────────────
best_auc, best_state, patience_cnt = -1.0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for x_fp_b, x_lm_b, y_b in train_dl:
        x_fp_b = x_fp_b.to(device)
        x_lm_b = x_lm_b.to(device)
        y_b    = y_b.to(device)

        optimizer.zero_grad()
        logit = model(x_fp_b, x_lm_b).squeeze(1)
        loss  = criterion(logit, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(y_b)
    epoch_loss /= len(train_ds)

    # 검증 (test AUC - 기존 파이프라인과 동일 기준)
    model.eval()
    with torch.no_grad():
        logit_test = model(fp_test_d, lm_test_d).squeeze(1)
        proba_test = torch.sigmoid(logit_test).cpu().numpy()
    auc = roc_auc_score(y_test_np, proba_test)
    scheduler.step(auc)

    if epoch % 20 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]["lr"]
        best_mark = " ★" if auc > best_auc else ""
        print(
            f"  Epoch {epoch:>4d} | loss={epoch_loss:.4f} | "
            f"AUC={auc:.4f} | lr={lr_now:.2e}{best_mark}"
        )

    if auc > best_auc:
        best_auc     = auc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  (best AUC={best_auc:.4f})")
            break

# ── 7. 최적 모델 저장 ─────────────────────────────────────────────
save_path = os.path.join(OUT_DIR, "group_model_best.pt")
torch.save(best_state, save_path)

print(f"\nBest test AUC : {best_auc:.4f}")
print(f"Saved         : {save_path}")
print(f"Scalers saved : {scaler_path}")
print("Step 5 OK")
