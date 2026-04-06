import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import DATA_DIR, DATA_PATH, FEAT_PATH
from utils import load_dataset

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: FP Preprocessing (StandardScaler)")
print("  LinearProjection은 model 내부에서 학습 (fp_proj)")
print("=" * 60)

smiles_all, X_fp_all, y_all, ref_all, feat_cols = load_dataset(DATA_PATH, FEAT_PATH)

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

fp_dim = X_fp_all.shape[1]
y_tr, y_te = y_all[train_mask], y_all[test_mask]
print(f"FP: {fp_dim}-dim  |  Train: {train_mask.sum()}  |  Test: {test_mask.sum()}")
print(f"Train labels: pos={int(y_tr.sum())}, neg={int((1-y_tr).sum())}")
print(f"Test  labels: pos={int(y_te.sum())}, neg={int((1-y_te).sum())}")

# FP StandardScaler (SelectKBest 없음 — 모델 내 fp_proj가 중요도 학습)
scaler_fp  = StandardScaler()
X_fp_train = scaler_fp.fit_transform(X_fp_all[train_mask]).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
print(f"\nFP shape: train={X_fp_train.shape}, test={X_fp_test.shape}")

# 저장
np.save(os.path.join(DATA_DIR, "fp_full_train.npy"), X_fp_train)
np.save(os.path.join(DATA_DIR, "fp_full_test.npy"),  X_fp_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_tr)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_te)
np.save(os.path.join(DATA_DIR, "smiles_train.npy"), smiles_all[train_mask])
np.save(os.path.join(DATA_DIR, "smiles_test.npy"),  smiles_all[test_mask])

with open(os.path.join(DATA_DIR, "scalers.pkl"), "wb") as f:
    pickle.dump({"fp": scaler_fp, "fp_dim": fp_dim}, f)

with open(os.path.join(DATA_DIR, "fp_feature_names.json"), "w") as f:
    json.dump(feat_cols, f, indent=2)

# 검증
for name, exp in [
    ("fp_full_train.npy", (train_mask.sum(), fp_dim)),
    ("fp_full_test.npy",  (test_mask.sum(),  fp_dim)),
]:
    arr = np.load(os.path.join(DATA_DIR, name))
    assert arr.shape == exp and not np.isnan(arr).any(), f"Verify failed: {name}"

print(f"\nSaved: fp_full_train/test.npy, smiles_train/test.npy, y_train/test.npy, scalers.pkl, fp_feature_names.json")
print("Step 1 OK")
