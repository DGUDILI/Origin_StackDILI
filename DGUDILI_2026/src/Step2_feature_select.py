import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
DATA_PATH = r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv"
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
K    = 16
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
args = parser.parse_args()
POOLING = args.pooling

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Step 2: FP SelectKBest(k=16) + Full ChemBERTa Scaling")
print("=" * 60)
print(f"Pooling: {POOLING}")

df_meta = pd.read_csv(DATA_PATH)
df_feat = pd.read_csv(FEAT_PATH)

assert list(df_meta["SMILES"]) == list(df_feat["SMILES"]), \
    "SMILES order mismatch between Dataset.csv and Dataset_feature.csv"

feat_cols = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all  = df_feat[feat_cols].values.astype(np.float32)
y_all     = df_feat["Label"].values.astype(np.float32)
ref_all   = df_feat["ref"].values

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

print(f"FP features total: {len(feat_cols)}")
print(f"Train: {train_mask.sum()}  |  Test: {test_mask.sum()}")
y_tr, y_te = y_all[train_mask], y_all[test_mask]
print(f"Train labels: p={int(y_tr.sum())}, n={int((1-y_tr).sum())}")
print(f"Test  labels: p={int(y_te.sum())}, n={int((1-y_te).sum())}")

# FP SelectKBest
np.random.seed(SEED)   # mutual_info_classif 재현성 보장
print(f"\n[FP] SelectKBest(mutual_info_classif, k={K}) fit on train...")
sel_fp = SelectKBest(mutual_info_classif, k=K)
sel_fp.fit(X_fp_all[train_mask], y_all[train_mask])
fp_idx = sel_fp.get_support(indices=True)
selected_fp_names = [feat_cols[i] for i in fp_idx]

X_fp_train = X_fp_all[train_mask][:, fp_idx]
X_fp_test  = X_fp_all[test_mask][:, fp_idx]
scaler_fp  = RobustScaler()
X_fp_train = scaler_fp.fit_transform(X_fp_train).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_test).astype(np.float32)
print(f"  Selected FP features: {selected_fp_names}")

# ChemBERTa full embedding
emb_path   = os.path.join(DATA_DIR, f"chemberta_embeddings_{POOLING}.npy")
order_path = os.path.join(DATA_DIR, "smiles_order.npy")
assert os.path.exists(emb_path), f"Missing: {emb_path}\nRun Step1 first."

embeddings   = np.load(emb_path)
smiles_order = np.load(order_path, allow_pickle=True)
assert list(smiles_order) == list(df_meta["SMILES"]), "Embedding SMILES order mismatch"

hidden_dim = embeddings.shape[1]

print(f"\n[ChemBERTa] Full embedding + StandardScaler (no SelectKBest)")
X_cham_train = embeddings[train_mask].astype(np.float32)
X_cham_test  = embeddings[test_mask].astype(np.float32)

scaler_cham  = StandardScaler()
X_cham_train = scaler_cham.fit_transform(X_cham_train).astype(np.float32)
X_cham_test  = scaler_cham.transform(X_cham_test).astype(np.float32)
print(f"  ChemBERTa full shape: train={X_cham_train.shape}, test={X_cham_test.shape}")
print(f"  Hidden dim: {hidden_dim}")

# Save
np.save(os.path.join(DATA_DIR, "fp_k16_train.npy"), X_fp_train)
np.save(os.path.join(DATA_DIR, "fp_k16_test.npy"),  X_fp_test)

np.save(os.path.join(DATA_DIR, f"cham_full_train_{POOLING}.npy"), X_cham_train)
np.save(os.path.join(DATA_DIR, f"cham_full_test_{POOLING}.npy"),  X_cham_test)

np.save(os.path.join(DATA_DIR, "y_train.npy"), y_all[train_mask])
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_all[test_mask])

with open(os.path.join(DATA_DIR, "selected_fp_features.json"), "w") as f:
    json.dump(selected_fp_names, f, indent=2)

with open(os.path.join(DATA_DIR, f"scalers_full_{POOLING}.pkl"), "wb") as f:
    pickle.dump({"fp": scaler_fp, "cham": scaler_cham}, f)

# Verify
for name, exp in [
    ("fp_k16_train.npy",               (train_mask.sum(), K)),
    ("fp_k16_test.npy",                (test_mask.sum(),  K)),
    (f"cham_full_train_{POOLING}.npy", (train_mask.sum(), hidden_dim)),
    (f"cham_full_test_{POOLING}.npy",  (test_mask.sum(),  hidden_dim)),
]:
    arr = np.load(os.path.join(DATA_DIR, name))
    assert arr.shape == exp and not np.isnan(arr).any(), f"Verify failed: {name}"

print("\nAll files saved and verified.")
print("Step 2 OK")