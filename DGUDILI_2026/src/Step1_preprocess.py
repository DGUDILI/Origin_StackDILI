import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import DATA_DIR, DATA_PATH, FEAT_PATH
from utils import load_dataset

os.makedirs(DATA_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="캐시 파일이 있어도 강제 재실행")
_args, _ = parser.parse_known_args()
FORCE = _args.force

# ─────────────────────────────────────────────────────────────────────────────
# Step 1-A: FP 전처리 (StandardScaler, fp_full_train/test.npy 저장)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1-A: FP Preprocessing (StandardScaler, fp_full_train/test.npy)")
print("=" * 60)

smiles_all, X_fp_all, y_all, ref_all, feat_cols = load_dataset(DATA_PATH, FEAT_PATH)

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

fp_dim = X_fp_all.shape[1]
y_tr, y_te = y_all[train_mask], y_all[test_mask]
print(f"FP: {fp_dim}-dim  |  Train: {train_mask.sum()}  |  Test: {test_mask.sum()}")
print(f"Train labels: pos={int(y_tr.sum())}, neg={int((1-y_tr).sum())}")
print(f"Test  labels: pos={int(y_te.sum())}, neg={int((1-y_te).sum())}")

scaler_fp  = StandardScaler()
X_fp_train = scaler_fp.fit_transform(X_fp_all[train_mask]).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
print(f"\nFP shape: train={X_fp_train.shape}, test={X_fp_test.shape}")

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

for name, exp in [
    ("fp_full_train.npy", (train_mask.sum(), fp_dim)),
    ("fp_full_test.npy",  (test_mask.sum(),  fp_dim)),
]:
    arr = np.load(os.path.join(DATA_DIR, name))
    assert arr.shape == exp and not np.isnan(arr).any(), f"Verify failed: {name}"

print(f"Saved: fp_full_train/test.npy, smiles_train/test.npy, y_train/test.npy, scalers.pkl")
print("Step 1-A OK\n")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1-B: GraphMACCSEncoder 전용 — MACCS + PyG 그래프 캐시 저장
#   Windows CPU 병목 해소: DataLoader num_workers=0 권장 환경에서
#   on-the-fly RDKit 변환 대신 미리 .pt 파일로 저장
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1-B: Graph + MACCS 캐시 저장 (GraphMACCSEncoder용)")
print("  SMILES → PyG Data + MACCS 167-bit → .pt")
print("=" * 60)

import torch
from graph_utils import smiles_to_pyg, get_maccs, verify_maccs_mapping

_train_pt = os.path.join(DATA_DIR, "train_graphs.pt")
_test_pt  = os.path.join(DATA_DIR, "test_graphs.pt")
if not FORCE and os.path.exists(_train_pt) and os.path.exists(_test_pt):
    print(f"[SKIP] 캐시 파일이 이미 존재합니다. 재생성하려면 --force 플래그를 사용하세요.")
    print(f"  {_train_pt}")
    print(f"  {_test_pt}")
    print("Step 1-B SKIP (cached)\n\nStep 1 전체 완료")
    sys.exit(0)

verify_maccs_mapping()

def _build_graph_cache(smiles_list, labels, split_name: str):
    data_list = []
    skip = 0
    for smi, label in zip(smiles_list, labels):
        pyg = smiles_to_pyg(smi)
        if pyg is None:
            print(f"  [WARN] 파싱 실패 — SMILES: {smi[:40]}...")
            skip += 1
            continue
        maccs = get_maccs(smi)
        data_list.append({
            "pyg":   pyg,
            "maccs": maccs,                         # (167,) float32
            "label": torch.tensor(label, dtype=torch.float32),
            "smiles": smi,
        })
    path = os.path.join(DATA_DIR, f"{split_name}_graphs.pt")
    torch.save(data_list, path)
    print(f"  [{split_name}] {len(data_list)}개 저장 → {path}  (skip={skip})")
    return len(data_list)

smiles_tr = smiles_all[train_mask]
smiles_te = smiles_all[test_mask]

n_tr = _build_graph_cache(smiles_tr, y_tr, "train")
n_te = _build_graph_cache(smiles_te, y_te, "test")

print(f"\nGraph cache: train={n_tr}, test={n_te}")
print("Step 1-B OK")
print("\nStep 1 전체 완료")
