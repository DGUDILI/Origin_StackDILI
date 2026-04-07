"""
preprocess.py — 데이터 전처리

Step 1-A: FP StandardScaler 전처리 → data/ 저장
Step 1-B: SMILES → PyG Data + MACCS 167-bit → {train,test}_graphs.pt 저장

사용법:
    python src/pipeline/preprocess.py
    python src/pipeline/preprocess.py --config config_clean.yaml
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.config import load_cfg
from core.utils import load_dataset


def main():
    _ROOT = os.path.dirname(_SRC)  # DGUDILI_2026/
    default_cfg = os.path.join(_ROOT, "config.yaml")

    parser = argparse.ArgumentParser(description="DGUDILI_2026 데이터 전처리")
    parser.add_argument("--config", default=default_cfg,
                        help=f"YAML 설정 파일 경로 (기본값: {default_cfg})")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(cfg.data_dir, exist_ok=True)

    # ── Step 1-A: FP 전처리 ───────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1-A: FP Preprocessing (StandardScaler)")
    print("=" * 60)

    smiles_all, X_fp_all, y_all, ref_all, feat_cols = load_dataset(cfg.data_csv, cfg.feat_csv)

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

    np.save(os.path.join(cfg.data_dir, "fp_full_train.npy"), X_fp_train)
    np.save(os.path.join(cfg.data_dir, "fp_full_test.npy"),  X_fp_test)
    np.save(os.path.join(cfg.data_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(cfg.data_dir, "y_test.npy"),  y_te)
    np.save(os.path.join(cfg.data_dir, "smiles_train.npy"), smiles_all[train_mask])
    np.save(os.path.join(cfg.data_dir, "smiles_test.npy"),  smiles_all[test_mask])

    with open(os.path.join(cfg.data_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({"fp": scaler_fp, "fp_dim": fp_dim}, f)
    with open(os.path.join(cfg.data_dir, "fp_feature_names.json"), "w") as f:
        json.dump(feat_cols, f, indent=2)

    for name, exp in [
        ("fp_full_train.npy", (train_mask.sum(), fp_dim)),
        ("fp_full_test.npy",  (test_mask.sum(),  fp_dim)),
    ]:
        arr = np.load(os.path.join(cfg.data_dir, name))
        assert arr.shape == exp and not np.isnan(arr).any(), f"Verify failed: {name}"

    print("Saved: fp_full_train/test.npy, smiles_train/test.npy, y_train/test.npy, scalers.pkl")
    print("Step 1-A OK\n")

    # ── Step 1-B: Graph + MACCS 캐시 ─────────────────────────────────────────
    print("=" * 60)
    print("Step 1-B: Graph + MACCS 캐시 저장 (GraphMACCSEncoder용)")
    print("  SMILES → PyG Data + MACCS 167-bit → .pt")
    print("=" * 60)

    import torch
    from core.graph_utils import smiles_to_pyg, get_maccs, verify_maccs_mapping

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
                "pyg":    pyg,
                "maccs":  maccs,
                "label":  torch.tensor(label, dtype=torch.float32),
                "smiles": smi,
            })
        path = os.path.join(cfg.data_dir, f"{split_name}_graphs.pt")
        torch.save(data_list, path)
        print(f"  [{split_name}] {len(data_list)}개 저장 → {path}  (skip={skip})")
        return len(data_list)

    n_tr = _build_graph_cache(smiles_all[train_mask], y_tr, "train")
    n_te = _build_graph_cache(smiles_all[test_mask],  y_te, "test")

    print(f"\nGraph cache: train={n_tr}, test={n_te}")
    print("Step 1-B OK")
    print("\n전처리 완료")


if __name__ == "__main__":
    main()
