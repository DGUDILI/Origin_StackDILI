import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)
from transformers import AutoTokenizer
from torch_geometric.data import Batch as PyGBatch
from xgboost import XGBClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import (
    K, D_MODEL, NUM_HEADS, MODEL_NAME,
    BATCH_SIZE, MAX_LENGTH, SEED,
    DATA_DIR, OUT_DIR,
    MACCS_DIM, MAX_ATOMS, SAGE_LAYERS, SAGE_HIDDEN, ATOM_FEAT_DIM,
)
from model import GraphMACCSEncoder


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attn_mask": torch.stack([b["attn_mask"] for b in batch]),
        "maccs":     torch.stack([b["maccs"] for b in batch]),
        "graph":     PyGBatch.from_data_list([b["pyg_data"] for b in batch]),
    }


def extract_features(model, data_list, tokenizer, device):
    # 토크나이징
    smiles_list = [d["smiles"] for d in data_list]
    enc = tokenizer(
        smiles_list,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, data_list, enc):
            self.data = data_list
            self.ids  = enc["input_ids"]
            self.mask = enc["attention_mask"]
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            return {
                "input_ids": self.ids[i],
                "attn_mask": self.mask[i],
                "maccs":     self.data[i]["maccs"],
                "pyg_data":  self.data[i]["pyg"],
            }

    ds = _DS(data_list, enc)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)
    feats = []
    with torch.no_grad():
        for batch in dl:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attn_mask"].to(device)
            maccs = batch["maccs"].to(device)
            graph = batch["graph"].to(device)
            feats.append(model.encode(ids, mask, maccs, graph).cpu().numpy())
    return np.concatenate(feats, axis=0)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("Step 3: Feature Extraction + Stacking OOF (Stage 2)")
    print(f"  Encoder: GraphMACCSEncoder  |  feature_dim={K}  |  d_model={D_MODEL}")
    print("=" * 65)

    enc_path = os.path.join(OUT_DIR, "pretrained_graph_encoder.pt")
    assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step2 first: bash run.sh step2"

    train_cache = os.path.join(DATA_DIR, "train_graphs.pt")
    test_cache  = os.path.join(DATA_DIR, "test_graphs.pt")
    assert os.path.exists(train_cache), f"Missing: {train_cache}\nRun Step1 first."
    assert os.path.exists(test_cache),  f"Missing: {test_cache}\nRun Step1 first."

    train_data = torch.load(train_cache, weights_only=False)
    test_data  = torch.load(test_cache,  weights_only=False)
    y_train    = np.array([d["label"].item() for d in train_data])
    y_test     = np.array([d["label"].item() for d in test_data])
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")

    # ── Tokenizer & 모델 로드 ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder   = GraphMACCSEncoder(
        model_name=MODEL_NAME,
        atom_feat_dim=ATOM_FEAT_DIM,
        sage_hidden=SAGE_HIDDEN,
        sage_layers=SAGE_LAYERS,
        maccs_dim=MACCS_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        k=K,
        max_atoms=MAX_ATOMS,
    ).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    print(f"Encoder loaded ({device})")

    # ── Feature 추출 ──────────────────────────────────────────────────────────
    X_train = extract_features(encoder, train_data, tokenizer, device)
    X_test  = extract_features(encoder, test_data,  tokenizer, device)
    print(f"Feature shape: train={X_train.shape}, test={X_test.shape}")

    # ── 5-Fold OOF Stacking ───────────────────────────────────────────────────
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    xgb_spw = n_neg / n_pos
    BASE_MODELS = [
        ("RF",     RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          n_jobs=-1, random_state=SEED)),
        ("ET",     ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                        n_jobs=-1, random_state=SEED)),
        ("HistGB", HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                  random_state=SEED)),
        ("XGB",    XGBClassifier(
                       n_estimators=300, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8,
                       scale_pos_weight=xgb_spw,
                       eval_metric="logloss", random_state=SEED)),
    ]

    kf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_base    = len(BASE_MODELS)
    oof_probs  = np.zeros((len(X_train), n_base))
    test_probs = np.zeros((len(X_test),  n_base))

    print(f"\n[Stacking] 5-Fold OOF ({n_base} base models)...")
    for i, (name, clf_template) in enumerate(BASE_MODELS):
        fold_test_preds = []
        for tr_idx, val_idx in kf.split(X_train, y_train):
            clf = clone(clf_template)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            oof_probs[val_idx, i] = clf.predict_proba(X_train[val_idx])[:, 1]
            fold_test_preds.append(clf.predict_proba(X_test)[:, 1])
        test_probs[:, i] = np.mean(fold_test_preds, axis=0)
        print(f"  {name:8s}: OOF AUC={roc_auc_score(y_train, oof_probs[:, i]):.4f}")

    # ── Meta-model ────────────────────────────────────────────────────────────
    print("\n[Meta] LogisticRegression on OOF probs...")
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    meta.fit(oof_probs, y_train)

    final_proba    = meta.predict_proba(test_probs)[:, 1]
    oof_meta_proba = meta.predict_proba(oof_probs)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    oof_mccs   = [matthews_corrcoef(y_train, oof_meta_proba >= t) for t in thresholds]
    best_thr   = float(thresholds[np.argmax(oof_mccs)])
    print(f"\n[Threshold] OOF MCC-optimal: {best_thr:.2f}  (OOF MCC={max(oof_mccs):.4f})")

    final_pred       = (final_proba >= best_thr).astype(int)
    tn, fp_, fn, tp  = confusion_matrix(y_test, final_pred).ravel()

    # ── 평가 ──────────────────────────────────────────────────────────────────
    import pandas as pd
    dgudili = {
        "AUC":         roc_auc_score(y_test, final_proba),
        "MCC":         matthews_corrcoef(y_test, final_pred),
        "F1":          f1_score(y_test, final_pred),
        "ACC":         accuracy_score(y_test, final_pred),
        "Precision":   precision_score(y_test, final_pred),
        "Sensitivity": recall_score(y_test, final_pred),
        "Specificity": tn / (tn + fp_),
    }
    baseline = {
        "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
        "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
    }

    cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
    W = 75
    print("\n" + "=" * W)
    print("DGUDILI_2026 vs StackDILI  |  DILIrank test set (N=452)")
    print("=" * W)
    print(f"{'':25s}" + "".join(f"{c:>10s}" for c in cols))
    print("-" * W)
    print(f"{'StackDILI (target)':25s}" + "".join(f"{baseline[c]:>10.4f}" for c in cols))
    print(f"{'DGUDILI_2026':25s}"        + "".join(f"{dgudili[c]:>10.4f}" for c in cols))
    print("-" * W)
    print(f"{'vs StackDILI':25s}"        + "".join(f"{dgudili[c]-baseline[c]:>+10.4f}" for c in cols))
    print("=" * W)
    print(f"\nPipeline: SMILES → GraphSAGE + MACCS DiffCrossAttn (d={D_MODEL}) → {K}-dim → Stacking")

    results_df = pd.DataFrame([baseline, dgudili], index=["StackDILI", "DGUDILI_2026"])
    csv_path   = os.path.join(OUT_DIR, "results.csv")
    results_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    print("Step 3 OK")
