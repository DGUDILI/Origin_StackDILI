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
from xgboost import XGBClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from config import (
    K, D_MODEL, NUM_HEADS, MODEL_NAME,
    BATCH_SIZE, MAX_LENGTH, SEED,
    DATA_DIR, OUT_DIR,
)
from utils import SMILESDataset
from model import E2E_MHAResidualEncoder


def extract_features(model, smiles_list, fp_scaled, tokenizer, device):
    ds = SMILESDataset(smiles_list, fp_scaled, tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for ids, mask, fp in dl:
            ids, mask, fp = ids.to(device), mask.to(device), fp.to(device)
            feats.append(model.encode(ids, mask, fp).cpu().numpy())
    return np.concatenate(feats, axis=0)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("Step 3: Feature Extraction + Stacking OOF (Stage 2)")
    print(f"  Encoder: E2E_MHAResidualEncoder  |  feature_dim={K}  |  d_model={D_MODEL}")
    print("=" * 65)

    enc_path = os.path.join(OUT_DIR, "pretrained_encoder.pt")
    assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step2 first."

    # ── 데이터 로드 (Step1 저장 npy 사용) ────────────────────────────────────────
    fp_train_path = os.path.join(DATA_DIR, "fp_full_train.npy")
    assert os.path.exists(fp_train_path), f"Missing: {fp_train_path}\nRun Step1 first."

    fp_train     = np.load(fp_train_path)
    fp_test      = np.load(os.path.join(DATA_DIR, "fp_full_test.npy"))
    y_train      = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test       = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    smiles_train = np.load(os.path.join(DATA_DIR, "smiles_train.npy"), allow_pickle=True)
    smiles_test  = np.load(os.path.join(DATA_DIR, "smiles_test.npy"),  allow_pickle=True)
    fp_in_dim    = fp_train.shape[1]
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}  |  FP: {fp_in_dim}-dim")

    # ── Tokenizer & 모델 로드 ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = E2E_MHAResidualEncoder(
        fp_in_dim=fp_in_dim, k=K, d_model=D_MODEL, num_heads=NUM_HEADS,
        model_name=MODEL_NAME,
    ).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    print(f"Encoder loaded ({device})")

    # ── Feature 추출 (encoder 16-dim + raw FP 425-dim 연결) ──────────────────────
    enc_tr = extract_features(encoder, smiles_train, fp_train, tokenizer, device)
    enc_te = extract_features(encoder, smiles_test,  fp_test,  tokenizer, device)
    X_train_feat = np.hstack([enc_tr, fp_train])   # (N_train, 16+425=441)
    X_test_feat  = np.hstack([enc_te, fp_test])    # (N_test,  441)
    print(f"Feature shape: train={X_train_feat.shape}, test={X_test_feat.shape}")

    # ── 5-Fold OOF Stacking ──────────────────────────────────────────────────────
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    xgb_spw = n_neg / n_pos   # scale_pos_weight < 1 → 모델이 positive 예측 보수적
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

    kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_base = len(BASE_MODELS)

    oof_probs  = np.zeros((len(X_train_feat), n_base))
    test_probs = np.zeros((len(X_test_feat),  n_base))

    print(f"\n[Stacking] 5-Fold OOF ({n_base} base models)...")
    for i, (name, clf_template) in enumerate(BASE_MODELS):
        fold_test_preds = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_feat, y_train)):
            clf_fold = clone(clf_template)
            clf_fold.fit(X_train_feat[tr_idx], y_train[tr_idx])
            oof_probs[val_idx, i] = clf_fold.predict_proba(X_train_feat[val_idx])[:, 1]
            fold_test_preds.append(clf_fold.predict_proba(X_test_feat)[:, 1])
        test_probs[:, i] = np.mean(fold_test_preds, axis=0)
        oof_auc = roc_auc_score(y_train, oof_probs[:, i])
        print(f"  {name:8s}: OOF AUC={oof_auc:.4f}")

    # ── LR Meta-model ────────────────────────────────────────────────────────────
    print("\n[Meta] LogisticRegression on OOF probs...")
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    meta.fit(oof_probs, y_train)

    final_proba    = meta.predict_proba(test_probs)[:, 1]
    oof_meta_proba = meta.predict_proba(oof_probs)[:, 1]

    # OOF에서 MCC 최적 threshold 탐색
    thresholds = np.arange(0.10, 0.91, 0.01)
    oof_mccs   = [matthews_corrcoef(y_train, oof_meta_proba >= t) for t in thresholds]
    best_thr   = float(thresholds[np.argmax(oof_mccs)])
    print(f"\n[Threshold] OOF MCC-optimal: {best_thr:.2f}  (OOF MCC={max(oof_mccs):.4f})")

    final_pred  = (final_proba >= best_thr).astype(int)
    tn, fp_, fn, tp = confusion_matrix(y_test, final_pred).ravel()

    # ── 평가 ─────────────────────────────────────────────────────────────────────
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
    print(f"DGUDILI_2026 vs StackDILI  |  DILIrank test set (N=452)")
    print("=" * W)
    print(f"{'':25s}" + "".join(f"{c:>10s}" for c in cols))
    print("-" * W)
    print(f"{'StackDILI (target)':25s}" + "".join(f"{baseline[c]:>10.4f}" for c in cols))
    print(f"{'DGUDILI_2026':25s}"        + "".join(f"{dgudili[c]:>10.4f}" for c in cols))
    print("-" * W)
    print(f"{'vs StackDILI':25s}"        + "".join(f"{dgudili[c]-baseline[c]:>+10.4f}" for c in cols))
    print("=" * W)
    print(f"\nPipeline: SMILES -> ChemBERTa(E2E) + FP {fp_in_dim}-dim -> MHA Residual (d={D_MODEL}) -> {K}-dim -> Stacking")

    results_df = pd.DataFrame(
        [baseline, dgudili],
        index=["StackDILI", "DGUDILI_2026"]
    )
    csv_path = os.path.join(OUT_DIR, "results.csv")
    results_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    print("Step 3 OK")
