"""
Step_CV.py — DGUDILI 2026 10-Fold CV (env2)

encoder는 train set으로 학습된 pretrained_encoder.pt 사용 (Step2 결과).
전체 데이터(train+test)에 대해 10-fold stratified CV를 수행하여
stacking 파이프라인의 일반화 성능을 평가.

Note: encoder는 train set 기반으로 학습되므로, fold에서 test로 사용되는
      train 샘플에 대해 경미한 낙관 편향이 있을 수 있음.
      DILIrank test 샘플은 encoder 학습에 사용되지 않았으므로 편향 없음.
"""
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
    DATA_PATH, FEAT_PATH, _USE_CLEAN,
)
from utils import set_seed, load_dataset, SMILESDataset
from model import E2E_MHAResidualEncoder

N_FOLDS = 10


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
    set_seed(SEED)

    label = "CLEAN" if _USE_CLEAN else "ORIGINAL"
    print("=" * 65)
    print(f"Step CV: DGUDILI 2026  10-Fold CV  ({label} data)")
    print(f"  Encoder: E2E_MHAResidualEncoder  |  k={K}  |  d_model={D_MODEL}")
    print("=" * 65)

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    for p in [DATA_PATH, FEAT_PATH]:
        assert os.path.exists(p), f"Missing: {p}"

    smiles_all, X_fp_all, y_all, ref_all, _ = load_dataset(DATA_PATH, FEAT_PATH)
    from sklearn.preprocessing import StandardScaler
    scaler_fp = StandardScaler()
    scaler_fp.fit(X_fp_all[ref_all != "DILIrank"])  # train 기준 fit
    X_fp_scaled = scaler_fp.transform(X_fp_all).astype(np.float32)

    print(f"Total: {len(y_all)}  |  FP: {X_fp_all.shape[1]}-dim")

    # ── encoder 로드 ───────────────────────────────────────────────────────────
    enc_path = os.path.join(OUT_DIR, "pretrained_encoder.pt")
    assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step2 first."

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder   = E2E_MHAResidualEncoder(
        fp_in_dim=X_fp_all.shape[1], k=K, d_model=D_MODEL,
        num_heads=NUM_HEADS, model_name=MODEL_NAME,
    ).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    print(f"Encoder loaded ({device})")

    # ── 전체 데이터 feature 추출 ────────────────────────────────────────────────
    print(f"\nExtracting features for all {len(y_all)} samples...")
    X_feat = extract_features(encoder, smiles_all, X_fp_scaled, tokenizer, device)
    print(f"Feature shape: {X_feat.shape}")

    # ── 10-Fold CV ────────────────────────────────────────────────────────────
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_pos = int((y_all == 1).sum())
    n_neg = len(y_all) - n_pos
    xgb_spw = n_neg / n_pos

    BASE_MODELS = [
        ("RF",     RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          n_jobs=-1, random_state=SEED)),
        ("ET",     ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                        n_jobs=-1, random_state=SEED)),
        ("HistGB", HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                  random_state=SEED)),
        ("XGB",    XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=xgb_spw,
                                 eval_metric="logloss", random_state=SEED)),
    ]

    fold_metrics = []

    print(f"\n[CV] {N_FOLDS}-Fold Stratified CV...")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_feat, y_all)):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y_all[tr_idx],  y_all[te_idx]

        # 5-fold OOF stacking on this fold's train
        inner_kf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        n_base    = len(BASE_MODELS)
        oof_p     = np.zeros((len(X_tr), n_base))
        test_p    = np.zeros((len(X_te), n_base))

        for i, (name, clf_tpl) in enumerate(BASE_MODELS):
            fold_te_preds = []
            for _, (inner_tr, inner_val) in enumerate(inner_kf.split(X_tr, y_tr)):
                clf = clone(clf_tpl)
                clf.fit(X_tr[inner_tr], y_tr[inner_tr])
                oof_p[inner_val, i] = clf.predict_proba(X_tr[inner_val])[:, 1]
                fold_te_preds.append(clf.predict_proba(X_te)[:, 1])
            test_p[:, i] = np.mean(fold_te_preds, axis=0)

        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        meta.fit(oof_p, y_tr)

        final_proba = meta.predict_proba(test_p)[:, 1]
        oof_meta    = meta.predict_proba(oof_p)[:, 1]

        # OOF MCC 최적 threshold
        thresholds = np.arange(0.10, 0.91, 0.01)
        oof_mccs   = [matthews_corrcoef(y_tr, oof_meta >= t) for t in thresholds]
        best_thr   = float(thresholds[np.argmax(oof_mccs)])
        final_pred = (final_proba >= best_thr).astype(int)

        tn, fp_, fn, tp = confusion_matrix(y_te, final_pred).ravel()
        m = {
            "AUC":         roc_auc_score(y_te, final_proba),
            "MCC":         matthews_corrcoef(y_te, final_pred),
            "F1":          f1_score(y_te, final_pred),
            "ACC":         accuracy_score(y_te, final_pred),
            "Precision":   precision_score(y_te, final_pred, zero_division=0),
            "Sensitivity": recall_score(y_te, final_pred),
            "Specificity": tn / (tn + fp_),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold+1:2d}/{N_FOLDS}  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}  F1={m['F1']:.4f}  thr={best_thr:.2f}")

    # ── 집계 ──────────────────────────────────────────────────────────────────
    cols  = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
    means = {c: float(np.mean([m[c] for m in fold_metrics])) for c in cols}
    stds  = {c: float(np.std( [m[c] for m in fold_metrics])) for c in cols}

    print("\n" + "=" * 65)
    print(f"DGUDILI 2026  10-Fold CV Average  ({label})")
    print("=" * 65)
    print("  " + "  ".join(f"{c}: {means[c]:.4f} ±{stds[c]:.4f}" for c in cols))

    # ── 저장 ──────────────────────────────────────────────────────────────────
    import pandas as pd
    rows = [{"model": f"fold_{i+1}", **m} for i, m in enumerate(fold_metrics)]
    rows.append({"model": "DGUDILI_2026", **means})
    rows.append({"model": "DGUDILI_2026_std", **stds})
    df = pd.DataFrame(rows)

    cv_out_dir = OUT_DIR.replace("outputs", "outputs_cv")
    os.makedirs(cv_out_dir, exist_ok=True)
    out_path = os.path.join(cv_out_dir, "results_cv.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Step CV OK")
