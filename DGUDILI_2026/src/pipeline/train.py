"""
train.py — GraphMACCSEncoder 통합 학습 엔트리포인트

사용법:
    python src/pipeline/train.py --stage pretrain            # Stage 1: encoder 사전학습
    python src/pipeline/train.py --stage stack               # Stage 2: feature 추출 + stacking
    python src/pipeline/train.py --stage cv                  # 10-Fold CV 평가
    python src/pipeline/train.py --stage pretrain --config config_clean.yaml

--config: YAML 설정 파일 경로 (기본값: DGUDILI_2026/config.yaml)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.config import load_cfg, Cfg
from core.dataset import MolGraphDataset, collate_fn, make_loader
from core.model import GraphMACCSEncoder
from core.utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokenizer(cfg: Cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.model_name)


def _make_encoder(cfg: Cfg, device: torch.device) -> GraphMACCSEncoder:
    return GraphMACCSEncoder(
        atom_feat_dim=cfg.atom_feat_dim,
        maccs_dim=cfg.maccs_dim,
        sage_hidden=cfg.sage_hidden,
        sage_layers=cfg.sage_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        k=cfg.k,
        max_atoms=cfg.max_atoms,
        dropout=cfg.dropout,
        model_name=cfg.model_name,
    ).to(device)


def _load_encoder(cfg: Cfg, device: torch.device) -> GraphMACCSEncoder:
    ckpt = os.path.join(cfg.output_dir, "pretrained_graph_encoder.pt")
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}\n--stage pretrain 먼저 실행하세요."
    encoder = _make_encoder(cfg, device)
    encoder.load_state_dict(torch.load(ckpt, map_location=device))
    encoder.eval()
    print(f"Encoder loaded: {ckpt}  ({device})")
    return encoder


def _build_base_models(cfg: Cfg):
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )
    from xgboost import XGBClassifier
    return [
        ("RF",     RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          n_jobs=-1, random_state=cfg.seed)),
        ("ET",     ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                        n_jobs=-1, random_state=cfg.seed)),
        ("HistGB", HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                  random_state=cfg.seed)),
        ("XGB",    XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=cfg.seed)),
    ]


def extract_features(encoder: GraphMACCSEncoder, data_list: list,
                     tokenizer, cfg: Cfg, device: torch.device) -> np.ndarray:
    from torch.utils.data import DataLoader
    ds = MolGraphDataset(data_list, tokenizer, cfg.max_length)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)
    feats = []
    with torch.no_grad():
        for batch in dl:
            feats.append(encoder.encode(
                batch.input_ids.to(device),
                batch.attn_mask.to(device),
                batch.maccs.to(device),
                batch.graph.to(device),
            ).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _oof_stacking(X_train, y_train, X_test, cfg, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    xgb_spw = n_neg / n_pos

    BASE_MODELS = _build_base_models(cfg)
    for _, clf in BASE_MODELS:
        if hasattr(clf, "scale_pos_weight"):
            clf.scale_pos_weight = xgb_spw

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    n_base = len(BASE_MODELS)
    oof_probs  = np.zeros((len(X_train), n_base))
    test_probs = np.zeros((len(X_test),  n_base))

    print(f"  [{n_splits}-Fold OOF] {n_base} base models...")
    for i, (name, clf_template) in enumerate(BASE_MODELS):
        fold_test_preds = []
        for tr_idx, val_idx in kf.split(X_train, y_train):
            clf = clone(clf_template)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            oof_probs[val_idx, i] = clf.predict_proba(X_train[val_idx])[:, 1]
            fold_test_preds.append(clf.predict_proba(X_test)[:, 1])
        test_probs[:, i] = np.mean(fold_test_preds, axis=0)
        print(f"    {name:8s}: OOF AUC={roc_auc_score(y_train, oof_probs[:, i]):.4f}")

    return oof_probs, test_probs


def _evaluate(y_true, y_pred_proba, y_pred_label) -> dict:
    from sklearn.metrics import (
        roc_auc_score, matthews_corrcoef, f1_score,
        accuracy_score, precision_score, recall_score, confusion_matrix,
    )
    tn, fp_, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
    return {
        "AUC":         roc_auc_score(y_true, y_pred_proba),
        "MCC":         matthews_corrcoef(y_true, y_pred_label),
        "F1":          f1_score(y_true, y_pred_label),
        "ACC":         accuracy_score(y_true, y_pred_label),
        "Precision":   precision_score(y_true, y_pred_label, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred_label),
        "Specificity": tn / (tn + fp_),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Pretrain
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(cfg: Cfg) -> None:
    from sklearn.model_selection import train_test_split

    print("=" * 65)
    print("Stage 1: GraphMACCSEncoder Pre-training")
    print(f"  ChemBERTa lr={cfg.lr_chem}  |  Graph/DiffAttn lr={cfg.lr_other}")
    print(f"  k={cfg.k}, d_model={cfg.d_model}, num_heads={cfg.num_heads}, max_atoms={cfg.max_atoms}")
    print(f"  epochs={cfg.epochs}, batch={cfg.batch_size}, patience={cfg.patience}")
    print("=" * 65)

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_cache = os.path.join(cfg.data_dir, "train_graphs.pt")
    assert os.path.exists(train_cache), (
        f"Missing: {train_cache}\n python src/pipeline/preprocess.py 먼저 실행하세요."
    )
    all_data   = torch.load(train_cache, weights_only=False)
    all_labels = np.array([d["label"].item() for d in all_data])
    n_pos = int(all_labels.sum()); n_neg = len(all_labels) - n_pos
    print(f"Train cache: {len(all_data)}개  |  pos={n_pos}, neg={n_neg}")

    idx = np.arange(len(all_data))
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.15, stratify=all_labels.astype(int), random_state=cfg.seed
    )
    print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

    tokenizer = _load_tokenizer(cfg)

    train_dl = make_loader(
        [all_data[i] for i in idx_tr], tokenizer, cfg.max_length, cfg.batch_size,
        shuffle=True,
        cache_path=os.path.join(cfg.data_dir, f"graph_train_tokens_{cfg.seed}.pt"),
    )
    val_dl = make_loader(
        [all_data[i] for i in idx_val], tokenizer, cfg.max_length, cfg.batch_size,
        shuffle=False,
        cache_path=os.path.join(cfg.data_dir, f"graph_val_tokens_{cfg.seed}.pt"),
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _make_encoder(cfg, device)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: trainable={trainable:,} / total={total:,}  |  Device: {device}")

    n_layers      = len(encoder.chemberta.encoder.layer)
    chem_last_ids = {id(p) for p in encoder.chemberta.encoder.layer[n_layers - 1].parameters()}
    chem_all_ids  = {id(p) for p in encoder.chemberta.parameters()}

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if id(p) in chem_last_ids],
             "lr": cfg.lr_chem},
            {"params": [p for p in encoder.parameters() if id(p) not in chem_all_ids],
             "lr": cfg.lr_other},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.sched_min_lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    y_val_np    = np.array([all_data[i]["label"].item() for i in idx_val])
    y_val_t     = torch.tensor(y_val_np, dtype=torch.float32)
    best_val_auc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        for batch in train_dl:
            ids  = batch.input_ids.to(device)
            mask = batch.attn_mask.to(device)
            mac  = batch.maccs.to(device)
            grph = batch.graph.to(device)
            y_b  = batch.labels.to(device)

            optimizer.zero_grad()
            loss = criterion(encoder(ids, mask, mac, grph).squeeze(1), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)

        epoch_loss /= len(idx_tr)

        encoder.eval()
        all_logits, all_proba = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = encoder(
                    batch.input_ids.to(device),
                    batch.attn_mask.to(device),
                    batch.maccs.to(device),
                    batch.graph.to(device),
                ).squeeze(1)
                all_logits.append(logits.cpu())
                all_proba.append(torch.sigmoid(logits).cpu().numpy())
        val_logits = torch.cat(all_logits)
        val_proba  = np.concatenate(all_proba)

        val_loss = criterion(val_logits, y_val_t).item()
        val_auc  = roc_auc_score(y_val_np, val_proba)
        scheduler.step(val_auc)

        if epoch % 10 == 0 or epoch == 1:
            cur_lr    = optimizer.param_groups[0]["lr"]
            best_mark = " * best" if val_auc > best_val_auc else ""
            print(
                f"  Epoch {epoch:>4d} | train_loss={epoch_loss:.4f}"
                f" | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
                f" | lr={cur_lr:.2e}{best_mark}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"\n  Early stop at epoch {epoch} (best val_AUC={best_val_auc:.4f})")
                break

    save_path = os.path.join(cfg.output_dir, "pretrained_graph_encoder.pt")
    torch.save(best_state, save_path)
    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Saved: {save_path}")
    print("Stage 1 (pretrain) OK")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Stacking
# ─────────────────────────────────────────────────────────────────────────────

def stack(cfg: Cfg) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef
    import pandas as pd

    print("=" * 65)
    print("Stage 2: Feature Extraction + Stacking OOF")
    print(f"  Encoder: GraphMACCSEncoder  |  feature_dim={cfg.k}  |  d_model={cfg.d_model}")
    print("=" * 65)

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_cache = os.path.join(cfg.data_dir, "train_graphs.pt")
    test_cache  = os.path.join(cfg.data_dir, "test_graphs.pt")
    for p in [train_cache, test_cache]:
        assert os.path.exists(p), f"Missing: {p}\n python src/pipeline/preprocess.py 먼저 실행하세요."

    train_data = torch.load(train_cache, weights_only=False)
    test_data  = torch.load(test_cache,  weights_only=False)
    y_train    = np.array([d["label"].item() for d in train_data])
    y_test     = np.array([d["label"].item() for d in test_data])
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")

    tokenizer = _load_tokenizer(cfg)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder   = _load_encoder(cfg, device)

    print("\nExtracting features...")
    X_train = extract_features(encoder, train_data, tokenizer, cfg, device)
    X_test  = extract_features(encoder, test_data,  tokenizer, cfg, device)
    print(f"Feature shape: train={X_train.shape}, test={X_test.shape}")

    oof_probs, test_probs = _oof_stacking(X_train, y_train, X_test, cfg, n_splits=5)

    print("\n[Meta] LogisticRegression on OOF probs...")
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.seed)
    meta.fit(oof_probs, y_train)

    final_proba    = meta.predict_proba(test_probs)[:, 1]
    oof_meta_proba = meta.predict_proba(oof_probs)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    oof_mccs   = [matthews_corrcoef(y_train, oof_meta_proba >= t) for t in thresholds]
    best_thr   = float(thresholds[np.argmax(oof_mccs)])
    print(f"\n[Threshold] OOF MCC-optimal: {best_thr:.2f}  (OOF MCC={max(oof_mccs):.4f})")

    final_pred = (final_proba >= best_thr).astype(int)
    metrics    = _evaluate(y_test, final_proba, final_pred)

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
    print(f"{'DGUDILI_2026':25s}"        + "".join(f"{metrics[c]:>10.4f}" for c in cols))
    print("-" * W)
    print(f"{'vs StackDILI':25s}"        + "".join(f"{metrics[c]-baseline[c]:>+10.4f}" for c in cols))
    print("=" * W)

    results_df = pd.DataFrame([baseline, metrics], index=["StackDILI", "DGUDILI_2026"])
    csv_path   = os.path.join(cfg.output_dir, "results.csv")
    results_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    print("Stage 2 (stack) OK")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: 10-Fold CV
# ─────────────────────────────────────────────────────────────────────────────

def cv(cfg: Cfg) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef
    import pandas as pd

    print("=" * 65)
    print(f"Stage CV: DGUDILI 2026  10-Fold CV")
    print(f"  Encoder: GraphMACCSEncoder  |  k={cfg.k}  |  d_model={cfg.d_model}")
    print("=" * 65)

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_cache = os.path.join(cfg.data_dir, "train_graphs.pt")
    test_cache  = os.path.join(cfg.data_dir, "test_graphs.pt")
    for p in [train_cache, test_cache]:
        assert os.path.exists(p), f"Missing: {p}\n python src/pipeline/preprocess.py 먼저 실행하세요."

    train_data = torch.load(train_cache, weights_only=False)
    test_data  = torch.load(test_cache,  weights_only=False)
    all_data   = train_data + test_data
    y_all      = np.array([d["label"].item() for d in all_data])
    print(f"Total: {len(all_data)}  (train={len(train_data)}, test={len(test_data)})")

    tokenizer = _load_tokenizer(cfg)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder   = _load_encoder(cfg, device)

    print(f"\nExtracting features for all {len(all_data)} samples...")
    X_feat = extract_features(encoder, all_data, tokenizer, cfg, device)
    print(f"Feature shape: {X_feat.shape}")

    N_FOLDS = 10
    kf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.seed)
    fold_metrics = []

    print(f"\n[CV] {N_FOLDS}-Fold Stratified CV...")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_feat, y_all)):
        X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
        y_tr, y_te = y_all[tr_idx],  y_all[te_idx]

        oof_p, test_p = _oof_stacking(X_tr, y_tr, X_te, cfg, n_splits=5)

        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.seed)
        meta.fit(oof_p, y_tr)

        final_proba = meta.predict_proba(test_p)[:, 1]
        oof_meta    = meta.predict_proba(oof_p)[:, 1]

        thresholds = np.arange(0.10, 0.91, 0.01)
        oof_mccs   = [matthews_corrcoef(y_tr, oof_meta >= t) for t in thresholds]
        best_thr   = float(thresholds[np.argmax(oof_mccs)])
        final_pred = (final_proba >= best_thr).astype(int)

        m = _evaluate(y_te, final_proba, final_pred)
        fold_metrics.append(m)
        print(f"  Fold {fold+1:2d}/{N_FOLDS}  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}"
              f"  F1={m['F1']:.4f}  thr={best_thr:.2f}")

    cols  = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
    means = {c: float(np.mean([m[c] for m in fold_metrics])) for c in cols}
    stds  = {c: float(np.std( [m[c] for m in fold_metrics])) for c in cols}

    print("\n" + "=" * 65)
    print("DGUDILI 2026  10-Fold CV Average")
    print("=" * 65)
    print("  " + "  ".join(f"{c}: {means[c]:.4f} ±{stds[c]:.4f}" for c in cols))

    rows = [{"model": f"fold_{i+1}", **m} for i, m in enumerate(fold_metrics)]
    rows.append({"model": "DGUDILI_2026", **means})
    rows.append({"model": "DGUDILI_2026_std", **stds})
    df = pd.DataFrame(rows)

    out_path = os.path.join(cfg.output_dir, "results_cv.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Stage CV OK")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _ROOT = os.path.dirname(_SRC)  # DGUDILI_2026/
    default_cfg = os.path.join(_ROOT, "config.yaml")

    parser = argparse.ArgumentParser(description="DGUDILI_2026 GraphMACCSEncoder 학습")
    parser.add_argument("--stage",  required=True, choices=["pretrain", "stack", "cv"],
                        help="실행할 스테이지")
    parser.add_argument("--config", default=default_cfg,
                        help=f"YAML 설정 파일 경로 (기본값: {default_cfg})")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    STAGES = {"pretrain": pretrain, "stack": stack, "cv": cv}
    STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()
