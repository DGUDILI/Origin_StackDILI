"""
compare_all.py — 저장된 모든 모델을 로드해 테스트셋 성능 비교

체크포인트가 존재하는 모델만 자동으로 평가하며, 없는 모델은 건너뜁니다.
"""

import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

# ── 공통 데이터 로드 ──────────────────────────────────────────────
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
EMB_PATH  = os.path.join(DATA_DIR, "chemberta_embeddings.npy")

df_feat    = pd.read_csv(FEAT_PATH)
feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
embeddings = np.load(EMB_PATH).astype(np.float32)

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"
y_test     = y_all[test_mask]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "AUC":         roc_auc_score(y_test, proba),
        "MCC":         matthews_corrcoef(y_test, pred),
        "F1":          f1_score(y_test, pred),
        "ACC":         accuracy_score(y_test, pred),
        "Precision":   precision_score(y_test, pred, zero_division=0),
        "Sensitivity": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp_),
        "_proba":      proba,
        "_TP": tp, "_FN": fn, "_FP": fp_, "_TN": tn,
    }


def load_scalers(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_test_tensors(scaler_fp, scaler_lm):
    X_fp = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
    X_lm = scaler_lm.transform(embeddings[test_mask]).astype(np.float32)
    return torch.tensor(X_fp).to(device), torch.tensor(X_lm).to(device)


def infer(model, fp_t, lm_t) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(fp_t, lm_t).squeeze(1)).cpu().numpy()


# ── 결과 수집 ─────────────────────────────────────────────────────
results = {}

# 0. StackDILI (하드코딩 — 별도 Jupyter 노트북 모델)
results["StackDILI"] = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
    "_proba": None,
}

# 1. DGUDILI_2026 prev (CrossAttentionEncoder + LR)
_enc_pt  = os.path.join(OUT_DIR, "pretrained_encoder.pt")
_k16_dir = DATA_DIR
if os.path.exists(_enc_pt) and os.path.exists(os.path.join(_k16_dir, "fp_k16_train.npy")):
    try:
        from model import CrossAttentionEncoder
        enc = CrossAttentionEncoder(k=16, d_k=32).to(device)
        enc.load_state_dict(torch.load(_enc_pt, map_location=device))
        enc.eval()

        def _load_t(name):
            return torch.tensor(np.load(os.path.join(_k16_dir, name)),
                                dtype=torch.float32).to(device)

        with torch.no_grad():
            X_tr16 = enc.encode(_load_t("cham_k16_train.npy"),
                                 _load_t("fp_k16_train.npy")).cpu().numpy()
            X_te16 = enc.encode(_load_t("cham_k16_test.npy"),
                                 _load_t("fp_k16_test.npy")).cpu().numpy()

        y_tr = y_all[train_mask]
        lr   = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr16, y_tr)
        proba = lr.predict_proba(X_te16)[:, 1]
        results["DGUDILI_2026_prev"] = compute_metrics(proba)
        print("[OK] DGUDILI_2026_prev")
    except Exception as e:
        print(f"[SKIP] DGUDILI_2026_prev: {e}")
else:
    print("[SKIP] DGUDILI_2026_prev: checkpoint not found")

# 2. GroupCrossAttn v1 (single seed=42)
_v1_pt  = os.path.join(OUT_DIR, "group_model_best.pt")
_v1_sc  = os.path.join(DATA_DIR, "scalers_group.pkl")
if os.path.exists(_v1_pt) and os.path.exists(_v1_sc):
    try:
        from group_model import GroupCrossAttentionDILI
        m = GroupCrossAttentionDILI(d=16, dropout=0.0).to(device)
        m.load_state_dict(torch.load(_v1_pt, map_location=device))
        sc = load_scalers(_v1_sc)
        fp_t, lm_t = get_test_tensors(sc["fp"], sc["lm"])
        results["GroupCrossAttn_v1"] = compute_metrics(infer(m, fp_t, lm_t))
        print("[OK] GroupCrossAttn_v1")
    except Exception as e:
        print(f"[SKIP] GroupCrossAttn_v1: {e}")
else:
    print("[SKIP] GroupCrossAttn_v1: checkpoint not found")

# 3. Ensemble x5
_ens_seeds = [42, 0, 7, 21, 99]
_ens_sc    = os.path.join(OUT_DIR, "ensemble_scalers.pkl")
_ens_pts   = [os.path.join(OUT_DIR, f"ensemble_seed{s}.pt") for s in _ens_seeds]
if all(os.path.exists(p) for p in _ens_pts) and os.path.exists(_ens_sc):
    try:
        from group_model import GroupCrossAttentionDILI
        sc = load_scalers(_ens_sc)
        fp_t, lm_t = get_test_tensors(sc["fp"], sc["lm"])
        all_proba = np.zeros(len(y_test))
        for s, pt in zip(_ens_seeds, _ens_pts):
            m = GroupCrossAttentionDILI(d=16, dropout=0.0).to(device)
            m.load_state_dict(torch.load(pt, map_location=device))
            all_proba += infer(m, fp_t, lm_t)
        results[f"Ensemble_x{len(_ens_seeds)}"] = compute_metrics(
            all_proba / len(_ens_seeds)
        )
        print(f"[OK] Ensemble_x{len(_ens_seeds)}")
    except Exception as e:
        print(f"[SKIP] Ensemble: {e}")
else:
    print("[SKIP] Ensemble: checkpoint(s) not found")

# 4. ModeB
_mb_pt = os.path.join(OUT_DIR, "modeB_best.pt")
_mb_sc = os.path.join(DATA_DIR, "scalers_modeB.pkl")
if os.path.exists(_mb_pt) and os.path.exists(_mb_sc):
    try:
        from group_model_b import GroupCrossAttentionModB
        m = GroupCrossAttentionModB(d_model=64, d_out=16, dropout=0.0).to(device)
        m.load_state_dict(torch.load(_mb_pt, map_location=device))
        sc = load_scalers(_mb_sc)
        fp_t, lm_t = get_test_tensors(sc["fp"], sc["lm"])
        results["ModeB"] = compute_metrics(infer(m, fp_t, lm_t))
        print("[OK] ModeB")
    except Exception as e:
        print(f"[SKIP] ModeB: {e}")
else:
    print("[SKIP] ModeB: checkpoint not found")

# 5. ModeBV2 (체크포인트가 있을 경우)
_mbv2_pt = os.path.join(OUT_DIR, "modeBV2_best.pt")
_mbv2_sc = os.path.join(DATA_DIR, "scalers_modeBV2.pkl")
if os.path.exists(_mbv2_pt) and os.path.exists(_mbv2_sc):
    try:
        from group_model_b import GroupCrossAttentionModB_V2
        m = GroupCrossAttentionModB_V2(d_model=128, d_out=32, nhead=4, dropout=0.0).to(device)
        m.load_state_dict(torch.load(_mbv2_pt, map_location=device))
        sc = load_scalers(_mbv2_sc)
        fp_t, lm_t = get_test_tensors(sc["fp"], sc["lm"])
        results["ModeBV2"] = compute_metrics(infer(m, fp_t, lm_t))
        print("[OK] ModeBV2")
    except Exception as e:
        print(f"[SKIP] ModeBV2: {e}")
else:
    print("[SKIP] ModeBV2: checkpoint not found")

# ── 결과 테이블 출력 ──────────────────────────────────────────────
COLS = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
col_w = 8
name_w = 24

print("\n" + "=" * (name_w + col_w * len(COLS)))
print(f"  Test set: DILIrank (N=452,  pos=184, neg=268)")
print("=" * (name_w + col_w * len(COLS)))
print(f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in COLS))
print("-" * (name_w + col_w * len(COLS)))

for name, d in results.items():
    row = f"{name:<{name_w}}"
    for c in COLS:
        val = d.get(c)
        row += f"{val:>{col_w}.4f}" if val is not None else f"{'N/A':>{col_w}}"
    print(row)

    if "_TP" in d:
        print(f"  {'':>{name_w-2}}  TP={d['_TP']:>3}  FN={d['_FN']:>3}  "
              f"FP={d['_FP']:>3}  TN={d['_TN']:>3}")

print("=" * (name_w + col_w * len(COLS)))

# ── CSV 저장 ──────────────────────────────────────────────────────
save_cols = {name: {c: d.get(c) for c in COLS} for name, d in results.items()}
df_out = pd.DataFrame(save_cols).T
csv_path = os.path.join(OUT_DIR, "compare_all_results.csv")
df_out.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── ROC 곡선 시각화 ────────────────────────────────────────────────
colors = ["steelblue", "salmon", "orange", "seagreen", "purple", "brown"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1)

for (name, d), col in zip(results.items(), colors):
    proba = d.get("_proba")
    if proba is None:
        continue
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, color=col,
            label=f"{name}  AUC={d['AUC']:.4f}")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models (DILIrank test set)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "compare_all_roc.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")
print("\ncompare_all OK")
