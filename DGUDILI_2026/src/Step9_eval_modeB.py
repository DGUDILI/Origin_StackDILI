import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from group_model_b import GroupCrossAttentionModB

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("Step 9: Mode B - Evaluation & Comparison")
print("=" * 65)

# ── 1. 모델 & 스케일러 로드 ────────────────────────────────────────
model_path  = os.path.join(OUT_DIR, "modeB_best.pt")
scaler_path = os.path.join(DATA_DIR, "scalers_modeB.pkl")
assert os.path.exists(model_path),  f"Missing: {model_path}\nRun Step8 first."
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step8 first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GroupCrossAttentionModB(d_model=64, d_out=16, dropout=0.0).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"ModeB model loaded ({device})")

with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)

# ── 2. 테스트 데이터 ───────────────────────────────────────────────
FEAT_PATH  = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
EMB_PATH   = os.path.join(DATA_DIR, "chemberta_embeddings.npy")

df_feat    = pd.read_csv(FEAT_PATH)
feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
embeddings = np.load(EMB_PATH).astype(np.float32)

test_mask  = ref_all == "DILIrank"
y_test     = y_all[test_mask]

X_fp_test  = scalers["fp"].transform(X_fp_all[test_mask]).astype(np.float32)
X_lm_test  = scalers["lm"].transform(embeddings[test_mask]).astype(np.float32)

fp_test_t  = torch.tensor(X_fp_test).to(device)
lm_test_t  = torch.tensor(X_lm_test).to(device)

# ── 3. 예측 ───────────────────────────────────────────────────────
with torch.no_grad():
    proba = torch.sigmoid(model(fp_test_t, lm_test_t).squeeze(1)).cpu().numpy()

pred = (proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

modeB = {
    "AUC":         roc_auc_score(y_test, proba),
    "MCC":         matthews_corrcoef(y_test, pred),
    "F1":          f1_score(y_test, pred),
    "ACC":         accuracy_score(y_test, pred),
    "Precision":   precision_score(y_test, pred, zero_division=0),
    "Sensitivity": recall_score(y_test, pred),
    "Specificity": tn / (tn + fp_),
}

# ── 4. 비교 기준값 ─────────────────────────────────────────────────
baseline = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}
dgudili_prev = {
    "AUC": 0.7312, "MCC": 0.3045, "F1": 0.6362, "ACC": 0.6128,
    "Precision": 0.5152, "Sensitivity": 0.8315, "Specificity": 0.4627,
}
group_v1 = {
    "AUC": 0.8410, "MCC": 0.5078, "F1": 0.7246, "ACC": 0.7478,
    "Precision": 0.6522, "Sensitivity": 0.8152, "Specificity": 0.7015,
}
ensemble_x5 = {
    "AUC": 0.8729, "MCC": 0.5966, "F1": 0.7652, "ACC": 0.8031,
    "Precision": 0.7436, "Sensitivity": 0.7880, "Specificity": 0.8134,
}

# ── 5. 결과 테이블 ─────────────────────────────────────────────────
cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 88)
print("Comparison  |  Test set: DILIrank (N=452,  pos=184, neg=268)")
print("=" * 88)
print(f"{'Model':<30s}" + "".join(f"{c:>8s}" for c in cols))
print("-" * 88)
for name, d in [
    ("StackDILI (baseline)",    baseline),
    ("DGUDILI_2026 (prev)",     dgudili_prev),
    ("GroupCrossAttn v1",       group_v1),
    ("Ensemble x5",             ensemble_x5),
    ("ModeB (new)",             modeB),
]:
    print(f"{name:<30s}" + "".join(f"{d[c]:>8.4f}" for c in cols))
print("-" * 88)
print(f"{'Delta ModeB vs Ensemble':30s}" +
      "".join(f"{modeB[c]-ensemble_x5[c]:>+8.4f}" for c in cols))
print(f"{'Delta ModeB vs baseline':30s}" +
      "".join(f"{modeB[c]-baseline[c]:>+8.4f}" for c in cols))
print("=" * 88)

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"  TP={tp:>4d}  FN={fn:>4d}")
print(f"  FP={fp_:>4d}  TN={tn:>4d}")

# ── 6. Attention Weight 분석 ──────────────────────────────────────
attn_w    = model.get_attention_weights(fp_test_t, lm_test_t).cpu().numpy()
mean_attn = attn_w.mean(axis=0)

print(f"\n[Group Attention Weights - mean over test set]")
prev_attn = {"Const+CalcCATS": 0.2154+0.2820,  # v1 const+calcCATS 합산
             "PC1-6": 0.0,
             "MACCS": 0.2682, "E-state": 0.2345}
for gname, w in zip(GroupCrossAttentionModB.GROUP_NAMES, mean_attn):
    bar = "#" * int(w * 40)
    print(f"  {gname:<18s}: {w:.4f}  {bar}")

# ── 7. CSV 저장 ────────────────────────────────────────────────────
all_results = pd.DataFrame(
    [baseline, dgudili_prev, group_v1, ensemble_x5, modeB],
    index=["StackDILI", "DGUDILI_prev", "GroupCrossAttn_v1",
           "Ensemble_x5", "ModeB"]
)
csv_path = os.path.join(OUT_DIR, "results_modeB_comparison.csv")
all_results.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── 8. 시각화 (1x3) ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) ROC 곡선
fpr_b, tpr_b, _ = roc_curve(y_test, proba)
axes[0].plot(fpr_b, tpr_b, "b-", lw=2,
             label=f"ModeB AUC={modeB['AUC']:.4f}")
fpr_e = 1 - ensemble_x5["Specificity"]
axes[0].scatter([fpr_e], [ensemble_x5["Sensitivity"]], color="seagreen",
                s=80, zorder=5, label=f"Ensemble x5 AUC={ensemble_x5['AUC']:.4f}")
axes[0].scatter([1-group_v1["Specificity"]], [group_v1["Sensitivity"]],
                color="orange", s=80, zorder=5,
                label=f"v1 AUC={group_v1['AUC']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (test set)")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].grid(True, alpha=0.3)

# (b) 메트릭 비교 막대
metric_sub = ["AUC", "MCC", "Specificity", "Sensitivity"]
x = np.arange(len(metric_sub))
w = 0.17
for i, (d, lbl, col) in enumerate(zip(
    [baseline, dgudili_prev, group_v1, ensemble_x5, modeB],
    ["StackDILI", "DGUDILI_prev", "v1", "Ensemble", "ModeB"],
    ["steelblue", "salmon", "orange", "seagreen", "purple"],
)):
    axes[1].bar(x + (i - 2) * w, [d[m] for m in metric_sub], w,
                label=lbl, color=col, alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_sub)
axes[1].set_ylim(0, 1.12)
axes[1].set_title("Metrics Comparison")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

# (c) Group Attention Weight
bars = axes[2].bar(
    GroupCrossAttentionModB.GROUP_NAMES, mean_attn,
    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
)
for bar, val in zip(bars, mean_attn):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
axes[2].set_ylim(0, max(mean_attn) * 1.35)
axes[2].set_title("Mean Group Attention Weight (ModeB)")
axes[2].set_ylabel("Attention Weight")
axes[2].tick_params(axis="x", rotation=15)
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "modeB_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")
print("Step 9 OK")
