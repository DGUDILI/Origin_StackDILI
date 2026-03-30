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
from group_model_v2 import GroupCrossAttentionDILIv2

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("Step 6 v2: GroupCrossAttentionDILIv2 - Evaluation & Comparison")
print("=" * 65)

# ── 1. v2 모델 & 스케일러 로드 ────────────────────────────────────
model_path  = os.path.join(OUT_DIR, "group_model_v2_best.pt")
scaler_path = os.path.join(DATA_DIR, "scalers_group_v2.pkl")
assert os.path.exists(model_path),  f"Missing: {model_path}\nRun Step5_v2 first."
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step5_v2 first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GroupCrossAttentionDILIv2(d=16, h=4, dropout=0.0).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"v2 model loaded ({device})")

with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)

# ── 2. 테스트 데이터 준비 ─────────────────────────────────────────
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
EMB_PATH  = os.path.join(DATA_DIR, "chemberta_embeddings.npy")

df_feat    = pd.read_csv(FEAT_PATH)
feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
embeddings = np.load(EMB_PATH).astype(np.float32)

test_mask = ref_all == "DILIrank"
y_test    = y_all[test_mask]

X_fp_test = scalers["fp"].transform(X_fp_all[test_mask]).astype(np.float32)
X_lm_test = scalers["lm"].transform(embeddings[test_mask]).astype(np.float32)

fp_test_t = torch.tensor(X_fp_test).to(device)
lm_test_t = torch.tensor(X_lm_test).to(device)

# ── 3. 예측 ───────────────────────────────────────────────────────
with torch.no_grad():
    proba = torch.sigmoid(model(fp_test_t, lm_test_t).squeeze(1)).cpu().numpy()

pred = (proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

v2 = {
    "AUC":         roc_auc_score(y_test, proba),
    "MCC":         matthews_corrcoef(y_test, pred),
    "F1":          f1_score(y_test, pred),
    "ACC":         accuracy_score(y_test, pred),
    "Precision":   precision_score(y_test, pred, zero_division=0),
    "Sensitivity": recall_score(y_test, pred),
    "Specificity": tn / (tn + fp_),
}

# ── 4. 비교 기준값 (하드코딩) ─────────────────────────────────────
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

# ── 5. 결과 테이블 출력 ───────────────────────────────────────────
cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 84)
print("Comparison  |  Test set: DILIrank (N=452,  pos=184, neg=268)")
print("=" * 84)
print(f"{'Model':<28s}" + "".join(f"{c:>8s}" for c in cols))
print("-" * 84)
for name, d in [
    ("StackDILI (baseline)", baseline),
    ("DGUDILI_2026 (prev)",  dgudili_prev),
    ("GroupCrossAttn v1",    group_v1),
    ("GroupCrossAttn v2",    v2),
]:
    print(f"{name:<28s}" + "".join(f"{d[c]:>8.4f}" for c in cols))
print("-" * 84)
print(f"{'Delta v2 vs v1 (up)':28s}" +
      "".join(f"{v2[c]-group_v1[c]:>+8.4f}" for c in cols))
print(f"{'Delta v2 vs baseline (up)':28s}" +
      "".join(f"{v2[c]-baseline[c]:>+8.4f}" for c in cols))
print("=" * 84)

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"  TP={tp:>4d}  FN={fn:>4d}")
print(f"  FP={fp_:>4d}  TN={tn:>4d}")

# ── 6. Attention Weight 분석 ──────────────────────────────────────
attn_mean  = model.get_attention_weights(fp_test_t, lm_test_t).cpu().numpy()
attn_heads = model.get_attention_weights_per_head(fp_test_t, lm_test_t).cpu().numpy()
# attn_mean  : (N, 4)
# attn_heads : (N, h=4, 4)

mean_v2 = attn_mean.mean(axis=0)   # (4,)
mean_v1 = np.array([0.2154, 0.2820, 0.2682, 0.2345])  # v1 결과 (Step6 출력값)

print(f"\n[Group Attention Weights - mean over test set]")
print(f"  {'Group':<16s}  {'v1':>6s}  {'v2':>6s}  {'delta':>7s}")
for gname, w1, w2 in zip(GroupCrossAttentionDILIv2.GROUP_NAMES, mean_v1, mean_v2):
    print(f"  {gname:<16s}  {w1:>6.4f}  {w2:>6.4f}  {w2-w1:>+7.4f}")

# ── 7. 결과 CSV 저장 ──────────────────────────────────────────────
all_results = pd.DataFrame(
    [baseline, dgudili_prev, group_v1, v2],
    index=["StackDILI", "DGUDILI_2026_prev", "GroupCrossAttn_v1", "GroupCrossAttn_v2"]
)
csv_path = os.path.join(OUT_DIR, "results_group_v2_comparison.csv")
all_results.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── 8. 시각화 (2x2 subplot) ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (0,0) ROC 곡선
fpr_v2, tpr_v2, _ = roc_curve(y_test, proba)
axes[0, 0].plot(fpr_v2, tpr_v2, "b-", lw=2,
                label=f"v2 AUC={v2['AUC']:.4f}")
# v1 ROC는 저장된 수치 없으므로 점으로 표시
axes[0, 0].scatter([1-group_v1["Specificity"]], [group_v1["Sensitivity"]],
                   color="orange", s=100, zorder=5,
                   label=f"v1 (AUC={group_v1['AUC']:.4f})")
axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curve (test set)")
axes[0, 0].legend(loc="lower right")
axes[0, 0].grid(True, alpha=0.3)

# (0,1) 메트릭 비교 막대 (4모델)
metric_sub = ["AUC", "MCC", "Specificity", "Sensitivity"]
x = np.arange(len(metric_sub))
w = 0.2
all_dicts  = [baseline, dgudili_prev, group_v1, v2]
all_labels = ["StackDILI", "DGUDILI_prev", "v1", "v2"]
colors     = ["steelblue", "salmon", "orange", "seagreen"]
for i, (d, lbl, col) in enumerate(zip(all_dicts, all_labels, colors)):
    axes[0, 1].bar(x + (i - 1.5) * w, [d[m] for m in metric_sub],
                   w, label=lbl, color=col)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metric_sub)
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].set_title("Metrics Comparison")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis="y")

# (1,0) 그룹 어텐션 평균 v1 vs v2 비교
x2 = np.arange(len(GroupCrossAttentionDILIv2.GROUP_NAMES))
w2 = 0.35
axes[1, 0].bar(x2 - w2/2, mean_v1, w2, label="v1", color="orange",  alpha=0.8)
axes[1, 0].bar(x2 + w2/2, mean_v2, w2, label="v2", color="seagreen", alpha=0.8)
axes[1, 0].set_xticks(x2)
axes[1, 0].set_xticklabels(GroupCrossAttentionDILIv2.GROUP_NAMES)
axes[1, 0].set_ylim(0, max(max(mean_v1), max(mean_v2)) * 1.35)
axes[1, 0].set_title("Mean Group Attention Weight: v1 vs v2")
axes[1, 0].set_ylabel("Attention Weight")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis="y")
for xi, (w1, w2_val) in enumerate(zip(mean_v1, mean_v2)):
    axes[1, 0].text(xi - w2/2, w1 + 0.003, f"{w1:.3f}", ha="center", fontsize=8)
    axes[1, 0].text(xi + w2/2, w2_val + 0.003, f"{w2_val:.3f}", ha="center", fontsize=8)

# (1,1) v2 헤드별 어텐션 히트맵 (test set 평균)
mean_per_head = attn_heads.mean(axis=0)  # (h=4, 4)
im = axes[1, 1].imshow(mean_per_head, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=mean_per_head.max())
axes[1, 1].set_xticks(range(4))
axes[1, 1].set_xticklabels(GroupCrossAttentionDILIv2.GROUP_NAMES, rotation=20, ha="right")
axes[1, 1].set_yticks(range(4))
axes[1, 1].set_yticklabels([f"Head {i+1}" for i in range(4)])
axes[1, 1].set_title("v2: Per-Head Attention Weight (test mean)")
plt.colorbar(im, ax=axes[1, 1])
for i in range(4):
    for j in range(4):
        axes[1, 1].text(j, i, f"{mean_per_head[i,j]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if mean_per_head[i,j] < 0.35 else "white")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "group_model_v2_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")
print("Step 6 v2 OK")
