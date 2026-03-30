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
from group_model import GroupCrossAttentionDILI

os.makedirs(OUT_DIR, exist_ok=True)

D = 16

print("=" * 65)
print("Step 6: Group Cross-Attention DILI - Evaluation & Comparison")
print("=" * 65)

# ── 1. 모델 & 스케일러 로드 ────────────────────────────────────────
model_path  = os.path.join(OUT_DIR, "group_model_best.pt")
scaler_path = os.path.join(DATA_DIR, "scalers_group.pkl")
assert os.path.exists(model_path),  f"Missing: {model_path}\nRun Step5 first."
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step5 first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GroupCrossAttentionDILI(d=D, dropout=0.0).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded  ({device})")

with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
scaler_fp = scalers["fp"]
scaler_lm = scalers["lm"]

# ── 2. 테스트 데이터 준비 ─────────────────────────────────────────
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
EMB_PATH  = os.path.join(DATA_DIR, "chemberta_embeddings.npy")

df_feat    = pd.read_csv(FEAT_PATH)
feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
embeddings = np.load(EMB_PATH).astype(np.float32)

test_mask  = ref_all == "DILIrank"
train_mask = ref_all != "DILIrank"

X_fp_test  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
X_lm_test  = scaler_lm.transform(embeddings[test_mask]).astype(np.float32)
y_test     = y_all[test_mask]

fp_test_t  = torch.tensor(X_fp_test).to(device)
lm_test_t  = torch.tensor(X_lm_test).to(device)

# ── 3. 예측 ───────────────────────────────────────────────────────
with torch.no_grad():
    logits = model(fp_test_t, lm_test_t).squeeze(1)
    proba  = torch.sigmoid(logits).cpu().numpy()

pred = (proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

results = {
    "AUC":         roc_auc_score(y_test, proba),
    "MCC":         matthews_corrcoef(y_test, pred),
    "F1":          f1_score(y_test, pred),
    "ACC":         accuracy_score(y_test, pred),
    "Precision":   precision_score(y_test, pred, zero_division=0),
    "Sensitivity": recall_score(y_test, pred),
    "Specificity": tn / (tn + fp_),
}

baseline = {
    "AUC": 0.9736, "MCC": 0.8304, "F1": 0.9010, "ACC": 0.9159,
    "Precision": 0.8650, "Sensitivity": 0.9402, "Specificity": 0.8993,
}
dgudili_prev = {
    "AUC": 0.7312, "MCC": 0.3045, "F1": 0.6362, "ACC": 0.6128,
    "Precision": 0.5152, "Sensitivity": 0.8315, "Specificity": 0.4627,
}

# ── 4. 결과 테이블 출력 ───────────────────────────────────────────
cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 80)
print("Comparison  |  Test set: DILIrank (N=452,  pos=184, neg=268)")
print("=" * 80)
print(f"{'Model':<28s}" + "".join(f"{c:>8s}" for c in cols))
print("-" * 80)
print(f"{'StackDILI (baseline)':<28s}" + "".join(f"{baseline[c]:>8.4f}" for c in cols))
print(f"{'DGUDILI_2026 (prev)':<28s}" + "".join(f"{dgudili_prev[c]:>8.4f}" for c in cols))
print(f"{'GroupCrossAttn (new)':<28s}" + "".join(f"{results[c]:>8.4f}" for c in cols))
print("-" * 80)
print(f"{'Delta vs prev (↑)':<28s}" +
      "".join(f"{results[c]-dgudili_prev[c]:>+8.4f}" for c in cols))
print(f"{'Delta vs baseline (↑)':<28s}" +
      "".join(f"{results[c]-baseline[c]:>+8.4f}" for c in cols))
print("=" * 80)

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"  TP={tp:>4d}  FN={fn:>4d}")
print(f"  FP={fp_:>4d}  TN={tn:>4d}")
print(f"  Sensitivity (Recall) = {tp/(tp+fn):.4f}")
print(f"  Specificity          = {tn/(tn+fp_):.4f}")

# ── 5. Attention Weight 분석 ──────────────────────────────────────
attn_weights = model.get_attention_weights(fp_test_t, lm_test_t).cpu().numpy()
# attn_weights: (N_test, 4)
mean_attn = attn_weights.mean(axis=0)
print(f"\n[Group Attention Weights - mean over test set]")
for gname, w in zip(GroupCrossAttentionDILI.GROUP_NAMES, mean_attn):
    bar = "#" * int(w * 40)
    print(f"  {gname:<16s}: {w:.4f}  {bar}")

# ── 6. 결과 CSV 저장 ──────────────────────────────────────────────
all_results = pd.DataFrame(
    [baseline, dgudili_prev, results],
    index=["StackDILI", "DGUDILI_2026_prev", "GroupCrossAttn"]
)
csv_path = os.path.join(OUT_DIR, "results_group_comparison.csv")
all_results.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── 7. 시각화 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) ROC 곡선
fpr_new, tpr_new, _ = roc_curve(y_test, proba)
axes[0].plot(fpr_new, tpr_new, "b-", lw=2,
             label=f"GroupCrossAttn AUC={results['AUC']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (test set)")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# (b) 메트릭 비교 막대 그래프
metric_subset = ["AUC", "MCC", "Specificity", "Sensitivity"]
x = np.arange(len(metric_subset))
w = 0.25
axes[1].bar(x - w, [baseline[m]    for m in metric_subset], w, label="StackDILI",    color="steelblue")
axes[1].bar(x,     [dgudili_prev[m] for m in metric_subset], w, label="Prev (2026)", color="salmon")
axes[1].bar(x + w, [results[m]      for m in metric_subset], w, label="GroupCrossAttn", color="seagreen")
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_subset)
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Metrics Comparison")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# (c) Group Attention Weight (평균)
bars_c = axes[2].bar(
    GroupCrossAttentionDILI.GROUP_NAMES, mean_attn,
    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
)
for bar, val in zip(bars_c, mean_attn):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", va="bottom", fontsize=10
    )
axes[2].set_ylim(0, max(mean_attn) * 1.3)
axes[2].set_title("Mean Group Attention Weight")
axes[2].set_ylabel("Attention Weight")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "group_model_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")
print("Step 6 OK")
