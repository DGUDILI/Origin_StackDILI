import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, confusion_matrix, roc_curve
)
from sklearn.decomposition import PCA

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import CrossAttentionEncoder

K, D_K = 16, 32

def load_np(name): return np.load(os.path.join(DATA_DIR, name))
def load_t(name):  return torch.tensor(load_np(name), dtype=torch.float32)

fp_train   = load_t("fp_k16_train.npy")
fp_test    = load_t("fp_k16_test.npy")
cham_train = load_t("cham_k16_train.npy")
cham_test  = load_t("cham_k16_test.npy")
y_train    = load_np("y_train.npy")
y_test     = load_np("y_test.npy")

encoder = CrossAttentionEncoder(k=K, d_k=D_K)
encoder.load_state_dict(torch.load(os.path.join(OUT_DIR, "pretrained_encoder.pt"), map_location="cpu"))
encoder.eval()

with torch.no_grad():
    X_train_16 = encoder.encode(cham_train, fp_train).numpy()
    X_test_16  = encoder.encode(cham_test,  fp_test).numpy()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_16, y_train)
proba = lr.predict_proba(X_test_16)[:, 1]
pred  = lr.predict(X_test_16)

tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

print("=" * 60)
print("[1] Confusion Matrix Analysis")
print("=" * 60)
print(f"  TP={tp:3d}  FP={fp_:3d}")
print(f"  FN={fn:3d}  TN={tn:3d}")
print(f"  Sensitivity = {tp/(tp+fn):.4f}")
print(f"  Specificity = {tn/(tn+fp_):.4f}  *** PROBLEM")
print(f"  FPR         = {fp_/(fp_+tn):.4f}")
print(f"  MCC         = {matthews_corrcoef(y_test, pred):.4f}")
num = tp*tn - fp_*fn
den = math.sqrt((tp+fp_)*(tp+fn)*(tn+fp_)*(tn+fn))
print(f"  MCC = ({tp}*{tn} - {fp_}*{fn}) / sqrt(...) = {num}/{den:.1f} = {num/den:.4f}")
print(f"  Key: FP={fp_} ~ TP={tp} -> numerator collapses")

print()
print("=" * 60)
print("[2] Prediction Probability Distribution")
print("=" * 60)
proba_pos = proba[y_test == 1]
proba_neg = proba[y_test == 0]
print(f"  Positive mean={proba_pos.mean():.4f}, std={proba_pos.std():.4f}")
print(f"  Negative mean={proba_neg.mean():.4f}, std={proba_neg.std():.4f}")
print(f"  Mean gap: {proba_pos.mean()-proba_neg.mean():.4f}  (should be large)")
print(f"  Neg with prob>0.5: {int((proba_neg>0.5).sum())}/{len(proba_neg)} = {(proba_neg>0.5).mean():.4f}  -> FP source")

print()
print("=" * 60)
print("[3] Feature Space (16-dim) Quality")
print("=" * 60)
mu_p  = X_train_16[y_train==1].mean(axis=0)
mu_n  = X_train_16[y_train==0].mean(axis=0)
std_p = X_train_16[y_train==1].std(axis=0)
std_n = X_train_16[y_train==0].std(axis=0)
fisher = np.abs(mu_p - mu_n) / (std_p + std_n + 1e-8)
print(f"  Fisher ratio (train): mean={fisher.mean():.4f}, max={fisher.max():.4f}")
print(f"  Dims with Fisher<0.1: {(fisher<0.1).sum()}/16  (class overlap)")

print()
print("=" * 60)
print("[4] d_v=1 Bottleneck Check")
print("=" * 60)
with torch.no_grad():
    q_in  = cham_test[:50].unsqueeze(-1)
    k_in  = fp_test[:50].unsqueeze(-1)
    Q     = encoder.W_Q(q_in)
    K_mat = encoder.W_K(k_in)
    V     = encoder.W_V(k_in)
    scores   = torch.bmm(Q, K_mat.transpose(1,2)) / math.sqrt(D_K)
    weights  = torch.softmax(scores, dim=-1)
    attn_out = torch.bmm(weights, V).squeeze(-1)

entropy = -(weights * (weights + 1e-9).log()).sum(dim=-1).mean().item()
max_ent = math.log(K)
print(f"  Attention entropy: {entropy:.4f} / {max_ent:.4f} = {entropy/max_ent:.4f}")
print(f"  W_V weight: {encoder.W_V.weight.data.item():.6f}  bias: {encoder.W_V.bias.data.item():.6f}")
print(f"  attn_out std: {attn_out.std().item():.6f}  (near 0 = no discriminative info)")

print()
print("=" * 60)
print("[5] Class Balance")
print("=" * 60)
print(f"  Train ratio: {y_train.mean():.3f}  (>0.5 -> bias toward positive)")
print(f"  Test  ratio: {y_test.mean():.3f}")

print()
print("=" * 60)
print("[6] Stage1 vs Stage2 AUC Gap")
print("=" * 60)
with torch.no_grad():
    enc_tr = torch.sigmoid(encoder(cham_train, fp_train)).squeeze(1).numpy()
    enc_te = torch.sigmoid(encoder(cham_test,  fp_test)).squeeze(1).numpy()
print(f"  Encoder head AUC train: {roc_auc_score(y_train, enc_tr):.4f}")
print(f"  Encoder head AUC test:  {roc_auc_score(y_test,  enc_te):.4f}  <- Stage1 ceiling")
print(f"  LR on 16-dim AUC test:  {roc_auc_score(y_test,  proba):.4f}  <- Stage2 result")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("MCC Low Cause Analysis -- DGUDILI 2026", fontsize=14, fontweight="bold")

ax = axes[0]
ax.hist(proba_neg, bins=30, alpha=0.6, color="#3366CC", label=f"Negative (n={len(proba_neg)})", density=True)
ax.hist(proba_pos, bins=30, alpha=0.6, color="#CC3333", label=f"Positive (n={len(proba_pos)})", density=True)
ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="threshold=0.5")
ax.set_title(f"Prob Distribution\nNeg={proba_neg.mean():.3f}  Pos={proba_pos.mean():.3f}", fontsize=10)
ax.set_xlabel("Predicted Probability"); ax.legend(fontsize=9)

ax = axes[1]
cm_arr = np.array([[tp, fn], [fp_, tn]])
labels = [["TP","FN"],["FP","TN"]]
im = ax.imshow(cm_arr, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm_arr[i,j]}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm_arr[i,j] > cm_arr.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred Pos","Pred Neg"]); ax.set_yticklabels(["Actual Pos","Actual Neg"])
ax.set_title(f"Confusion Matrix\nMCC={matthews_corrcoef(y_test,pred):.4f}  Spec={tn/(tn+fp_):.4f}", fontsize=10)
plt.colorbar(im, ax=ax)

ax = axes[2]
fpr_v, tpr_v, _ = roc_curve(y_test, proba)
auc_val = roc_auc_score(y_test, proba)
ax.plot(fpr_v, tpr_v, color="#CC3333", lw=2, label=f"DGUDILI 2026 (AUC={auc_val:.4f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.axvline(fp_/(fp_+tn), color="#CC3333", linestyle=":", alpha=0.8, label=f"FPR={fp_/(fp_+tn):.3f}")
ax.set_title("ROC Curve\nHigh FPR = Low Specificity = Low MCC", fontsize=10)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=9)
ax.text(0.38, 0.1, "StackDILI AUC=0.9736", fontsize=9, color="#3366CC", transform=ax.transAxes)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "mcc_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
