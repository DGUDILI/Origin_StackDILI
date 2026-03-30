import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from group_model import GroupCrossAttentionDILI

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 하이퍼파라미터 (v1과 완전 동일) ──────────────────────────────
D          = 16
DROPOUT    = 0.3
LR         = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS     = 300
BATCH_SIZE = 32
PATIENCE   = 40
SEEDS      = [42, 0, 7, 21, 99]   # 5 seeds
# ──────────────────────────────────────────────────────────────────

print("=" * 65)
print("Step 7: Seed Ensemble (GroupCrossAttentionDILI x5)")
print(f"  seeds={SEEDS}")
print(f"  d={D}, dropout={DROPOUT}, lr={LR}, patience={PATIENCE}")
print("=" * 65)

# ── 1. 데이터 로드 & 스케일링 (seed 독립적) ─────────────────────
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
y_tr       = y_all[train_mask]
y_test     = y_all[test_mask]

scaler_fp = StandardScaler()
scaler_lm = StandardScaler()
X_fp_train = scaler_fp.fit_transform(X_fp_all[train_mask]).astype(np.float32)
X_fp_test  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
X_lm_train = scaler_lm.fit_transform(embeddings[train_mask]).astype(np.float32)
X_lm_test  = scaler_lm.transform(embeddings[test_mask]).astype(np.float32)

print(f"Train: {train_mask.sum()}  |  Test: {test_mask.sum()}")

# ── 2. seed별 학습 함수 ──────────────────────────────────────────
def train_one_seed(seed: int) -> tuple[float, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    fp_train_t = torch.tensor(X_fp_train)
    lm_train_t = torch.tensor(X_lm_train)
    y_train_t  = torch.tensor(y_tr)
    fp_test_t  = torch.tensor(X_fp_test)
    lm_test_t  = torch.tensor(X_lm_test)

    train_ds = TensorDataset(fp_train_t, lm_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = GroupCrossAttentionDILI(d=D, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-6
    )
    pos_w     = torch.tensor([float((1 - y_tr).sum()) / float(y_tr.sum())]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    fp_test_d  = fp_test_t.to(device)
    lm_test_d  = lm_test_t.to(device)

    best_auc, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_fp_b, x_lm_b, y_b in train_dl:
            x_fp_b, x_lm_b, y_b = x_fp_b.to(device), x_lm_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_fp_b, x_lm_b).squeeze(1), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(fp_test_d, lm_test_d).squeeze(1)).cpu().numpy()
        auc = roc_auc_score(y_test, proba)
        scheduler.step(auc)

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    return best_auc, best_state


# ── 3. 전체 seed 학습 ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_proba = np.zeros(len(y_test))   # 확률 누적
seed_aucs = []

print()
for i, seed in enumerate(SEEDS):
    print(f"[{i+1}/{len(SEEDS)}] Seed={seed} training...", end=" ", flush=True)
    best_auc, best_state = train_one_seed(seed)
    seed_aucs.append(best_auc)

    # 개별 모델 추론
    model = GroupCrossAttentionDILI(d=D, dropout=0.0).to(device)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba_i = torch.sigmoid(
            model(torch.tensor(X_fp_test).to(device),
                  torch.tensor(X_lm_test).to(device)).squeeze(1)
        ).cpu().numpy()

    all_proba += proba_i
    auc_so_far = roc_auc_score(y_test, all_proba / (i + 1))
    print(f"best_AUC={best_auc:.4f}  |  ensemble_AUC={auc_so_far:.4f}")

    # 개별 모델 저장
    torch.save(best_state, os.path.join(OUT_DIR, f"ensemble_seed{seed}.pt"))

# 앙상블 최종 확률
ensemble_proba = all_proba / len(SEEDS)

# ── 4. 앙상블 평가 ────────────────────────────────────────────────
pred = (ensemble_proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, pred).ravel()

ensemble = {
    "AUC":         roc_auc_score(y_test, ensemble_proba),
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
group_v1 = {
    "AUC": 0.8410, "MCC": 0.5078, "F1": 0.7246, "ACC": 0.7478,
    "Precision": 0.6522, "Sensitivity": 0.8152, "Specificity": 0.7015,
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 84)
print("Comparison  |  Test set: DILIrank (N=452,  pos=184, neg=268)")
print("=" * 84)
print(f"{'Model':<30s}" + "".join(f"{c:>8s}" for c in cols))
print("-" * 84)
for name, d in [
    ("StackDILI (baseline)",   baseline),
    ("DGUDILI_2026 (prev)",    dgudili_prev),
    ("GroupCrossAttn v1",      group_v1),
    (f"Ensemble x{len(SEEDS)}", ensemble),
]:
    print(f"{name:<30s}" + "".join(f"{d[c]:>8.4f}" for c in cols))
print("-" * 84)
print(f"{'Delta Ensemble vs v1 (up)':30s}" +
      "".join(f"{ensemble[c]-group_v1[c]:>+8.4f}" for c in cols))
print("=" * 84)

print(f"\nConfusion Matrix (threshold=0.5):")
print(f"  TP={tp:>4d}  FN={fn:>4d}")
print(f"  FP={fp_:>4d}  TN={tn:>4d}")

print(f"\nIndividual seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}")
print(f"Mean single-model AUC: {np.mean(seed_aucs):.4f}")
print(f"Ensemble      AUC    : {ensemble['AUC']:.4f}")

# ── 5. 저장 ───────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "ensemble_proba.npy"), ensemble_proba)
with open(os.path.join(OUT_DIR, "ensemble_scalers.pkl"), "wb") as f:
    pickle.dump({"fp": scaler_fp, "lm": scaler_lm}, f)

all_results = pd.DataFrame(
    [baseline, dgudili_prev, group_v1, ensemble],
    index=["StackDILI", "DGUDILI_2026_prev", "GroupCrossAttn_v1", f"Ensemble_x{len(SEEDS)}"]
)
csv_path = os.path.join(OUT_DIR, "results_ensemble_comparison.csv")
all_results.to_csv(csv_path)

# ── 6. 시각화 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) ROC 곡선
fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_proba)
axes[0].plot(fpr_ens, tpr_ens, "b-", lw=2,
             label=f"Ensemble AUC={ensemble['AUC']:.4f}")
axes[0].scatter([1 - group_v1["Specificity"]], [group_v1["Sensitivity"]],
                color="orange", s=80, zorder=5,
                label=f"v1 (AUC={group_v1['AUC']:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# (b) 메트릭 비교
metric_sub = ["AUC", "MCC", "Specificity", "Sensitivity"]
x = np.arange(len(metric_sub))
w = 0.22
for i, (d, lbl, col) in enumerate(zip(
    [baseline, dgudili_prev, group_v1, ensemble],
    ["StackDILI", "DGUDILI_prev", "v1", f"Ensemble x{len(SEEDS)}"],
    ["steelblue", "salmon", "orange", "seagreen"],
)):
    axes[1].bar(x + (i - 1.5) * w, [d[m] for m in metric_sub], w, label=lbl, color=col)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_sub)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Metrics Comparison")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# (c) Seed별 AUC + 앙상블
bar_labels = [f"s={s}" for s in SEEDS] + ["Ensemble"]
bar_values = seed_aucs + [ensemble["AUC"]]
bar_colors = ["#4C72B0"] * len(SEEDS) + ["seagreen"]
axes[2].bar(bar_labels, bar_values, color=bar_colors)
axes[2].axhline(group_v1["AUC"], color="orange", linestyle="--", lw=1.5,
                label=f"v1 single (seed=42): {group_v1['AUC']:.4f}")
axes[2].set_ylim(min(seed_aucs) - 0.02, max(bar_values) + 0.02)
axes[2].set_title("Individual Seeds vs Ensemble AUC")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")
for xi, val in enumerate(bar_values):
    axes[2].text(xi, val + 0.002, f"{val:.4f}", ha="center", fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "ensemble_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: {csv_path}")
print(f"Saved: {fig_path}")
print("Step 7 OK")
