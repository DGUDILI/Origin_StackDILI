import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
SRC_DIR  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
from model import E2E_FTV6StyleEncoder

DATA_PATH = r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv"
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"

os.makedirs(OUT_DIR, exist_ok=True)

K          = 16
D_K        = 32
D_V        = 4
BATCH_SIZE = 16
MAX_LENGTH = 256
SEED       = 42
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

print("=" * 65)
print("Step 4 E2E: 16-dim extract + Stacking OOF (Stage 2)")
print("=" * 65)

enc_path = os.path.join(OUT_DIR, "pretrained_encoder_e2e_cls.pt")
assert os.path.exists(enc_path), f"Missing: {enc_path}\nRun Step3_e2e_pretrain.py first."

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
df_meta = pd.read_csv(DATA_PATH)
df_feat = pd.read_csv(FEAT_PATH)
assert list(df_meta["SMILES"]) == list(df_feat["SMILES"]), "SMILES order mismatch"

feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
y_all      = df_feat["Label"].values.astype(np.float32)
ref_all    = df_feat["ref"].values
smiles_all = df_meta["SMILES"].values

train_mask = ref_all != "DILIrank"
test_mask  = ref_all == "DILIrank"

smiles_train = smiles_all[train_mask]
smiles_test  = smiles_all[test_mask]
y_train      = y_all[train_mask]
y_test       = y_all[test_mask]

scaler_path = os.path.join(DATA_DIR, "scalers_ftv6_cls.pkl")
assert os.path.exists(scaler_path), f"Missing: {scaler_path}\nRun Step2 first."
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
scaler_fp = scalers["fp"]

fp_train_scaled = scaler_fp.transform(X_fp_all[train_mask]).astype(np.float32)
fp_test_scaled  = scaler_fp.transform(X_fp_all[test_mask]).astype(np.float32)
fp_in_dim = fp_train_scaled.shape[1]

print(f"Train: {len(y_train)}  |  Test: {len(y_test)}  |  FP: {fp_in_dim}-dim")

# ── Tokenizer & Dataset ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, fp_array):
        self.smiles = smiles_list
        self.fp     = fp_array

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.smiles[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.fp[idx], dtype=torch.float32),
        )


# ── 모델 로드 ────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = E2E_FTV6StyleEncoder(
    fp_in_dim=fp_in_dim, k=K, d_k=D_K, d_v=D_V,
    model_name=MODEL_NAME,
).to(device)
encoder.load_state_dict(torch.load(enc_path, map_location=device))
encoder.eval()
print(f"Encoder loaded ({device})")


# ── 16-dim Feature 추출 ──────────────────────────────────────────────────────
def extract_features(smiles_list, fp_scaled):
    ds = SMILESDataset(smiles_list, fp_scaled)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for ids, mask, fp in dl:
            ids, mask, fp = ids.to(device), mask.to(device), fp.to(device)
            feats.append(encoder.encode(ids, mask, fp).cpu().numpy())
    return np.concatenate(feats, axis=0)


X_train_feat = extract_features(smiles_train, fp_train_scaled)
X_test_feat  = extract_features(smiles_test,  fp_test_scaled)
print(f"Feature Space: train={X_train_feat.shape}, test={X_test_feat.shape}")

# ── Base Models ──────────────────────────────────────────────────────────────
BASE_MODELS = [
    ("RF",     RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
    ("ET",     ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
    ("HistGB", HistGradientBoostingClassifier(max_iter=300, random_state=SEED)),
    ("XGB",    XGBClassifier(
                   n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", random_state=SEED)),
]

kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
n_base = len(BASE_MODELS)

oof_probs  = np.zeros((len(X_train_feat), n_base))
test_probs = np.zeros((len(X_test_feat),  n_base))

# ── 5-Fold OOF Stacking ──────────────────────────────────────────────────────
print(f"\n[Stacking] 5-Fold OOF training ({n_base} base models)...")
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
print("\n[Meta-model] LogisticRegression on OOF probs...")
meta = LogisticRegression(max_iter=1000, random_state=SEED)
meta.fit(oof_probs, y_train)

final_proba = meta.predict_proba(test_probs)[:, 1]
final_pred  = (final_proba >= 0.5).astype(int)
tn, fp_, fn, tp = confusion_matrix(y_test, final_pred).ravel()

# ── 평가 결과 ────────────────────────────────────────────────────────────────
dgudili_e2e = {
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
ftv6_dv1 = {  # d_v=1 이전 결과
    "AUC": 0.9196, "MCC": 0.6682, "F1": 0.8106, "ACC": 0.8252,
    "Precision": 0.7253, "Sensitivity": 0.9185, "Specificity": 0.7612,
}
e2e_dv1 = {   # E2E d_v=1 이전 결과
    "AUC": 0.9219, "MCC": 0.7270, "F1": 0.8426, "ACC": 0.8628,
    "Precision": 0.7905, "Sensitivity": 0.9022, "Specificity": 0.8358,
}

cols = ["AUC", "MCC", "F1", "ACC", "Precision", "Sensitivity", "Specificity"]
print("\n" + "=" * 82)
print("E2E DGUDILI (d_v=4, val_loss) vs 이전 실험  |  DILIrank N=452")
print("=" * 82)
print(f"{'':32s}" + "".join(f"{c:>10s}" for c in cols))
print("-" * 82)
print(f"{'StackDILI (목표)':32s}"    + "".join(f"{baseline[c]:>10.4f}" for c in cols))
print(f"{'ftv6_cls (d_v=1)':32s}"    + "".join(f"{ftv6_dv1[c]:>10.4f}" for c in cols))
print(f"{'E2E_cls (d_v=1, val_auc)':32s}" + "".join(f"{e2e_dv1[c]:>10.4f}" for c in cols))
print(f"{'E2E_cls (d_v=4, val_loss)':32s}" + "".join(f"{dgudili_e2e[c]:>10.4f}" for c in cols))
print("-" * 82)
print(f"{'vs StackDILI':32s}" + "".join(f"{dgudili_e2e[c]-baseline[c]:>+10.4f}" for c in cols))
print(f"{'vs E2E d_v=1':32s}" + "".join(f"{dgudili_e2e[c]-e2e_dv1[c]:>+10.4f}" for c in cols))
print("=" * 82)

results_df = pd.DataFrame(
    [baseline, ftv6_dv1, e2e_dv1, dgudili_e2e],
    index=["StackDILI", "DGUDILI_ftv6_dv1", "DGUDILI_E2E_dv1", "DGUDILI_E2E_dv4"]
)
csv_path = os.path.join(OUT_DIR, "results_e2e_cls.csv")
results_df.to_csv(csv_path)
print(f"\nSaved: {csv_path}")
print("Step 4 E2E OK")
