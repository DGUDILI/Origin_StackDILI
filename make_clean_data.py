"""
make_clean_data.py

SMILES 문자열은 달라도 RDKit canonical SMILES로 변환했을 때
train/test 간 동일 분자가 존재하는 경우, train에서 해당 행을 제거한다.
(test = DILIrank, benchmark이므로 test는 건드리지 않음)

출력:
  Data/Dataset_clean.csv
  Code/Dataset_feature_clean.csv
"""

import os
import sys
import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("[ERROR] RDKit not installed. Run: pip install rdkit")
    sys.exit(1)

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_CSV  = os.path.join(ROOT, "Data", "Dataset.csv")
FEAT_CSV  = os.path.join(ROOT, "Code", "Dataset_feature.csv")
DATA_OUT  = os.path.join(ROOT, "Data", "Dataset_clean.csv")
FEAT_OUT  = os.path.join(ROOT, "Code", "Dataset_feature_clean.csv")

for p in [DATA_CSV, FEAT_CSV]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}")
        sys.exit(1)

# ── Canonicalize ─────────────────────────────────────────────────────────────
def canonical(smi):
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol) if mol else None

print("=" * 60)
print("make_clean_data: removing train/test molecular duplicates")
print("=" * 60)

df = pd.read_csv(DATA_CSV)
assert {"SMILES", "Label", "ref"}.issubset(df.columns), \
    "Dataset.csv must have SMILES, Label, ref columns"

df["_canon"] = df["SMILES"].apply(canonical)

invalid = df["_canon"].isna().sum()
if invalid:
    print(f"[WARN] {invalid} SMILES could not be parsed by RDKit (will be kept as-is)")
    df.loc[df["_canon"].isna(), "_canon"] = df.loc[df["_canon"].isna(), "SMILES"]

train_mask = df["ref"] != "DILIrank"
test_mask  = df["ref"] == "DILIrank"

train_canon = set(df.loc[train_mask, "_canon"])
test_canon  = set(df.loc[test_mask,  "_canon"])
overlap     = train_canon & test_canon

print(f"\nTotal   : {len(df)}")
print(f"Train   : {train_mask.sum()}  (ref != DILIrank)")
print(f"Test    : {test_mask.sum()}   (ref == DILIrank)")
print(f"\nCanonical SMILES 중복 (train ∩ test): {len(overlap)}")

if overlap:
    print("\n중복 분자 목록 (canonical SMILES):")
    for s in sorted(overlap):
        print(f"  {s}")

# train에서만 제거 (test는 유지)
remove_mask = train_mask & df["_canon"].isin(overlap)
df_clean = df[~remove_mask].drop(columns=["_canon"]).reset_index(drop=True)

print(f"\n제거된 train 샘플: {remove_mask.sum()}")
print(f"Clean train : {(df_clean['ref'] != 'DILIrank').sum()}")
print(f"Clean test  : {(df_clean['ref'] == 'DILIrank').sum()}")

df_clean.to_csv(DATA_OUT, index=False)
print(f"\nSaved: {DATA_OUT}")

# ── Feature CSV에도 동일 필터 적용 ────────────────────────────────────────────
df_feat = pd.read_csv(FEAT_CSV)
assert list(df_feat["SMILES"]) == list(pd.read_csv(DATA_CSV)["SMILES"]), \
    "SMILES order mismatch between Dataset.csv and Dataset_feature.csv"

keep_idx = df.index[~remove_mask].tolist()
df_feat_clean = df_feat.iloc[keep_idx].reset_index(drop=True)
df_feat_clean.to_csv(FEAT_OUT, index=False)
print(f"Saved: {FEAT_OUT}")

print("\nmake_clean_data OK")
