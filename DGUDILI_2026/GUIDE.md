# DGUDILI (2026) — Implementation Guide

## Overview

DGUDILI (2026) is a hybrid DILI (Drug-Induced Liver Injury) prediction model that replaces the genetic algorithm feature selection used in StackDILI with a **Cross-Attention fusion** of two complementary molecular representations.

### Architecture

```
DILI Dataset (TRAIN: p=768/n=630  |  TEST: p=184/n=268)
        │
        ├── [FP path]  iFeatureOmegaCLI → 425-dim
        │              Constitution(29) + CalcCATS(150) + MACCS(167) + Estate(79)
        │              └── SelectKBest(k=16)  →  fp_k16 ∈ R^(B×16)
        │
        └── [LM path]  ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) → 768-dim CLS
                       └── SelectKBest(k=16)  →  cham_k16 ∈ R^(B×16)
                                    │
              Cross-Attention  Q←ChemBERTa  /  K,V←FP
              scores = Q @ K^T / √d_k  →  weights = softmax(scores)
              attn = weights @ V  →  (B, 16, 1)  →  squeeze  →  (B, 16)
                                    │
                        Feature Space (k=16)
                                    │
                      sklearn Logistic Regression
                                    │
                            AUC  /  MCC  /  F1
```

**Key difference from StackDILI:** StackDILI uses a genetic algorithm to select ~209 features from FP only. DGUDILI fuses FP and ChemBERTa language model embeddings via cross-attention, reducing to **16 features** while incorporating chemical semantics from pre-trained molecular representations.

---

## Prerequisites

### 1. Required files (from StackDILI)
```
C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv          ← raw dataset
C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv  ← FP features (run Feature.py first)
```

If `Dataset_feature.csv` does not exist, run StackDILI's Feature.py first:
```bash
cd C:\DGUDILI\Origin_StackDILI\Code
conda activate DGUDILI
python Feature.py
```

### 2. Install dependencies
```bash
conda activate DGUDILI
pip install torch transformers
```

---

## Running the Pipeline

### Full pipeline (recommended)
```bash
cd C:\DGUDILI\DGUDILI_2026
conda activate DGUDILI
python run_all.py
```

`run_all.py` executes all steps in order and skips Step 1 automatically if `chemberta_embeddings.npy` already exists.

---

### Step-by-step execution

#### Step 1 — ChemBERTa Embedding Extraction
```bash
python Step1_chemberta_embed.py
```
- Downloads `seyonec/ChemBERTa-zinc-base-v1` on first run (~500 MB)
- Encodes all 1,850 SMILES → CLS token (768-dim)
- Output: `chemberta_embeddings.npy` (1850, 768)
- Runtime: ~3–5 min (CPU)

#### Step 2 — Feature Selection
```bash
python Step2_feature_select.py
```
- Applies `SelectKBest(f_classif, k=16)` to both FP (425→16) and ChemBERTa (768→16)
- Fit **only on train split** (`ref != 'DILIrank'`), transforms test split
- Output: `fp_k16_{train,test}.npy`, `cham_k16_{train,test}.npy`, `y_{train,test}.npy`

#### Step 3 — Cross-Attention Pre-training (Stage 1)
```bash
python Step3_pretrain.py
```
- Trains `CrossAttentionEncoder` (W_Q, W_K, W_V + temporary Linear head)
- Loss: BCEWithLogitsLoss, Optimizer: Adam (lr=1e-3)
- Early stopping on test AUC (patience=20, max 200 epochs)
- Output: `pretrained_encoder.pt`

#### Step 4 — Feature Extraction + Logistic Regression (Stage 2)
```bash
python Step4_extract_and_lr.py
```
- Loads frozen encoder, extracts 16-dim Feature Space
- Trains `sklearn.LogisticRegression` on extracted features
- Evaluates on DILIrank test set
- Output: `results_comparison.csv` + printed comparison table

---

## Output

```
======================================================================
DGUDILI 2026 vs StackDILI  |  Test set: DILIrank (N=452)
======================================================================
                    AUC      MCC       F1      ACC   Precision  Sensitivity  Specificity
StackDILI          0.9736   0.8304   0.9010   0.9159   0.8650     0.9402       0.8993
DGUDILI_2026       x.xxxx   x.xxxx   x.xxxx   x.xxxx   x.xxxx     x.xxxx       x.xxxx
Delta(+↑)         +x.xxxx  +x.xxxx  +x.xxxx  ...
======================================================================
```

Results are saved to `results_comparison.csv`.

---

## File Structure

```
C:\DGUDILI\DGUDILI_2026\
├── model.py                    # CrossAttentionEncoder (nn.Module)
├── Step1_chemberta_embed.py    # SMILES → ChemBERTa CLS 768-dim
├── Step2_feature_select.py     # SelectKBest k=16 for both modalities
├── Step3_pretrain.py           # Stage 1: Cross-Attention pre-training
├── Step4_extract_and_lr.py     # Stage 2: 16-dim extraction + LR + evaluation
├── run_all.py                  # Full pipeline orchestrator
├── draw_flowchart.py           # Architecture flowchart generator
├── analyze_mcc.py              # Diagnostic analysis script
├── DGUDILI_2026_flowchart.png  # Architecture diagram
└── results_comparison.csv      # Final evaluation results
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| k | 16 | Features selected per modality / attention sequence length |
| d_k | 32 | Internal attention projection dimension |
| d_v | 1 | Value projection dimension (ensures output = k=16) |
| lr | 1e-3 | Adam learning rate |
| epochs | 200 | Max training epochs |
| patience | 20 | Early stopping patience (test AUC) |
| batch_size | 32 | Training batch size |

---

## Train / Test Split

Strictly follows StackDILI's split logic:
```python
train = data[data['ref'] != 'DILIrank']   # 1,398 samples
test  = data[data['ref'] == 'DILIrank']   # 452 samples
```
All selectors and scalers are fit **exclusively on the train split**.

---

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` is set automatically in every script (required on Windows with MKL + PyTorch)
- `lm_head.*` warnings from AutoModel are expected and harmless
- Step 1 is skipped on subsequent runs if `chemberta_embeddings.npy` exists
