# DGUDILI (2026) — Implementation Guide

## Overview

DGUDILI (2026) is a hybrid DILI (Drug-Induced Liver Injury) prediction model that replaces the genetic algorithm feature selection used in StackDILI with a **Cross-Attention fusion** of two complementary molecular representations.

```
FP 425-dim     → SelectKBest(k=16) → fp_k16   (B, 16)
ChemBERTa 768-dim → SelectKBest(k=16) → cham_k16 (B, 16)

Cross-Attention (Q ← ChemBERTa, K/V ← FP, d_k=32, d_v=1):
  scores = Q @ K^T / √d_k → weights = softmax(scores)
  attn = weights @ V → (B, 16) Feature Space
  → Logistic Regression → logit
```

**Key difference from StackDILI:** StackDILI uses a genetic algorithm to select ~209 features from FP only. DGUDILI fuses FP and ChemBERTa embeddings via cross-attention, reducing to **16 features** while incorporating chemical semantics from a pre-trained molecular language model.

### Performance (DILIrank test set, N=452)

| Model | AUC | MCC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| StackDILI *(baseline)* | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| DGUDILI_2026 | 0.7313 | 0.3045 | 0.6362 | 0.8315 | 0.4627 |

---

## Prerequisites

### 1. 필수 파일 (StackDILI에서 생성)
```
C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv
C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv  ← Feature.py 실행 후 생성
```

`Dataset_feature.csv`가 없으면 먼저 실행:
```bash
cd C:\DGUDILI\Origin_StackDILI\Code
conda activate DGUDILI
python Feature.py
```

### 2. 패키지 설치
```bash
conda activate DGUDILI
pip install torch transformers scikit-learn
```

---

## Pipeline 실행

```bash
cd C:\DGUDILI\DGUDILI_2026
conda run DGUDILI
```

### Step 1 — ChemBERTa Embedding 추출
```bash
python src/Step1_chemberta_embed.py
```
- 최초 실행 시 `seyonec/ChemBERTa-zinc-base-v1` 다운로드 (~500 MB)
- 전체 1,850개 SMILES → CLS 토큰 768-dim 인코딩
- 출력: `data/chemberta_embeddings.npy` (1850, 768)
- 소요 시간: 약 3~5분 (CPU)

### Step 2 — Feature Selection
```bash
python src/Step2_feature_select.py
```
- `SelectKBest(f_classif, k=16)` 을 FP (425→16), ChemBERTa (768→16)에 각각 적용
- train split 기준으로만 fit, test split에는 transform만 적용
- 출력: `fp_k16_{train,test}.npy`, `cham_k16_{train,test}.npy`, `y_{train,test}.npy`

### Step 3 — Cross-Attention 사전학습 (Stage 1)
```bash
python src/Step3_pretrain.py
```
- `CrossAttentionEncoder` (W_Q, W_K, W_V + 분류 헤드) 학습
- Loss: BCEWithLogitsLoss, Optimizer: Adam (lr=1e-3)
- Early stopping: test AUC 기준, patience=20, max 200 epochs
- 출력: `outputs/pretrained_encoder.pt`

### Step 4 — Feature 추출 + Logistic Regression (Stage 2)
```bash
python src/Step4_extract_and_lr.py
```
- 저장된 인코더로 16-dim Feature Space 추출 (frozen)
- `sklearn.LogisticRegression` 학습 후 DILIrank test set 평가
- 출력: `outputs/results_comparison.csv` + 비교 테이블 출력

---

## File Structure

```
C:\DGUDILI\DGUDILI_2026\
├── GUIDE.md
│
├── src/
│   ├── model.py                 # CrossAttentionEncoder (nn.Module)
│   ├── Step1_chemberta_embed.py # SMILES → ChemBERTa CLS 768-dim
│   ├── Step2_feature_select.py  # SelectKBest k=16 (FP & ChemBERTa)
│   ├── Step3_pretrain.py        # Stage 1: Cross-Attention 사전학습
│   └── Step4_extract_and_lr.py  # Stage 2: 16-dim 추출 + LR + 평가
│
├── data/
│   ├── chemberta_embeddings.npy   # (1850, 768) ChemBERTa CLS
│   ├── fp_k16_{train,test}.npy    # SelectKBest 선택된 FP (k=16)
│   ├── cham_k16_{train,test}.npy  # SelectKBest 선택된 ChemBERTa (k=16)
│   ├── y_{train,test}.npy         # 레이블
│   ├── scalers.pkl                # StandardScaler (FP, ChemBERTa)
│   ├── selected_fp_features.json  # 선택된 FP feature 이름
│   └── selected_cham_dims.json    # 선택된 ChemBERTa 차원 인덱스
│
└── outputs/
    ├── pretrained_encoder.pt      # 학습된 CrossAttentionEncoder 가중치
    ├── results_comparison.csv     # 평가 결과 (StackDILI vs DGUDILI_2026)
    ├── mcc_analysis.png           # MCC 진단 분석 그래프
    └── DGUDILI_2026_flowchart.png # 아키텍처 다이어그램
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| k | 16 | 모달리티별 선택 피처 수 / Attention 시퀀스 길이 |
| d_k | 32 | Attention Q/K 내부 투영 차원 |
| d_v | 1 | Attention V 투영 차원 (출력이 k=16 되도록 고정) |
| lr | 1e-3 | Adam learning rate |
| epochs | 200 | 최대 학습 epoch |
| patience | 20 | Early stopping patience (test AUC 기준) |
| batch_size | 32 | 학습 배치 크기 |

---

## Train / Test Split

StackDILI와 동일한 기준 사용:
```python
train = data[data['ref'] != 'DILIrank']   # 1,398 samples (pos=768, neg=630)
test  = data[data['ref'] == 'DILIrank']   # 452 samples  (pos=184, neg=268)
```
모든 스케일러 및 SelectKBest는 **train 기준으로만 fit**, test에는 transform만 적용.

---

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` 모든 스크립트에 자동 설정 (Windows MKL + PyTorch 충돌 방지)
- `lm_head.*` 경고는 AutoModel 사용 시 정상 동작으로 무시 가능
- Step 1은 `chemberta_embeddings.npy`가 이미 존재해도 항상 재실행됨 (재사용하려면 Step 2부터 실행)
- Step 2에서 일부 FP 피처가 constant feature 경고를 출력하는 것은 정상 (해당 피처는 선택에서 자동 제외)
