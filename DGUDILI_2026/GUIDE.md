# DGUDILI (2026) — Implementation Guide

## Overview

DGUDILI (2026) is a hybrid DILI (Drug-Induced Liver Injury) prediction model that fuses molecular fingerprints (FP) and ChemBERTa language model embeddings via **Group-wise Cross-Attention**, producing a 16-dim feature space for binary classification.

---

## Model Lineup

| Model | Script | AUC | MCC | Specificity |
|---|---|---|---|---|
| StackDILI *(baseline)* | *(Origin_StackDILI)* | 0.9736 | 0.8304 | 0.8993 |
| DGUDILI_2026_prev | Step3 + Step4 | 0.7312 | 0.3045 | 0.4627 |
| GroupCrossAttn v1 | Step5 + Step6 | 0.8410 | 0.5078 | 0.7015 |
| Ensemble x5 | Step7 | 0.8729 | 0.5966 | 0.8134 |
| **ModeB** | **Step8 + Step9** | **0.9171** | **0.7067** | **0.8060** |

---

## Quick Start

```bash
cd C:\DGUDILI\DGUDILI_2026
conda activate DGUDILI
```

### 모델별 실행 (run.py)

```bash
python run.py prev       # DGUDILI_2026_prev  학습 + 평가  (Step2→3→4)
python run.py v1         # GroupCrossAttn v1  학습 + 평가
python run.py ensemble   # Seed Ensemble x5   학습 + 평가
python run.py modeB      # Mode B             학습 + 평가
python run.py compare    # 저장된 모든 모델 비교 (재학습 없음)
```

> `compare`는 체크포인트가 존재하는 모델만 자동 감지해 평가합니다.

---

## Prerequisites

### 필수 파일 (StackDILI에서 생성)
```
C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv
C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv  ← Feature.py 실행 후 생성
```

`Dataset_feature.csv`가 없으면 먼저 실행:
```bash
cd C:\DGUDILI\Origin_StackDILI\Code
python Feature.py
```

### ChemBERTa 임베딩 (최초 1회)
```bash
python src/Step1_chemberta_embed.py
```
- `seyonec/ChemBERTa-zinc-base-v1` 다운로드 (~500 MB, 최초 1회)
- 출력: `data/chemberta_embeddings.npy` (1850, 768)
- 이후 실행 시 자동 스킵

---

## Architecture

### DGUDILI_2026_prev (Steps 1–4)

```
FP 425-dim → SelectKBest(k=16) → fp_k16 (B, 16)
ChemBERTa 768-dim → SelectKBest(k=16) → cham_k16 (B, 16)

Cross-Attention (d_k=32, d_v=1):
  Q ← ChemBERTa, K/V ← FP
  attn → (B, 16) Feature Space
  → Logistic Regression → logit
```

### GroupCrossAttn v1 / Ensemble (Steps 5–7)

```
FP 425-dim을 화학적 의미 단위로 4 그룹 분리 (고정 인덱스):
  G0 Constitution : [  0: 29]  (29-dim)
  G1 CalcCATS     : [ 29:179] (150-dim)
  G2 MACCS        : [179:346] (167-dim)
  G3 E-state      : [346:425]  (79-dim)

각 그룹 → Linear(group_dim, 16) → token (B, 16)
FP_tokens K,V : (B, 4, 16)
ChemBERTa → Linear(768, 16) → LM_token Q : (B, 1, 16)

Cross-Attention:
  scores = Q @ K^T / √16 → (B, 1, 4)
  fused  = softmax @ V   → (B, 16)
  → LayerNorm + Dropout → Linear(16, 1) → logit

Ensemble: 동일 아키텍처를 seed=[42,0,7,21,99]로 5회 학습,
          test 확률 평균
```

### ModeB (Steps 8–9) ← 현재 최고 성능

```
FP 425-dim을 5 그룹으로 재분할:
  Const+CalcCATS : [0:23] + [29:179]  (173-dim)
  PC1-6          : [23:29]             (  6-dim)
  MACCS          : [179:346]           (167-dim)
  E-state        : [346:425]           ( 79-dim)

FP branch  : Linear(in, 64) + LayerNorm + ReLU + Dropout
             → FP_emb (B, 4, 64)  [K, V]
CB branch  : Linear(768, 64) + LayerNorm + ReLU + Dropout
             → CB_emb (B, 1, 64)  [Q]

Cross-Attention (d_model=64):
  scores = Q @ K^T / √64 → (B, 1, 4)
  output : (B, 1, 64) → Residual(+CB_emb) + LayerNorm
         → squeeze → Linear(64, 16) → (B, 16) → Linear(16, 1)
```

---

## File Structure

```
C:\DGUDILI\DGUDILI_2026\
├── run.py                          # 모델 선택 실행 진입점
├── GUIDE.md
│
├── src/
│   ├── model.py                    # CrossAttentionEncoder (prev 아키텍처)
│   ├── group_model.py              # GroupCrossAttentionDILI (v1)
│   ├── group_model_b.py            # GroupCrossAttentionModB / ModB_V2
│   │
│   ├── Step1_chemberta_embed.py    # SMILES → ChemBERTa CLS 768-dim
│   ├── Step2_feature_select.py     # SelectKBest k=16 (prev용)
│   ├── Step3_pretrain.py           # Cross-Attention 사전학습 (prev)
│   ├── Step4_extract_and_lr.py     # 16-dim 추출 + LR (prev)
│   │
│   ├── Step5_group_train.py        # GroupCrossAttn v1 학습
│   ├── Step6_group_eval.py         # v1 평가
│   ├── Step7_seed_ensemble.py      # Seed Ensemble x5 학습 + 평가
│   │
│   ├── Step8_train_modeB.py        # Mode B 학습
│   ├── Step9_eval_modeB.py         # Mode B 평가
│   │
│   └── compare_all.py              # 전체 모델 비교 (run.py compare)
│
├── data/
│   ├── chemberta_embeddings.npy    # (1850, 768) ChemBERTa CLS
│   ├── scalers_group.pkl           # v1 StandardScaler
│   ├── scalers_modeB.pkl           # ModeB StandardScaler
│   └── ...
│
└── outputs/
    ├── group_model_best.pt         # v1 체크포인트
    ├── ensemble_seed{N}.pt         # Ensemble 체크포인트 (seed별)
    ├── ensemble_scalers.pkl
    ├── modeB_best.pt               # ModeB 체크포인트
    ├── compare_all_results.csv     # 전체 비교 결과
    └── compare_all_roc.png         # ROC 곡선 비교 그래프
```

---

## Train / Test Split

StackDILI 동일 기준:
```python
train = data[data['ref'] != 'DILIrank']   # 1,398 samples (pos=768, neg=630)
test  = data[data['ref'] == 'DILIrank']   # 452 samples  (pos=184, neg=268)
```
모든 스케일러는 **train 기준으로만 fit**, test에는 transform만 적용.

---

## Key Hyperparameters

| | prev | v1 | ModeB |
|---|---|---|---|
| 내부 차원 | d_k=32 | d=16 | d_model=64 |
| 출력 차원 | 16 | 16 | 16 |
| Optimizer | Adam | AdamW | AdamW |
| LR | 1e-3 | 5e-4 | 1e-3 |
| Scheduler | — | ReduceLROnPlateau | CosineAnnealingLR |
| Dropout | — | 0.3 | 0.2 |
| Epochs (max) | 200 | 300 | 200 |
| Early stop patience | 20 | 40 | 30 |
| Batch size | 32 | 32 | 32 |

---

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` 모든 스크립트에 자동 설정 (Windows MKL + PyTorch 충돌 방지)
- `lm_head.*` 경고는 AutoModel 사용 시 정상 동작으로 무시해도 됨
- `python run.py compare` 실행 시 체크포인트가 없는 모델은 자동 스킵
