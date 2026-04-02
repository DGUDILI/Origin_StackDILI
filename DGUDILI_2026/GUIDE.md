# DGUDILI (2026) — Implementation Guide

## Overview

DGUDILI (2026) is a hybrid DILI (Drug-Induced Liver Injury) prediction model combining a pre-trained molecular language model (ChemBERTa) with hand-crafted molecular fingerprints (FP) via Cross-Attention fusion.

### Architecture Evolution

| 버전 | 모델 | FP 처리 | 분류기 | AUC | MCC |
|------|------|---------|--------|-----|-----|
| StackDILI *(baseline)* | GA + Stacking | GA 선택 ~209 피처 | RF/ET/HistGB/XGB | 0.9736 | 0.8304 |
| CrossAttentionEncoder | SelectKBest k=16 | d_v=1 | LR | 0.7313 | 0.3045 |
| **FTV6StyleEncoder** | **Linear Projection (425→k)** | **d_v=1** | **Stacking OOF** | **0.9196** | **0.6682** |
| **E2E_FTV6** | **Linear Projection** | **d_v=1, ChemBERTa fine-tune** | **Stacking OOF** | **0.9219** | **0.7270** |

---

## ChemBERTa 모델 스펙

```
모델명: DeepChem/ChemBERTa-77M-MLM
hidden_size: 384 (※ "77M"은 파라미터 수가 아닌 MLM 학습 토큰 수)
num_hidden_layers: 3
total_params: ~3.5M
```

> **주의:** GUIDE 이전 버전에 기재된 "768-dim", "seyonec/ChemBERTa-zinc-base-v1"은 오류.
> 현재 코드는 DeepChem/ChemBERTa-77M-MLM (384-dim) 사용.

---

## Prerequisites

### 필수 파일 (StackDILI에서 생성)
```
C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv
C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv
```

`Dataset_feature.csv`가 없으면:
```bash
cd C:\DGUDILI\Origin_StackDILI\Code
conda activate DGUDILI
python Feature.py
```

### 패키지 설치
```bash
conda activate DGUDILI
pip install torch transformers scikit-learn xgboost
```

---

## Pipeline A — FTV6StyleEncoder (현재 권장)

```
SMILES → ChemBERTa(384-dim, frozen) → StandardScaler
FP 425-dim                           → StandardScaler
                                     ↓
            FTV6StyleEncoder
              FP:   Linear(425, k) + LayerNorm  → (B, k)  [soft projection]
              Chem: LN → Linear(384,128) → GELU → Drop → Linear(128,k) → (B, k)
              CrossAttn: Q=Chem, K/V=FP, d_k=32, d_v=4
              attn: (B, k, d_v) → reshape → (B, k*d_v)
                                     ↓
            Stacking OOF (RF / ET / HistGB / XGB, 5-Fold)
                                     ↓
            LR Meta-model → 최종 예측
```

### 실행 순서

```bash
conda activate DGUDILI
cd C:\DGUDILI

# Step 1: ChemBERTa 임베딩 추출 (CLS pooling)
python DGUDILI_2026/src/Step1_chemberta_embed.py --pooling cls

# Step 2: FP 전처리 (SelectKBest 없음, 전체 425-dim StandardScaler)
python DGUDILI_2026/src/Step2_feature_select.py --pooling cls

# Step 3: FTV6StyleEncoder Stage 1 학습 (Early stop: val_AUC)
python DGUDILI_2026/src/Step3_pretrain.py --pooling cls

# Step 4: 피처 추출 + Stacking 앙상블 평가
python DGUDILI_2026/src/Step4_extract_and_lr.py --pooling cls
```

---

## Pipeline B — E2E_FTV6StyleEncoder (ChemBERTa 파인튜닝 통합)

```
SMILES (원본 문자열)
  → Tokenizer → input_ids, attention_mask
  → ChemBERTa (layer[0,1] frozen, layer[2] unfrozen)
  → CLS token (B, 384)
                                     ↓
            E2E_FTV6StyleEncoder (Pipeline A와 동일 CrossAttn 구조)
            차등 LR: ChemBERTa layer[2] = 1e-4 / CrossAttn+Proj = 3e-4
                                     ↓
            Stacking OOF → LR Meta-model → 최종 예측
```

### 실행 순서

```bash
# Step 1, 2는 Pipeline A와 동일하게 먼저 실행 (FP scaler 생성 필요)

# Step 3 E2E: 배치마다 SMILES 토크나이징, ChemBERTa 파인튜닝
python DGUDILI_2026/src/Step3_e2e_pretrain.py

# Step 4 E2E: E2E 인코더로 피처 추출 + Stacking 평가
python DGUDILI_2026/src/Step4_e2e_extract_and_lr.py
```

---

## File Structure

```
C:\DGUDILI\DGUDILI_2026\
├── GUIDE.md
│
├── src/
│   ├── model.py                    # CrossAttentionEncoder, FTV6StyleEncoder,
│   │                               # GroupedCrossAttentionEncoder, E2E_FTV6StyleEncoder
│   ├── Step1_chemberta_embed.py    # SMILES → ChemBERTa CLS/Mean 384-dim
│   ├── Step2_feature_select.py     # FP 425-dim StandardScaler (SelectKBest 제거)
│   ├── Step3_pretrain.py           # FTV6StyleEncoder Stage 1 (val_AUC early stop)
│   ├── Step4_extract_and_lr.py     # 피처 추출 + Stacking OOF + 평가
│   ├── Step3_e2e_pretrain.py       # E2E ChemBERTa 파인튜닝 (val_AUC early stop)
│   ├── Step4_e2e_extract_and_lr.py # E2E 피처 추출 + Stacking OOF + 평가
│   └── inspect_chemberta.py        # ChemBERTa 모델 구조 확인 유틸리티
│
├── data/
│   ├── chemberta_embeddings_cls.npy  # (1850, 384) ChemBERTa CLS
│   ├── fp_full_train.npy             # (1398, 425) 전체 FP, StandardScaler 적용
│   ├── fp_full_test.npy              # (452,  425)
│   ├── cham_full_train_cls.npy       # (1398, 384) ChemBERTa, StandardScaler 적용
│   ├── cham_full_test_cls.npy        # (452,  384)
│   ├── y_train.npy / y_test.npy      # 레이블
│   ├── scalers_ftv6_cls.pkl          # FP + ChemBERTa StandardScaler
│   └── fp_feature_names.json         # 425개 FP 피처 이름
│
└── outputs/
    ├── pretrained_encoder_ftv6_cls.pt   # FTV6StyleEncoder 가중치
    ├── pretrained_encoder_e2e_cls.pt    # E2E_FTV6StyleEncoder 가중치
    ├── results_ftv6_cls.csv             # FTV6 실험 결과
    └── results_e2e_cls.csv              # E2E 실험 결과
```

---

## Key Hyperparameters

### FTV6StyleEncoder (Step3_pretrain.py)

| Parameter | Value | Description |
|---|---|---|
| k | 16 | Attention 시퀀스 길이 (Query/Key/Value 토큰 수) |
| d_k | 32 | Q/K 투영 차원 |
| d_v | 4 | V 투영 차원 (encode 출력: B × k×d_v = B × 64) |
| lr | 3e-4 | AdamW learning rate |
| weight_decay | 1e-4 | |
| epochs | 300 | 최대 epoch |
| patience | 40 | Early stopping patience |
| sched_patience | 10 | ReduceLROnPlateau patience |
| batch_size | 32 | |
| dropout | 0.3 | chem_proj 내 dropout |
| **early_stop** | **val_AUC (mode=max)** | **val_loss는 분포 이동으로 부적합** |

### E2E_FTV6StyleEncoder (Step3_e2e_pretrain.py)

| Parameter | Value | Description |
|---|---|---|
| k, d_k, d_v | 16, 32, 4 | FTV6StyleEncoder와 동일 |
| lr_chem | 1e-4 | ChemBERTa last layer (layer[2]) |
| lr_other | 3e-4 | CrossAttn + Projection |
| batch_size | 16 | ChemBERTa forward 메모리 고려 |
| max_length | 256 | SMILES tokenizer max length |
| **early_stop** | **val_AUC (mode=max)** | |

---

## Train / Test Split

```python
train = data[data['ref'] != 'DILIrank']   # 1,398 samples (pos=768, neg=630)
test  = data[data['ref'] == 'DILIrank']   # 452 samples  (pos=184, neg=268)

# Stage 1 내부 Val 분리 (조기종료용)
val = train_stratified_split(train, test_size=0.15)  # 210 samples
```

---

## 실험 결과 (DILIrank test, N=452)

| 모델 | d_v | 조기종료 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|---------|-----|-----|----|-------------|-------------|
| StackDILI | — | — | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| CrossAttentionEncoder | 1 | test_AUC (누수) | 0.7313 | 0.3045 | 0.6362 | 0.8315 | 0.4627 |
| GroupedCrossAttention | 1 | val_AUC | 0.8013 | 0.4484 | 0.6950 | 0.7989 | 0.6567 |
| FTV6StyleEncoder | 1 | val_AUC | 0.9196 | 0.6682 | 0.8106 | 0.9185 | 0.7612 |
| **E2E_FTV6StyleEncoder** | **1** | **val_AUC** | **0.9219** | **0.7270** | **0.8426** | **0.9022** | **0.8358** |
| FTV6StyleEncoder | 4 | val_loss ⚠️ | 0.8252 | 0.5316 | 0.7377 | 0.9402 | 0.5821 |
| E2E_FTV6StyleEncoder | 4 | val_loss ⚠️ | 0.9006 | 0.6522 | 0.8010 | 0.8641 | 0.7985 |

> ⚠️ val_loss 기준 실험은 조기종료가 너무 이르게 발생하여 d_v=4의 실제 효과를 반영하지 못함.
> **다음 실험: d_v=4 + val_AUC 재실행 필요.**

---

## 알려진 문제점 및 고려사항

### 1. Train/Test 분포 이동

- Train: 다중 데이터베이스 혼합 (ChEMBL, SIDER 등) — 노이즈 多
- Test: DILIrank 단일 큐레이션 — 더 깔끔하게 분리됨
- 결과: val_AUC(~0.68) << test_AUC(~0.92) 역전 현상
- 대응: Domain Adversarial Training (P3, 미실행)

### 2. val_loss 조기종료 부적합

val_loss는 epoch 2~5에 최솟값 도달 후 즉시 증가(분포 차이 과적합 신호).
이 시점 저장 모델은 사실상 미학습 상태 → test AUC 급락.
**반드시 val_AUC (mode=max) 사용.**

### 3. E2E 파라미터 과다

trainable 1M params / 1,188 samples ≈ 840 params/sample.
과적합 위험. LoRA(rank=4) 적용 시 ~50K params로 축소 가능 (미실현).

### 4. ChemBERTa LOAD REPORT 경고 무시

```
lm_head.* | UNEXPECTED  ← AutoModel이 MLM head 제외하여 정상
pooler.*   | MISSING     ← 분류 목적으로 재초기화되어 정상
```

---

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` 모든 스크립트에 자동 설정 (Windows MKL 충돌 방지)
- `--pooling cls` (기본값): CLS 토큰. `--pooling mean` 옵션도 지원
- Step1은 `chemberta_embeddings_cls.npy` 존재해도 항상 재실행됨
- .npy / .pkl / .pt 파일은 gitignore 권장 (용량)
