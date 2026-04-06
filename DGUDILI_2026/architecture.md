# DGUDILI_2026 아키텍처 문서

## 목표

SMILES 문자열과 분자 지문(Fingerprint)을 융합하여 DILI(Drug-Induced Liver Injury) 발생 여부를 이진 분류.

---

## 파이프라인 개요

```
Step 1: FP 전처리 (StandardScaler)
Step 2: E2E_FTV6StyleEncoder 학습 (Stage 1)
Step 3: Feature 추출 + Stacking OOF (Stage 2)
```

```
[SMILES]  [Fingerprint 425-dim]
   │              │
   ▼              ▼
[Stage 1: E2E_FTV6StyleEncoder 학습]
   │
   ▼
[16-dim Fused Feature 추출]
   │
   ▼
[Stage 2: Stacking OOF → LR Meta → 최종 예측]
```

---

## Stage 1: E2E_FTV6StyleEncoder

### 전체 구조도

```
SMILES 문자열
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  ChemBERTa-77M-MLM  (DeepChem/ChemBERTa-77M-MLM)     │
│                                                      │
│  Tokenizer → [CLS, tok1, tok2, ..., SEP] (max 256)   │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │ Embedding Layer            [Frozen]         │     │
│  └──────────────────────┬──────────────────────┘     │
│                         ▼                            │
│  ┌──────────────────────────────────────────────┐    │
│  │ Encoder Layer[0]           [Frozen]          │    │
│  └──────────────────────┬───────────────────────┘    │
│                         ▼                            │
│  ┌──────────────────────────────────────────────┐    │
│  │ Encoder Layer[1]           [Frozen]          │    │
│  └──────────────────────┬───────────────────────┘    │
│                         ▼                            │
│  ┌──────────────────────────────────────────────┐    │
│  │ Encoder Layer[2]  ← [Unfrozen, lr=1e-4]      │    │
│  └──────────────────────┬───────────────────────┘    │
│                         ▼                            │
│            CLS token hidden state                    │
│                   (B, 384)                           │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  chem_proj            │
              │  LayerNorm(384)       │
              │  Linear(384 → 128)    │
              │  GELU                 │
              │  Dropout(0.3)         │
              │  Linear(128 → 16)     │
              └───────────┬───────────┘
                          │
                          ▼
                     q : (B, 16)          ← Query 토큰
                          │
                          ▼
              W_Q: Linear(1 → 32)
                          │
                          ▼
                    Q : (B, 16, 32)
```

```
분자 지문 FP (425-dim)
    │
    ▼
┌──────────────────────┐
│  fp_proj             │
│  Linear(425 → 16)    │
│  LayerNorm(16)       │
└──────────┬───────────┘
           │
           ▼
      kv : (B, 16)       ← Key/Value 토큰
           │
      ┌────┴────┐
      ▼         ▼
W_K: Linear   W_V: Linear
  (1 → 32)     (1 → 1)
      │              │
      ▼              ▼
K : (B, 16, 32)   V : (B, 16, 1)
```

```
Cross-Attention Fusion
─────────────────────────────────────────────────
Q : (B, 16, 32)   K : (B, 16, 32)   V : (B, 16, 1)

scores  = Q @ K^T / √32         → (B, 16, 16)
weights = softmax(scores, dim=-1) → (B, 16, 16)
attn    = weights @ V            → (B, 16, 1)

flatten → (B, 16)    [k=16 × d_v=1]
─────────────────────────────────────────────────

Stage 1 학습:
  head: Linear(16 → 1) → BCEWithLogitsLoss

Stage 2 추출:
  encode() → (B, 16) fused feature vector
```

### 텐서 플로우 요약

| 단계 | 입력 shape | 출력 shape | 설명 |
|------|-----------|-----------|------|
| Tokenize | SMILES str | (B, 256) | max_length=256, padding |
| ChemBERTa | (B, 256) | (B, 384) | CLS token 추출 |
| chem_proj | (B, 384) | (B, 16) | Query 토큰 생성 |
| fp_proj | (B, 425) | (B, 16) | Key/Value 토큰 생성 |
| W_Q | (B, 16, 1) | (B, 16, 32) | Query 행렬 |
| W_K | (B, 16, 1) | (B, 16, 32) | Key 행렬 |
| W_V | (B, 16, 1) | (B, 16, 1) | Value 행렬 (d_v=1) |
| Cross-Attn | Q,K,V | (B, 16, 1) | Attention 가중합 |
| Flatten | (B, 16, 1) | (B, 16) | 융합 피처 벡터 |
| head | (B, 16) | (B, 1) | Stage 1 예측값 |

### 학습 설정 (Stage 1)

| 항목 | 값 |
|------|---|
| ChemBERTa | DeepChem/ChemBERTa-77M-MLM (hidden=384, layers=3) |
| Frozen | embeddings + encoder.layer[0], [1] |
| Unfrozen | encoder.layer[2] (마지막), lr=1e-4 |
| CrossAttn/Proj lr | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping | val_AUC (mode=max, patience=30) |
| Scheduler | ReduceLROnPlateau(val_AUC, mode=max, factor=0.5, patience=8) |
| Loss | BCEWithLogitsLoss |
| Train/Val split | 85% / 15% stratified |

---

## Stage 2: Stacking OOF → Logistic Regression

```
Train 전체 데이터 (1398 샘플)
        │
        ▼
[E2E_FTV6StyleEncoder.encode()]
        │
        ▼
 16-dim Fused Features
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  5-Fold Stratified CV (OOF Stacking)                       │
│                                                            │
│  Base Models:                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ RF     : RandomForest(n_estimators=300)              │  │
│  │ ET     : ExtraTrees(n_estimators=300)                │  │
│  │ HistGB : HistGradientBoosting(max_iter=300)          │  │
│  │ XGB    : XGBoost(n_est=300, lr=0.05, max_depth=4)    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  OOF 결과: oof_probs  shape = (1398, 4)                    │
│  Test 결과: test_probs shape = (452, 4)                    │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Meta Model                   │
              │  LogisticRegression(C=1.0)    │
              │  fit(oof_probs, y_train)      │
              │  predict_proba(test_probs)    │
              └───────────────┬───────────────┘
                              │
                              ▼
                  최종 예측 확률 (452,)
                              │
                              ▼
                    threshold = 0.5 → 이진 분류
```

---

## FP Soft Projection의 의미

기존 접근(SelectKBest k=16)은 통계적 기준으로 16개 피처를 하드 선택하여 나머지 정보를 폐기.

FTV6 방식은 `Linear(425, 16)` 가중치 행렬이 학습을 통해 중요도를 배우는 **Soft Projection**:

```
FP 425개 → W ∈ ℝ^{425×16} → 16개 토큰
             ↑
     각 토큰 = FP 전체의 가중합
     (중요한 FP에 큰 가중치, 덜 중요한 FP에 작은 가중치)
```

---

## Cross-Attention의 역할

ChemBERTa(언어적 구조 정보)를 **Query**, 분자 지문(물리화학적 특성)을 **Key/Value**로 사용:

```
Query(ChemBERTa) → "이 분자의 구조적 특성에서 어떤 물리화학 피처가 중요한가?"
Key/Value(FP)    → "DILI 예측에 관련된 물리화학적 신호를 제공"

Attention weight (B, 16, 16):
  rows = 16개의 ChemBERTa query 토큰
  cols = 16개의 FP key 토큰
  → 각 구조적 토큰이 어느 FP 정보에 주목할지 학습
```

---

## 실험 결과 (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | **0.9736** | **0.8304** | **0.9010** | **0.9402** | **0.8993** |
| **DGUDILI_2026** | **0.9224** | **0.7270** | **0.8426** | 0.9022 | 0.8358 |
