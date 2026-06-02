# DGUDILI_2026 — Code Analysis Report

**Date:** 2026-04-10  
**Model:** GraphMACCSEncoder (GINEConv + DifferentialCrossAttention + ChemBERTa)  
**Task:** DILI (Drug-Induced Liver Injury) binary classification  
**Reference baseline:** StackDILI (AUC=0.9736, MCC=0.8304)

---

## 1. Codebase Structure

```
src/
  config.py                  — 하이퍼파라미터 및 경로 중앙 관리
  model.py                   — GraphMACCSEncoder (핵심 모델)
  differential_attention.py  — DifferentialCrossAttention (Differential Transformer 변형)
  graph_utils.py             — smiles_to_pyg, get_maccs, MACCS_NAMES, 런타임 검증
  features.py                — extract_features, collate_fn (Step3/CV 공유)
  utils.py                   — set_seed, load_dataset
  Step1_preprocess.py        — FP StandardScaler + Graph/MACCS .pt 캐시 저장
  Step2_pretrain.py          — GraphMACCSEncoder Stage 1 학습 (val_AUC early stop)
  Step3_stacking.py          — 32-dim 피처 추출 + 5-Fold OOF Stacking → LR meta
  Step4_xai.py               — DiffAttn XAI 히트맵 (원자-MACCS bit 상관관계)
  Step5_mHeadHeatmap.py      — 멀티헤드별 독립 어텐션 히트맵
  Step_CV.py                 — 10-Fold CV (env2, 전체 N=1,850)
```

---

## 2. Cleaned Issues (이번 분석에서 수정된 항목)

### 2.1 크리티컬 버그 (ImportError — 실행 즉시 크래시)

| 파일 | 증상 | 원인 | 수정 |
|------|------|------|------|
| `Step4_xai.py` | `ImportError: cannot import name 'SAGE_LAYERS'` | SAGEConv → GINEConv 전환 시 config import 미반영 | `SAGE_LAYERS/SAGE_HIDDEN` → `GINE_LAYERS/GINE_HIDDEN/BOND_FEAT_DIM` |
| `Step4_xai.py` | `TypeError: __init__() got unexpected keyword argument 'sage_hidden'` | `load_model()` 내 모델 인스턴스화 파라미터 구 인터페이스 사용 | `sage_hidden/sage_layers` → `gine_hidden/gine_layers/bond_feat_dim` |
| `Step5_mHeadHeatmap.py` | 동일 | 동일 | 동일 |
| `Step_CV.py` | 동일 | 동일 | 동일 |

### 2.2 데드 코드 (Dead Code)

| 파일 | 항목 | 이유 |
|------|------|------|
| `utils.py` | `SMILESDataset` 클래스 (44줄) | FP 기반 구 파이프라인 잔재. 현재 Step2는 `MolGraphDataset`(Step2 내부 정의), Step3/CV는 `features.py::extract_features` 사용. `SMILESDataset`을 import하는 파일 없음. |
| `utils.py` | `import os`, `from torch.utils.data import Dataset` | `SMILESDataset` 삭제 후 불필요 |
| `Step_CV.py` | `DATA_PATH, FEAT_PATH` import | `__main__` 블록에서 한 번도 사용 안 됨. CV는 graph 캐시(`.pt`) 직접 로드. |
| `Step5_mHeadHeatmap.py` | `from graph_utils2 import get_correct_maccs_name` | `graph_utils.py`의 `MACCS_NAMES`로 대체 가능한 중복 매핑. |

### 2.3 데드 파일

| 파일 | 상태 | 설명 |
|------|------|------|
| `graph_utils2.py` | **미사용** (삭제 권장) | `Step5_mHeadHeatmap.py`에서만 import됐으나, Step5 수정 후 완전 고아. `HUMAN_READABLE_MACCS` 딕셔너리는 `graph_utils.py::MACCS_NAMES`와 동일 정보(다른 문자열 표현)의 중복. |

### 2.4 주석 정리

| 파일 | 항목 |
|------|------|
| `Step1_preprocess.py:24` | "E2E_FTV6StyleEncoder / E2E_MHAResidualEncoder 용" → 구 모델명 삭제 |
| `Step5_mHeadHeatmap.py` | `# ✅ 우리가 방금 새로 만든...` 등 디버깅용 이모지 주석 4개 제거 |
| `architecture.md` | MACCS inactive bit 처리 방식 오기 (`kv_padding_mask -1e9` → binary gate) 수정, lambda clamp `max=2.0` 누락 수정 |

---

## 3. Architecture Analysis

### 3.1 전체 파이프라인 Flow

```
SMILES
  │
  ├─[ChemBERTa-77M-MLM]─────────────────────────────────────────────────┐
  │   embeddings + layer[0,1]: frozen                                    │
  │   layer[2]: unfrozen (lr=1e-4)                                       │
  │   CLS token (B,384) → LayerNorm → Linear(384,64) → chem_feat (B,64) │
  │                                                                      │
  ├─[GINEConv Graph Encoder]────────────────────────────────────────────┤
  │   atom feat 43-dim → Linear(43,64)                                   │
  │   bond feat  9-dim → Linear(9,64)  [edge_proj]                       │
  │   GINEConv(64,64,edge_dim=64) × 2 + BatchNorm + ReLU                │
  │   to_dense_batch → (B,100,64) + pad_mask                             │
  │   Linear(64,64) → node_q (B,100,64)               [Query]           │
  │                                                                      │
  ├─[MACCS Embedding]───────────────────────────────────────────────────┤
  │   MACCSkeys (B,167) binary                                           │
  │   Embedding(167,64)[arange] × maccs → maccs_kv (B,167,64)           │
  │   inactive bits: binary gate → 0-vector            [Key/Value]       │
  └──────────────────────────┬───────────────────────────────────────────┘
                             ▼
           DifferentialCrossAttention
           ────────────────────────────────────────
           W_q: (B,100,128) → Q1,Q2   [h=4, d_head=16]
           W_k: (B,167,128) → K1,K2
           W_v: (B,167, 64) → V
           scores1 = softmax(Q1@K1ᵀ/√16)
           scores2 = softmax(Q2@K2ᵀ/√16)
           λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + 0.8
           λ = clamp(λ, min=1e-4, max=2.0)
           attn = scores1 - λ·scores2    ← Differential attention
           out  = attn @ V → GroupNorm → ×(1-0.8) → W_o
           attn_weights = mean(attn, dim=heads)   → XAI 저장
           ────────────────────────────────────────
                             ▼
           masked_mean_pool (pad_mask 기준)
           → graph_feat (B,64)
                             ▼
           concat([chem_feat, graph_feat]) (B,128)
           → Linear(128,64) → LayerNorm → fused (B,64)
           → LayerNorm → Linear(64,32) → GELU → Dropout(0.3)
           → encode_out (B,32)
                    ┌────────┴────────┐
              Stage 1              Stage 2
           Linear(32,1)         encode() → (B,32)
           BCEWithLogitsLoss      Stacking OOF
```

### 3.2 Stage 2: Stacking OOF Flow

```
train_graphs.pt (N=1,398)
        │
        ▼
GraphMACCSEncoder.encode()   ← eval mode, frozen
        │
        ▼
  32-dim fused feature
        │
        ▼
5-Fold Stratified OOF
┌─────────────────────────────────────┐
│  RF  (n=300, class_weight=balanced) │
│  ET  (n=300, class_weight=balanced) │
│  HistGB (n=300, balanced)           │
│  XGB (n=300, lr=0.05, scale_pos_w) │
│  OOF probs: (1398,4)                │
│  Test probs: (452,4)  [fold avg]    │
└─────────────────────────────────────┘
        │
        ▼
LogisticRegression(class_weight=balanced)
  fit(oof_probs, y_train)
        │
        ▼
OOF MCC-optimal threshold (0.10~0.90 탐색)
        │
        ▼
최종 예측 (DILIrank N=452)
```

---

## 4. Key Design Decisions

### 4.1 DifferentialCrossAttention — 설계 근거

**기반:** Differential Transformer (Ye et al., arxiv 2410.05258)의 cross-attention 변형.

**핵심 아이디어:**
- standard softmax attention은 모든 key에 분산된 noisy attention 문제가 있음
- 두 개의 attention map(a1, a2)을 학습하여 a2가 포착하는 공통 패턴(노이즈)을 a1에서 차감
- 분자 맥락: a2는 공통적으로 활성화되는 MACCS 키(noise)를 포착, a1에서 차감 → 분자 특이적 구조-활성 상관관계 강조

**구현 특이사항:**
- **binary gate** 방식으로 inactive MACCS bit 처리: `maccs_kv = Embedding(idx) * maccs.unsqueeze(-1)`. 0-vector 자체가 softmax에 균등 기여하는 문제가 있으나, 두 attention head가 대칭적으로 처리하여 차분에서 상쇄됨. `kv_padding_mask`(-1e9 마스킹) 경로가 구현되어 있으나 현재 미활성.
- **λ 초기화 0.8, clamp(1e-4, 2.0):** 학습 초기 a1 ≈ a2 → 차분값 작음 → gradient signal 약할 수 있음. 실험적으로 λ_init=0.8이 안정.
- **GroupNorm + (1-λ_init) 스케일 보정:** 논문 권장 방식으로 출력 분산 안정화.

### 4.2 ChemBERTa 부분 동결 전략

| 레이어 | 상태 | 학습률 |
|--------|------|--------|
| embeddings | frozen | — |
| encoder.layer[0], [1] | frozen | — |
| encoder.layer[2] (마지막) | unfrozen | 1e-4 |
| Graph/DiffAttn/Fusion | trainable | 3e-4 |

**근거:** 전체 unfreezing은 도메인 오버피팅(ChemBERTa 사전학습 도메인 ≠ DILI 레이블 분포) 유발. 2-layer unfreeze 실험에서 AUC 하락 확인. 1-layer가 regularization 강도 상 최적.

### 4.3 val_AUC vs val_loss Early Stopping

**val_loss 조기종료가 역효과인 이유:**
- train 데이터: 다중 DB 혼합 (노이즈 많음)
- test 데이터: DILIrank (더 깔끔하게 분리)
- val_loss는 epoch 2~5에서 최솟값 도달 후 증가 → loss 기준 체크포인트 = 사실상 미학습 상태
- **결론:** val_AUC (mode=max, patience=30) 사용. 현재 Step2에 적용됨.

### 4.4 Meta-model LR (ET 미사용 이유)

OOF 피처 차원이 4(base model 수)에 불과한 상황에서:
- **ExtraTrees meta:** n_estimators=300 → OOF MCC=1.0 과적합, test Specificity 0.40 붕괴
- **LogisticRegression meta:** 4차원 선형 경계에서 일반화 유지. 현재 채택.

---

## 5. Strengths (장점)

| 항목 | 설명 |
|------|------|
| **삼중 표현 융합** | 그래프 구조(GINEConv), 구조 지문(MACCS DiffAttn), 언어 임베딩(ChemBERTa)의 상호보완적 정보 활용 |
| **edge feature 활용** | GINEConv의 9-dim bond feature (결합 종류, 공액, 고리 여부, 입체화학) → SAGEConv 대비 env2 AUC +0.065, MCC +0.171 향상 |
| **Differential attention** | Standard softmax attention 대비 노이즈 억제, 모델이 분자 특이적 구조-MACCS 상관관계에 집중할 수 있도록 유도 |
| **XAI 내장** | `attn_weights (B,MAX_ATOMS,167)` 자동 저장 → 원자-MACCS 상관관계 히트맵, 멀티헤드 독립 분석 모두 지원 |
| **2-Stage 설계** | Stage 1 encoder 학습과 Stage 2 stacking 분리 → 피처 재사용성, 엔드투엔드 과적합 회피 |
| **재현성** | `set_seed()`로 random/numpy/torch/cuda/cudnn 완전 고정 |
| **캐시 시스템** | 토큰/그래프 `.pt` 캐시로 반복 실험 시 전처리 비용 제거 |
| **그라디언트 클리핑** | `clip_grad_norm_(max_norm=1.0)` → 학습 안정성 |

---

## 6. Weaknesses / Limitations (단점 및 한계)

| 항목 | 설명 |
|------|------|
| **Binary gate vs kv_padding_mask** | 비활성 MACCS bit를 0-vector로 처리하면 softmax 분모에 균등 기여. Differential 구조에서 부분 상쇄되나 이론적으로 `kv_padding_mask=-1e9`가 더 엄밀. 현재 `kv_padding_mask=None`이 하드코딩됨. |
| **val_AUC 편향** | val set이 train과 동일 분포(혼합 DB)이므로 val_AUC ≈ 0.63~0.76. Early stopping 신호가 약하고 patience=30이 사실상 완전 학습에 가까움. |
| **λ 포화 위험** | λ_init=0.8 고정 후 head-wise λ 학습으로 max=2.0까지 증가 가능. λ=2.0 근방에서 a2 기여가 과도해질 수 있음 (단, clamp로 제한). |
| **encoder 고정 CV** | Step_CV에서 encoder는 train set으로만 학습되어 있고, 10-fold 내 train 샘플에 대해 경미한 낙관 편향 존재 (DILIrank 샘플은 편향 없음). |
| **MAX_ATOMS=100 하드코딩** | 100개 이상의 원자를 가진 분자는 truncation됨. 제약 조건이 config에만 있고 경고 없음. |
| **Step 1-A 산출물 미활용** | `fp_full_train/test.npy`, `scalers.pkl` 등 FP 관련 파일이 Step 1-A에서 생성되지만 현재 파이프라인(Step2~CV)에서 소비되지 않음. 구 FP 기반 모델 잔재. |

| **단일 seed 결과** | env1(Fixed Split)은 seed=42 단일 실험. seed variance 미보고. |

---

## 7. Experiment Results Summary

### env1 — Fixed Split (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | **0.9736** | **0.8304** | **0.9010** | **0.9402** | **0.8993** |
| SAGEConv + LR | 0.8426 | 0.5275 | 0.7102 | 0.6793 | 0.8396 |
| **GINEConv + LR (DGUDILI_2026)** | **0.8808** | **0.5776** | **0.7565** | 0.7935 | 0.7910 |
| vs 목표 (GINEConv) | −0.0928 | −0.2528 | −0.1445 | −0.1467 | −0.1083 |

### env2 — 10-Fold CV (전체 N=1,850)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| SAGEConv + LR | 0.8909 ±0.030 | 0.6366 ±0.072 | 0.8230 ±0.037 | 0.831 | 0.802 |
| **GINEConv + LR (DGUDILI_2026)** | **0.9558 ±0.015** | **0.8080 ±0.051** | **0.9068 ±0.024** | 0.908 | 0.900 |

**관찰:** env2(CV)에서 AUC=0.9558 ±0.015는 목표(0.9736)에 근접하지만, env1(Fixed Split)에서 0.8808로 격차가 큼. env1과 env2의 차이는 encoder가 train set 기반으로만 학습된 상황에서 DILIrank 분포 이동의 영향을 반영.

---

## 8. Ablation Summary (배제된 전략)

| 전략 | 결과 | 결론 |
|------|------|------|
| ChemBERTa 2-layer unfreeze | AUC 하락 | 도메인 오버피팅. 1-layer가 최적 regularization |
| DANN/DAT domain adaptation | AUC 하락 | train/test 분포 차이가 signal quality shift이지 domain shift가 아님 |
| ET meta-model | Specificity 0.40 붕괴 | 4-dim OOF 입력에서 과적합 (OOF MCC=1.0) |
| val_loss early stopping | AUC 급락 | epoch 2~5에서 val_loss 최소 → 사실상 미학습 체크포인트 |
| MLP 2-layer | −0.015 | Dropout×2 과도 정규화 |
| BatchNorm → GraphNorm | −0.014 | 소규모 배치에서 역효과 |

---

## 9. Next Steps (Phase 4 계획)

**비MACCS FP 혼합 (현재 AUC ceiling 도달 시 시도):**
- MACCS(167)는 encoder가 이미 cross-attention으로 처리 → 추가 입력 제외
- 추가 대상: E-state(79) + CalcCATS(150) + Constitutional(22) + 기타(7) = **258-dim**
- pipeline: `encode_out(32) ‖ fp_non_maccs(258)` = **290-dim** → Stacking 입력
- 예상 env1 AUC +0.02~0.04

---

## 10. File-level Summary Table

| 파일 | 역할 | 상태 |
|------|------|------|
| `config.py` | 하이퍼파라미터/경로 중앙 관리 | 정상 |
| `model.py` | GraphMACCSEncoder 정의 | 정상 |
| `differential_attention.py` | DifferentialCrossAttention | 정상 |
| `graph_utils.py` | SMILES→PyG, MACCS, 검증 | 정상 |
| `features.py` | extract_features, collate_fn | 정상 |
| `utils.py` | set_seed, load_dataset | 정상 (SMILESDataset 제거 완료) |
| `Step1_preprocess.py` | FP 전처리 + Graph 캐시 저장 | 정상 (주석 수정 완료) |
| `Step2_pretrain.py` | Stage 1 학습 | 정상 |
| `Step3_stacking.py` | Stage 2 Stacking | 정상 |
| `Step4_xai.py` | DiffAttn XAI | **수정 완료** (SAGE→GINE) |
| `Step5_mHeadHeatmap.py` | 멀티헤드 히트맵 | **수정 완료** (SAGE→GINE, graph_utils2 제거) |
| `Step_CV.py` | 10-Fold CV | **수정 완료** (SAGE→GINE, 데드 import 제거) |

