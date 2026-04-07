# DGUDILI 알고리즘 현황 분석 및 개선 보고서

> **분석 일시:** 2026-04-08  
> **대상 경로:** `C:\DiffDILI\Origin_StackDILI\` (GraphMACCSEncoder v2026)  
> **현재 성능:** AUC 0.9224, MCC 0.7270, F1 0.8426 (DILIrank test, N=452)

---

## 1. 프로젝트 구조 개요

```
Origin_StackDILI/
├── Data/Dataset.csv               ← 1,850개 화합물 데이터셋 (SMILES, Label, ref)
├── Code/
│   ├── Dataset_feature.csv        ← 425-dim FP (iFeatureOmegaCLI)
│   └── Dataset_feature_clean.csv  ← clean 버전 (run-clean 실행 시 생성)
│
├── DGUDILI_2026/
│   └── src/
│       ├── config.py                  ← 하이퍼파라미터 및 경로
│       ├── model.py                   ← GraphMACCSEncoder (현재), 레거시 E2E 모델 포함
│       ├── graph_utils.py             ← smiles_to_pyg, get_maccs
│       ├── differential_attention.py  ← DifferentialCrossAttention
│       ├── utils.py                   ← set_seed, load_dataset
│       ├── Step1_preprocess.py        ← FP 전처리 + PyG/MACCS .pt 캐시
│       ├── Step2_pretrain.py          ← GraphMACCSEncoder Stage 1 학습
│       ├── Step3_stacking.py          ← Feature 추출 + Stacking OOF (env1)
│       ├── Step4_xai.py               ← DiffAttn XAI 히트맵
│       └── Step_CV.py                 ← 10-Fold CV (env2)
│
└── Origin_StackDILI/             ← StackDILI 원본 논문 재현 코드 (베이스라인)
```

---

## 2. 알고리즘 파이프라인 분석

### 2.1 Origin_StackDILI (베이스라인)

**전체 흐름:**
```
SMILES → iFeatureOmegaCLI (Constitution + Pharmacophore + MACCS + E-state)
       → 425-dim FP 생성
       → Genetic Algorithm (GA) → ~209개 피처 선택
       → Stacking Ensemble (ET + HistGB + RF + XGBoost → MetaLearner)
       → 최종 예측
```

**성능 (DILIrank test, N=452):**
| 지표 | 값 |
|------|----|
| AUC  | 0.9736 |
| MCC  | 0.8304 |
| F1   | 0.9010 |
| Sensitivity | 0.9402 |
| Specificity | 0.8993 |

---

### 2.2 DGUDILI_2026 — GraphMACCSEncoder (현재 개발 버전)

**전체 흐름:**
```
SMILES
  ├─→ ChemBERTa-77M-MLM (last-layer unfreeze, lr=1e-4)
  │   → CLS (B, 384) → LN → Linear(384, 64) → chem_feat (B, 64)
  │
  ├─→ RDKit mol → 43-dim atom features → atom_proj → SAGEConv×2 (hidden=64)
  │   → to_dense_batch → node_q (B, 100, 64)
  │
  └─→ MACCSkeys (B, 167) → Embedding(167, 64) → maccs_kv (B, 167, 64)
      inactive bits masked -1e9

           ↓
  DifferentialCrossAttention (Q=node_q, K/V=maccs_kv)
  λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + 0.8  ;  λ.clamp(min=1e-4)
  scores = softmax(Q1@K1^T/√d) − λ · softmax(Q2@K2^T/√d)

           ↓
  masked_mean_pool → graph_feat (B, 64)
  concat([chem_feat, graph_feat]) → fuse_proj → MLP → encode_out (B, 32)

           ↓
  Stage 2: Stacking OOF
    Base: RF / ET / HistGB / XGB (n_estimators=300, 5-fold)
    Meta: LogisticRegression
    Threshold: OOF MCC-optimal (0.10~0.90 탐색)
```

**성능 (DILIrank test, N=452):**
| 지표 | 현재 (GraphMACCSEncoder) | 목표 (StackDILI) | gap |
|------|--------------------------|-----------------|-----|
| AUC  | **0.9224** | 0.9736 | -0.0512 |
| MCC  | **0.7270** | 0.8304 | -0.1034 |
| F1   | **0.8426** | 0.9010 | -0.0584 |
| Sensitivity | 0.9022 | 0.9402 | -0.0380 |
| Specificity | 0.8358 | 0.8993 | -0.0635 |

---

### 2.3 아키텍처 진화 이력

| 버전 | 모델 | AUC | MCC | 해결된 문제 |
|------|------|-----|-----|------------|
| StackDILI *(baseline)* | GA + Stacking | 0.9736 | 0.8304 | — |
| CrossAttentionEncoder | CLS + FP CrossAttn | 0.7313 | 0.3045 | — |
| FTV6StyleEncoder | FP soft projection | 0.9196 | 0.6682 | Early stop leakage 수정 |
| E2E_FTV6StyleEncoder | ChemBERTa fine-tune | 0.9219 | 0.7270 | ChemBERTa unfreeze |
| **GraphMACCSEncoder** | **GraphSAGE + DiffAttn** | **0.9224** | **0.7270** | **MACCS 구조 정보 통합** |

---

## 3. 현재 남은 문제점

### 🔴 문제 1: Train/Test 분포 이동 (핵심 병목)

**현상:**
```
val_AUC  ≈ 0.63~0.70  (train 내부 val, 다중 DB 혼합)
test_AUC ≈ 0.90~0.93  (DILIrank — 단일 큐레이션 데이터)
```

**원인 분석:**
- Train: ChEMBL, SIDER 등 다중 데이터베이스 혼합 → 레이블 노이즈 多, 화학적 다양성 大
- Test: DILIrank 단일 큐레이션 → 명확히 분리된 DILI/비DILI 분포
- 결과적으로 val_AUC로 조기종료 기준을 삼으면 train 분포에 과도하게 맞춰짐

**영향:**
- 모델이 test 분포에 최적화되지 않은 채로 조기종료
- 잠재적으로 더 좋은 checkpoint가 존재할 수 있으나 val_AUC 기준으로는 탐색 불가

**현재 대응:** val_AUC (mode=max) 기준 조기종료로 최악의 경우는 방지됨  
**미해결:** 분포 이동 자체를 줄이는 도메인 적응 미적용 (P3)

---

### 🟠 문제 2: Step_CV encoder 낙관 편향

**위치:** `Step_CV.py`

**현상:**
- encoder(GraphMACCSEncoder)는 train set 전체로 Stage 1 학습
- 10-Fold CV에서 encoder 학습에 사용된 train 샘플이 CV test fold에도 포함
- train 샘플에 대한 CV test 결과는 경미하게 낙관적 편향 존재

**범위 제한:**
- DILIrank 452개 샘플은 encoder 학습에 사용되지 않으므로 편향 없음
- env1(fixed split)의 test 결과는 완전히 신뢰 가능

---

### 🟡 문제 3: Sensitivity/Specificity gap

**현상:**
```
Sensitivity: 0.9022  (StackDILI: 0.9402, gap -0.0380)
Specificity: 0.8358  (StackDILI: 0.8993, gap -0.0635)
```

Specificity(비독성 올바른 분류) gap이 Sensitivity보다 큼. 비독성 화합물을 독성으로 잘못 예측하는 False Positive가 상대적으로 많음. OOF MCC-optimal threshold 탐색으로 일부 완화되어 있으나 한계 존재.

---

## 4. 핵심 문제 요약

```
┌─────────────────────────────────────────────────────────────────────┐
│              AUC -0.051 gap의 주원인 분석 (vs StackDILI)            │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 분포 이동    → train(혼합 DB 노이즈) vs test(DILIrank 정제)      │
│    val_AUC 0.68 vs test_AUC 0.92 — val로 최적화 불가               │
│ 2. FP 미활용   → GraphMACCSEncoder는 FP를 Stage 2 입력으로만 사용  │
│    (Stacking base에 FP 직접 입력하지 않음)                          │
│ 3. Specificity 취약 → 비독성 FP 비율이 상대적으로 높음             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 개선 방안

### 5.1 P3 — Domain Adversarial Training (분포 이동 대응, 미구현)

**아이디어:**
```python
# model.py 수정안
class GraphMACCSEncoder_DAT(GraphMACCSEncoder):
    def __init__(self, ...):
        super().__init__(...)
        self.domain_clf = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)   # train(0) vs test(1) 도메인 분류
        )

# Step2_pretrain.py 수정안
loss = bce_loss + lambda_d * domain_loss
# test SMILES(레이블 없음)를 도메인 분류 학습에만 활용
```

**기대 효과:** train/test 분포 차이 완화 → val_AUC 향상 → 조기종료 품질 개선 → AUC +0.02~0.05

**전제 조건:** test SMILES를 레이블 없이 도메인 적응에 활용하는 것이 방법론적으로 허용되는 맥락인지 확인 필요

---

### 5.2 FP 직접 통합 (Stage 2 개선)

**현재:**
```
encode_out(B, 32) → Stacking [RF/ET/HistGB/XGB] → LR meta
```

**개선안:**
```
concat([encode_out(B, 32), fp_scaled(B, 425)]) → Stacking [RF/ET/HistGB/XGB] → LR meta
```

FP 정보가 Stage 2 입력에 직접 포함되어 Stacking 분류기가 FP 특징을 추가로 활용 가능.

---

## 6. 우선순위별 Action Plan

| 우선순위 | 작업 | 예상 효과 | 난이도 |
|----------|------|-----------|--------|
| **P3-A** | Domain Adversarial Training | AUC +0.02~0.05 | 높음 |
| **P3-B** | Stage 2 입력에 FP 직접 통합 | AUC +0.01~0.03 | 낮음 |
| **P3-C** | Specificity 집중 threshold 최적화 (F-beta 등) | 균형 개선 | 낮음 |

---

## 7. 목표 성능 로드맵

```
현재 (GraphMACCSEncoder)    P3-B (FP 통합)      P3-A (DAT)          목표
AUC: 0.9224              →  AUC: 0.93~0.94   →  AUC: 0.95+      →  0.9736
MCC: 0.7270              →  MCC: 0.74~0.76   →  MCC: 0.78+      →  0.8304
```

---

## 8. 결론

DGUDILI_2026 GraphMACCSEncoder는 이전 CrossAttentionEncoder(AUC 0.7313) 대비 크게 개선된 AUC 0.9224를 달성했으며, 구조적 결함(데이터 누수, 모델 용량 부족, 선형 분류기)은 모두 해결된 상태입니다.

현재 StackDILI와의 gap(AUC -0.051, MCC -0.103)은 주로 **train/test 분포 이동** 문제에서 기인합니다. 이를 해결하기 위한 Domain Adversarial Training이 주요 미해결 과제이며, Stage 2에 FP를 직접 통합하는 단기 개선과 병행 시 StackDILI 수준의 성능(AUC 0.95+) 달성이 가능할 것으로 판단합니다.

모델의 핵심 아이디어인 **GraphSAGE + MACCS DifferentialCrossAttention + ChemBERTa 3-way fusion**은 개념적으로 타당하며, 현재 병목은 아키텍처 자체가 아닌 분포 이동 적응 메커니즘의 부재입니다.
