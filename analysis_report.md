# DGUDILI 알고리즘 현황 분석 및 개선 보고서

> **분석 일시:** 2026-04-02  
> **대상 경로:** `C:\DGUDILI\` (Origin_StackDILI / DGUDILI_2026)

---

## 1. 프로젝트 구조 개요

```
C:\DGUDILI\
├── Origin_StackDILI/      ← 베이스라인 (논문 원본 재현)
│   ├── Data/Dataset.csv   ← 1,850개 화합물 데이터셋
│   ├── Code/
│   │   ├── Feature.py     ← iFeatureOmegaCLI 분자 기술자 추출
│   │   ├── GA.ipynb       ← 유전 알고리즘 피처 선택
│   │   ├── stacking.ipynb ← 스태킹 앙상블 학습
│   │   ├── ML_model.ipynb ← 개별 ML 모델 학습
│   │   └── Model/         ← 저장된 pkl 모델 (ET, HistGB, RF, XGBoost, Stacking)
│   └── Feature/Feature.csv
│
└── DGUDILI_2026/          ← 개선 시도 버전
    ├── src/
    │   ├── model.py                   ← CrossAttentionEncoder (핵심 모델)
    │   ├── Step1_chemberta_embed.py   ← ChemBERTa 임베딩 추출
    │   ├── Step2_feature_select.py    ← 피처 선택 (SelectKBest)
    │   ├── Step3_pretrain.py          ← Cross-Attention 학습
    │   └── Step4_extract_and_lr.py   ← 피처 추출 + LR 분류
    └── data/              ← 중간 산출물 (npy 파일)
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

### 2.2 DGUDILI_2026 (현재 개발 버전)

**전체 흐름:**
```
SMILES → ChemBERTa-77M-MLM (768-dim CLS/Mean 임베딩)
       → iFeatureOmegaCLI (425-dim FP)
       → SelectKBest(f_classif, k=16): FP만 선별
       → CrossAttentionEncoder
           Q ← ChemBERTa proj(768→16)
           K/V ← FP(16)
           Attention → 16-dim Feature Space
       → Logistic Regression → 최종 예측
```

**성능 (DILIrank test, N=452):**
| 지표 | 현재 | 베이스라인 |
|------|------|-----------|
| AUC  | 0.7313 | 0.9736 |
| MCC  | 0.3045 | 0.8304 |
| F1   | 0.6362 | 0.9010 |
| Sensitivity | 0.8315 | 0.9402 |
| Specificity | 0.4627 | 0.8993 |

> [!CAUTION]
> **DGUDILI_2026은 모든 지표에서 베이스라인(StackDILI) 대비 심각하게 낮은 성능을 보입니다.**  
> AUC -0.2423, MCC -0.5259로 단순 개선이 아닌 구조적 재설계가 필요합니다.

---

## 3. 발견된 문제점 (심각도 순)

### 🔴 문제 1: 치명적 데이터 누수 (Data Leakage) — Early Stopping

**위치:** `Step3_pretrain.py`, Line 96

```python
# 현재 코드 (문제)
proba = torch.sigmoid(encoder(cham_test_d, fp_test_d)).squeeze(1).cpu().numpy()
auc = roc_auc_score(y_test_np, proba)
...
if auc > best_auc:
    best_auc = auc
    best_state = ...
```

**문제 설명:**  
Early stopping 기준이 **Test 세트(DILIrank)의 AUC**를 직접 모니터링합니다. 이는 Test 세트에 대한 정보가 모델 선택에 직접 사용되는 심각한 **데이터 누수**입니다. 이 때문에 Step3에서 보이는 AUC와 Step4에서 최종 보고되는 AUC가 일관되지 않을 수 있으며, 모델이 Test 세트에 간접 과적합됩니다.

**올바른 방법:** Train 세트를 일부 분리하여 Validation 세트를 만들고, Val AUC 또는 Val Loss를 기준으로 Early Stopping을 수행해야 합니다.

---

### 🔴 문제 2: 심각한 모델 용량 부족 (Underfitting)

**위치:** `model.py` — `CrossAttentionEncoder`

```python
class CrossAttentionEncoder(nn.Module):
    # 핵심 파라미터:
    self.W_Q = nn.Linear(1, d_k)   # 1 → 32
    self.W_K = nn.Linear(1, d_k)   # 1 → 32
    self.W_V = nn.Linear(1, 1)     # 1 → 1
    self.head = nn.Linear(k, 1)    # 16 → 1
```

**문제 설명:**  
현재 Cross-Attention 구조는 **element-wise 처리 방식**으로, 각 피처 차원을 독립적인 1차원 토큰으로 취급합니다. 이 설계의 근본적 문제는:

1. **`d_v=1`**: Value 투영이 1차원으로 극도로 제한되어 정보 표현력이 없음
2. **각 피처 차원 간 상호작용 없음**: Q와 K의 cross-attention이 서로 다른 모달리티 간 관계를 학습하기 어려운 구조
3. **총 파라미터 수가 극소**: 효과적인 표현 학습에 필요한 용량 부족

ChemBERTa projection에 LayerNorm + GELU + Dropout으로 충분한 처리를 하는 반면, FP는 선형 매핑 1개만 적용됩니다.

---

### 🟠 문제 3: SelectKBest 적용 불일치 (모달리티 비대칭)

**위치:** `Step2_feature_select.py`, Line 75-83

```python
# FP: SelectKBest로 16개 선별
sel_fp = SelectKBest(f_classif, k=K)
sel_fp.fit(X_fp_all[train_mask], y_all[train_mask])

# ChemBERTa: SelectKBest 없이 전체 768-dim 사용!
print("[ChemBERTa] Full embedding + StandardScaler (no SelectKBest)")
X_cham_train = embeddings[train_mask].astype(np.float32)  # (1398, 768)
```

**문제 설명:**  
- FP: 425차원 → 16차원 (선택률 3.8%)
- ChemBERTa: 768차원 → **768차원 그대로** (선택 없음)

GUIDE.md에는 "ChemBERTa 768-dim → SelectKBest(k=16)"라고 명시되어 있지만, **실제 코드는 전체 768차원을 그대로 사용**합니다. 이로 인해 CrossAttentionEncoder의 `chem_in_dim` 파라미터가 384로 하드코딩되어 있어 **shape 불일치가 발생할 수 있습니다.**

> [!WARNING]
> `model.py` Line 22: `chem_in_dim: int = 384` — ChemBERTa-77M의 실제 hidden dim은 **384**이며, 코드는 DeepChem/ChemBERTa-77M-MLM(384-dim)을 사용하고 있습니다. GUIDE.md의 "768-dim" 설명은 문서와 코드 간 불일치입니다.

---

### 🟠 문제 4: 단일 분류기의 표현력 한계

**위치:** `Step4_extract_and_lr.py`, Line 64

```python
lr_model = LogisticRegression(max_iter=1000, random_state=42)
```

**문제 설명:**  
16-dim Cross-Attention 피처 공간으로부터 Logistic Regression만 사용합니다. 이는 선형 분리 가능성을 전제하는 모델로, 분자 독성 예측과 같은 복잡한 비선형 패턴 학습에 부적합합니다. StackDILI가 ET + HistGB + RF + XGBoost 앙상블을 사용하는 것과 대조됩니다.

---

### 🟡 문제 5: 하이퍼파라미터 불일치 (문서 vs 코드)

**GUIDE.md의 명세:**
| 파라미터 | 문서 값 | 실제 코드 값 |
|---------|---------|------------|
| lr | 1e-3 | **3e-4** (Step3) |
| ChemBERTa dim | 768 | **384** (model.py) |
| patience | 20 | 20 (일치) |

이는 문서가 업데이트되지 않은 채로 코드만 수정된 결과로, 재현성과 협업에 어려움을 줍니다.

---

### 🟡 문제 6: 교차검증(Cross-Validation) 미적용

**현재 Train/Test Split:**
```python
train = data[data['ref'] != 'DILIrank']  # 1,398 samples
test  = data[data['ref'] == 'DILIrank']  # 452 samples (고정)
```

모든 하이퍼파라미터 선택과 Early Stopping이 **동일한 고정 Test 세트에 대해 반복 평가**됩니다. K-Fold Cross-Validation을 도입하여 더 신뢰 가능한 성능 추정이 필요합니다.

---

### 🟡 문제 7: 재현성 미보장 (GPU 비결정성)

**위치:** `Step3_pretrain.py`, Line 39-40

```python
torch.manual_seed(SEED)
np.random.seed(SEED)
# 누락:
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
```

CUDA 환경에서 GPU 연산은 Random Seed만으로 완전히 제어되지 않습니다. 과거 대화 기록에서도 환경 간 AUC 불일치 문제가 보고된 바 있습니다.

---

### 🔵 문제 8: Step 1의 불필요한 재실행 강제

**GUIDE.md, Line 154:**
> "Step 1은 `chemberta_embeddings.npy`가 이미 존재해도 항상 재실행됨"

임베딩 추출은 3~5분 소요되는 연산입니다. 파일 존재 여부 확인 후 스킵하는 캐싱 로직이 없어 개발 사이클이 느립니다.

---

## 4. 핵심 문제 요약

```
┌─────────────────────────────────────────────────────────┐
│               AUC 0.2423 하락의 주원인 분석              │
├─────────────────────────────────────────────────────────┤
│ 1. 데이터 누수    → Early stopping이 Test AUC 직접 관찰  │
│    (역설적으로 AUC가 낮으면 과적합 방향도 다름)          │
│ 2. 모델 용량 부족 → d_v=1이 표현력 병목 지점            │
│ 3. 16-dim 병목   → 피처 과도한 압축 (425+384 → 16)      │
│ 4. 선형 분류기   → 비선형 독성 패턴 학습 불가            │
│ 5. 검증 방법론   → 단일 Test 세트, CV 없음               │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 개선 방안

### 5.1 즉시 적용 가능 (단기)

#### [Fix-1] Early Stopping 수정 — 데이터 누수 제거

```python
# Step3_pretrain.py 수정안
from sklearn.model_selection import train_test_split

# Train 세트에서 Val 세트 분리 (stratified)
idx = np.arange(len(y_train))
idx_tr, idx_val = train_test_split(idx, test_size=0.15, 
                                    stratify=y_train.numpy(), 
                                    random_state=42)

# Val 세트를 기준으로 Early Stopping
val_auc = roc_auc_score(y_val_np, val_proba)
if val_auc > best_val_auc:
    best_val_auc = val_auc
    best_state = ...
```

#### [Fix-2] 재현성 완전 보장

```python
# Step3_pretrain.py에 추가
import random
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### [Fix-3] Step 1 캐싱

```python
# Step1_chemberta_embed.py 수정
emb_path = os.path.join(DATA_DIR, f"chemberta_embeddings_{POOLING}.npy")
if os.path.exists(emb_path):
    print(f"[SKIP] {emb_path} already exists. Use --force to regenerate.")
    sys.exit(0)
```

---

### 5.2 아키텍처 개선 (중기)

#### [Arch-1] `d_v` 확장 및 Multi-head Attention 도입

```python
# model.py 개선안
class CrossAttentionEncoder(nn.Module):
    def __init__(self, chem_in_dim=384, k=32, n_heads=4, d_k=32, d_v=16, dropout=0.2):
        # 4-head Attention, d_v=16 (현재 d_v=1 대비 16배 표현력)
        self.W_Q = nn.Linear(1, d_k * n_heads)
        self.W_K = nn.Linear(1, d_k * n_heads)  
        self.W_V = nn.Linear(1, d_v * n_heads)
        
        # Residual connection + LayerNorm 추가
        self.norm = nn.LayerNorm(k)
        
        # 비선형 분류 헤드
        self.head = nn.Sequential(
            nn.Linear(k * d_v, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
```

#### [Arch-2] 피처 차원 확장 (k=16 → k=32~64)

현재 k=16은 너무 과도한 압축입니다. 전체 입력(384+16=400)의 8~16% 수준인 **k=32~64**가 더 적절합니다.

#### [Arch-3] 분류기 업그레이드

```python
# Step4: LR 대신 앙상블 또는 MLP 사용
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

classifiers = {
    'RF': RandomForestClassifier(n_estimators=300, random_state=42),
    'XGB': XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}
```

---

### 5.3 방법론 개선 (장기)

#### [Method-1] 5-Fold Cross-Validation 도입

```python
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    # 각 fold에서 독립적으로 모델 학습 및 평가
    ...
```

#### [Method-2] End-to-End Fine-tuning (Stage 통합)

현재 Stage 1(Encoder 학습) → Stage 2(LR 학습)가 분리되어 있는데, Joint Training으로 통합하면 피처 공간이 최종 분류 목표에 최적화됩니다.

#### [Method-3] ChemBERTa Fine-tuning 고려

현재 ChemBERTa는 **frozen** 상태로만 사용합니다. DILI 데이터셋을 활용한 일부 Fine-tuning(마지막 2~3 레이어만)이 성능을 크게 향상시킬 수 있습니다.

---

## 6. 우선순위별 Action Plan

| 우선순위 | 작업 | 예상 효과 | 난이도 |
|----------|------|-----------|--------|
| **P0** | Fix-1: Early Stopping → Val 세트 기준 변경 | 누수 제거, 신뢰성 확보 | 낮음 |
| **P0** | Fix-2: 재현성 시드 완전 설정 | 환경 간 일관된 결과 | 낮음 |
| **P1** | Arch-1: d_v=1 → 16, Multi-head | AUC +0.05~0.10 예상 | 중간 |
| **P1** | Arch-2: k=16 → 32~64 | AUC +0.03~0.05 예상 | 낮음 |
| **P1** | Arch-3: LR → XGB/MLP | AUC +0.05~0.08 예상 | 낮음 |
| **P2** | Method-1: 5-Fold CV | 평가 신뢰도 향상 | 중간 |
| **P2** | Arch-4: Residual + LayerNorm | 학습 안정성 향상 | 중간 |
| **P3** | Method-2: End-to-End 학습 | AUC +0.10~0.15 예상 | 높음 |
| **P3** | Method-3: ChemBERTa Fine-tuning | AUC +0.05~0.10 예상 | 높음 |

---

## 7. 목표 성능 로드맵

```
현재 상태       단기 목표(P0+P1)     중기 목표(P2)       장기 목표(P3)
AUC: 0.7313 →   AUC: 0.85~0.88   →  AUC: 0.90~0.92  →  AUC: 0.93~0.97
MCC: 0.3045 →   MCC: 0.55~0.65   →  MCC: 0.65~0.72  →  MCC: 0.75~0.83
```

> [!IMPORTANT]
> **P0 수정(데이터 누수 제거)은 무조건 먼저 수행해야 합니다.** 현재 모델이 Test 세트를 간접적으로 "본" 상태에서 학습되었기 때문에, 누수 제거 후 성능이 오히려 더 낮게 나올 수 있습니다. 이 경우 실제 성능이 드러난 것이므로 긍정적으로 해석해야 합니다.

---

## 8. 결론

DGUDILI_2026은 **개념적으로 올바른 방향**(FP + 언어모델 임베딩 융합)을 가지고 있으나, 구현 수준에서 여러 치명적 결함이 복합적으로 작용하여 베이스라인 대비 크게 낮은 성능을 보입니다. 

특히 **Early Stopping의 Test 세트 직접 사용은 방법론적으로 허용될 수 없으며**, 이를 수정하는 것이 가장 시급합니다. 이후 모델 용량 확장(d_v, k, 분류기)을 통해 성능을 단계적으로 끌어올리는 전략이 현실적입니다.

ChemBERTa + FP의 Cross-Modal Attention 아이디어 자체는 분자 독성 예측에서 유망한 접근법이므로, 구현 결함을 수정하면 StackDILI 수준 또는 그 이상의 성능 달성이 가능할 것으로 판단합니다.
