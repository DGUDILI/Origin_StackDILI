# DGUDILI 2026 — 환경 구성 및 실행 가이드

> 모델: **GraphMACCSEncoder** (GraphSAGE + MACCS DifferentialCrossAttention + ChemBERTa)

---

## 모델 구조

```
SMILES
  ├─→ ChemBERTa-77M-MLM (last-layer fine-tune) → CLS (B, 384) → chem_feat (B, 64)
  ├─→ RDKit mol → atom_features (43-dim) → SAGEConv×2 (hidden=64)
  │   → to_dense_batch → node_q (B, 100, 64)
  └─→ MACCSkeys (B, 167) → Embedding(167, 64) → maccs_kv (B, 167, 64)
                        ↓
      DifferentialCrossAttention (Q=node_q, K/V=maccs_kv)
      inactive bits masked -1e9 | λ.clamp(min=1e-4)
                        ↓
      masked_mean_pool → graph_feat (B, 64)
      concat([chem_feat, graph_feat]) → fuse_proj → MLP → 32-dim
                        ↓
  Stacking: RF / ET / HistGB / XGB (5-fold OOF) + LR meta
                        ↓
             OOF MCC-optimal threshold
```

---

## 디렉토리 구조

```
Origin_StackDILI/
├── Dockerfile                   # Docker 이미지 정의
├── run.sh                       # 파이프라인 실행 진입점
├── make_clean_data.py           # train/test 중복 분자 제거
├── ENV_GUIDE.md                 # 이 파일
│
├── Data/
│   ├── Dataset.csv              # 전체 데이터 (SMILES, Label, ref)
│   └── Dataset_clean.csv        # run-clean 실행 후 자동 생성
│
├── Code/
│   ├── Dataset_feature.csv      # FP 피처 (Morgan fingerprint 등)
│   └── Dataset_feature_clean.csv  # run-clean 실행 후 자동 생성
│
└── DGUDILI_2026/
    ├── src/
    │   ├── config.py                  # 하이퍼파라미터 및 경로 설정
    │   ├── model.py                   # GraphMACCSEncoder (현재), E2E_FTV6/MHA 레거시
    │   ├── graph_utils.py             # smiles_to_pyg, get_maccs
    │   ├── differential_attention.py  # DifferentialCrossAttention
    │   ├── utils.py                   # set_seed, load_dataset
    │   ├── Step1_preprocess.py        # 1-A: FP 전처리 / 1-B: PyG+MACCS 캐시
    │   ├── Step2_pretrain.py          # GraphMACCSEncoder 학습 (Stage 1)
    │   ├── Step3_stacking.py          # Feature 추출 + Stacking + 평가 (env1)
    │   ├── Step4_xai.py               # DiffAttn XAI 히트맵
    │   └── Step_CV.py                 # 10-Fold CV 전체 파이프라인 (env2)
    │
    ├── data/             # env1 original 중간 결과물 (Step1 출력)
    ├── data_clean/       # env1 clean 중간 결과물
    ├── outputs/          # env1 original 최종 결과
    ├── outputs_clean/    # env1 clean 최종 결과
    ├── outputs_cv/       # env2 original 최종 결과
    └── outputs_cv_clean/ # env2 clean 최종 결과
```

---

## 사전 준비

### 1. 필수 데이터 파일 확인

```
Origin_StackDILI/Data/Dataset.csv
Origin_StackDILI/Code/Dataset_feature.csv
```

파일이 없는 경우 `env-new` 브랜치에서 복원:

```bash
git show env-new:Data/Dataset.csv > Data/Dataset.csv
git show env-new:Code/Dataset_feature.csv > Code/Dataset_feature.csv
```

### 2. Docker 설치 확인

```bash
docker --version
```

Docker Desktop (Mac/Windows) 또는 Docker Engine (Linux)이 실행 중이어야 합니다.

---

## Docker 이미지 빌드

**최초 1회만 실행** (약 5~10분, 이미지 크기 ~3 GB)

```bash
cd Origin_StackDILI
bash run.sh build
```

빌드 완료 확인:
```bash
docker images dgudili
```

빌드 포함 내용:
- Python 3.10-slim
- PyTorch CPU (>=2.6)
- scikit-learn==1.7.1, numpy==1.26.4 (버전 고정)
- XGBoost, RDKit, Transformers
- ChemBERTa-77M-MLM 사전 다운로드 (첫 실행 속도 개선)

> **팀원 공유 시**: 이미지 파일로 공유하면 빌드 없이 바로 사용 가능합니다.
> ```bash
> # 내보내기
> docker save dgudili:latest | gzip > dgudili.tar.gz
> # 불러오기
> docker load < dgudili.tar.gz
> ```

---

## 실험 환경 설명

| 환경 | 데이터 | 평가 방식 | 샘플 수 |
|---|---|---|---|
| **env1 / original** | 원본 전체 | Fixed Split | train 1,398 / test 452 |
| **env1 / clean** | 중복 제거 | Fixed Split | train 1,187 / test 452 |
| **env2 / original** | 원본 전체 | 10-Fold CV | 전체 1,850개 |
| **env2 / clean** | 중복 제거 | 10-Fold CV | 전체 1,639개 |

- **original**: train(non-DILIrank) / test(DILIrank) 고정 분할, StackDILI와 동일 조건
- **clean**: train ∩ test 중복 분자 211개를 train에서 제거 → 더 엄격한 평가
- **env1**: Step1 → Step2 → Step3 순서로 실행
- **env2**: env1에서 학습된 encoder를 재사용 + 전체 데이터 10-Fold Stacking CV

> env2를 실행하려면 env1(동일 데이터 버전)을 먼저 실행해 `pretrained_encoder.pt`를 생성해야 합니다.

---

## 실행 명령어

모든 명령어는 `Origin_StackDILI/` 루트에서 실행합니다.

```bash
cd Origin_StackDILI
```

### 전체 파이프라인 실행

```bash
# env1 — Fixed Split, 원본 데이터
bash run.sh run env1

# env1 — Fixed Split, clean 데이터  (make_clean_data.py 자동 실행)
bash run.sh run-clean env1

# env2 — 10-Fold CV, 원본 데이터  (env1 original 먼저 실행 필요)
bash run.sh run env2

# env2 — 10-Fold CV, clean 데이터  (env1 clean 먼저 실행 필요)
bash run.sh run-clean env2
```

### 개별 스텝 실행 (디버깅용, env1 original 기준)

```bash
bash run.sh step1   # Step1: FP 전처리 (StandardScaler fit)
bash run.sh step2   # Step2: E2E_MHAResidualEncoder 학습
bash run.sh step3   # Step3: Feature 추출 + Stacking + 평가
```

### Docker 컨테이너 직접 진입

```bash
bash run.sh shell
```

---

## 파이프라인 상세

### Step 1 — FP 전처리 (`Step1_preprocess.py`)
- train(non-DILIrank) 기준으로 `StandardScaler` fit
- test(DILIrank)는 transform only (leakage 방지)
- 출력: `data/fp_scaler.pkl`, `data/fp_scaled_{train,test}.npy`

### Step 2 — GraphMACCSEncoder 학습 (`Step2_pretrain.py`)
- ChemBERTa-77M-MLM last-layer: lr=1e-4
- GraphSAGE + DifferentialCrossAttention (나머지 파라미터): lr=3e-4
- 조기종료: val AUC 기준, patience=30, max 200 epochs
- 출력: `outputs/pretrained_graph_encoder.pt`

### Step 3 — Stacking + 평가 (`Step3_stacking.py`, env1 전용)
- Frozen encoder로 train/test 모두 32-dim 피처 추출
- Base models: RF / ET / HistGB / XGB (n_estimators=300)
- 5-fold OOF로 stacking 학습 → LR meta classifier
- OOF에서 MCC 최적 threshold 탐색 (0.10~0.90)
- 출력: `outputs/results_stacking.csv`

### Step CV — 10-Fold CV (`Step_CV.py`, env2 전용)
- Step2에서 학습된 encoder를 고정(frozen)으로 재사용
- 전체 데이터에 대해 32-dim 피처 한 번 추출 → 10-fold split
- encoder가 train set 기반 학습 → train 샘플 CV test fold 시 경미한 낙관 편향 존재
- 각 fold마다 5-fold inner OOF stacking + LR meta + OOF MCC threshold
- 출력: `outputs_cv/results_cv.csv`

---

## Docker 없이 로컬에서 직접 실행

Docker 없이 로컬 Python 환경에서도 실행 가능합니다.

### 패키지 설치

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 scikit-learn==1.7.1 xgboost transformers pandas rdkit
```

재현성을 위해 `scikit-learn==1.7.1`, `numpy==1.26.4`는 버전을 맞추는 것을 권장합니다.

### 실행

```bash
cd Origin_StackDILI

# 환경 변수 설정 (clean 데이터는 USE_CLEAN_DATA=1)
export STACKDILI_ROOT=$(pwd)
export USE_CLEAN_DATA=0

# env1
python DGUDILI_2026/src/Step1_preprocess.py
python DGUDILI_2026/src/Step2_pretrain.py
python DGUDILI_2026/src/Step3_stacking.py

# env2 (env1 실행 후)
python DGUDILI_2026/src/Step_CV.py
```

Windows (PowerShell):
```powershell
$env:STACKDILI_ROOT = (Get-Location).Path
$env:USE_CLEAN_DATA = "0"
```

---

## 환경 변수

`config.py`에서 사용하는 환경 변수:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `STACKDILI_ROOT` | `../` (프로젝트 루트) | `Data/`, `Code/` 폴더 위치 |
| `USE_CLEAN_DATA` | `0` | `1`이면 clean 데이터 사용 |

Docker 실행 시 `run.sh`가 자동으로 설정합니다. 직접 실행 시에는 수동 지정 필요:

```bash
export STACKDILI_ROOT=/path/to/Origin_StackDILI
export USE_CLEAN_DATA=1
python DGUDILI_2026/src/Step1_preprocess.py
```

---

## 주요 하이퍼파라미터 (`config.py`)

```python
K          = 32      # encode() 출력 차원 (Stacking 입력)
D_MODEL    = 64      # DiffAttn 내부 차원
NUM_HEADS  = 4       # Differential attention 헤드 수
DROPOUT    = 0.3
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

# GraphMACCSEncoder 전용
MACCS_DIM      = 167   # MACCSkeys 차원
MAX_ATOMS      = 100   # to_dense_batch 패딩 기준
SAGE_LAYERS    = 2
SAGE_HIDDEN    = 64
ATOM_FEAT_DIM  = 43

BATCH_SIZE     = 16
MAX_LENGTH     = 256
SEED           = 42
EPOCHS         = 200
PATIENCE       = 30
LR_CHEM        = 1e-4   # ChemBERTa last-layer lr
LR_OTHER       = 3e-4   # GraphSAGE + DiffAttn lr
```

---

## 실험 결과

### env1 — Fixed Split (Test: DILIrank N=452, GraphMACCSEncoder)

| 데이터 | AUC | MCC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Original | 0.9224 | 0.7270 | 0.8426 | 0.9022 | 0.8358 |

> StackDILI 목표: AUC 0.9736, MCC 0.8304, F1 0.9010

### 참고: 이전 모델 결과 (E2E_MHAResidualEncoder)

| 데이터 | AUC | MCC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Original (env1) | 0.9182 | 0.6670 | 0.8099 | 0.8913 | 0.7873 |
| Clean (env1) | 0.8130 | 0.4675 | 0.7067 | 0.8315 | 0.6418 |

---

## 재현성 주의사항

같은 코드여도 머신 환경에 따라 수치가 소폭 달라질 수 있습니다.

| 원인 | 영향 스텝 |
|---|---|
| PyTorch CPU 백엔드 차이 (MKL vs OpenBLAS) | Step2 학습 |
| XGBoost 버전별 트리 분할 차이 | Step3, Step CV |
| sklearn 버전별 HistGB 내부 차이 | Step3, Step CV |

Docker 이미지 버전 확인:
```bash
docker run --rm dgudili:latest python -c "
import sklearn, xgboost, torch, numpy
print('sklearn :', sklearn.__version__)
print('xgboost :', xgboost.__version__)
print('torch   :', torch.__version__)
print('numpy   :', numpy.__version__)
"
```

`Dockerfile`에 `scikit-learn==1.7.1`, `numpy==1.26.4`가 고정되어 있으므로 같은 이미지를 사용하면 재현성이 보장됩니다.

---

## 자주 발생하는 오류

### `Missing: .../Data/Dataset.csv` 또는 `Dataset_feature.csv`
데이터 파일이 없는 경우입니다.
```bash
git show env-new:Data/Dataset.csv > Data/Dataset.csv
git show env-new:Code/Dataset_feature.csv > Code/Dataset_feature.csv
```

### `Missing: pretrained_encoder.pt` (env2 실행 시)
Step2가 아직 실행되지 않아 encoder 파일이 없는 경우입니다.
```bash
bash run.sh run env1       # original의 경우
bash run.sh run-clean env1 # clean의 경우
```

### `AssertionError: Missing: .../outputs/pretrained_encoder.pt`
env2를 env1보다 먼저 실행한 경우입니다. env1을 먼저 실행하세요.

### `lm_head.* UNEXPECTED` / `pooler.dense.* MISSING` 경고
ChemBERTa를 AutoModel로 로드할 때 나오는 정상 경고입니다. 무시해도 됩니다.

### `KMP: Initializing libiomp5.dylib` 오류
`KMP_DUPLICATE_LIB_OK=TRUE` 환경 변수가 설정되지 않은 경우입니다.
Docker 실행 시에는 `run.sh`가 자동 설정합니다.
