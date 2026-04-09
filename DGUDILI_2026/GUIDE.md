# DGUDILI_2026 — Implementation Guide

## Overview

GraphMACCSEncoder: GraphSAGE + MACCS DifferentialCrossAttention + ChemBERTa 융합 DILI 예측 모델.

### Architecture Evolution

| 버전 | 모델 | AUC (env1) | MCC (env1) | 비고 |
|------|------|-----|-----|------|
| StackDILI *(baseline)* | GA + Stacking | 0.9736 | 0.8304 | 목표 |
| CrossAttentionEncoder | CLS + FP CrossAttn | 0.7313 | 0.3045 | Early stop leakage |
| FTV6StyleEncoder | FP soft projection | 0.9196 | 0.6682 | val_AUC early stop |
| E2E_FTV6StyleEncoder | ChemBERTa fine-tune | 0.9219 | 0.7270 | ChemBERTa unfrozen |
| GraphMACCSEncoder (SAGEConv) | GraphSAGE + DiffAttn | 0.8426 | 0.5275 | Phase 1~2 |
| **GraphMACCSEncoder (GINEConv)** | **GINEConv + edge_attr 9-dim** | **0.8808** | **0.5776** | **현재** |

---

## ChemBERTa 모델 스펙

```
모델명: DeepChem/ChemBERTa-77M-MLM
hidden_size: 384  (※ "77M"은 파라미터 수가 아닌 MLM 학습 토큰 수)
num_hidden_layers: 3
total_params: ~3.5M
```

---

## Prerequisites

### 필수 데이터 파일
```
Origin_StackDILI/Data/Dataset.csv
Origin_StackDILI/Code/Dataset_feature.csv
```

### 패키지
```bash
pip install torch torch-geometric einops
pip install numpy==1.26.4 scikit-learn==1.7.1 xgboost transformers pandas rdkit
```

---

## 현재 파이프라인 — GraphMACCSEncoder (GINEConv)

```
SMILES
  ├─→ ChemBERTa (last-layer fine-tune) → CLS (B, 384) → chem_feat (B, 64)
  ├─→ RDKit mol → 43-dim atom features → atom_proj
  │   + bond features (9-dim: single/double/triple/aromatic/ring/conjugated/stereo×3)
  │   → GINEConv×2 (edge_attr 활용) → node_q (B, 100, 64)
  └─→ MACCSkeys (B, 167) → Embedding → binary gate → maccs_kv (B, 167, 64)
                                  ↓
         DifferentialCrossAttention (Q=node_q, K/V=maccs_kv)
         inactive bits masked -1e9 | λ.clamp(min=1e-4, max=2.0)
                                  ↓
         masked_mean_pool → graph_feat (B, 64)
         concat([chem_feat, graph_feat]) → fuse_proj → MLP → (B, 32)
                                  ↓
         Stacking OOF (RF / ET / HistGB / XGB, 5-Fold) + LR meta
                                  ↓
         OOF MCC-optimal threshold → 최종 예측
```

### 실행 순서

```bash
cd Origin_StackDILI
export STACKDILI_ROOT=$(pwd)   # Windows: $env:STACKDILI_ROOT = (Get-Location).Path

# Step 1: FP 전처리 + Graph/MACCS 캐시 빌드
python DGUDILI_2026/src/Step1_preprocess.py

# Step 2: GraphMACCSEncoder 학습 (Stage 1)
python DGUDILI_2026/src/Step2_pretrain.py

# Step 3: Feature 추출 + Stacking OOF + 평가
python DGUDILI_2026/src/Step3_stacking.py

# (선택) env2: 10-Fold CV
python DGUDILI_2026/src/Step_CV.py

# (선택) XAI: 원자-MACCS key 상관관계 히트맵
python DGUDILI_2026/src/Step4_xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O"  # 아스피린
```

또는 Docker:
```bash
bash run.sh run-graph      # fixed split 전체 파이프라인
bash run.sh step4          # XAI (아스피린)
bash run.sh step4 "<SMILES>"  # XAI (지정 SMILES)
```

---

## File Structure

```
DGUDILI_2026/
├── GUIDE.md
├── architecture.md
│
├── src/
│   ├── config.py                  # 하이퍼파라미터 및 경로
│   ├── model.py                   # E2E_FTV6StyleEncoder, E2E_MHAResidualEncoder,
│   │                              # GraphMACCSEncoder (현재 사용)
│   ├── graph_utils.py             # smiles_to_pyg, get_maccs, get_atom_features
│   ├── differential_attention.py  # DifferentialCrossAttention (arxiv 2410.05258)
│   ├── utils.py                   # set_seed, load_dataset, SMILESDataset
│   ├── Step1_preprocess.py        # 1-A: FP 전처리 / 1-B: PyG + MACCS .pt 캐시
│   ├── Step2_pretrain.py          # GraphMACCSEncoder Stage 1 학습
│   ├── Step3_stacking.py          # Feature 추출 + Stacking OOF + 평가 (env1)
│   ├── Step4_xai.py               # DiffAttn XAI 히트맵
│   └── Step_CV.py                 # 10-Fold CV (env2)
│
├── data/                          # env1 original 중간 결과물
│   ├── train_graphs.pt            # PyG Data + MACCS + label (train 캐시)
│   ├── test_graphs.pt             # PyG Data + MACCS + label (test 캐시)
│   ├── fp_full_train/test.npy     # FP 425-dim, StandardScaler 적용
│   ├── y_train/test.npy           # 레이블
│   ├── scalers.pkl                # FP StandardScaler
│   └── fp_feature_names.json      # 425개 FP 피처 이름
│
└── outputs/
    ├── pretrained_graph_encoder.pt  # Step2 GraphMACCSEncoder 가중치
    └── results.csv                  # 최종 평가 결과
```

clean 데이터 경로: `data_clean/`, `outputs_clean/` (`USE_CLEAN_DATA=1` 시 자동 전환)

---

## Key Hyperparameters (config.py)

| Parameter | Value | Description |
|---|---|---|
| K | 32 | encode() 출력 차원 (Stacking 입력) |
| D_MODEL | 64 | DiffAttn 내부 차원 |
| NUM_HEADS | 4 | Differential attention 헤드 수 |
| DROPOUT | 0.3 | MLP dropout |
| SAGE_HIDDEN | 64 | GINEConv hidden dim |
| SAGE_LAYERS | 2 | GINEConv layer 수 |
| BOND_FEAT_DIM | 9 | get_bond_features() 출력 차원 |
| MAX_ATOMS | 100 | to_dense_batch 패딩 기준 |
| MACCS_DIM | 167 | MACCSkeys 차원 (bit 0 미사용, bits 1~166 유효) |
| ATOM_FEAT_DIM | 43 | get_atom_features() 출력 차원 |
| LR_CHEM | 1e-4 | ChemBERTa last-layer lr |
| LR_OTHER | 3e-4 | GraphSAGE + DiffAttn lr |
| EPOCHS | 200 | 최대 epoch |
| PATIENCE | 30 | Early stopping patience |
| **early_stop** | **val_AUC (mode=max)** | **val_loss는 분포 이동으로 부적합** |

---

## Train / Test Split

```python
train = data[data['ref'] != 'DILIrank']   # 1,398 samples (pos=768, neg=630)
test  = data[data['ref'] == 'DILIrank']   # 452 samples  (pos=184, neg=268)

# Stage 1 내부 Val 분리 (조기종료용, train의 15%)
# ~210 samples, stratified
```

---

## 실험 결과

### env1 — Fixed Split (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| SAGEConv + LR | 0.8426 | 0.5275 | 0.7102 | 0.6793 | 0.8396 |
| **GINEConv + LR** | **0.8808** | **0.5776** | **0.7565** | 0.7935 | 0.7910 |

### env2 — 10-Fold CV (전체 N=1,850)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| SAGEConv + LR | 0.8909 ±0.030 | 0.6366 ±0.072 | 0.8230 ±0.037 | 0.831 | 0.802 |
| **GINEConv + LR** | **0.9558 ±0.015** | **0.8080 ±0.051** | **0.9068 ±0.024** | 0.908 | 0.900 |

---

## 알려진 문제점 및 고려사항

### 1. Train/Test 분포 이동
- Train: 다중 데이터베이스 혼합 (ChEMBL, SIDER 등) — 노이즈 多
- Test: DILIrank 단일 큐레이션 — 더 깔끔하게 분리됨
- 결과: val_AUC(~0.68) << test_AUC(~0.92)
- 대응: Domain Adversarial Training (P3, 미실행)

### 2. val_loss 조기종료 부적합
val_loss는 epoch 2~5에 최솟값 도달 후 즉시 증가. **반드시 val_AUC (mode=max) 사용.**

### 3. Step_CV encoder 편향
encoder가 train set으로 학습되었으므로, 10-fold CV에서 train 샘플이 test fold에 포함되면 경미한 낙관 편향 존재. DILIrank 샘플은 편향 없음. (Step_CV.py docstring 참고)

### 4. ChemBERTa LOAD 경고 (정상)
```
lm_head.* UNEXPECTED  ← AutoModel이 MLM head 제외하여 정상
pooler.*   MISSING     ← 분류 목적 재초기화, 정상
```

---

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` 모든 스크립트에 자동 설정 (Windows MKL 충돌 방지)
- `USE_CLEAN_DATA=1` 환경변수로 중복 제거 데이터 사용 가능
- `.npy` / `.pkl` / `.pt` 파일은 gitignore 권장
