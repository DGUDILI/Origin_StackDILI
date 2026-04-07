# DGUDILI (2026) — 실행 가이드

---

## 알고리즘 흐름도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         입력: SMILES 문자열                              │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────────┐
          ▼                 ▼                       ▼
  ┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐
  │  ChemBERTa    │  │  RDKit 분자 그래프│  │  MACCSkeys (167-bit) │
  │  (77M-MLM)    │  │  PyG Data        │  │  Binary 벡터        │
  │  Tokenize     │  │  원자 피처 43-dim │  │                    │
  └───────┬───────┘  └────────┬────────┘  └──────────┬──────────┘
          │                   │                      │
          ▼                   ▼                      │
  ┌───────────────┐  ┌─────────────────┐             │
  │ CLS token     │  │  atom_proj      │             │
  │ (B, 384)      │  │  Linear(43→64)  │             │
  └───────┬───────┘  └────────┬────────┘             │
          │                   │                      │
          ▼                   ▼                      │
  ┌───────────────┐  ┌─────────────────┐             │
  │ LayerNorm     │  │  SAGEConv × 2   │             │
  │ Linear(384→64)│  │  (64→64)        │             │
  │ chem_feat     │  └────────┬────────┘             │
  │ (B, 64)       │           │                      │
  └───────┬───────┘  ┌────────▼────────┐             │
          │           │  to_dense_batch│             │
          │           │  (B, 100, 64)  │             │
          │           │  + pad_mask    │             │
          │           └────────┬────────┘            │
          │                    │                     │
          │           ┌────────▼────────┐  ┌─────────▼──────────┐
          │           │  node_proj      │  │  Embedding(167, 64) │
          │           │  node_q         │  │  + inactive mask    │
          │           │  (B, 100, 64)  │  │  maccs_kv           │
          │           │  [QUERY]        │  │  (B, 167, 64)       │
          │           └────────┬────────┘  │  [KEY / VALUE]      │
          │                    └─────────┐ └──────────┬──────────┘
          │                              ▼            │
          │                   ┌──────────────────────┐│
          │                   │ Differential Cross-  ││
          │                   │ Attention            ◄┘
          │                   │ (4 heads, λ per head)│
          │                   │                      │
          │                   │ attn_out (B,100,64)  │
          │                   │ attn_wts (B,100,167) │ ← XAI 시각화
          │                   └──────────┬───────────┘
          │                              │
          │                   ┌──────────▼───────────┐
          │                   │  Masked Mean Pooling  │
          │                   │  (pad_mask 기반)       │
          │                   │  graph_feat (B, 64)   │
          │                   └──────────┬────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌──────────────────────┐
              │  Concat Fusion        │
              │  [chem_feat|graph_feat│
              │  (B, 128) → Linear   │
              │  → LayerNorm (B, 64) │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  MLP Bottleneck      │
              │  Linear(64→32)→GELU  │
              │  → Dropout           │
              │  encode_out (B, 32)  │  ← Stacking 입력 피처
              └──────────┬───────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼ (Stage 1)                 ▼ (Stage 2)
  ┌─────────────────┐       ┌─────────────────────────┐
  │  head(32→1)     │       │  5-Fold OOF Stacking    │
  │  BCEWithLogits  │       │  RF / ET / HistGB / XGB │
  │  val_AUC 조기종료│       │  → LogisticRegression   │
  └─────────────────┘       │  → 최종 예측 (DILIrank)  │
                            └─────────────────────────┘
```

---

## 파일 구조

```
DGUDILI_2026/
│
├── config.yaml              ← 기본 실험 설정 (원본 데이터)
├── config_clean.yaml        ← 중복제거 데이터 실험 설정
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                ← 재사용 가능한 라이브러리 모듈
│   │   ├── __init__.py
│   │   ├── config.py        ← Cfg dataclass + load_cfg(yaml)
│   │   ├── dataset.py       ← MolBatch, MolGraphDataset, collate_fn
│   │   ├── model.py         ← GraphMACCSEncoder
│   │   ├── differential_attention.py  ← DifferentialCrossAttention
│   │   ├── graph_utils.py   ← smiles_to_pyg, get_maccs, MACCS_NAMES
│   │   └── utils.py         ← set_seed, load_dataset
│   │
│   └── pipeline/            ← 실행 스크립트
│       ├── __init__.py
│       ├── preprocess.py    ← Step 1: FP 전처리 + Graph/MACCS 캐시 생성
│       ├── train.py         ← Stage 1 (pretrain) / Stage 2 (stack) / CV
│       └── xai.py           ← Differential Attention XAI 히트맵
│
├── data/
│   ├── original/            ← 원본 데이터 전처리 결과
│   │   ├── train_graphs.pt  ← [{pyg, maccs, label, smiles}, ...]
│   │   ├── test_graphs.pt
│   │   ├── fp_full_train/test.npy
│   │   ├── y_train/test.npy
│   │   ├── smiles_train/test.npy
│   │   ├── scalers.pkl
│   │   ├── fp_feature_names.json
│   │   └── graph_*_tokens_42.pt   ← tokenizer 캐시
│   │
│   └── clean/               ← 중복제거 데이터 전처리 결과
│       ├── fp_full_train/test.npy
│       ├── y_train/test.npy
│       ├── smiles_train/test.npy
│       ├── scalers.pkl
│       └── fp_feature_names.json
│       (train/test_graphs.pt 없음 → preprocess.py --config config_clean.yaml 실행)
│
└── outputs/
    ├── original/            ← 원본 데이터 학습 결과
    │   ├── pretrained_graph_encoder.pt
    │   ├── results.csv
    │   └── xai_*.png        ← XAI 시각화 저장 위치
    │
    ├── clean/               ← 중복제거 데이터 학습 결과
    │   └── results.csv
    │
    └── cv/                  ← Cross-Validation 결과
        ├── results_cv.csv         ← 원본 데이터 10-Fold CV
        └── results_cv_clean.csv   ← 중복제거 데이터 10-Fold CV
```

---

## 각 파일 역할 요약

| 파일 | 용도 |
|------|------|
| `src/core/config.py` | 하이퍼파라미터 정의 + YAML 로더. `load_cfg("config.yaml")` → `Cfg` 반환 |
| `src/core/dataset.py` | `MolGraphDataset` (graph 캐시 로드 + tokenize), `collate_fn`, `MolBatch` |
| `src/core/model.py` | `GraphMACCSEncoder` 클래스 (ChemBERTa + GraphSAGE + DiffAttn) |
| `src/core/differential_attention.py` | `DifferentialCrossAttention` — head-wise λ, GroupNorm, XAI 어텐션 반환 |
| `src/core/graph_utils.py` | `smiles_to_pyg`, `get_maccs`, `MACCS_NAMES` 167개 매핑 |
| `src/core/utils.py` | `set_seed`, `load_dataset` (CSV 로드) |
| `src/pipeline/preprocess.py` | Step 1: raw CSV → FP .npy + graph .pt 캐시 생성 |
| `src/pipeline/train.py` | `--stage pretrain` / `--stage stack` / `--stage cv` 통합 실행 |
| `src/pipeline/xai.py` | 학습된 모델로 단일 분자 XAI 히트맵 생성 |

---

## 실행 순서

### 1. 전처리 (최초 1회)

```bash
cd DGUDILI_2026
python src/pipeline/preprocess.py
# 중복제거 실험 시:
python src/pipeline/preprocess.py --config config_clean.yaml
```

### 2. Encoder 사전학습 (Stage 1)

```bash
python src/pipeline/train.py --stage pretrain
# 중복제거:
python src/pipeline/train.py --stage pretrain --config config_clean.yaml
```

출력: `outputs/original/pretrained_graph_encoder.pt`

### 3. Stacking 평가 (Stage 2)

```bash
python src/pipeline/train.py --stage stack
```

출력: `outputs/original/results.csv`

### 4. 10-Fold CV

```bash
python src/pipeline/train.py --stage cv
```

출력: `outputs/original/results_cv.csv`

### 5. XAI 히트맵

```bash
python src/pipeline/xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
python src/pipeline/xai.py --smiles "..." --top_k 20 --mol
```

---

## 설정 파일 (config.yaml)

```yaml
data_dir: data/original      # 전처리 결과 위치
output_dir: outputs/original # 체크포인트 + 결과 저장 위치
data_csv: Data/Dataset.csv
feat_csv: Code/Dataset_feature.csv

k: 32          # encode() 출력 차원 (Stacking 입력)
d_model: 64    # GraphSAGE + DiffAttn hidden dim
num_heads: 4
sage_layers: 2
sage_hidden: 64
...
```

`config_clean.yaml`로 복사하고 `data_dir`, `output_dir`, `data_csv`, `feat_csv`만 변경하면 중복제거 실험 가능.

---

## 실험 결과 (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| **GraphMACCSEncoder** | **0.9224** | **0.7270** | **0.8426** | 0.9022 | 0.8358 |

---

## 핵심 설계 결정

**val_AUC 조기종료**: val_loss는 epoch 2~5에 최솟값 도달 후 즉시 증가 (train/test 분포 이동 탓). val_loss 기준 checkpoint는 사실상 미학습 상태 → **반드시 val_AUC (mode=max) 사용**.

**Windows num_workers=0**: DataLoader 멀티프로세싱 비활성화. on-the-fly RDKit 변환 대신 Step 1에서 `.pt` 캐시 파일로 미리 저장.

**Differential Cross-Attention**: 표준 softmax 대비 배경 노이즈 소거 → XAI 히트맵에서 중요 원자-MACCS 상관관계가 희소하게 강조됨.
