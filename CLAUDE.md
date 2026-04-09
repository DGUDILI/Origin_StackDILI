## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- User instructions always override this file.

## Project Context: DGUDILI_2026

**Goal:** DILI(Drug-Induced Liver Injury) 예측 — GINEConv + MACCS DifferentialCrossAttention + ChemBERTa 융합 모델

**Current Pipeline:** GraphMACCSEncoder (val_AUC early stopping) + Stacking OOF (RF/ET/HistGB/XGB → LR meta)

**ChemBERTa 모델:** DeepChem/ChemBERTa-77M-MLM, hidden_dim=**384**, layers=**3**
- "77M"은 MLM 사전학습 SMILES 토큰 수이며, 파라미터 수가 아님 (실제 ~3.5M)
- E2E: ChemBERTa가 모델 내부에 내장 (SMILES 실시간 처리)

## Pipeline Structure

```
src/
  model.py                  — GraphMACCSEncoder (GINEConv + edge_attr 9-dim)
  graph_utils.py            — smiles_to_pyg (edge_attr 포함), get_maccs (RDKit 기반)
  differential_attention.py — DifferentialCrossAttention (arxiv 2410.05258 cross-attn 변형)
  Step1_preprocess.py       — 1-A: FP StandardScaler / 1-B: Graph+MACCS .pt 캐시 (edge_attr 포함)
  Step2_pretrain.py         — GraphMACCSEncoder 학습 → outputs/pretrained_graph_encoder.pt
  Step3_stacking.py         — Feature 추출 + Stacking OOF → outputs/results.csv
  Step4_xai.py              — DiffAttn XAI 히트맵 (원자-MACCS key 상관관계)
  Step_CV.py                — 10-Fold CV (env2)
  config.py                 — 하이퍼파라미터 및 경로

data/
  train_graphs.pt / test_graphs.pt  — PyG 그래프 + MACCS (167-bit) + edge_attr (9-dim) 캐시
  fp_full_train/test.npy            — FP (425-dim), StandardScaler 적용
  y_train/test.npy                  — 레이블
  scalers.pkl                       — FP scaler

outputs/
  pretrained_graph_encoder.pt  — Step2 학습된 GraphMACCSEncoder 가중치
  results.csv                  — 최종 평가 결과
```

## Experiment Results

### env1 — Fixed Split (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| SAGEConv + LR | 0.8426 | 0.5275 | 0.7102 | 0.6793 | 0.8396 |
| **GINEConv + LR** | **0.8808** | **0.5776** | **0.7565** | 0.7935 | 0.7910 |

### env2 — 10-Fold CV (전체 N=1,850)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| SAGEConv + LR | 0.8909 ±0.030 | 0.6366 ±0.072 | 0.8230 ±0.037 | 0.831 ±0.064 | 0.802 ±0.064 |
| **GINEConv + LR** | **0.9558 ±0.015** | **0.8080 ±0.051** | **0.9068 ±0.024** | 0.908 ±0.025 | 0.900 ±0.041 |

## Critical Lessons (조기종료 기준)

**val_loss 조기종료는 이 문제에서 역효과:**
- Train(다중 DB 혼합) vs Test(DILIrank) 분포 이동으로 인해 val_loss는 epoch 2~5에 최소값 도달 후 즉시 증가
- val_loss 기준 checkpoint = 사실상 미학습 상태 → test AUC 급락
- **결론: 조기종료는 val_AUC(mode=max) 사용. 현재 Step2에 적용됨.**

**분포 이동 패턴:**
- val_AUC ≈ 0.63~0.76 (train 동일 분포)
- test_AUC ≈ 0.88~0.96 (DILIrank — 더 깔끔하게 분리됨)
- 역설적으로 DILIrank가 더 쉬운 데이터. train 혼합 DB에 노이즈가 많은 것이 원인.

**ET meta는 역효과:**
- OOF 피처가 4개(RF/ET/HistGB/XGB)뿐인 상황에서 ET 300그루 → OOF MCC=1.0 과적합
- test Specificity 0.40 붕괴. LR이 4차원 경계에서 일반화 유지.

## Model Details

**GraphMACCSEncoder (model.py):**
- SMILES → ChemBERTa(last layer unfreeze, lr=1e-4) → CLS (B, 384) → LayerNorm → Linear → chem_feat (B, d_model)
- SMILES → RDKit mol → 43-dim atom features → atom_proj → GINEConv×2 (edge_attr 9-dim, sage_hidden=64) → to_dense_batch → node_q (B, MAX_ATOMS, d_model)
- SMILES → MACCSkeys (B, 167) → Embedding(167, d_model) → maccs_kv (B, 167, d_model); 비활성 bit: binary gate로 0벡터화
- DifferentialCrossAttention(Q=node_q, K/V=maccs_kv) → attn_out (B, MAX_ATOMS, d_model)
- masked_mean_pool → graph_feat (B, d_model) → concat([chem_feat, graph_feat]) → fuse_proj → MLP → encode_out (B, k=32)
- Stage 1: head Linear(32, 1) + BCEWithLogitsLoss(pos_weight=n_neg/n_pos)
- Stage 2: encode() → (B, 32) fused feature → Stacking OOF

**Step2_pretrain.py:**
- Train 85% / Val 15% stratified split
- Early stopping: val_AUC (mode=max, patience=30)
- Scheduler: ReduceLROnPlateau(val_AUC, mode=max, factor=0.5, patience=8)
- Seed 완전 고정: random, numpy, torch, cuda, cudnn
- val_loss 계산 시 `val_logits.to(device)` 필수 (pos_weight device 일치)

**Key Hyperparameters (config.py):**
- K=32, D_MODEL=64, NUM_HEADS=4, DROPOUT=0.3
- SAGE_HIDDEN=64, SAGE_LAYERS=2, BOND_FEAT_DIM=9, MAX_ATOMS=100, MACCS_DIM=167, ATOM_FEAT_DIM=43

## Recent Changes (2026-04-10)

- `model.py` — SAGEConv → GINEConv + edge_proj(9→64): bond features (single/double/triple/aromatic/ring/conjugated/stereo×3) 활용
- `graph_utils.py` — smiles_to_pyg에 edge_attr 9-dim (get_bond_features) 추가
- `Step2_pretrain.py:239` — val_loss 계산 시 `val_logits.to(device)` 추가 (GPU/CPU 불일치 버그 수정)
- `Step3_stacking.py` — meta-model ET → LR (ET는 4-dim OOF에서 과적합 확인)
- `config.py` — BOND_FEAT_DIM=9 추가

## 배제된 전략 (재시도 금지)

| 전략 | 결과 | 이유 |
|---|---|---|
| ChemBERTa 2-layer unfreeze | AUC 하락 | 도메인 오버피팅. 1-layer가 적절한 regularization |
| DANN/DAT (λ>0) | AUC 하락 | train/test 분포 차이가 signal quality shift. DILIrank 신호 제거 |
| ET meta | AUC 하락, Specificity 붕괴 | 4-dim OOF 입력에서 과적합 (OOF MCC=1.0) |
| MLP 2-layer | -0.015 | Dropout×2 과도 정규화 |
| BatchNorm → GraphNorm | -0.014 | 소규모 배치에서 역효과 |

## Next Steps

**Phase 4: 비MACCS FP 혼합 (현재 ceiling 도달 시 시도)**
- MACCS(167)는 encoder가 이미 cross-attention 처리 → 제외
- 추가 대상: E-state(79) + CalcCATS(150) + Constitutional(22) + 기타(7) = **258-dim**
- encode_out(32) + fp_non_maccs(258) = **290-dim** → Stacking 입력
- 예상 env1 AUC +0.02~0.04
