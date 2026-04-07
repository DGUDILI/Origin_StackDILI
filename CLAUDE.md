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

**Goal:** DILI(Drug-Induced Liver Injury) 예측 — GraphSAGE + MACCS DifferentialCrossAttention + ChemBERTa 융합 모델

**Current Pipeline:** GraphMACCSEncoder (val_AUC early stopping) + Stacking OOF (RF/ET/HistGB/XGB → LR meta)

**ChemBERTa 모델:** DeepChem/ChemBERTa-77M-MLM, hidden_dim=**384**, layers=**3**
- "77M"은 MLM 사전학습 SMILES 토큰 수이며, 파라미터 수가 아님 (실제 ~3.5M)
- E2E: ChemBERTa가 모델 내부에 내장 (SMILES 실시간 처리)

## Pipeline Structure

```
src/
  model.py                  — E2E_FTV6StyleEncoder, E2E_MHAResidualEncoder, GraphMACCSEncoder
  graph_utils.py            — smiles_to_pyg, get_maccs (RDKit 기반)
  differential_attention.py — DifferentialCrossAttention (arxiv 2410.05258 cross-attn 변형)
  Step1_preprocess.py       — 1-A: FP StandardScaler / 1-B: Graph+MACCS .pt 캐시
  Step2_pretrain.py         — GraphMACCSEncoder 학습 → outputs/pretrained_graph_encoder.pt
  Step3_stacking.py         — Feature 추출 + Stacking OOF → outputs/results.csv
  Step4_xai.py              — DiffAttn XAI 히트맵 (원자-MACCS key 상관관계)
  Step_CV.py                — 10-Fold CV (env2)
  config.py                 — 하이퍼파라미터 및 경로

data/
  train_graphs.pt / test_graphs.pt  — PyG 그래프 + MACCS (167-bit) 캐시
  fp_full_train/test.npy            — FP (425-dim), StandardScaler 적용
  y_train/test.npy                  — 레이블
  scalers.pkl                       — FP scaler

outputs/
  pretrained_graph_encoder.pt  — Step2 학습된 GraphMACCSEncoder 가중치
  results.csv                  — 최종 평가 결과
```

## Experiment Results (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| **DGUDILI_2026** | **0.9224** | **0.7270** | **0.8426** | 0.9022 | 0.8358 |

## Critical Lessons (조기종료 기준)

**val_loss 조기종료는 이 문제에서 역효과:**
- Train(다중 DB 혼합) vs Test(DILIrank) 분포 이동으로 인해 val_loss는 epoch 2~5에 최소값 도달 후 즉시 증가
- val_loss 기준 checkpoint = 사실상 미학습 상태 → test AUC 급락
- **결론: 조기종료는 val_AUC(mode=max) 사용. 현재 Step2에 적용됨.**

**분포 이동 패턴:**
- val_AUC ≈ 0.63~0.70 (train 동일 분포)
- test_AUC ≈ 0.90~0.93 (DILIrank — 더 깔끔하게 분리됨)
- 역설적으로 DILIrank가 더 쉬운 데이터. train 혼합 DB에 노이즈가 많은 것이 원인.

## Model Details

**GraphMACCSEncoder (model.py):**
- SMILES → ChemBERTa(last layer unfreeze, lr=1e-4) → CLS (B, 384) → LayerNorm → Linear → chem_feat (B, d_model)
- SMILES → RDKit mol → 43-dim atom features → atom_proj → SAGEConv×2 (sage_hidden=64) → to_dense_batch → node_q (B, MAX_ATOMS, d_model)
- SMILES → MACCSkeys (B, 167) → Embedding(167, d_model) → maccs_kv (B, 167, d_model); inactive bits masked -1e9
- DifferentialCrossAttention(Q=node_q, K/V=maccs_kv) → attn_out (B, MAX_ATOMS, d_model)
- masked_mean_pool → graph_feat (B, d_model) → concat([chem_feat, graph_feat]) → fuse_proj → MLP → encode_out (B, k=32)
- Stage 1: head Linear(32, 1) + BCEWithLogitsLoss
- Stage 2: encode() → (B, 32) fused feature → Stacking OOF

**Step2_pretrain.py:**
- Train 85% / Val 15% stratified split
- Early stopping: val_AUC (mode=max, patience=30)
- Scheduler: ReduceLROnPlateau(val_AUC, mode=max, factor=0.5, patience=8)
- Seed 완전 고정: random, numpy, torch, cuda, cudnn

**Key Hyperparameters (config.py):**
- K=32, D_MODEL=64, NUM_HEADS=4, DROPOUT=0.3
- SAGE_HIDDEN=64, SAGE_LAYERS=2, MAX_ATOMS=100, MACCS_DIM=167, ATOM_FEAT_DIM=43

## Recent Changes (2026-04-08)

- `model.py:300` — MACCS Embedding 주석 수정: "binary mask 곱" → "-1e9 마스킹" (실제 동작과 일치)
- `differential_attention.py:77` — lambda `.clamp(min=1e-4)` 추가: lam < 0 방지, gradient flow 유지

## Next Steps

**P3: Domain Adversarial Training (분포 이동 대응)**
- 현재 gap: AUC -0.051, MCC -0.103 vs StackDILI
- test SMILES(레이블 없이)를 domain 적응 학습에 활용
- train(다중 DB 혼합) ↔ test(DILIrank) 분포 이동 완화 목표
- 목표: AUC 0.95+, MCC 0.80+
