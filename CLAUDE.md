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

**Goal:** DILI(Drug-Induced Liver Injury) 예측 — ChemBERTa(E2E) + 분자 지문(FP) Cross-Attention 융합 모델

**Current Pipeline:** E2E_FTV6StyleEncoder (d_v=1, val_AUC early stopping) + Stacking OOF (RF/ET/HistGB/XGB → LR meta)

**ChemBERTa 모델:** DeepChem/ChemBERTa-77M-MLM, hidden_dim=**384**, layers=**3**
- "77M"은 MLM 사전학습 SMILES 토큰 수를 의미하며, 파라미터 수가 아님 (실제 ~3.5M)
- E2E: ChemBERTa가 모델 내부에 내장 (SMILES를 실시간으로 처리, 사전 추출 불필요)

## Pipeline Structure

```
src/
  model.py           — E2E_FTV6StyleEncoder 단일 모델
  Step1_preprocess.py — FP StandardScaler 전처리 → data/ 저장
  Step2_pretrain.py  — E2E 학습 (Stage 1) → outputs/pretrained_encoder.pt
  Step3_stacking.py  — Feature 추출 + Stacking OOF (Stage 2) → outputs/results.csv

data/
  fp_full_train/test.npy   — 전체 FP (425-dim), StandardScaler 적용
  y_train/test.npy         — 레이블
  scalers.pkl              — FP scaler 저장
  fp_feature_names.json    — FP 피처 이름

outputs/
  pretrained_encoder.pt    — Step2 학습된 encoder 가중치
  results.csv              — 최종 평가 결과
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

**E2E_FTV6StyleEncoder (model.py):**
- SMILES → ChemBERTa(layer[2] unfreeze, lr=1e-4) → CLS (B, 384) → chem_proj → (B, 16)
- FP 425-dim → fp_proj: Linear(425,16)+LayerNorm → (B, 16)
- Cross-Attention: Q(ChemBERTa) × K/V(FP) → (B, 16, 1) → flatten → (B, 16)
- Stage 1: head Linear(16,1) + BCEWithLogitsLoss
- Stage 2: encode() → (B, 16) fused feature → Stacking OOF

**Step2_pretrain.py:**
- Train 85% / Val 15% stratified split
- Early stopping: val_AUC (mode=max, patience=30)
- Scheduler: ReduceLROnPlateau(val_AUC, mode=max, factor=0.5, patience=8)
- Seed 완전 고정: random, numpy, torch, cuda, cudnn

## Next Steps

**P3: Domain Adversarial Training (분포 이동 대응)**
- 현재 gap: AUC -0.051, MCC -0.103 vs StackDILI
- test SMILES(레이블 없이)를 domain 적응 학습에 활용
- train(다중 DB 혼합) ↔ test(DILIrank) 분포 이동 완화 목표
- 목표: AUC 0.95+, MCC 0.80+
