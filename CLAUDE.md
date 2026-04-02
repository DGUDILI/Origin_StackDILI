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

**Goal:** DILI(Drug-Induced Liver Injury) 예측 — ChemBERTa 임베딩 + 분자 지문(FP)의 Cross-Attention 융합 모델

**Current Best Pipeline:** FTV6StyleEncoder (full FP soft projection) + Stacking OOF (RF/ET/HistGB/XGB → LR meta)

**ChemBERTa 모델:** DeepChem/ChemBERTa-77M-MLM, hidden_dim=**384**, layers=**3**
- "77M"은 MLM 사전학습 SMILES 토큰 수를 의미하며, 파라미터 수가 아님 (실제 ~3.5M)

## Experiment Results (DILIrank test, N=452)

| 모델 | 조기종료 기준 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-------------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | — | 0.9736 | 0.8304 | 0.9010 | 0.9402 | 0.8993 |
| FTV6StyleEncoder (d_v=1) | val_AUC | **0.9196** | 0.6682 | 0.8106 | 0.9185 | 0.7612 |
| E2E_FTV6 (d_v=1) | val_AUC | **0.9219** | **0.7270** | 0.8426 | 0.9022 | 0.8358 |
| FTV6StyleEncoder (d_v=4) | val_loss | 0.8252 | 0.5316 | 0.7377 | 0.9402 | 0.5821 |
| E2E_FTV6 (d_v=4) | val_loss | 0.9006 | 0.6522 | 0.8010 | 0.8641 | 0.7985 |

**현재 최고:** E2E_FTV6 (d_v=1, val_AUC) — AUC 0.9219, MCC 0.7270

## Known Issues & Fix Status

| 우선순위 | 문제 | 상태 |
|----------|------|------|
| P0 | Fix-1: Early Stopping 데이터 누수 (test AUC 직접 사용) | ✅ 수정완료 |
| P0 | Fix-2: 재현성 시드 불완전 (cuda seed 누락) | ✅ 수정완료 |
| P0 | Fix-3: SelectKBest(k=16) 하드컷 → Soft Linear Projection | ✅ 완료 (FTV6StyleEncoder) |
| P1 | Arch-1: d_v=1 병목 → d_v=4 확장 | ⚠️ 구현완료, val_AUC 기준 재실험 필요 |
| P1 | Arch-2: LR → Stacking OOF (RF/ET/HistGB/XGB) | ✅ 완료 |
| P1 | E2E: ChemBERTa last-layer 파인튜닝 통합 | ✅ 완료 (Step3/4_e2e) |
| P2 | Method-1: val_AUC vs val_loss 조기종료 비교 | ✅ 실험완료, val_AUC 우위 확인 |
| P3 | Domain Adversarial Training (분포 이동 대응) | 미수행 |

## Critical Lessons (조기종료 기준)

**val_loss 조기종료는 이 문제에서 역효과:**
- Train(다중 DB 혼합) vs Test(DILIrank) 분포 이동으로 인해 val_loss는 epoch 2~5에 최소값 도달 후 즉시 증가
- val_loss 기준으로 저장된 checkpoint = 사실상 미학습 상태 → test AUC 급락
- val_AUC 기준: epoch 15~20까지 학습 허용 → test AUC 0.92+ 유지
- **결론: 조기종료는 val_AUC(mode=max) 사용. val_loss는 이 구조에 부적합.**

**분포 이동 패턴:**
- val_AUC ≈ 0.63~0.70 (train 동일 분포)
- test_AUC ≈ 0.90~0.93 (DILIrank — 더 깔끔하게 분리됨)
- 역설적으로 DILIrank가 더 쉬운 데이터. train 혼합 DB에 노이즈가 많은 것이 원인.

## Fix Details

**FTV6StyleEncoder (model.py):**
- SelectKBest 제거 → FP 425-dim 전체를 Linear(425, k) soft projection으로 학습
- W_V: Linear(1, d_v) — d_v=1(기존) vs d_v=4(확장판)
- head: Linear(k * d_v, 1) — encode() returns (B, k*d_v)

**E2E_FTV6StyleEncoder (model.py):**
- ChemBERTa-77M-MLM 내장, 3레이어 중 layer[2](마지막)만 unfreeze
- 차등 LR: ChemBERTa layer[2] = 1e-4, CrossAttn/Proj = 3e-4
- encode(input_ids, attention_mask, x_fp) → (B, k*d_v)

**Step3_pretrain.py / Step3_e2e_pretrain.py:**
- Train 85% / Val 15% stratified split, Early stopping = val_AUC (mode=max)
- Seed 완전 고정: random, numpy, torch, cuda, cudnn

## Performance Roadmap

```
완료(P0+Arch)    d_v=4(val_AUC)    P3(Domain Adapt.)
AUC 0.9219    →  AUC 0.93~0.95  →  AUC 0.95+
MCC 0.7270    →  MCC 0.75~0.80  →  MCC 0.80+
```

## Next Steps

1. **즉시:** FTV6 + E2E d_v=4 를 val_AUC 기준으로 재실행하여 d_v=4 순수 효과 측정
2. **이후:** d_v=4 결과가 MCC 0.75 미만이면 P3 (Domain Adversarial Training) 적용
3. P3: test SMILES(레이블 없이)를 domain 적응 학습에 활용, 분포 이동 완화
