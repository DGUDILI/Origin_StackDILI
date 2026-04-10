# DGUDILI_2026 — GraphMACCSEncoder 아키텍처

## 목표

SMILES 문자열로부터 세 가지 표현(그래프 구조, MACCS 지문, 언어 임베딩)을 융합하여 DILI 이진 분류.

---

## 전체 파이프라인

```
Step 1-A: FP 전처리 (StandardScaler, 425-dim)
Step 1-B: PyG 그래프 + MACCS 캐시 저장 (.pt)
Step 2:   GraphMACCSEncoder 학습 (Stage 1, val_AUC early stop)
Step 3:   Feature 추출 + Stacking OOF (Stage 2)
```

```
[SMILES]
   ├─────────────────────────────────────────────────────────┐
   │  ChemBERTa-77M-MLM                                      │
   │  (마지막 레이어 unfreeze, lr=1e-4)                       │
   │  → CLS (B, 384) → LN → Linear(384, 64) → chem_feat      │
   │                                           (B, 64)       │
   ├─────────────────────────────────────────────────────────┤
   │  RDKit mol → 43-dim atom features                       │
   │  + bond features (9-dim: single/double/triple/aromatic/ │
   │    ring/conjugated/stereo×3) → edge_proj(9, 64)         │
   │  → atom_proj Linear(43, 64)                             │
   │  → GINEConv(64, 64, edge_dim=64) × 2 + BN + ReLU       │
   │  → to_dense_batch → (B, 100, 64) + pad_mask            │
   │  → node_proj Linear(64, 64)  → node_q (B, 100, 64)     │
   ├─────────────────────────────────────────────────────────┤
   │  MACCSkeys (B, 167) binary                              │
   │  → Embedding(167, 64)[idx] * maccs (binary gate)        │
   │  → maccs_kv (B, 167, 64)                                │
   │    inactive bits → 0-vector (embedding zeroed out)      │
   └────────────────────────┬────────────────────────────────┘
                            ▼
         DifferentialCrossAttention
         ─────────────────────────────────────────────────────
         Q  = node_q    (B, 100, 64)   ← atom node embeddings
         KV = maccs_kv  (B, 167, 64)   ← MACCS key embeddings

         W_q: (B, 100, 128) → Q1, Q2  [h=4, d_head=16]
         W_k: (B, 167, 128) → K1, K2
         W_v: (B, 167, 64)  → V

         scores1 = Q1 @ K1^T / √16    (B, 4, 100, 167)
         scores2 = Q2 @ K2^T / √16
         [inactive bits already 0-vectors; kv_padding_mask=None]

         λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + 0.8
         λ.clamp(min=1e-4, max=2.0)    ← 음수화 + 포화 방지

         a1 = softmax(scores1)
         a2 = softmax(scores2)
         scores = a1 - λ·a2            ← Differential attention
         out = scores @ V              (B, 4, 100, 16)
         GroupNorm → W_o               (B, 100, 64)
         ─────────────────────────────────────────────────────
                            │
                            ▼
         masked_mean_pool (pad_mask 기반)
         → graph_feat (B, 64)
                            │
                            ▼
         concat([chem_feat, graph_feat])  (B, 128)
         → fuse_proj Linear(128, 64) → LayerNorm
         → MLP: LN → Linear(64, 32) → GELU → Dropout(0.3)
         → encode_out (B, 32)
                            │
                   ┌────────┴────────┐
             Stage 1             Stage 2
          Linear(32, 1)      encode() → (B, 32)
          BCEWithLogitsLoss      Stacking OOF
```

---

## Stage 1: GraphMACCSEncoder 학습

### 텐서 플로우 요약

| 단계 | 입력 shape | 출력 shape | 모듈 |
|------|-----------|-----------|------|
| ChemBERTa | (B, 256) token ids | (B, 384) | AutoModel CLS |
| chem_proj | (B, 384) | (B, 64) | LN → Linear |
| atom_proj | (N_total, 43) | (N_total, 64) | Linear |
| edge_proj | (E, 9) | (E, 64) | Linear |
| GINEConv×2 | (N_total, 64) + edge(E,64) | (N_total, 64) | GINEConv + BN + ReLU |
| to_dense_batch | (N_total, 64) | (B, 100, 64) | PyG dense pad |
| node_proj | (B, 100, 64) | (B, 100, 64) | Linear → node_q |
| MACCS embed | (B, 167) int | (B, 167, 64) | Embedding(167, 64) |
| DiffAttn | Q(B,100,64) KV(B,167,64) | (B, 100, 64) | DifferentialCrossAttention |
| masked_pool | (B, 100, 64) | (B, 64) | mean over valid atoms |
| fuse_proj | (B, 128) | (B, 64) | Linear → LN |
| MLP | (B, 64) | (B, 32) | LN → Linear → GELU → Drop |
| head | (B, 32) | (B, 1) | Linear (Stage 1 only) |

### 학습 설정

| 항목 | 값 |
|------|---|
| ChemBERTa | DeepChem/ChemBERTa-77M-MLM (hidden=384, layers=3) |
| Frozen | embeddings + encoder.layer[0], [1] |
| Unfrozen | encoder.layer[2] (마지막), lr=1e-4 |
| Graph/DiffAttn lr | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping | val_AUC (mode=max, patience=30) |
| Scheduler | ReduceLROnPlateau(val_AUC, mode=max, factor=0.5, patience=8) |
| Loss | BCEWithLogitsLoss |
| Train/Val split | 85% / 15% stratified |

---

## Stage 2: Stacking OOF → LR meta

```
Train 전체 (1,398 샘플)
        │
        ▼
[GraphMACCSEncoder.encode()]  ← frozen (eval mode)
        │
        ▼
 32-dim Fused Features
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  5-Fold Stratified OOF Stacking                          │
│  RF / ET / HistGB / XGB (n_estimators=300)               │
│  OOF probs: (1398, 4)   Test probs: (452, 4)             │
└──────────────────────────┬───────────────────────────────┘
                           ▼
         LogisticRegression meta
         fit(oof_probs, y_train)
                           ▼
         OOF MCC-optimal threshold (0.10~0.90 탐색)
                           ▼
              최종 예측 (DILIrank 452개)
```

---

## DifferentialCrossAttention 상세 (differential_attention.py)

논문 Differential Transformer (arxiv 2410.05258)의 cross-attention 변형.

```
λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init(0.8)
λ.clamp(min=1e-4, max=2.0)   ← 음수화 방지 + 포화 방지 (gradient 유지)

scores = softmax(Q1@K1^T/√d) - λ · softmax(Q2@K2^T/√d)
       = a1 - λ · a2
```

- a1이 주목하는 MACCS key에서 a2가 공통으로 주목하는 "노이즈"를 차감
- inactive MACCS bits: scores에 -1e9 마스킹 → attention ≈ 0
- GroupNorm + (1 - λ_init) 스케일 보정으로 출력 안정화
- attn_weights (head 평균) 저장 → Step4_xai.py에서 히트맵 생성

---

## XAI: Step4_xai.py

```python
encoder.encode(...)          # 추론 실행
attn = encoder.get_attn_weights()  # (1, MAX_ATOMS, 167)
# torch.relu(attn)[:, 1:]    → (n_atoms, 166) 양수 differential score
# seaborn heatmap: x=MACCS bits 1~166, y=원자 기호
```

---

## 실험 결과

### env1 — Fixed Split (DILIrank test, N=452)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| StackDILI (목표) | **0.9736** | **0.8304** | **0.9010** | **0.9402** | **0.8993** |
| SAGEConv + LR | 0.8426 | 0.5275 | 0.7102 | 0.6793 | 0.8396 |
| **GINEConv + LR** | **0.8808** | **0.5776** | **0.7565** | 0.7935 | 0.7910 |
| vs 목표 (GINEConv) | -0.0928 | -0.2528 | -0.1445 | -0.1467 | -0.1083 |

### env2 — 10-Fold CV (전체 N=1,850)

| 모델 | AUC | MCC | F1 | Sensitivity | Specificity |
|------|-----|-----|----|-------------|-------------|
| SAGEConv + LR | 0.8909 ±0.030 | 0.6366 ±0.072 | 0.8230 ±0.037 | 0.831 | 0.802 |
| **GINEConv + LR** | **0.9558 ±0.015** | **0.8080 ±0.051** | **0.9068 ±0.024** | 0.908 | 0.900 |
