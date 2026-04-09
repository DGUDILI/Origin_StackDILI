# DiffDILI_v3 env1 AUC 향상 계획

> 작성: 2026-04-09  
> 목표: clean env1 AUC 향상 (env2 성능 유지)  
> 원칙: MACCS 167-bit은 encoder DifferentialCrossAttention 전용. 핵심 구조 유지.

---

## 변경사항 Flowchart

```
SMILES
  │
  ├─[ChemBERTa]──────────────────────────────────────────────────────┐
  │   last-layer unfreeze (lr=1e-4)                                   │
  │   CLS (B, 384) → LayerNorm → Linear → chem_feat (B, 64)          │
  │                                                                   │
  ├─[RDKit mol]                                                       │
  │   atom features (N, 43-dim)                                       │
  │   + bond features (E, 9-dim) ← [Phase 3 신규]                    │
  │      single/double/triple/aromatic/ring/conjugated/stereo         │
  │                                                                   │
  │   atom_proj → GINEConv×2 (edge_attr 활용) ← [Phase 3: SAGEConv→] │
  │   + BatchNorm1d                                                   │
  │   → to_dense_batch → node_q (B, MAX_ATOMS, 64)  [Query]          │
  │                                                                   │
  ├─[MACCSkeys]──────────────────────────────────────────────────────┤
  │   167-bit binary                                                  │
  │   Embedding(167, 64) * maccs ← [Phase 2: Binary Gate 신규]       │
  │   → maccs_kv (B, 167, 64)  [Key/Value]                           │
  │                                                                   │
  │   DifferentialCrossAttention(Q=node_q, KV=maccs_kv)              │
  │     lambda clamp(max=2.0) ← [Phase 1 신규]                       │
  │     dropout=config.DROPOUT ← [Phase 1 신규]                      │
  │   → attn_out (B, MAX_ATOMS, 64)                                   │
  │   → masked_mean_pool → graph_feat (B, 64)                        │
  │                                                                   │
  └──────────────────── concat ──────────────────────────────────────┘
         fuse_proj(128→64) → LayerNorm
         MLP: Linear(64→32) → GELU → Dropout(0.3)
         → encode_out (B, 32)
         → head Linear(32→1) + BCEWithLogitsLoss(pos_weight) ← [Phase 1 신규]

  Stage 2: encode_out (B, 32)
         → Stacking OOF [RF / ET / HistGB / XGB → LR meta]
         → 최종 예측
```

---

## 실험 결과 요약

| Phase | 주요 변경 | env1 AUC | env2 AUC |
|---|---|---|---|
| 베이스라인 | GraphMACCSEncoder (SAGEConv) | ~0.805 (clean) | 0.9152 |
| Phase 1 | pos_weight + dropout연동 + lambda상한 + 로깅 | +0.04 | — |
| Phase 2 | Binary Gate만 채택 (MLP/GraphNorm/LabelSmoothing 역효과) | 중립 | — |
| **Phase 3** | **GINEConv + Edge Feature 9-dim** | **0.8286** | **0.9374** |

---

## 실험으로 배제된 전략 (재시도 금지)

| 전략 | 결과 | 이유 |
|---|---|---|
| ChemBERTa 2-layer unfreeze | test AUC 하락 | 도메인 오버피팅. 1-layer가 적절한 regularization 역할 |
| DANN/DAT (λ>0) | AUC 하락 | train/test 분포 차이가 signal quality shift. DANN이 유용한 DILIrank 신호 제거 |

---

## Phase 1 — 안정화 수정 ✅ 완료

> Step1 캐시 재생성 불필요. 재학습 1회 후 Step3/CV 실행.  
> **결과: env1 AUC +0.04 향상**

| Step | 파일 | 내용 | 상태 |
|---|---|---|---|
| 1-A | `Step2_pretrain.py:191` | BCE `pos_weight = n_neg/n_pos` 추가 | ✅ |
| 1-B | `model.py:89` | DiffAttn `dropout=0.1` → `dropout=dropout` | ✅ |
| 1-C | `differential_attention.py:77` | lambda `.clamp(min=1e-4)` → `.clamp(min=1e-4, max=2.0)` | ✅ |
| 1-D | `Step2_pretrain.py:244` | 로깅 `epoch % 10` → `epoch % 5` | ✅ |

---

## Phase 2 — 정규화 강화 ⚠️ 대부분 역효과, 개별 실험 완료

> 전체 일괄 적용 시 env1 AUC -0.026, env2 AUC -0.048 퇴보.
> 개별 실험으로 원인 분리 완료. clean train 1,187개 소규모 데이터에서 추가 정규화/용량 확장 한계 확인.

| Step | 파일 | 내용 | 결과 |
|---|---|---|---|
| 2-A | `config.py:7` | `DROPOUT 0.3 → 0.4` | ❌ 건너뜀 (2-B/C와 조합 역효과로 추정) |
| 2-B | `model.py` | MLP 1-layer → 2-layer | ❌ 단독 -0.015 역효과 (Dropout×2 과도) |
| 2-C | `model.py` | `BatchNorm1d` → `GraphNorm` | ❌ 단독 -0.014 역효과 |
| **2-D** | `model.py:146-153` | **MACCS Binary Gate** | **✅ 유지 (중립, gradient 안정화)** |
| 2-E | `Step2_pretrain.py` | Label Smoothing `smoothing=0.05` | ❌ 건너뜀 (pos_weight와 충돌 추정) |

**결론: Phase 2에서 Binary Gate(2-D)만 유지. 나머지는 소규모 clean 데이터에서 역효과.**
**현재 베이스라인: Phase 1 + Binary Gate = env1 AUC ~0.805**

### 2-B MLP 변경 내용
```python
# 변경 전
self.mlp = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, k),
    nn.GELU(),
    nn.Dropout(dropout),
)
# 변경 후
self.mlp = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, k),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

### 2-C GraphNorm 변경 내용
```python
from torch_geometric.nn import GraphNorm
self.sage_bns = nn.ModuleList([GraphNorm(sage_hidden) for _ in range(sage_layers)])
# forward: x = torch.relu(bn(conv(x, edge_index), graph_batch.batch))
```

---

## Phase 3 — 아키텍처 개선 ✅ 완료

> Step1 캐시 재생성(--force) + 재학습 필요.

| Step | 파일 | 내용 | 결과 |
|---|---|---|---|
| 3-A | `graph_utils.py` + `model.py` | Edge Feature 9-dim + `SAGEConv → GINEConv` | ✅ env1 +0.024 / env2 +0.022 |

**현재 베이스라인: env1 AUC 0.8286 / env2 AUC 0.9374**

### 3-A 추가할 bond feature (9-dim)
```python
def get_bond_features(bond) -> list[float]:
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    return [
        float(bt == rdchem.BondType.SINGLE),
        float(bt == rdchem.BondType.DOUBLE),
        float(bt == rdchem.BondType.TRIPLE),
        float(bt == rdchem.BondType.AROMATIC),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        float(bond.GetStereo() == rdchem.BondStereo.STEREONONE),
        float(bond.GetStereo() == rdchem.BondStereo.STEREOE),
        float(bond.GetStereo() == rdchem.BondStereo.STEREOZ),
    ]
```

---

## Phase 4 — 마지막 실험: 비MACCS FP 혼합

> Phase 1~3 완료 후, 구조 개선 ceiling 도달 시 시도.  
> MACCS(167)는 encoder가 이미 cross-attention 처리 → 제외.  
> 추가 대상: E-state(79) + CalcCATS(150) + Constitutional(22) + 기타(7) = **258-dim**

```python
maccs_cols = {c for c in feat_cols if c.startswith('MACCS')}
non_maccs_idx = [i for i, c in enumerate(feat_cols) if c not in maccs_cols]
fp_non_maccs = fp_all[:, non_maccs_idx]                        # (N, 258)
X = np.concatenate([X_enc_32, fp_non_maccs], axis=1)           # (N, 290)
```

---

## AUC 로드맵 (예상)

```
AUC 0.9224  현재 베이스라인
      │
      ▼  Phase 1 (재학습 1회)
      ~0.925~0.93
      │
      ▼  Phase 2 (재학습 1회)
      ~0.93~0.945
      │
      ▼  Phase 3 (Step1 재생성 + 재학습)
      ~0.945~0.955
      │
      ▼  Phase 4 (비MACCS FP 혼합)
      ~0.955+
```
