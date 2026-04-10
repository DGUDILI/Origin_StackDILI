# SMILES Augmentation — 구현 계획 v2 (확정)

> **작성일**: 2026-04-10
> **v1 → v2 주요 변경**: CSV 사전저장(Step1-C) 방식 폐기 → Step2 코드 내 Split-후-증강 방식으로 확정.
> GroupShuffleSplit 불필요. **N_AUG=2 (변경: 5→2)**. Step3/CV 무수정.

---

## 1. 배경 및 핵심 제약

`graph_utils.py`에 `augment_smiles()` 함수가 이미 구현되어 있고,
`config.py`에 `N_AUG` 설정도 이미 존재합니다.

**핵심 제약**: `num_workers=0` (Windows 환경) 때문에 현재 파이프라인은
Step1에서 PyG/MACCS를 `.pt`로 캐싱한 뒤 Step2에서 읽기만 합니다.
증강도 동일한 원칙으로, **Step2 내부에서 1회 증강 후 토큰 캐시에 저장 → 학습 중 재사용** 구조로 만듭니다.

**epoch 소요 시간 참고 (CPU 환경)**

| N_AUG | 샘플 배수 | epoch당 소요 |
|---|---|---|
| 0 | 원본 ×1 | ~3분 |
| **2 (채택)** | **원본 ×3** | **~7분** |
| 5 | 원본 ×6 | ~20분 |

---

## 2. 결정 사항 요약

| 항목 | 결정 | 이유 |
|---|---|---|
| N_AUG | **2** | epoch당 ~7분으로 적절한 속도/효과 균형 |
| GroupShuffleSplit | **불필요** | Split 먼저 → 증강 순서이므로 누수 없음 |
| Step3/CV 증강 | **미적용** | OOF fold 오염 방지. encoder 표현력 강화가 목적 |

---

## 3. 누수 방지 원리

```
[잘못된 방식]  증강 먼저 → split  →  동일 분자가 Train/Val 양쪽에 섞임 → 누수
[채택 방식]    split 먼저 → 증강  →  Train 서브셋에만 증강, Val은 원본만 → 안전
```

Step2에서 기존 `StratifiedSplit`으로 먼저 `idx_tr` / `idx_val`을 분리한 뒤,
`idx_tr`에 해당하는 샘플들에만 `augment_smiles()`를 적용하므로
`GroupShuffleSplit` 또는 `canonical_smiles` Group Key가 전혀 불필요합니다.

**단계별 증강 적용 근거**

| 단계 | 증강 적용 | 이유 |
|---|---|---|
| Step2 (사전학습) | ✅ 적용 | encoder가 다양한 SMILES 표현 학습 → 일반화 |
| Step3 (OOF Stacking) | ❌ 미적용 | 동일 분자 중복이 CV fold를 오염시킴 |
| Step_CV | ❌ 미적용 | 동일 이유 + GroupKFold 전면 교체 필요해짐 |

---

## 4. 확정 아키텍처 (전체 흐름)

```
[기존]
Step1 (train_graphs.pt 저장) → Step2 (StratifiedSplit 85/15 → 학습)

[변경 후]
Step1  ← 변경 없음 (train_graphs.pt 원본 그대로)
    ↓
Step2  [수정]:
  1. train_graphs.pt 로드 (원본)
  2. StratifiedSplit 85/15  ← 기존 그대로 유지
     → idx_tr (Train 서브셋) / idx_val (Val 서브셋)
  3. train_subset = [all_data[i] for i in idx_tr]   ← Split 완료 후
     ↑ 여기서만 증강 적용 (Val은 절대 건드리지 않음)
  4. [신규] augment_train_subset(train_subset, N_AUG=2)
     → aug_data_list (원본 + 최대 2배 증강, 총 최대 3배)
     → 토큰 캐시 저장: graph_train_tokens_{SEED}_aug{N_AUG}.pt
  5. MolGraphDataset(aug_data_list) / MolGraphDataset(val_subset)
  6. GraphMACCSEncoder 학습 (val_AUC Early Stop)
  7. pretrained_graph_encoder.pt 저장

Step3   ← 변경 없음 (train_graphs.pt 원본 사용)
Step_CV ← 변경 없음 (train+test_graphs.pt 원본 사용)
```

---

## 5. 수정 파일 목록

### 5-A. `config.py` — 완료 ✅

`N_AUG = 5 → 2`로 변경 및 주석 보강.

```python
# ── 데이터 증강 ──────────────────────────────────────────────────────────────
N_AUG = 2   # Train subset 당 랜덤 SMILES 증강 개수 (0 = 비활성화)
             # N_AUG=2 → 원본 1 + 증강 2 = 샘플 최대 3배 (train만, val/test 불변)
             # epoch 소요 시간 참고 (CPU 환경): N_AUG=0 → ~3분, N_AUG=2 → ~7분, N_AUG=5 → ~20분
```

---

### 5-B. `Step2_pretrain.py` — 미구현 (구현 필요)

**변경 포인트 3곳:**

#### 변경 1: import 추가 + `augment_train_subset()` 함수 추가 (상단)

```python
# 기존 import에 추가
from config import N_AUG
from graph_utils import augment_smiles, smiles_to_pyg, get_maccs


def augment_train_subset(data_list: list, n_aug: int, seed: int) -> list:
    """
    Split된 train 서브셋에만 SMILES 증강 적용.
    Val 서브셋은 이 함수를 호출하지 않으므로 Data Leakage 없음.

    Args:
        data_list: Split 이후의 train 서브셋 [{pyg, maccs, label, smiles}, ...]
        n_aug:     증강 개수 (config.N_AUG)
        seed:      재현성용 시드

    Returns:
        원본 + 증강 합산 리스트 (원본 순서 유지, 증강 후 추가)
        MACCS는 분자 동일 → 원본 maccs 재사용 (재계산 생략으로 속도 향상)
    """
    if n_aug == 0:
        return data_list

    augmented = list(data_list)  # 원본 유지
    for d in data_list:
        rand_smiles = augment_smiles(d["smiles"], n_aug=n_aug, seed=seed)
        for rand_smi in rand_smiles:
            pyg = smiles_to_pyg(rand_smi)
            if pyg is None:
                continue  # 파싱 실패 스킵 (희귀 케이스)
            augmented.append({
                "pyg":    pyg,
                "maccs":  d["maccs"],   # 동일 분자 → 원본 MACCS 재사용
                "label":  d["label"],
                "smiles": rand_smi,     # 실제 학습 SMILES (랜덤)
            })
    return augmented
```

#### 변경 2: StratifiedSplit 직후 증강 적용

```python
# [기존] ── 변경 없음
idx_tr, idx_val = train_test_split(
    idx, test_size=0.15, stratify=all_labels.astype(int), random_state=SEED
)
print(f"  train subset: {len(idx_tr)}, val subset: {len(idx_val)}")

# [신규 추가] ── Split 이후, train 서브셋에만 증강
train_subset_orig = [all_data[i] for i in idx_tr]
val_subset        = [all_data[i] for i in idx_val]

if N_AUG > 0:
    print(f"\n[Augment] N_AUG={N_AUG} → train {len(train_subset_orig)}개 × 최대 {N_AUG+1}배...")
    train_subset = augment_train_subset(train_subset_orig, n_aug=N_AUG, seed=SEED)
    print(f"  augmented train: {len(train_subset_orig)} → {len(train_subset)}개")
    print(f"  val (원본 유지): {len(val_subset)}개")
else:
    train_subset = train_subset_orig
    print(f"  N_AUG=0: 증강 없음 (train={len(train_subset)}, val={len(val_subset)})")
```

#### 변경 3: 토큰 캐시 파일명 분기 ⚠️ 필수 (미처리 시 런타임 에러)

```python
# 증강 여부에 따라 파일명 분기 → 기존 캐시(graph_train_tokens_42.pt)와 충돌 방지
aug_suffix = f"_aug{N_AUG}" if N_AUG > 0 else ""

train_ds = MolGraphDataset(
    train_subset, tokenizer, MAX_LENGTH,
    # 기존: f"graph_train_tokens_{SEED}.pt"  ← 이 줄을 아래로 교체
    cache_path=os.path.join(DATA_DIR, f"graph_train_tokens_{SEED}{aug_suffix}.pt"),
)
val_ds = MolGraphDataset(
    val_subset, tokenizer, MAX_LENGTH,
    cache_path=os.path.join(DATA_DIR, f"graph_val_tokens_{SEED}.pt"),  # val은 원본, 변경 없음
)
```

> **N_AUG=2 기준 생성 파일명**: `graph_train_tokens_42_aug2.pt`

---

### 5-C. 그 외 모든 파일 — 변경 없음

| 파일 | 이유 |
|---|---|
| `Step1_preprocess.py` | 원본 `train_graphs.pt` 그대로 유지. Step1-C 블록 미추가 |
| `graph_utils.py` | `augment_smiles()` 이미 구현됨. 호출 측(Step2)이 사용 |
| `Step3_stacking.py` | 원본 `train_graphs.pt` 사용. 증강 미관여 |
| `Step_CV.py` | 원본 `train+test_graphs.pt` 사용. 증강 미관여 |
| `model.py` | 입력 형태 동일 (SMILES만 바뀌고 dimension 불변) |
| `features.py` | 추출 로직 무간섭 |
| `differential_attention.py` | 완전 무관 |
| `Step4_xai.py` | 단일 SMILES 추론, 데이터셋 무관 |
| `run.sh` | 실행 순서 미변경 |

---

## 6. 수정 범위 최종 요약

| 파일 | 상태 | 변경 내용 | 변경량 |
|---|---|---|---|
| `config.py` | ✅ 완료 | N_AUG 5→2, 주석 보강 | 3줄 수정 |
| `Step2_pretrain.py` | ⬜ 미구현 | `augment_train_subset()` 추가 + 호출 + 토큰캐시 파일명 분기 | ~50줄 추가 |
| 그 외 9개 파일 | ✅ 무수정 | 변경 없음 | 0 |

---

## 7. 산출물 파일 구조 (변경 후)

```
data/
├── train_graphs.pt                      ← 기존 (원본만, Step1-B 산출)
│                                           Step3, Step_CV 사용
├── test_graphs.pt                       ← 기존 (변경 없음)
│
├── graph_train_tokens_42.pt             ← 기존 원본 토큰 캐시 (N_AUG=0 시 사용)
├── graph_val_tokens_42.pt               ← 기존 val 토큰 캐시 (변경 없음)
│
└── graph_train_tokens_42_aug2.pt  [신규] ← 증강 train 토큰 캐시 (N_AUG=2 시 자동 생성)
```

> `train_graphs_aug.pt`, `aug_manifest.csv` — **미생성** (Step1-C 방식 폐기됨)

---

## 8. 데이터 흐름 분기 (Step2 내부)

```
all_data  (train_graphs.pt 전체, N개)
    ↓
StratifiedSplit(0.85/0.15, stratify=labels)   ← 기존 그대로
    ↓
idx_tr (0.85 × N개)  ─── augment_train_subset(N_AUG=2) ───→  train_subset (최대 0.85N × 3개)
idx_val (0.15 × N개) ──────────────────────────────────────→  val_subset   (0.15N개, 원본만)
    ↓                                                              ↓
MolGraphDataset                                            MolGraphDataset
(graph_train_tokens_42_aug2.pt)                           (graph_val_tokens_42.pt)
    ↓
GraphMACCSEncoder 학습 (val_AUC Early Stop)
    ↓
pretrained_graph_encoder.pt
    ↓
Step3:   extract_features(train_graphs.pt 원본)  ← 증강 미관여
Step_CV: extract_features(all_graphs.pt 원본)    ← 증강 미관여
```

---

## 9. 검증 계획

### Step2 실행 로그 확인 (N_AUG=2 기준)

```
[Augment] N_AUG=2 → train 1215개 × 최대 3배...
  augmented train: 1215 → 3645개
  val (원본 유지): 215개
```

### 누수 자동 검증 (Step2 내 assert)

```python
# augment_train_subset() 완료 직후 삽입
train_orig_smiles = {d["smiles"] for d in train_subset_orig}
val_smiles        = {d["smiles"] for d in val_subset}
overlap = train_orig_smiles & val_smiles
assert len(overlap) == 0, f"Leakage detected: {len(overlap)} SMILES overlap"
print(f"[OK] No data leakage (train_orig={len(train_orig_smiles)}, val={len(val_smiles)})")
```

### 성능 비교 기준

- `best_val_auc` 수렴 속도 및 최종값 (증강 전후 비교)
- `Step3/results.csv`의 AUC, MCC 값 비교
  (증강은 encoder 품질에만 영향, Stacking 학습은 무관)
