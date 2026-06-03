"""
services/xai.py — Differential Cross-Attention XAI 파이프라인

Step4_xai.py의 알고리즘을 웹 서비스용으로 재구성.

핵심 알고리즘 (Step4_xai.py 완전 호환):
  1. model.diff_attn.multihead_scores (B, h, MAX_ATOMS, 167) 추출
  2. torch.relu() → 음수 differential 노이즈 제거 (differential 메커니즘 artifact)
  3. bit 0 제외 (RDKit 1-based indexing dummy, always 0) → 유효 bits 1~166
  4. 실제 원자 수 슬라이싱 (MAX_ATOMS 패딩 제거)
  5. 원자 중요도: sum over (heads, MACCS_bits) → Min-Max 정규화 → [0, 1]
  6. MACCS 패턴 기여도: sum over (heads, atoms) → 전체 합 대비 비율
  7. RDKit rdMolDraw2D로 원자 색상 오버레이 SVG 생성
     - prob > 45.0 (DILI 예측): 빨강 그라디언트
     - prob ≤ 45.0 (안전 예측): 파랑 그라디언트
  8. SMARTS 작용기 매핑으로 독성 원인 TOP 3 추출 (폴백: 원자 레벨)

스레드 안전성:
  이 모듈의 함수들은 순수 numpy/RDKit 연산입니다.
  model 접근은 inference.py 내 Lock 구간에서 완료 후 numpy 배열로 전달됩니다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 원자 중요도 하이라이트 최소 임계값 (Min-Max 정규화 기준 0~1)
HIGHLIGHT_THRESHOLD: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 내부 타입
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaccsPatternScore:
    """MACCS 패턴 하나의 어텐션 기여 정보."""
    bit_index: int    # 1 ~ 166 (MACCS bit 번호)
    name: str         # Durant et al. 2002 기반 인간 가독 이름
    importance: float # 전체 어텐션 대비 기여 비율 (0 ~ 1)


@dataclass(frozen=True)
class ToxicReasonScore:
    """SMARTS 작용기 기반 독성 원인 기여 정보."""
    name: str         # 작용기 이름 또는 'Atom #N (X)' 폴백 형태
    importance: float # TOP-K 내 합산 기준 정규화된 기여도 (0 ~ 1)
    rank: int         # 1 = 가장 높은 기여도


# ─────────────────────────────────────────────────────────────────────────────
# SMARTS 작용기 패턴 목록 (18가지 독성 관련 주요 작용기)
# ─────────────────────────────────────────────────────────────────────────────
# 출처: DILI 관련 문헌 (Kaplowitz 2005, Chen 2016) 기반 간독성 관련 작용기
# SMARTS 문법: RDKit rdkit.Chem.MolFromSmarts 호환

SMARTS_FG_LIST: list[tuple[str, str]] = [
    # 질소 기반 (주요 간독성 관련)
    ("Nitro Group",         "[N+](=O)[O-]"),              # -NO2: 생체 활성화, 반응성 대사체
    ("Aromatic Amine",      "[NH2]c"),                    # ArNH2: CYP1A2 기질, 생체 활성화
    ("Aliphatic Amine",     "[NH2][CX4]"),                # 지방족 1차 아민
    ("Secondary Amine",     "[NX3;H1;!$(NC=O)]"),         # R-NH-R (아마이드 제외)
    ("Amide",               "[NX3][CX3](=[OX1])"),        # -CONH-: 간 대사 부담
    # 산소 기반
    ("Phenol",              "[OX2H1]c"),                  # ArOH: glucuronidation/sulfation 부담
    ("Hydroxyl",            "[OX2H1][CX4]"),              # 지방족 -OH
    ("Carboxylic Acid",     "[CX3](=[OX1])[OX2H1]"),      # -COOH: acyl-glucuronide 형성
    ("Ester",               "[CX3](=[OX1])[OX2H0][#6]"), # -COO-: 가수분해, 반응성 대사체
    ("Aldehyde",            "[CX3H1](=[OX1])"),           # -CHO: 단백질 직접 부가체 형성
    ("Ketone",              "[#6][CX3](=[OX1])[#6]"),     # -C(=O)-: 친전자성 카보닐
    ("Epoxide",             "C1OC1"),                     # 3원 환: 고반응성
    # 황 기반
    ("Thiol",               "[SX2H1]"),                   # -SH: 단백질과 공유 반응
    ("Thioether",           "[#6][SX2;!H1][#6]"),         # R-S-R: CYP2C19 기질
    ("Sulfoxide",           "[SX3](=[OX1])([#6])[#6]"),   # sulfoxidation 대사체
    ("Sulfonamide",         "[SX4](=[OX1])(=[OX1])[NX3]"),# sulfa 약물: GST 기질
    # 할로겐/기타
    ("Halide",              "[F,Cl,Br,I][#6]"),           # C-X 결합
    ("Imine",               "[CX3]=[NX2]"),               # -C=N-: 단백질 반응성 Schiff 염기
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 모델 어텐션 텐서 → numpy 변환
# ─────────────────────────────────────────────────────────────────────────────

def extract_mh_scores_numpy(
    mh_raw: Optional["torch.Tensor"],  # type: ignore[type-arg]
    n_atoms: int,
) -> np.ndarray:
    """
    모델에서 캡처한 multihead_scores 텐서를 XAI 처리용 numpy 배열로 변환.

    Args:
        mh_raw : model.diff_attn.multihead_scores.clone() — shape (B, h, MAX_ATOMS, 167)
                 또는 fallback용 head-averaged (1, MAX_ATOMS, 167) / None
        n_atoms: 분자 실제 원자 수 (MAX_ATOMS 패딩 제거 기준)

    Returns:
        numpy float32, shape (num_heads, n_atoms, 166)
          - axis 0: 어텐션 헤드 (4개)
          - axis 1: 실제 원자 (n_atoms개)
          - axis 2: 유효 MACCS bits (bits 1~166, bit 0 제외)
          - 모든 값 ≥ 0 (relu 적용으로 음수 differential 노이즈 제거)

    Algorithm:
        1. squeeze batch dim: (B, h, MAX, 167) → (h, MAX, 167)
        2. relu: 음수 differential 어텐션 노이즈 제거 (Step4_xai.py 동일)
        3. bit 0 제외: [:, :, 1:] → (h, MAX, 166)
        4. 실제 원자 슬라이싱: [:, :n_atoms, :] → (h, n_atoms, 166)
    """
    import torch

    if mh_raw is None:
        logger.warning("mh_raw is None — returning zero scores for %d atoms", n_atoms)
        return np.zeros((1, n_atoms, 166), dtype=np.float32)

    # ── shape 정규화 ───────────────────────────────────────────────────────
    # (B, h, MAX_ATOMS, 167) → squeeze batch dim → (h, MAX_ATOMS, 167)
    t = mh_raw.squeeze(0)  # (h, MAX_ATOMS, 167) if B=1

    if t.ndim == 2:
        # fallback: head-averaged (MAX_ATOMS, 167)
        t = t.unsqueeze(0)  # (1, MAX_ATOMS, 167)

    # ── relu → bit 0 제외 → 원자 슬라이싱 ────────────────────────────────
    # relu로 음수 differential 노이즈를 제거해야 원자별 변별력이 살아남.
    # 제거하면 음/양이 섞인 합산 결과가 대부분 양수로 수렴해 모든 원자가 동일 강도로 강조됨.
    t = torch.relu(t)
    t = t[:, :n_atoms, 1:]  # (h, n_atoms, 166) — bit 0 제외

    arr = t.detach().cpu().numpy().astype(np.float32)

    logger.debug(
        "extract_mh_scores_numpy: output shape=%s, max=%.4f",
        arr.shape, arr.max() if arr.size > 0 else 0.0,
    )
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 원자별 중요도 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_atom_importance(mh_scores: np.ndarray) -> np.ndarray:
    """
    원자별 총 어텐션 중요도 계산 후 Min-Max 정규화.

    Args:
        mh_scores: (num_heads, n_atoms, 166) float32, 모든 값 ≥ 0 (relu 적용됨)

    Returns:
        atom_importance: (n_atoms,) float32, 범위 [0, 1]
          0.0 = 어텐션 없는 원자 / 1.0 = 가장 강하게 주목받은 원자

    Algorithm:
        - 헤드 4개 × MACCS 166비트에 걸쳐 각 원자의 differential 어텐션 합산
          importance[i] = Σ_h Σ_j mh_scores[h, i, j]
        - Min-Max 정규화: (x - min) / (max - min) → [0, 1]
          가장 주목받은 원자 = 1.0, 가장 덜 주목받은 원자 = 0.0
          (대칭 L∞ 대신 Min-Max를 써야 분자 내 원자 간 상대적 변별력이 유지됨)
    """
    if mh_scores.size == 0:
        return np.zeros(0, dtype=np.float32)

    # (h, n_atoms, 166) → sum MACCS → (h, n_atoms) → sum heads → (n_atoms,)
    importance: np.ndarray = mh_scores.sum(axis=2).sum(axis=0)

    # Min-Max 정규화 → [0, 1]
    min_val = float(importance.min())
    max_val = float(importance.max())
    if max_val - min_val > 1e-9:
        importance = (importance - min_val) / (max_val - min_val)
    else:
        # 모든 원자가 동일한 중요도 → 변별 불가 → 전부 0 처리 (하이라이트 없음)
        importance = np.zeros_like(importance)

    return importance.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MACCS 비트 → 정확한 이름 변환 (실제 RDKit smartsPatts 기반)
# ─────────────────────────────────────────────────────────────────────────────

# RDKit MACCSkeys.smartsPatts의 실제 SMARTS 패턴 → 화학적 이름 매핑
# MACCS_NAMES(Durant 2002 기반) 딕셔너리가 RDKit 구현과 비트 번호 불일치이므로
# 실제 smartsPatts SMARTS 문자열에서 직접 이름 파생.
_SMARTS_TO_READABLE: dict[str, str] = {
    # 탄소 패턴
    '[CH3]':                   'Methyl group (CH3)',
    '[CH2]':                   'Methylene (CH2)',
    '[C;H3,H4]':               'Terminal carbon (CH3)',
    '[CH3]~*~[CH2]~*':        'CH3-x-CH2 chain',
    '[CH3]~*~[CH3]':          'Geminal dimethyl (CH3-x-CH3)',
    '[CH3]~[CH2]~*':          'Ethyl chain (CH3-CH2)',
    '[CH3]~*~*~*~[CH2]~*':    'Long carbon chain',
    '*~*(~*)(~*)~*':           'Quaternary carbon',
    # 산소 패턴
    '[#8]':                    'Oxygen (any)',
    '[#8]~[#6]~[#8]':         'O-C-O (ester/acid/acetal)',
    '[#6]=[#8]':               'Carbonyl C=O',
    '[#6]-[#8]':               'C-O bond',
    '[O;!H0]':                 'Hydroxyl (-OH)',
    '[#8]=*':                  'Double-bond oxygen (C=O)',
    '[#8]~*~[CH2]~*':         'O adjacent to CH2',
    '[#8]~[#6](~[#6])~[#6]':  'Ketone C(=O)',
    '*!@[#8]!@*':              'Ether (non-ring C-O-C)',
    '*@*!@[#8]':               'Ring-attached oxygen',
    '*~[CH2]~[#8]':           'CH2 next to oxygen',
    # 질소 패턴
    '[#7]':                    'Nitrogen (any)',
    '[#7;R]':                  'Ring nitrogen',
    '[#7;!H0]':                'N-H (amine/amide)',
    '[#7]=*':                  'Imine/oxime (C=N)',
    '[#7]!:*:*':               'N adjacent to aromatic',
    '[#7]~[#6]~[#8]':         'N-C-O (amide/carbamate)',
    '[#7]~*~[#8]':            'N-x-O',
    '[#7]~*~[CH2]~*':         'N adjacent to CH2',
    '*~[CH2]~[#7]':           'CH2-N',
    '*~[#7](~*)~*':           'Tertiary nitrogen',
    '[#7]~*(~*)~*':           'Tertiary N (branched)',
    '[#6]-[#7]':              'C-N bond',
    '[#7]~[#6]~[#8]':         'Amide N-C=O',
    '[#7]*~*':                'N-C chain',
    # 황 패턴
    '[#16]':                   'Sulfur (any)',
    '[SH]':                    'Thiol (-SH)',
    # 할로겐 패턴
    '[F,Cl,Br,I]':             'Halogen (F/Cl/Br/I)',
    '[F,Cl,Br,I]~*(~*)~*':    'Branched halide',
    'Cl':                      'Chlorine',
    # 방향족/고리 패턴
    'a':                       'Aromatic atom',
    '[R]':                     'Ring atom',
    '[!C;!c;R]':               'Heteroatom in ring',
    '[!#6;R]':                 'Non-C ring atom',
    '*@*(@*)@*':               'Ring branch point',
    '*1~*~*~*~*~*~1':         '6-membered ring',
    '*1~*~*~*~*~1':           '5-membered ring',
    '*1~*~*~*~1':             '4-membered ring',
    # 기타 헤테로 패턴
    '[!#6;!#1]':               'Heteroatom (non-C/H)',
    '[!#6;!#1;!H0]':          'Heteroatom with H',
    '[!#6;!#1]~[#8]':         'Heteroatom bonded to O',
    '[!#6;!#1]~[CH2]~*':      'Heteroatom-CH2',
    '[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]': 'Multiple heteroatoms',
    '*!@*@*!@*':              'Cross-ring connectivity',
    '*!:*:*!:*':              'Mixed aromatic/non-aromatic',
    '*@*!@[#7]':              'Ring-N at junction',
    '*@*!@[#8]':              'Ring-O at junction',
    '*~[!#6;!#1](~*)~*':      'Branched heteroatom',
    '[#8]!:*:*':              'Non-aromatic O in aromatic',
    '[#7]!:*:*':              'Non-aromatic N in aromatic',
    '[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]': 'Two-carbon chain/ring',
}


def _get_maccs_display_name(bit_idx: int) -> str:
    """
    MACCS 비트 인덱스 → 표시 이름 반환.

    RDKit smartsPatts의 실제 SMARTS에서 이름을 파생하여
    MACCS_NAMES 딕셔너리의 비트 번호 불일치 문제를 우회.
    """
    try:
        from rdkit.Chem.MACCSkeys import smartsPatts
        info = smartsPatts.get(bit_idx)
        if info is None:
            return f"MACCS Key {bit_idx}"
        smarts_str, min_count = info
        display = _SMARTS_TO_READABLE.get(smarts_str)
        if display:
            suffix = f" ({min_count}+)" if min_count > 1 else ""
            return f"{display}{suffix}"
        # 매핑 없으면 SMARTS 직접 표시 (진단용)
        suffix = f" ({min_count}+)" if min_count > 0 else ""
        return f"Key {bit_idx}: {smarts_str}{suffix}"
    except Exception:
        return f"MACCS Key {bit_idx}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: MACCS 패턴 기여도 추출
# ─────────────────────────────────────────────────────────────────────────────

def extract_top_maccs_patterns(
    mh_scores: np.ndarray,
    smiles: str,
    top_k: int = 3,
) -> list[MaccsPatternScore]:
    """
    MACCS 패턴별 총 어텐션 기여도에서 상위 k개 추출.

    Args:
        mh_scores: (num_heads, n_atoms, 166) float32, 모든 값 ≥ 0 (relu 적용됨)
                   axis-2 인덱스 0 == MACCS bit 1, ..., 인덱스 165 == MACCS bit 166
        smiles   : Canonical SMILES — 실제 분자의 MACCS 활성 비트 마스킹에 사용
        top_k    : 반환할 패턴 수 (기본 3)

    Returns:
        list[MaccsPatternScore] — importance 내림차순.
        분자가 실제로 보유하지 않은 MACCS 비트(활성=0)는 어텐션 점수와 무관하게 제외.

    Algorithm:
        1. mh_scores (h, n_atoms, 166) → heads×atoms 합산 → pattern_sum (166,)
        2. RDKit MACCSkeys로 입력 분자의 실제 활성 비트 벡터(0/1) 계산
        3. pattern_sum × active_mask → 분자에 없는 비트 강제 0
        4. 마스킹 후 잔존 합 대비 정규화 → 기여 비율
        5. importance > 0 인 비트 중 top_k 반환
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from app.ml.graph_utils import MACCS_NAMES

    if mh_scores.size == 0:
        return []

    relu_scores = np.maximum(mh_scores, 0.0)

    # (h, n_atoms, 166) → sum(heads, atoms) → (166,)
    pattern_sum: np.ndarray = relu_scores.sum(axis=(0, 1))

    if pattern_sum.sum() < 1e-9:
        logger.debug("extract_top_maccs_patterns: all-zero raw scores, returning []")
        return []

    # ── 실제 분자 MACCS 비트 마스킹 ──────────────────────────────────────
    # 분자가 보유하지 않은 비트(fp[j]=0)는 어텐션 점수가 아무리 높아도 제외.
    # array index j → MACCS bit (j+1): fp[1]~fp[166]
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            active_mask = np.array(
                [int(fp[j + 1]) for j in range(166)], dtype=np.float32
            )
            pattern_sum = pattern_sum * active_mask
        except Exception as exc:
            logger.warning("MACCS masking failed for %r: %s", smiles[:60], exc)

    masked_total = float(pattern_sum.sum())
    if masked_total < 1e-9:
        logger.debug(
            "extract_top_maccs_patterns: no active MACCS bits after masking for %r",
            smiles[:60],
        )
        return []

    normalized = pattern_sum / masked_total  # (166,) — 활성 비트 내 기여 비율

    # importance > 0 인 비트만 (=분자가 실제로 보유한 비트) 중 top_k
    sorted_indices = np.argsort(normalized)[::-1]
    top_indices = [i for i in sorted_indices if normalized[i] > 0.0][:top_k]

    results: list[MaccsPatternScore] = []
    for arr_idx in top_indices:
        bit_idx = int(arr_idx) + 1   # array index 0 → MACCS bit 1
        # MACCS_NAMES 대신 실제 SMARTS 기반 이름 사용 (비트 번호 불일치 방지)
        name = _get_maccs_display_name(bit_idx)
        results.append(
            MaccsPatternScore(
                bit_index=bit_idx,
                name=name,
                importance=float(round(float(normalized[arr_idx]), 6)),
            )
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: SMARTS 작용기 기반 독성 원인 추출 (폴백 포함)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fg_toxic_reasons(
    smiles: str,
    atom_importance: np.ndarray,
    top_k: int = 3,
) -> list[ToxicReasonScore]:
    """
    SMARTS 작용기 매핑으로 독성 원인 TOP K 추출.

    Args:
        smiles         : Canonical SMILES
        atom_importance: (n_atoms,) float32, Min-Max 정규화된 [0, 1]
        top_k          : 반환할 최대 수 (기본 3)

    Returns:
        list[ToxicReasonScore] — importance 내림차순 정렬, 최대 top_k개

    Normalization (Bug 2 fix):
        FG 매칭 점수와 원자 레벨 폴백 점수를 구분하지 않고 모두 같은 raw score로
        수집한 뒤, TOP-K 전체 합산 단일 분모로 나눠 정규화함.
        → rank 1 + rank 2 + rank 3 importance 합 = 정확히 1.0 (100%) 수학적 보장.

    Robustness:
        - SMARTS 컴파일/매칭 실패는 개별 except, 전체 처리 계속
        - FG 매칭 수 < top_k 시 원자 레벨 raw score로 나머지 슬롯 보충
        - all-zero atom_importance 또는 FG 미매칭도 안전하게 처리
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("compute_fg_toxic_reasons: cannot parse %r", smiles[:60])
        return []

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []

    # atom_importance 길이 보정
    if len(atom_importance) > n_atoms:
        imp = atom_importance[:n_atoms]
    elif len(atom_importance) < n_atoms:
        pad = np.zeros(n_atoms - len(atom_importance), dtype=np.float32)
        imp = np.concatenate([atom_importance, pad])
    else:
        imp = atom_importance

    # ── Step 1: SMARTS FG 매칭 — raw score 수집 ──────────────────────────
    # raw score = 해당 FG에 속한 원자들의 atom_importance 합산
    fg_candidates: list[tuple[str, float]] = []

    for fg_name, smarts_str in SMARTS_FG_LIST:
        try:
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern, uniquify=True)
            if not matches:
                continue

            matched_atoms: set[int] = {
                idx for match in matches for idx in match if 0 <= idx < n_atoms
            }
            if not matched_atoms:
                continue

            raw_score = float(imp[list(matched_atoms)].sum())
            fg_candidates.append((fg_name, raw_score))

        except Exception as exc:
            logger.debug("SMARTS match error for %r: %s", fg_name, exc)
            continue

    fg_candidates.sort(key=lambda x: x[1], reverse=True)

    # ── Step 2: TOP-K 후보 수집 ────────────────────────────────────────
    # FG 결과를 먼저 채우고, 부족분은 원자 레벨 폴백(raw atom_importance[idx])으로 보충.
    # 원자 폴백도 FG와 동일한 raw score 스케일을 사용해야 통합 정규화가 성립함.
    candidates: list[tuple[str, float]] = list(fg_candidates[:top_k])

    if len(candidates) < top_k:
        remaining = top_k - len(candidates)
        fallback_filled = 0
        for atom_idx in np.argsort(imp)[::-1]:
            if fallback_filled >= remaining:
                break
            idx = int(atom_idx)
            try:
                symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            except Exception:
                symbol = "?"
            # 원자 폴백 raw score = 해당 원자의 atom_importance (이미 [0,1] Min-Max)
            candidates.append((f"Atom #{idx} ({symbol})", float(imp[idx])))
            fallback_filled += 1

    if not candidates:
        return []

    # FG + 원자 폴백 혼합 후 raw score 내림차순 재정렬 (퍼센티지 높은 순 보장)
    candidates.sort(key=lambda x: x[1], reverse=True)

    # ── Step 3: 통합 정규화 (TOP-K 전체 합 = 1.0 수학적 보장) ─────────────
    # FG와 원자 폴백 구분 없이 공통 분모(total_raw)로 나눠
    # rank 1 + rank 2 + rank 3 합산이 반드시 100%가 되도록 함.
    total_raw = sum(score for _, score in candidates)

    results: list[ToxicReasonScore] = []
    for rank, (name, score) in enumerate(candidates, start=1):
        if total_raw > 1e-9:
            importance = round(score / total_raw, 6)
        else:
            importance = round(1.0 / len(candidates), 6)
        results.append(ToxicReasonScore(name=name, importance=importance, rank=rank))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: RDKit SVG 하이라이트 렌더링
# ─────────────────────────────────────────────────────────────────────────────

def render_xai_svg(
    smiles: str,
    atom_importance: np.ndarray,
    prob: float,
    threshold: float = HIGHLIGHT_THRESHOLD,
    width: int = 600,
    height: int = 400,
) -> str:
    """
    원자 중요도 기반 단색 그라디언트 하이라이트 분자 SVG 생성.

    Args:
        smiles         : Canonical SMILES
        atom_importance: (n_atoms,) float32, Min-Max 정규화된 [0, 1]
        prob           : DILI 예측 확률 (0.0 ~ 100.0 %)
                         > 45.0  → 빨강 계열 하이라이트 (독성 예측)
                         ≤ 45.0  → 파랑 계열 하이라이트 (안전 예측)
        threshold      : 하이라이트 최소 중요도 임계값 — 이하 원자는 무색 처리
        width, height  : SVG 픽셀 크기

    Returns:
        SVG 문자열 (<?xml ...> 헤더 포함, 직접 <div>에 삽입 가능)

    Color scheme:
        color_val = max(0.0, 1.0 - imp * 0.8)   (imp ∈ (threshold, 1.0])

        독성 예측 (prob > 45.0) → 빨강:
          imp=0.15 → color_val=0.88 → (1.0, 0.88, 0.88)  연한 분홍
          imp=0.50 → color_val=0.60 → (1.0, 0.60, 0.60)  중간 빨강
          imp=1.00 → color_val=0.20 → (1.0, 0.20, 0.20)  진한 빨강

        안전 예측 (prob ≤ 45.0) → 파랑:
          imp=0.15 → color_val=0.88 → (0.88, 0.88, 1.0)  연한 파랑
          imp=0.50 → color_val=0.60 → (0.60, 0.60, 1.0)  중간 파랑
          imp=1.00 → color_val=0.20 → (0.20, 0.20, 1.0)  진한 파랑

        imp ≤ threshold → 무색 (하이라이트 제외)

    Raises:
        ValueError: SMILES 파싱 실패 시
        RuntimeError: RDKit 렌더링 내부 오류 시
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES for XAI SVG: {smiles!r}")

    n_atoms = mol.GetNumAtoms()
    is_toxic = prob > 45.0

    # ── 원자 중요도 배열 길이 보정 ─────────────────────────────────────────
    if len(atom_importance) < n_atoms:
        pad = np.zeros(n_atoms - len(atom_importance), dtype=np.float32)
        atom_importance = np.concatenate([atom_importance, pad])
    elif len(atom_importance) > n_atoms:
        atom_importance = atom_importance[:n_atoms]

    # ── 색상 계산 ──────────────────────────────────────────────────────────
    highlight_atoms: list[int] = []
    atom_color_map: dict[int, tuple[float, float, float]] = {}
    atom_radius_map: dict[int, float] = {}

    for idx in range(n_atoms):
        imp = float(atom_importance[idx])
        if imp <= threshold:
            continue

        color_val = max(0.0, 1.0 - imp * 0.8)

        if is_toxic:
            atom_color_map[idx] = (1.0, color_val, color_val)  # 빨강
        else:
            atom_color_map[idx] = (color_val, color_val, 1.0)  # 파랑

        atom_radius_map[idx] = 0.25 + imp * 0.20
        highlight_atoms.append(idx)

    # ── 결합 색상 (양 끝 원자가 모두 하이라이트된 경우) ─────────────────
    highlight_bonds: list[int] = []
    bond_color_map: dict[int, tuple[float, float, float]] = {}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in atom_color_map and j in atom_color_map:
            ci, cj = atom_color_map[i], atom_color_map[j]
            bond_color_map[bond.GetIdx()] = (
                (ci[0] + cj[0]) / 2.0,
                (ci[1] + cj[1]) / 2.0,
                (ci[2] + cj[2]) / 2.0,
            )
            highlight_bonds.append(bond.GetIdx())

    # ── SVG 렌더링 ───────────────────────────────────────────────────────
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False
    opts.padding = 0.15

    try:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_color_map,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_color_map,
        )
        drawer.FinishDrawing()
    except Exception as exc:
        raise RuntimeError(
            f"RDKit DrawMolecule failed for {smiles!r}: {exc}"
        ) from exc

    svg = drawer.GetDrawingText()
    logger.debug(
        "render_xai_svg: %d/%d atoms highlighted (threshold=%.2f, prob=%.1f%%, %s)",
        len(highlight_atoms), n_atoms, threshold, prob,
        "DILI/RED" if is_toxic else "SAFE/BLUE",
    )
    return svg


# ─────────────────────────────────────────────────────────────────────────────
# 퍼사드: 전체 XAI 산출물 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_xai_outputs(
    mh_raw: Optional["torch.Tensor"],  # type: ignore[type-arg]
    smiles: str,
    n_atoms: int,
    prob: float,
    top_k: int = 3,
) -> tuple[list[MaccsPatternScore], list[ToxicReasonScore], str]:
    """
    모델 멀티헤드 어텐션 텐서로부터 XAI 산출물 전체를 생성하는 파사드.

    inference.py에서 Lock 구간 종료 후 이 함수를 호출합니다.
    (mh_raw는 Lock 내부에서 .clone()된 안전한 복사본)

    Args:
        mh_raw  : model.diff_attn.multihead_scores.clone() — (B, h, MAX_ATOMS, 167)
                  또는 None (모델 실패 시 빈 결과 반환)
        smiles  : Canonical SMILES (원자 인덱스 기준으로 사용)
        n_atoms : 실제 원자 수 (PyG 그래프에서 파생)
        prob    : DILI 예측 확률 (0.0 ~ 100.0 %) — SVG 색상 결정 (빨강/파랑)
        top_k   : 상위 패턴/작용기 반환 수

    Returns:
        (top_maccs, toxic_reasons, svg_string)
        - top_maccs      : list[MaccsPatternScore], importance 내림차순
        - toxic_reasons  : list[ToxicReasonScore], SMARTS 작용기 기반 독성 원인 TOP K
        - svg_string     : str, XAI 하이라이트 SVG. 실패 시 plain SVG로 fallback.

    Raises:
        RuntimeError: mh_raw 처리 및 SVG 렌더링 모두 실패 시
    """
    # Step 1: tensor → numpy (relu 적용)
    mh_scores = extract_mh_scores_numpy(mh_raw, n_atoms)

    # Step 2: 원자 중요도 (Min-Max [0, 1])
    atom_imp = compute_atom_importance(mh_scores)

    # Step 3: MACCS 패턴 기여도 (분자 실제 활성 비트 마스킹 포함)
    try:
        top_maccs = extract_top_maccs_patterns(mh_scores, smiles, top_k=top_k)
    except Exception as exc:
        logger.warning("extract_top_maccs_patterns failed: %s — returning []", exc)
        top_maccs = []

    # Step 3b: SMARTS 작용기 기반 독성 원인 (폴백 포함)
    try:
        toxic_reasons = compute_fg_toxic_reasons(smiles, atom_imp, top_k=top_k)
    except Exception as exc:
        logger.warning("compute_fg_toxic_reasons failed: %s — returning []", exc)
        toxic_reasons = []

    # Step 4: SVG 렌더링 (prob 기반 색상, 실패 시 plain SVG fallback)
    try:
        svg = render_xai_svg(smiles, atom_imp, prob)
    except Exception as exc:
        logger.warning(
            "render_xai_svg failed for %r: %s — falling back to plain SVG", smiles[:60], exc
        )
        try:
            from app.services.chemistry import smiles_to_svg_plain
            svg = smiles_to_svg_plain(smiles)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both XAI and plain SVG rendering failed for {smiles!r}. "
                f"XAI error: {exc}. Plain SVG error: {fallback_exc}"
            ) from fallback_exc

    return top_maccs, toxic_reasons, svg


async def abuild_xai_outputs(
    mh_raw: Optional["torch.Tensor"],  # type: ignore[type-arg]
    smiles: str,
    n_atoms: int,
    prob: float,
    top_k: int = 3,
) -> tuple[list[MaccsPatternScore], list[ToxicReasonScore], str]:
    """build_xai_outputs()의 비동기 래퍼."""
    return await asyncio.to_thread(build_xai_outputs, mh_raw, smiles, n_atoms, prob, top_k)
