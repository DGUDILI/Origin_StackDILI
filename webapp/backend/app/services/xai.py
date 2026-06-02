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
# Step 3: MACCS 패턴 기여도 추출
# ─────────────────────────────────────────────────────────────────────────────

def extract_top_maccs_patterns(
    mh_scores: np.ndarray,
    top_k: int = 3,
) -> list[MaccsPatternScore]:
    """
    MACCS 패턴별 총 어텐션 기여도에서 상위 k개 추출.

    Args:
        mh_scores: (num_heads, n_atoms, 166) float32, 모든 값 ≥ 0 (relu 적용됨)
                   axis-2 인덱스 0 == MACCS bit 1, ..., 인덱스 165 == MACCS bit 166
        top_k    : 반환할 패턴 수 (기본 3)

    Returns:
        list[MaccsPatternScore] — importance 내림차순

    Algorithm:
        1. 모든 헤드 × 모든 원자에 걸쳐 각 MACCS 비트의 어텐션 합산
           pattern_sum[j] = Σ_h Σ_i mh_scores[h, i, j]
        2. 전체 합 대비 정규화 → 기여 비율 (0~1)
        3. 상위 top_k 인덱스 선택 및 MACCS_NAMES 매핑
    """
    from app.ml.graph_utils import MACCS_NAMES

    if mh_scores.size == 0:
        return []

    # mh_scores는 extract_mh_scores_numpy()에서 이미 relu 적용됨.
    # np.maximum으로 부동소수 오차 보호 차원에서 한 번 더 적용.
    relu_scores = np.maximum(mh_scores, 0.0)

    # (h, n_atoms, 166) → sum(heads, atoms) → (166,)
    pattern_sum: np.ndarray = relu_scores.sum(axis=(0, 1))

    total = pattern_sum.sum()
    if total < 1e-9:
        logger.debug("extract_top_maccs_patterns: all-zero scores, returning []")
        return []

    normalized = pattern_sum / total  # (166,) — 전체 어텐션 대비 각 비트 기여 비율

    # 상위 top_k (내림차순)
    top_indices = np.argsort(normalized)[::-1][:top_k]

    results: list[MaccsPatternScore] = []
    for arr_idx in top_indices:
        bit_idx = int(arr_idx) + 1   # array index 0 → MACCS bit 1
        raw_name = MACCS_NAMES.get(bit_idx)
        name = raw_name if raw_name else f"MACCS Key {bit_idx}"
        results.append(
            MaccsPatternScore(
                bit_index=bit_idx,
                name=name,
                importance=float(round(float(normalized[arr_idx]), 6)),
            )
        )

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
) -> tuple[list[MaccsPatternScore], str]:
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
        top_k   : 상위 MACCS 패턴 반환 수

    Returns:
        (top_maccs, svg_string)
        - top_maccs  : list[MaccsPatternScore], importance 내림차순
        - svg_string : str, XAI 하이라이트 SVG. 실패 시 plain SVG로 fallback.

    Raises:
        RuntimeError: mh_raw 처리 및 SVG 렌더링 모두 실패 시
    """
    # Step 1: tensor → numpy (relu 적용)
    mh_scores = extract_mh_scores_numpy(mh_raw, n_atoms)

    # Step 2: 원자 중요도 (Min-Max [0, 1])
    atom_imp = compute_atom_importance(mh_scores)

    # Step 3: MACCS 패턴 기여도
    try:
        top_maccs = extract_top_maccs_patterns(mh_scores, top_k=top_k)
    except Exception as exc:
        logger.warning("extract_top_maccs_patterns failed: %s — returning []", exc)
        top_maccs = []

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

    return top_maccs, svg


async def abuild_xai_outputs(
    mh_raw: Optional["torch.Tensor"],  # type: ignore[type-arg]
    smiles: str,
    n_atoms: int,
    prob: float,
    top_k: int = 3,
) -> tuple[list[MaccsPatternScore], str]:
    """build_xai_outputs()의 비동기 래퍼."""
    return await asyncio.to_thread(build_xai_outputs, mh_raw, smiles, n_atoms, prob, top_k)
