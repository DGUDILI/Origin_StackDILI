"""
services/chemistry.py — RDKit 기반 분자 검증, 물성 추출, 기본 SVG 렌더링

모든 RDKit 연산은 동기(blocking)이므로, FastAPI 비동기 핸들러에서는
asyncio.to_thread() 래퍼(aget_*)를 사용하세요.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 타입 정의
# ─────────────────────────────────────────────────────────────────────────────

class PhysChemData(TypedDict):
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    qed: float
    ring_count: int
    aromatic_rings: int


# ─────────────────────────────────────────────────────────────────────────────
# SMILES 검증 및 정규화
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_canonicalize(smiles: str) -> str | None:
    """
    SMILES 유효성 검사 후 Canonical SMILES 반환.

    Canonical화가 필수인 이유:
      - PyG 그래프 생성(smiles_to_pyg)과 XAI의 원자 인덱스가 동일 기준으로 정렬됨
      - 동일 분자가 다른 SMILES 표기로 입력될 때 일관된 결과 보장
      - RDKit이 원자 재번호를 표준화하여 예측값 재현성 확보

    Returns:
        Canonical SMILES 문자열, 또는 파싱 불가 시 None
    """
    from rdkit import Chem

    if not smiles or not smiles.strip():
        logger.debug("validate_and_canonicalize: empty input")
        return None

    stripped = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(stripped)
    except Exception as exc:
        logger.warning("RDKit MolFromSmiles raised exception for %r: %s", stripped[:80], exc)
        return None

    if mol is None:
        logger.debug("Invalid SMILES (mol is None): %r", stripped[:80])
        return None

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        logger.warning("MolToSmiles failed for %r: %s", stripped[:80], exc)
        return None

    if not canonical:
        return None

    return canonical


# ─────────────────────────────────────────────────────────────────────────────
# 물리화학 특성 계산
# ─────────────────────────────────────────────────────────────────────────────

def get_physicochemical_props(smiles: str) -> PhysChemData:
    """
    RDKit Descriptors 기반 9가지 물리화학 특성 계산.

    Args:
        smiles: validate_and_canonicalize()를 통과한 유효 SMILES

    Returns:
        PhysChemData TypedDict

    Raises:
        ValueError: SMILES 파싱 실패 시
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.QED import qed as calc_qed

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES for physicochemical calculation: {smiles!r}")

    try:
        mw      = round(Descriptors.ExactMolWt(mol), 2)
        logp    = round(Descriptors.MolLogP(mol), 3)
        hbd     = int(rdMolDescriptors.CalcNumHBD(mol))
        hba     = int(rdMolDescriptors.CalcNumHBA(mol))
        tpsa    = round(Descriptors.TPSA(mol), 2)
        rot_b   = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        qed_val = round(calc_qed(mol), 4)
        rings   = int(rdMolDescriptors.CalcNumRings(mol))
        aro     = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception as exc:
        raise ValueError(
            f"Descriptor calculation failed for {smiles!r}: {exc}"
        ) from exc

    return PhysChemData(
        molecular_weight=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        rotatable_bonds=rot_b,
        qed=qed_val,
        ring_count=rings,
        aromatic_rings=aro,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 기본 SVG 렌더링 (하이라이트 없음 — XAI 불가 시 fallback)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_svg_plain(
    smiles: str,
    width: int = 600,
    height: int = 400,
) -> str:
    """
    하이라이트 없이 분자 구조만 SVG로 렌더링.

    XAI 처리 실패 시 inference.py에서 fallback으로 호출됩니다.

    Returns:
        SVG 문자열 (<?xml ...> 헤더 포함)

    Raises:
        ValueError: SMILES 파싱 실패 시
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES for SVG: {smiles!r}")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False
    opts.padding = 0.15

    try:
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        raise ValueError(f"RDKit SVG rendering failed for {smiles!r}: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# 비동기 래퍼 (FastAPI 이벤트 루프 블로킹 방지)
# ─────────────────────────────────────────────────────────────────────────────

async def aget_physicochemical_props(smiles: str) -> PhysChemData:
    """get_physicochemical_props()의 비동기 래퍼."""
    return await asyncio.to_thread(get_physicochemical_props, smiles)


async def asmiles_to_svg_plain(smiles: str) -> str:
    """smiles_to_svg_plain()의 비동기 래퍼."""
    return await asyncio.to_thread(smiles_to_svg_plain, smiles)
