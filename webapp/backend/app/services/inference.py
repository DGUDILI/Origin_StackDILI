"""
services/inference.py — 단일 SMILES 추론 오케스트레이터

이 모듈은 chemistry / xai / model_loader 서비스를 결합하는
유일한 공개 진입점입니다.

실행 흐름 (predict_single 기준):
  1) SMILES 유효성 검사 및 Canonical화          [asyncio.to_thread]
  2) PyG 그래프, MACCS 벡터 생성
  3) ChemBERTa 토크나이저 인코딩
  4) PyGBatch 구성 + 디바이스 이동
  5) 모델 Forward pass (torch.no_grad)           ─┐ threading.Lock 구간
  6) sigmoid → DILI 확률 계산                   ─┤ (멀티스레드 경쟁 방지)
  7) multihead_scores 즉시 복사 후 Lock 해제    ─┘
  8) XAI 처리 (numpy/RDKit, Lock 외부)          [asyncio.to_thread]
  9) 물성치 계산 (RDKit Descriptors)            [asyncio.to_thread]
 10) SinglePredictResponse 조립 및 반환

스레드 안전성 설계:
  model.diff_attn.multihead_scores는 모델 인스턴스의 가변 속성이므로
  동시 요청이 Forward pass를 동시에 수행하면 데이터 경쟁이 발생합니다.
  model_loader._inference_lock (threading.Lock)으로 Forward + 캡처를
  원자적으로 처리하여 이를 방지합니다.
  numpy/RDKit 처리는 Lock 외부에서 수행되므로 추론 처리량을 극대화합니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import torch

from app.core.config import settings
from app.core.model_loader import get_device, get_inference_lock, get_model, get_tokenizer
from app.ml.graph_utils import get_maccs, smiles_to_pyg
from app.schemas.predict import MaccsPattern, PhysChemProps, SinglePredictResponse
from app.services.chemistry import (
    get_physicochemical_props,
    smiles_to_svg_plain,
    validate_and_canonicalize,
)
from app.services.xai import MaccsPatternScore, abuild_xai_outputs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 도메인 예외
# ─────────────────────────────────────────────────────────────────────────────

class InvalidSmilesError(ValueError):
    """SMILES 파싱 또는 유효성 검사 실패."""


class InferenceError(RuntimeError):
    """모델 추론 중 복구 불가능한 오류."""


# ─────────────────────────────────────────────────────────────────────────────
# 동기 추론 코어 (asyncio.to_thread로 오프로드)
# ─────────────────────────────────────────────────────────────────────────────

def _run_forward_pass(
    canonical: str,
) -> tuple[float, Optional[torch.Tensor], int]:
    """
    모델 Forward pass 수행 및 결과 캡처.

    threading.Lock 구간에서만 호출됩니다.
    multihead_scores는 .clone()으로 즉시 복사하여
    Lock 해제 후 다른 요청의 Forward pass에 의해 덮어씌워지는 것을 방지합니다.

    Args:
        canonical: Canonical SMILES

    Returns:
        (probability_pct, mh_scores_cloned, n_atoms)
        - probability_pct   : float, [0.0, 100.0]
        - mh_scores_cloned  : (B, h, MAX_ATOMS, 167) Tensor, CPU에서 복사됨
        - n_atoms           : 분자 실제 원자 수

    Raises:
        InferenceError: PyG 그래프 생성 실패, 또는 Forward pass 예외 시
    """
    from rdkit import Chem
    from torch_geometric.data import Batch as PyGBatch

    model     = get_model()
    tokenizer = get_tokenizer()
    device    = get_device()

    # ── 분자 파싱 → 원자 수 확인 ───────────────────────────────────────────
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise InferenceError(f"Canonical SMILES 파싱 실패 (예기치 않음): {canonical!r}")
    n_atoms = mol.GetNumAtoms()

    # ── PyG 그래프 생성 (Step1_preprocess와 동일한 smiles_to_pyg 호출) ────
    pyg_data = smiles_to_pyg(canonical)
    if pyg_data is None:
        raise InferenceError(
            f"PyG 그래프 생성 실패: {canonical!r}\n"
            "원자 특성 추출(get_atom_features) 또는 결합 특성 추출(get_bond_features) 확인 필요"
        )

    # ── MACCS 167-dim 벡터 ────────────────────────────────────────────────
    maccs_tensor = get_maccs(canonical).unsqueeze(0).to(device)   # (1, 167)

    # ── ChemBERTa 토크나이저 (Step2_pretrain과 동일 파라미터) ────────────
    enc = tokenizer(
        [canonical],
        max_length=settings.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids  = enc["input_ids"].to(device)        # (1, MAX_LENGTH)
    attn_mask  = enc["attention_mask"].to(device)   # (1, MAX_LENGTH)

    # ── PyGBatch 구성 및 디바이스 이동 ───────────────────────────────────
    graph_batch = PyGBatch.from_data_list([pyg_data]).to(device)

    # ── Forward pass + 멀티헤드 어텐션 캡처 (Lock 내부) ──────────────────
    with torch.no_grad():
        logit = model(input_ids, attn_mask, maccs_tensor, graph_batch)  # (1, 1)

    # sigmoid(logit) = P(DILI). 모델은 label 1=DILI 관례로 학습됨.
    probability_pct = round(torch.sigmoid(logit).item() * 100.0, 2)

    # multihead_scores를 즉시 복사: (B, h, MAX_ATOMS, 167)
    # .clone()하지 않으면 다음 요청의 Forward pass가 이 텐서를 덮어씀
    mh_raw: Optional[torch.Tensor] = None
    if (
        hasattr(model, "diff_attn")
        and hasattr(model.diff_attn, "multihead_scores")
        and model.diff_attn.multihead_scores is not None
    ):
        mh_raw = model.diff_attn.multihead_scores.detach().clone().cpu()
    elif model.get_attn_weights() is not None:
        # fallback: head-averaged attn_weights → (1, MAX_ATOMS, 167)로 확장
        aw = model.get_attn_weights().clone().cpu()  # (1, MAX_ATOMS, 167)
        mh_raw = aw.unsqueeze(1)                     # (1, 1, MAX_ATOMS, 167)
        logger.debug("Using fallback head-averaged attn_weights (no multihead_scores)")

    logger.debug(
        "Forward pass done: prob=%.2f%%, n_atoms=%d, mh_raw=%s",
        probability_pct, n_atoms,
        tuple(mh_raw.shape) if mh_raw is not None else None,
    )

    return probability_pct, mh_raw, n_atoms


def _run_model_inference(
    smiles: str,
    canonical: str,
    include_xai: bool,
) -> tuple[float, list[MaccsPatternScore], str]:
    """
    동기 추론 함수 전체. asyncio.to_thread()로 오프로드됩니다.

    Lock 구간:  Forward pass + multihead_scores 복사 (최소화)
    Lock 외부:  XAI numpy 처리, RDKit SVG 렌더링, 물성치 계산

    Returns:
        (probability_pct, top_maccs, molecule_svg)

    Raises:
        InferenceError
    """
    lock = get_inference_lock()

    # ── Lock: Forward pass는 직렬화 (멀티스레드 안전) ─────────────────────
    with lock:
        try:
            probability_pct, mh_raw, n_atoms = _run_forward_pass(canonical)
        except InferenceError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in _run_forward_pass for %s", canonical[:60])
            raise InferenceError(f"Forward pass 중 오류: {exc}") from exc

    # ── Lock 해제 후: XAI (numpy/RDKit, thread-safe) ──────────────────────
    top_maccs: list[MaccsPatternScore] = []
    svg = ""

    if include_xai:
        try:
            # probability_pct를 함께 전달 — SVG 색상(빨강/파랑) 결정에 사용
            top_maccs, svg = _run_xai_sync(mh_raw, canonical, n_atoms, probability_pct)
        except Exception as exc:
            logger.warning(
                "XAI processing failed for %r: %s — using plain SVG fallback",
                canonical[:60], exc,
            )
            try:
                svg = smiles_to_svg_plain(canonical)
            except Exception as svg_exc:
                logger.error("Plain SVG fallback also failed: %s", svg_exc)
                svg = ""
    else:
        # include_xai=False: plain SVG만 생성
        try:
            svg = smiles_to_svg_plain(canonical)
        except Exception as exc:
            logger.warning("Plain SVG generation failed: %s", exc)

    return probability_pct, top_maccs, svg


def _run_xai_sync(
    mh_raw: Optional[torch.Tensor],
    canonical: str,
    n_atoms: int,
    prob: float,
) -> tuple[list[MaccsPatternScore], str]:
    """
    XAI 처리 동기 버전 (Lock 외부에서 호출).
    build_xai_outputs와 동일 로직이나, asyncio.to_thread 없이 직접 호출.

    Args:
        mh_raw   : multihead_scores 클론 텐서 (또는 None)
        canonical: Canonical SMILES
        n_atoms  : 실제 원자 수
        prob     : DILI 예측 확률 (0.0 ~ 100.0 %) — SVG 색상 결정 (>45 → 빨강, ≤45 → 파랑)
    """
    from app.services.xai import build_xai_outputs
    return build_xai_outputs(mh_raw, canonical, n_atoms, prob, top_k=3)


# ─────────────────────────────────────────────────────────────────────────────
# 공개 비동기 API
# ─────────────────────────────────────────────────────────────────────────────

async def predict_single(
    smiles: str,
    include_xai: bool = True,
) -> SinglePredictResponse:
    """
    단일 SMILES DILI 예측 — 전체 파이프라인 공개 비동기 진입점.

    FastAPI 엔드포인트에서 직접 await 합니다.

    Args:
        smiles     : 사용자 입력 SMILES (유효하지 않으면 InvalidSmilesError)
        include_xai: False 시 SVG와 MACCS 패턴 생략 (빠른 응답이 필요한 경우)

    Returns:
        SinglePredictResponse — JSON 직렬화 가능한 Pydantic 모델

    Raises:
        InvalidSmilesError: SMILES 파싱 실패 시 (HTTP 422로 매핑)
        InferenceError    : 모델 추론 실패 시 (HTTP 500으로 매핑)
    """

    # ── Step 1: SMILES 유효성 검사 및 Canonical화 ─────────────────────────
    canonical: Optional[str] = await asyncio.to_thread(validate_and_canonicalize, smiles)
    if canonical is None:
        raise InvalidSmilesError(
            f"유효하지 않은 SMILES입니다: {smiles!r}\n"
            "올바른 SMILES 표기법을 확인하세요. (예: CCO, c1ccccc1, CC(=O)O)"
        )
    logger.info("predict_single: %r → canonical %r", smiles[:60], canonical[:60])

    # ── Step 2~7: 모델 추론 + XAI (스레드풀 오프로드) ─────────────────────
    try:
        probability_pct, top_maccs, svg = await asyncio.to_thread(
            _run_model_inference, smiles, canonical, include_xai
        )
    except (InvalidSmilesError, InferenceError):
        raise
    except Exception as exc:
        logger.exception("Unhandled exception in _run_model_inference for %s", canonical[:60])
        raise InferenceError(f"추론 파이프라인 예외: {exc}") from exc

    # ── Step 8: 물성치 계산 (RDKit, 추론과 독립적이므로 별도 오프로드) ────
    try:
        phys_data = await asyncio.to_thread(get_physicochemical_props, canonical)
    except Exception as exc:
        logger.warning("Physicochemical calculation failed for %r: %s", canonical[:60], exc)
        # 물성치 실패는 치명적이지 않음 — 기본값으로 채워 응답 유지
        phys_data = {
            "molecular_weight": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "tpsa": 0.0,
            "rotatable_bonds": 0,
            "qed": 0.0,
            "ring_count": 0,
            "aromatic_rings": 0,
        }

    # ── Step 9: 응답 조립 ─────────────────────────────────────────────────
    risk_level = "HIGH" if probability_pct >= settings.dili_threshold * 100.0 else "LOW"

    response = SinglePredictResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        probability=probability_pct,
        risk_level=risk_level,
        top_maccs_patterns=[
            MaccsPattern(
                bit_index=p.bit_index,
                name=p.name,
                importance=p.importance,
            )
            for p in top_maccs
        ],
        physicochemical=PhysChemProps(**phys_data),
        molecule_svg=svg,
    )

    logger.info(
        "predict_single complete: prob=%.2f%%, risk=%s, lipinski_violations=%d",
        probability_pct, risk_level, response.physicochemical.lipinski_violations,
    )

    return response
