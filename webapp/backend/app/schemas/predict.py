"""
schemas/predict.py — 단일 예측 API의 Request / Response Pydantic 모델

FastAPI 엔드포인트가 이 모델을 사용해 JSON 직렬화/역직렬화를 수행합니다.
pydantic v2 기준 작성.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────

class SinglePredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="입력 SMILES 문자열 (예: CCO, CC(=O)Oc1ccccc1C(=O)O)",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    include_xai: bool = Field(
        default=True,
        description="True 시 원자 하이라이트 SVG 및 MACCS 패턴 포함 반환",
    )

    @field_validator("smiles")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Response — 서브 모델
# ─────────────────────────────────────────────────────────────────────────────

class MaccsPattern(BaseModel):
    """상위 기여 MACCS 구조적 키 하나의 정보."""
    bit_index: int = Field(..., ge=1, le=166, description="MACCS bit 인덱스 (1~166)")
    name: str = Field(..., description="Durant et al. 2002 기반 인간 가독 이름")
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="전체 어텐션 대비 해당 패턴의 기여 비율 (0~1)",
    )


class ToxicReason(BaseModel):
    """SMARTS 작용기 기반 독성 원인 기여 정보 (XAI 해석 결과)."""
    name: str = Field(
        ...,
        description="작용기 이름 (예: 'Nitro Group') 또는 폴백 시 'Atom #N (X)' 형태",
    )
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="TOP-K 내 합산 기준 정규화된 기여도 (0~1). UI Progress Bar 폭으로 사용.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="기여도 순위 (1 = 가장 높음)",
    )


class PhysChemProps(BaseModel):
    """RDKit Descriptors 기반 물리화학 특성."""
    molecular_weight: float = Field(..., description="정확 분자량 (ExactMolWt, Da)")
    logp: float = Field(..., description="Wildman-Crippen LogP")
    hbd: int = Field(..., ge=0, description="수소 결합 공여체(HBD) 수")
    hba: int = Field(..., ge=0, description="수소 결합 수용체(HBA) 수")
    tpsa: float = Field(..., ge=0.0, description="위상 극성 표면적 (Å²)")
    rotatable_bonds: int = Field(..., ge=0, description="회전 가능 결합 수")
    qed: float = Field(..., ge=0.0, le=1.0, description="QED 약물 유사성 점수 (0~1)")
    ring_count: int = Field(..., ge=0, description="총 링 수")
    aromatic_rings: int = Field(..., ge=0, description="방향족 링 수")

    @computed_field
    @property
    def lipinski_violations(self) -> int:
        """
        Lipinski Rule-of-Five 위반 항목 수.
        MW > 500, LogP > 5, HBD > 5, HBA > 10 각각을 위반으로 집계.
        """
        count = 0
        if self.molecular_weight > 500.0:
            count += 1
        if self.logp > 5.0:
            count += 1
        if self.hbd > 5:
            count += 1
        if self.hba > 10:
            count += 1
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Response — 최상위
# ─────────────────────────────────────────────────────────────────────────────

class SinglePredictResponse(BaseModel):
    """POST /api/v1/predict/single 응답."""
    smiles: str = Field(..., description="요청에 입력된 원본 SMILES")
    canonical_smiles: str = Field(..., description="RDKit Canonical SMILES (원자 인덱스 고정)")
    probability: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="DILI 발생 확률 (0.0 ~ 100.0 %)",
    )
    risk_level: str = Field(
        ...,
        pattern="^(HIGH|LOW)$",
        description="DILI 위험 등급: 'HIGH' (≥ threshold) 또는 'LOW'",
    )
    top_maccs_patterns: list[MaccsPattern] = Field(
        default_factory=list,
        description="어텐션 기여도 상위 3개 MACCS 구조 패턴",
    )
    toxic_reasons: list[ToxicReason] = Field(
        default_factory=list,
        description=(
            "SMARTS 작용기 매핑 기반 독성 원인 TOP 3. "
            "include_xai=False 또는 분자 구조 특이 케이스 시 빈 배열 또는 "
            "원자 레벨 폴백('Atom #N (X)') 포함."
        ),
    )
    physicochemical: PhysChemProps = Field(..., description="물리화학 특성 세트")
    molecule_svg: str = Field(
        default="",
        description="XAI 원자 하이라이트가 포함된 RDKit SVG 문자열. include_xai=False 시 빈 문자열.",
    )
