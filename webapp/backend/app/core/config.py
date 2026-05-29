from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# webapp/backend/  (app/core/ 기준 두 단계 상위)
_BACKEND_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    서버 전역 설정.

    우선순위 (높음 → 낮음):
      1. 환경 변수 (대소문자 무시)
      2. .env 파일 (webapp/backend/.env)
      3. 아래 기본값

    필드 이름과 환경 변수 매핑 예시:
      k                    ← K
      d_model              ← D_MODEL
      model_weights_path   ← MODEL_WEIGHTS_PATH
      dili_threshold       ← DILI_THRESHOLD
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── GraphMACCSEncoder 아키텍처 (ml/config.py 값과 반드시 일치) ─────────────
    # 학습에 사용된 값과 다르면 가중치 로드 실패 또는 예측 오류 발생
    k: int = 32                # encode() 출력 차원
    d_model: int = 64          # MHA / Fusion 내부 차원
    num_heads: int = 4         # DifferentialCrossAttention 헤드 수
    dropout: float = 0.3       # 추론 시 eval()로 비활성화됨
    gine_layers: int = 2       # GINEConv 레이어 수
    gine_hidden: int = 64      # GINEConv hidden dim
    atom_feat_dim: int = 43    # get_atom_features() 출력 차원
    bond_feat_dim: int = 9     # get_bond_features() 출력 차원
    maccs_dim: int = 167       # RDKit MACCSkeys 벡터 길이
    max_atoms: int = 100       # to_dense_batch 패딩 상한
    max_length: int = 256      # ChemBERTa 토크나이저 최대 토큰 길이

    # ── ChemBERTa 모델 식별자 ────────────────────────────────────────────────
    # 로컬 경로 또는 HuggingFace Hub ID 모두 허용
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MLM"

    # ── 가중치 파일 경로 ──────────────────────────────────────────────────────
    # Docker 환경: COPY weights/ /app/weights/ → 기본값 그대로 사용
    # 로컬 개발:   .env에 MODEL_WEIGHTS_PATH=../../DGUDILI_2026/outputs/pretrained_graph_encoder.pt
    model_weights_path: Path = _BACKEND_DIR / "weights" / "pretrained_graph_encoder.pt"

    # ── 연산 장치 ─────────────────────────────────────────────────────────────
    # "auto"  → 런타임에서 CUDA 가용 여부 확인 후 자동 선택
    # "cpu"   → CPU 강제 (ECS Fargate 기본)
    # "cuda"  → GPU 강제 (로컬 GPU 개발 환경)
    device: str = "auto"

    # ── DILI 분류 임계값 ─────────────────────────────────────────────────────
    # P(DILI) >= dili_threshold → HIGH RISK
    # Step3_stacking.py의 OOF MCC-최적 임계값과 정렬 (기본 0.45)
    dili_threshold: float = 0.45

    # ── 배치 처리 ─────────────────────────────────────────────────────────────
    max_batch_rows: int = 5000    # 초과 시 422 Unprocessable Entity 반환
    batch_chunk_size: int = 32    # 내부 추론 루프 단위 크기

    # ── AWS ──────────────────────────────────────────────────────────────────
    s3_bucket_name: str = "dgudili-batch-results"
    aws_region: str = "ap-northeast-2"
    presigned_url_expiry: int = 3600  # S3 presigned URL 유효 시간(초)

    # ── 검증 ──────────────────────────────────────────────────────────────────

    @field_validator("d_model")
    @classmethod
    def _d_model_divisible(cls, v: int, info) -> int:
        num_heads = info.data.get("num_heads", 4)
        if v % num_heads != 0:
            raise ValueError(
                f"d_model({v}) must be divisible by num_heads({num_heads})"
            )
        return v

    @field_validator("dili_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"dili_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("device")
    @classmethod
    def _device_choices(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got '{v}'")
        return v

    # ── 편의 프로퍼티 ──────────────────────────────────────────────────────────

    @property
    def resolved_device(self) -> str:
        """
        실제 사용할 장치 문자열 반환.
        "auto"일 경우 torch.cuda.is_available()로 결정.
        model_loader.py에서 torch.device(settings.resolved_device) 로 사용.
        """
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def weights_path_str(self) -> str:
        """torch.load()에 전달할 절대 경로 문자열."""
        return str(self.model_weights_path.resolve())


# 모듈 임포트 시 즉시 생성 — 이후 `from app.core.config import settings` 로 접근
settings = Settings()
