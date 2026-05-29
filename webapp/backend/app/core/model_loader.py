"""
model_loader.py — GraphMACCSEncoder + AutoTokenizer 싱글턴 관리

FastAPI lifespan 컨텍스트 매니저를 통해 서버 시작 시 모델을 한 번만 로드하고,
서버 종료 시 메모리를 정리합니다.

사용법:
    # main.py
    from app.core.model_loader import lifespan
    app = FastAPI(lifespan=lifespan)

    # 엔드포인트 / 서비스 레이어
    from app.core.model_loader import get_model, get_tokenizer, get_device
    model = get_model()
    tokenizer = get_tokenizer()
    device = get_device()

임포트 설계 주의사항:
    ml/model.py는 내부에서 `from differential_attention import ...`를 수행합니다
    (flat import — DGUDILI_2026/src/ 기준 경로).
    이 파일 로드 시점에 sys.path에 app/ml/ 디렉터리를 추가하여
    differential_attention.py를 찾을 수 있게 합니다.
    app/ml/ 내 원본 ML 파일을 수정하지 않기 위한 조치입니다.
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── sys.path 패치: app/ml/ 내 flat import(differential_attention) 해결 ─────────
# model.py.__init__() 실행 시 `from differential_attention import ...`가 호출됨.
# Python은 sys.path에서 differential_attention.py를 탐색하므로
# app/ml/ 디렉터리를 경로에 추가해 두어야 합니다.
# 이 처리는 GraphMACCSEncoder 클래스 임포트(아래) 이전에 반드시 완료되어야 합니다.
_ML_DIR = Path(__file__).resolve().parent.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

# sys.path 패치 후 임포트 — noqa: E402 (모듈 레벨 조건 설치 패턴)
from app.ml.model import GraphMACCSEncoder  # noqa: E402

# ── 모듈 레벨 싱글턴 ──────────────────────────────────────────────────────────
_model: Optional[GraphMACCSEncoder] = None
_tokenizer = None                         # transformers.AutoTokenizer
_device: torch.device = torch.device("cpu")

# ── 추론 직렬화 Lock ──────────────────────────────────────────────────────────
# model.diff_attn.multihead_scores는 Forward pass마다 덮어쓰이는 가변 속성.
# 동시 요청이 Forward pass를 병렬 실행하면 데이터 경쟁이 발생하므로
# Forward pass + multihead_scores 캡처 구간을 직렬화합니다.
# numpy/RDKit 후처리는 Lock 외부에서 수행하여 처리량을 유지합니다.
_inference_lock: threading.Lock = threading.Lock()


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    """settings.resolved_device 문자열을 torch.device 객체로 변환."""
    device_str = settings.resolved_device
    device = torch.device(device_str)
    logger.info("Using device: %s", device)
    return device


def _load_tokenizer():
    """
    ChemBERTa AutoTokenizer 로드.
    settings.chemberta_model_name은 HuggingFace Hub ID 또는 로컬 디렉터리 경로.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", settings.chemberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.chemberta_model_name,
        use_fast=True,
    )
    logger.info("Tokenizer loaded (vocab_size=%d)", tokenizer.vocab_size)
    return tokenizer


def _load_model(device: torch.device) -> GraphMACCSEncoder:
    """
    GraphMACCSEncoder 초기화 → 가중치 로드 → eval 모드 전환.

    Args:
        device: 연산 장치 (torch.device("cpu") 또는 torch.device("cuda"))

    Raises:
        FileNotFoundError: 가중치 파일이 존재하지 않을 때.
        RuntimeError: state_dict 로드 중 아키텍처 불일치 시.
    """
    weights_path = Path(settings.model_weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"모델 가중치 파일을 찾을 수 없습니다: {weights_path}\n"
            f"다음 중 하나를 확인하세요:\n"
            f"  1) DGUDILI_2026/outputs/pretrained_graph_encoder.pt 를\n"
            f"     webapp/backend/weights/ 에 복사했는지 확인\n"
            f"  2) .env 파일의 MODEL_WEIGHTS_PATH 환경 변수가 올바른지 확인"
        )

    logger.info("Initializing GraphMACCSEncoder with config: "
                "k=%d, d_model=%d, num_heads=%d, gine_layers=%d, gine_hidden=%d",
                settings.k, settings.d_model, settings.num_heads,
                settings.gine_layers, settings.gine_hidden)

    encoder = GraphMACCSEncoder(
        model_name=settings.chemberta_model_name,
        d_model=settings.d_model,
        num_heads=settings.num_heads,
        k=settings.k,
        gine_layers=settings.gine_layers,
        gine_hidden=settings.gine_hidden,
        atom_feat_dim=settings.atom_feat_dim,
        bond_feat_dim=settings.bond_feat_dim,
        maccs_dim=settings.maccs_dim,
        max_atoms=settings.max_atoms,
        dropout=settings.dropout,
    )

    logger.info("Loading weights from: %s", weights_path)
    state_dict = torch.load(
        str(weights_path),
        map_location=device,
        weights_only=True,   # 신뢰할 수 없는 pickle 실행 방지 (PyTorch 2.0+)
    )

    missing, unexpected = encoder.load_state_dict(state_dict, strict=True)
    if missing:
        logger.warning("Missing keys in state_dict: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in state_dict: %s", unexpected)

    encoder.to(device)
    encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(
        "GraphMACCSEncoder ready — total params: %.2fM, trainable: %.2fM, device: %s",
        n_params / 1e6, n_trainable / 1e6, device,
    )
    return encoder


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 생명주기 관리.

    Startup:  장치 감지 → 토크나이저 로드 → 모델 로드 → eval 모드
    Shutdown: 싱글턴 해제 → CUDA 메모리 정리

    사용:
        app = FastAPI(lifespan=lifespan)
    """
    global _model, _tokenizer, _device

    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTUP: DGUDILI Backend 초기화 시작")
    logger.info("=" * 60)

    _device = _resolve_device()
    _tokenizer = _load_tokenizer()
    _model = _load_model(_device)

    logger.info("=" * 60)
    logger.info("STARTUP COMPLETE: 서버가 요청을 받을 준비가 됐습니다.")
    logger.info("  - 모델: GraphMACCSEncoder (k=%d, d_model=%d)", settings.k, settings.d_model)
    logger.info("  - 가중치: %s", settings.weights_path_str)
    logger.info("  - 장치: %s", _device)
    logger.info("  - DILI 임계값: %.2f", settings.dili_threshold)
    logger.info("=" * 60)

    yield  # ← 서버 실행 구간 (요청 처리)

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("SHUTDOWN: 리소스 해제 중...")
    _model = None
    _tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA 캐시 비움 완료")

    logger.info("SHUTDOWN COMPLETE")


# ── 공개 접근자 (서비스 레이어에서 호출) ─────────────────────────────────────

def get_model() -> GraphMACCSEncoder:
    """
    로드된 GraphMACCSEncoder 싱글턴 반환.

    Raises:
        RuntimeError: lifespan startup이 완료되지 않았을 때.
    """
    if _model is None:
        raise RuntimeError(
            "GraphMACCSEncoder가 초기화되지 않았습니다. "
            "서버 startup 로그를 확인하세요."
        )
    return _model


def get_tokenizer():
    """
    로드된 AutoTokenizer 싱글턴 반환.

    Raises:
        RuntimeError: lifespan startup이 완료되지 않았을 때.
    """
    if _tokenizer is None:
        raise RuntimeError(
            "AutoTokenizer가 초기화되지 않았습니다. "
            "서버 startup 로그를 확인하세요."
        )
    return _tokenizer


def get_device() -> torch.device:
    """현재 사용 중인 torch.device 반환."""
    return _device


def get_inference_lock() -> threading.Lock:
    """
    모델 Forward pass 직렬화용 Lock 반환.

    inference.py의 _run_forward_pass()에서 사용:
        with get_inference_lock():
            logit = model(...)
            mh_raw = model.diff_attn.multihead_scores.clone()
    """
    return _inference_lock


def is_ready() -> bool:
    """모델과 토크나이저가 모두 로드됐는지 확인. /health 엔드포인트에서 사용."""
    return _model is not None and _tokenizer is not None
