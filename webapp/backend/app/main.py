"""
app/main.py — FastAPI 애플리케이션 진입점

실행 방법:
  개발: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  프로덕션(Docker): uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
                   (workers > 1 불가: 모델 싱글턴이 프로세스 공유 안 됨)

환경 변수:
  ALLOWED_ORIGINS  — 쉼표 구분 Origin 목록 (기본: "*", 프로덕션 시 CloudFront 도메인으로 제한)
  ALLOW_CREDENTIALS— "true" / "false" (기본: "false", "*" origin 시 반드시 false)
"""

from __future__ import annotations

import logging
import logging.config
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.model_loader import is_ready, lifespan

# ─────────────────────────────────────────────────────────────────────────────
# 로깅 설정 (uvicorn 포맷과 통일)
# ─────────────────────────────────────────────────────────────────────────────

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class"    : "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level"   : os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console"],
    },
    "loggers": {
        # uvicorn 자체 로그는 INFO 유지
        "uvicorn"        : {"level": "INFO",    "propagate": True},
        "uvicorn.access" : {"level": "WARNING", "propagate": True},  # 접속 로그 억제
        # 프로젝트 내부 모듈 레벨 제어
        "app"            : {"level": os.getenv("LOG_LEVEL", "INFO").upper(), "propagate": True},
    },
})

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CORS 설정 (환경 변수로 제어)
# ─────────────────────────────────────────────────────────────────────────────

_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "*").strip()

# "*" 이면 모든 origin 허용 (개발/스테이징용)
# 프로덕션: ALLOWED_ORIGINS=https://xxxxxx.cloudfront.net,https://custom-domain.com
if _ORIGINS_ENV == "*":
    _ALLOWED_ORIGINS: list[str] = ["*"]
else:
    _ALLOWED_ORIGINS = [o.strip() for o in _ORIGINS_ENV.split(",") if o.strip()]

# allow_credentials=True는 allow_origins=["*"] 와 함께 사용 불가 (CORS 스펙)
_ALLOW_CREDENTIALS: bool = (
    os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"
    and _ALLOWED_ORIGINS != ["*"]
)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 애플리케이션 생성
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DGUDILI DILI Prediction API",
    description = (
        "GINEConv + ChemBERTa + Differential Cross-Attention 기반 "
        "약물 유발 간독성(DILI) 예측 REST API.\n\n"
        "- **단일 예측**: `POST /api/v1/predict/single`\n"
        "- **배치 스크리닝**: `POST /api/v1/predict/batch`\n"
        "- **배치 진행률**: `GET /api/v1/predict/batch/status/{task_id}`\n"
        "- **결과 다운로드**: `GET /api/v1/predict/batch/download/{task_id}`"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",    # Swagger UI (개발 편의)
    redoc_url   = "/redoc",
    lifespan    = lifespan,   # Startup(모델 로드) / Shutdown(리소스 해제)
)


# ─────────────────────────────────────────────────────────────────────────────
# 미들웨어
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _ALLOWED_ORIGINS,
    allow_credentials = _ALLOW_CREDENTIALS,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    expose_headers    = ["Content-Disposition"],   # 파일 다운로드 헤더 노출
    max_age           = 600,                       # preflight 캐시 10분
)

logger.info(
    "CORSMiddleware configured: origins=%s, credentials=%s",
    _ALLOWED_ORIGINS, _ALLOW_CREDENTIALS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 전역 예외 핸들러
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    핸들링되지 않은 예외를 500 JSON으로 변환.
    스택 트레이스는 서버 로그에만 기록되며 클라이언트에 노출되지 않습니다.
    """
    logger.exception(
        "Unhandled exception: %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도하세요.",
            "path"  : str(request.url.path),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 시스템 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="서버 헬스 체크",
    description="ECS 헬스 체크, ALB 타깃 그룹 체크, 프론트엔드 백엔드 가용성 확인에 사용.",
)
async def health_check() -> dict:
    """
    모델 로드 완료 여부를 포함한 서버 상태 반환.

    - `status: "ready"` — 모델 로드 완료, 요청 처리 가능
    - `status: "loading"` — 서버 시작 중 (lifespan 아직 완료 안 됨)

    ECS 헬스 체크 설정 권장:
      Path: /health, Healthy threshold: 2, Unhealthy threshold: 3,
      Interval: 30s, Timeout: 5s
    """
    ready = is_ready()
    return {
        "status" : "ready" if ready else "loading",
        "model"  : "GraphMACCSEncoder",
        "version": "1.0.0",
    }


@app.get(
    "/",
    tags=["System"],
    include_in_schema=False,  # Swagger에서 숨김
)
async def root() -> dict:
    return {
        "message": "DGUDILI API is running.",
        "docs"   : "/docs",
        "health" : "/health",
    }


# ─────────────────────────────────────────────────────────────────────────────
# API 라우터 등록
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(
    api_v1_router,
    prefix="/api/v1",
)

logger.info(
    "Routes registered: /api/v1/predict/single | /api/v1/predict/batch | "
    "/api/v1/predict/batch/status/{id} | /api/v1/predict/batch/download/{id}"
)
