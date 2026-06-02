"""
api/v1/router.py — API v1 라우터 집합

main.py에서 prefix="/api/v1" 로 마운트됩니다.
새 도메인 엔드포인트(예: /molecules, /reports)는 여기에 include_router로 추가하세요.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import predict

api_v1_router = APIRouter()

api_v1_router.include_router(
    predict.router,
    prefix="/predict",
    tags=["Prediction"],
)
