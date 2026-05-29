"""
schemas/batch.py — 배치 처리 API Request / Response Pydantic 모델
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"   # 태스크 등록됨, 아직 시작 전
    RUNNING = "running"   # 백그라운드 워커 실행 중
    DONE    = "done"      # 정상 완료, 결과 다운로드 가능
    FAILED  = "failed"    # 에러로 중단됨


class BatchStatusResponse(BaseModel):
    """GET /batch/status/{task_id} 응답."""
    task_id     : str        = Field(..., description="배치 태스크 고유 ID (UUID)")
    status      : TaskStatus = Field(..., description="현재 태스크 상태")
    total       : int        = Field(..., ge=0, description="처리 예정 총 행 수")
    processed   : int        = Field(..., ge=0, description="처리 완료된 행 수")
    progress_pct: float      = Field(..., ge=0.0, le=100.0, description="진행률 (0.0 ~ 100.0 %)")
    has_result  : bool       = Field(
        default=False,
        description="True 이면 /batch/download/{task_id} 로 결과 CSV 다운로드 가능",
    )
    error       : str | None = Field(default=None, description="FAILED 상태 시 오류 메시지")


class BatchUploadResponse(BaseModel):
    """POST /batch 응답."""
    task_id: str = Field(..., description="프론트엔드가 폴링에 사용할 태스크 ID")
    message: str = Field(default="배치 분석이 시작되었습니다.")
