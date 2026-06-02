"""
api/v1/endpoints/predict.py — DILI 예측 HTTP 엔드포인트

라우터 접두사: /api/v1/predict  (main.py → router.py 에서 마운트)

엔드포인트 목록:
  POST  /single                  — 단일 SMILES 예측
  POST  /batch                   — CSV 업로드, 배치 분석 시작 (task_id 즉시 반환)
  GET   /batch/status/{task_id}  — 배치 진행률 및 상태 조회
  GET   /batch/download/{task_id}— 완료된 배치 결과 CSV 다운로드

에러 HTTP 코드 매핑:
  InvalidSmilesError → 422 Unprocessable Entity
  InferenceError     → 500 Internal Server Error
  task not found     → 404 Not Found
  task not done      → 400 Bad Request
  file too large     → 413 Request Entity Too Large
  file type invalid  → 422 Unprocessable Entity
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse

from app.schemas.batch import BatchStatusResponse, BatchUploadResponse, TaskStatus
from app.schemas.predict import SinglePredictRequest, SinglePredictResponse
from app.services.batch_runner import (
    create_task,
    generate_presigned_url,
    get_task,
    run_batch,
)
from app.services.chemistry import name_to_smiles_via_pubchem, validate_and_canonicalize
from app.services.inference import InferenceError, InvalidSmilesError, predict_single

logger = logging.getLogger(__name__)

router = APIRouter()

# ── 업로드 제한 ──────────────────────────────────────────────────────────────
_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
_ACCEPTED_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",           # 일부 OS가 .csv를 text/plain으로 전송
    "application/octet-stream",  # 브라우저가 타입 감지 실패 시
})


# ─────────────────────────────────────────────────────────────────────────────
# POST /single — 단일 SMILES 예측
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/single",
    response_model=SinglePredictResponse,
    summary="단일 SMILES DILI 예측",
    description=(
        "SMILES 문자열 하나를 입력받아 DILI 확률, 위험 등급, "
        "물리화학 특성, XAI 원자 하이라이트 SVG, 상위 3개 MACCS 패턴을 반환합니다."
    ),
)
async def predict_single_endpoint(
    request: SinglePredictRequest,
) -> SinglePredictResponse:
    """
    단일 분자 DILI 예측.

    - 입력값이 유효한 SMILES이면 그대로 사용
    - 유효하지 않은 SMILES이면 PubChem PUG REST API로 분자 이름 조회 시도
    - 두 경우 모두 실패하면 400 Bad Request 반환
    - `include_xai=true` (기본): SVG 하이라이트 + MACCS 패턴 포함
    - `include_xai=false`: 확률/물성치만 반환 (빠른 응답)
    """
    # ── 입력 검증: SMILES 또는 영문 분자 이름 처리 ────────────────────────
    smiles_to_use = request.smiles

    # validate_and_canonicalize 내부에서 이미 모든 예외를 잡아 None을 반환하지만,
    # asyncio.to_thread 호출 자체에서 예기치 않은 오류가 발생할 경우에도
    # PubChem 경로로 안전하게 폴백하도록 한 겹 더 방어한다.
    try:
        canonical_check = await asyncio.to_thread(validate_and_canonicalize, request.smiles)
    except Exception as exc:
        logger.warning("validate_and_canonicalize raised unexpectedly for %r: %s", request.smiles[:40], exc)
        canonical_check = None

    if canonical_check is None:
        logger.info("[PubChem] %r is not valid SMILES — querying PubChem by name", request.smiles[:40])
        pubchem_smiles = await name_to_smiles_via_pubchem(request.smiles)
        if pubchem_smiles is None:
            logger.info("[PubChem] lookup failed for %r — returning 400", request.smiles[:40])
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 SMILES 구조식이거나 존재하지 않는 분자 이름입니다.",
            )
        logger.info("[PubChem] resolved %r → SMILES %r", request.smiles[:40], pubchem_smiles[:60])
        smiles_to_use = pubchem_smiles

    try:
        result = await predict_single(smiles_to_use, include_xai=request.include_xai)
    except InvalidSmilesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        logger.error("InferenceError for SMILES %r: %s", smiles_to_use[:60], exc)
        raise HTTPException(
            status_code=500,
            detail=f"모델 추론 중 오류가 발생했습니다: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in predict_single_endpoint")
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류입니다. 잠시 후 다시 시도하세요.",
        ) from exc

    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /batch — CSV 업로드, 배치 분석 시작
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=202,          # 202 Accepted: 요청은 수락됐으나 처리는 비동기
    summary="CSV 배치 DILI 스크리닝 시작",
    description=(
        "SMILES 컬럼이 포함된 CSV 파일을 업로드하면 배치 분석이 백그라운드로 시작되고 "
        "task_id를 즉시 반환합니다. "
        f"최대 파일 크기: 10 MB / 최대 행 수: 5,000행."
    ),
)
async def predict_batch_upload(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> BatchUploadResponse:
    """
    CSV 파일 업로드 → 배치 태스크 등록 → task_id 즉시 반환.

    프론트엔드는 반환된 task_id로 `/batch/status/{task_id}` 를 3초 간격으로 폴링하여
    진행률을 확인하고, 완료 후 `/batch/download/{task_id}` 로 결과를 다운로드합니다.
    """
    # ── 파일 타입 검증 ────────────────────────────────────────────────────
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").split(";")[0].strip().lower()

    if content_type not in _ACCEPTED_CONTENT_TYPES and not filename.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"CSV 파일만 허용됩니다. "
                f"수신된 Content-Type: {file.content_type!r}, 파일명: {file.filename!r}"
            ),
        )

    # ── 파일 크기 검증 ────────────────────────────────────────────────────
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="업로드된 파일이 비어 있습니다.")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"파일 크기({len(content) / 1024 / 1024:.1f} MB)가 "
                f"최대 허용 크기({_MAX_UPLOAD_BYTES // 1024 // 1024} MB)를 초과합니다."
            ),
        )

    # ── 태스크 등록 및 백그라운드 워커 시작 ──────────────────────────────
    task_id = create_task()
    background_tasks.add_task(run_batch, task_id, content)

    logger.info(
        "Batch task %s registered: file=%r, size=%d bytes",
        task_id, file.filename, len(content),
    )

    return BatchUploadResponse(
        task_id=task_id,
        message="배치 분석이 시작되었습니다. /batch/status/{task_id} 로 진행률을 확인하세요.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /batch/status/{task_id} — 진행률 및 상태 조회
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/batch/status/{task_id}",
    response_model=BatchStatusResponse,
    summary="배치 태스크 상태 조회",
    description="배치 분석의 현재 진행률(%)과 상태를 반환합니다. 완료 시 `has_result=true`.",
)
async def get_batch_status(task_id: str) -> BatchStatusResponse:
    """
    폴링용 상태 엔드포인트.

    프론트엔드 권장 폴링 주기: 3초.
    `status == "done"` 이고 `has_result == true` 이면 다운로드 버튼 활성화.
    """
    state = get_task(task_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"태스크 ID '{task_id}'를 찾을 수 없습니다. TTL(1시간) 초과 또는 잘못된 ID일 수 있습니다.",
        )

    progress_pct = round(
        state.processed / max(state.total, 1) * 100.0, 1
    ) if state.total > 0 else (100.0 if state.status == TaskStatus.DONE else 0.0)

    return BatchStatusResponse(
        task_id      = state.task_id,
        status       = state.status,
        total        = state.total,
        processed    = state.processed,
        progress_pct = progress_pct,
        has_result   = state.result_csv is not None or state.result_s3_key is not None,
        error        = state.error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /batch/download/{task_id} — 결과 CSV 다운로드
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/batch/download/{task_id}",
    summary="배치 결과 CSV 다운로드",
    description=(
        "완료된 배치 분석 결과를 CSV 파일로 다운로드합니다. "
        "S3가 설정된 경우 presigned URL로 리디렉션되고, "
        "그렇지 않으면 서버에서 직접 스트리밍됩니다."
    ),
    responses={
        200: {"description": "CSV 파일 스트리밍 응답"},
        302: {"description": "S3 presigned URL로 리디렉션"},
        400: {"description": "태스크가 아직 완료되지 않음"},
        404: {"description": "태스크를 찾을 수 없음"},
        500: {"description": "결과 데이터 없음 (내부 오류)"},
    },
)
async def download_batch_result(task_id: str):
    """
    배치 결과 CSV 다운로드.

    다운로드 우선순위:
      1. S3 presigned URL 발급 가능 → 302 리디렉션
      2. In-memory CSV 존재       → 직접 스트리밍 (StreamingResponse)
    """
    state = get_task(task_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"태스크 ID '{task_id}'를 찾을 수 없습니다.",
        )

    if state.status == TaskStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"태스크가 실패 상태입니다: {state.error}",
        )

    if state.status != TaskStatus.DONE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"태스크가 아직 완료되지 않았습니다. "
                f"현재 상태: {state.status.value} "
                f"({state.processed}/{max(state.total, 1)}행 처리 중)"
            ),
        )

    # ── S3 presigned URL 우선 ─────────────────────────────────────────────
    if state.result_s3_key:
        presigned_url = generate_presigned_url(state.result_s3_key)
        if presigned_url:
            logger.info("Redirecting task %s download to S3 presigned URL", task_id)
            return RedirectResponse(url=presigned_url, status_code=302)
        logger.warning("S3 presigned URL generation failed for task %s — falling back to in-memory", task_id)

    # ── In-memory CSV 스트리밍 ────────────────────────────────────────────
    if state.result_csv:
        filename = f"dili_results_{task_id[:8]}.csv"
        csv_bytes = state.result_csv.encode("utf-8")

        logger.info(
            "Streaming batch result for task %s (%d bytes)", task_id, len(csv_bytes)
        )
        return StreamingResponse(
            content=io.BytesIO(csv_bytes),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(csv_bytes)),
            },
        )

    # 여기 도달하면 DONE이지만 결과가 없는 비정상 상태
    logger.error("Task %s is DONE but has no result data", task_id)
    raise HTTPException(
        status_code=500,
        detail="결과 데이터를 찾을 수 없습니다. 배치 워커 로그를 확인하세요.",
    )
