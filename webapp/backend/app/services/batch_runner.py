"""
services/batch_runner.py — CSV 배치 스크리닝 백그라운드 워커

설계 원칙:
  - FastAPI BackgroundTasks는 동기 함수를 자동으로 ThreadPoolExecutor에서 실행.
    → run_batch()는 sync 함수로 작성하여 추가 이벤트루프 없이 직접 ML 추론 호출.
  - 진행률(state.processed)은 각 행 처리 후 즉시 갱신 → 프론트 폴링에서 실시간 반영.
  - 각 행의 에러는 전체 배치를 중단하지 않고 Error 컬럼에 기록 후 계속 진행.
  - 완료된 결과는 in-memory CSV 문자열로 보관 (task_store).
    S3 버킷이 설정되어 있으면 추가로 업로드하고 presigned URL 제공.
  - TTL(1시간)이 지난 완료/실패 태스크는 다음 요청 시 lazy-GC로 메모리 회수.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import pandas as pd

from app.core.config import settings
from app.schemas.batch import TaskStatus
from app.services.chemistry import get_physicochemical_props, validate_and_canonicalize
from app.services.inference import InferenceError, InvalidSmilesError, _run_model_inference

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 상태 및 레지스트리
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskState:
    """배치 태스크 하나의 전체 생명주기 상태."""
    task_id      : str
    created_at   : float        = field(default_factory=time.time)
    status       : TaskStatus   = TaskStatus.PENDING
    total        : int          = 0      # 처리 예정 총 행 수 (파싱 후 확정)
    processed    : int          = 0      # 완료 행 수 (실시간 갱신)
    error        : Optional[str] = None  # 배치 전체 실패 시 사유
    result_csv   : Optional[str] = None  # 완료 후 in-memory CSV 문자열
    result_s3_key: Optional[str] = None  # S3 업로드 성공 시 object key


# 태스크 레지스트리 — 프로세스 범위 싱글턴
task_store: dict[str, TaskState] = {}
_task_lock: Lock = Lock()

# 완료/실패 태스크 보관 기간
_TASK_TTL: int = 3600  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# 태스크 수명주기 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def create_task() -> str:
    """새 태스크 등록 후 task_id(UUID) 반환."""
    task_id = str(uuid.uuid4())
    with _task_lock:
        _cleanup_stale_unsafe()            # lazy GC
        task_store[task_id] = TaskState(task_id=task_id)
    logger.debug("Task created: %s", task_id)
    return task_id


def get_task(task_id: str) -> Optional[TaskState]:
    """task_id로 TaskState 조회. 없으면 None."""
    return task_store.get(task_id)


def _cleanup_stale_unsafe() -> None:
    """_task_lock 보유 상태에서만 호출. TTL 초과 태스크 제거."""
    cutoff = time.time() - _TASK_TTL
    stale = [
        tid for tid, s in task_store.items()
        if s.status in (TaskStatus.DONE, TaskStatus.FAILED)
        and s.created_at < cutoff
    ]
    for tid in stale:
        del task_store[tid]
    if stale:
        logger.info("GC: removed %d stale tasks", len(stale))


# ─────────────────────────────────────────────────────────────────────────────
# CSV 파싱 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

_SMILES_COL_CANDIDATES = [
    "smiles", "SMILES", "Smiles", "smi", "SMI",
    "canonical_smiles", "Canonical_SMILES", "molecule", "mol",
]


def _parse_csv(content: bytes) -> pd.DataFrame:
    """
    CSV bytes → DataFrame. UTF-8 / UTF-8-BOM / Latin-1 순으로 인코딩 시도.

    Raises:
        ValueError: 모든 인코딩 실패, 또는 pandas 파싱 오류 시
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                dtype=str,         # 전체 문자열 읽기: SMILES 파싱 전 타입 충돌 방지
                keep_default_na=False,
            )
            logger.debug("CSV parsed with encoding=%s, shape=%s", encoding, df.shape)
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV 파싱 실패: {exc}") from exc
    raise ValueError(
        "CSV 파일 인코딩을 인식할 수 없습니다. UTF-8 또는 Latin-1 CSV를 사용하세요."
    )


def _detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """
    DataFrame에서 SMILES 컬럼명 자동 탐지.
    우선순위: 사전 정의된 후보명 → 첫 번째 문자열 컬럼.
    """
    for candidate in _SMILES_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    # Fallback: 첫 번째 object(문자열) 컬럼
    for col in df.columns:
        if df[col].dtype == object:
            logger.warning("SMILES 컬럼 자동 탐지: '%s' 사용 (후보명 불일치)", col)
            return col
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 단일 행 추론 (동기)
# ─────────────────────────────────────────────────────────────────────────────

_ERROR_ROW_DEFAULTS: dict = {
    "Canonical_SMILES" : "",
    "DILI_Probability" : None,
    "Risk_Level"       : "ERROR",
    "Molecular_Weight" : None,
    "LogP"             : None,
    "HBD"              : None,
    "HBA"              : None,
    "TPSA"             : None,
    "Rotatable_Bonds"  : None,
    "QED"              : None,
    "Lipinski_Violations": None,
    "Error"            : "",
}


def _process_row(smiles_raw: str) -> dict:
    """
    단일 SMILES를 추론하고 결과 행(dict)을 반환.
    에러 발생 시에도 Error 컬럼에 메시지를 담아 반환 (예외 비전파).

    배치에서는 include_xai=False로 호출하여 SVG 생성을 생략합니다.
    """
    smiles = smiles_raw.strip()

    # ── SMILES 유효성 검사 ─────────────────────────────────────────────────
    canonical = validate_and_canonicalize(smiles)
    if canonical is None:
        return {**_ERROR_ROW_DEFAULTS, "Error": f"Invalid SMILES: {smiles!r}"}

    # ── 모델 추론 (include_xai=False → SVG/MACCS 생략, 속도 최적화) ───────
    try:
        prob_pct, *_ = _run_model_inference(smiles, canonical, include_xai=False)
    except (InvalidSmilesError, InferenceError) as exc:
        return {
            **_ERROR_ROW_DEFAULTS,
            "Canonical_SMILES": canonical,
            "Error": f"추론 실패: {exc}",
        }
    except Exception as exc:
        logger.exception("Unexpected inference error for SMILES %r", smiles[:60])
        return {
            **_ERROR_ROW_DEFAULTS,
            "Canonical_SMILES": canonical,
            "Error": f"예기치 않은 오류: {exc}",
        }

    # ── 물성치 계산 ────────────────────────────────────────────────────────
    try:
        phys = get_physicochemical_props(canonical)
    except Exception as exc:
        logger.warning("PhysChem failed for %r: %s", canonical[:60], exc)
        phys = {
            "molecular_weight": None, "logp": None,
            "hbd": None, "hba": None, "tpsa": None,
            "rotatable_bonds": None, "qed": None,
        }

    risk_level = "HIGH" if prob_pct >= settings.dili_threshold * 100.0 else "LOW"

    mw  = phys.get("molecular_weight") or 0.0
    lp  = phys.get("logp") or 0.0
    hbd = phys.get("hbd") or 0
    hba = phys.get("hba") or 0
    violations = int(mw > 500) + int(lp > 5) + int(hbd > 5) + int(hba > 10)

    return {
        "Canonical_SMILES"   : canonical,
        "DILI_Probability"   : prob_pct,
        "Risk_Level"         : risk_level,
        "Molecular_Weight"   : phys.get("molecular_weight"),
        "LogP"               : phys.get("logp"),
        "HBD"                : phys.get("hbd"),
        "HBA"                : phys.get("hba"),
        "TPSA"               : phys.get("tpsa"),
        "Rotatable_Bonds"    : phys.get("rotatable_bonds"),
        "QED"                : phys.get("qed"),
        "Lipinski_Violations": violations,
        "Error"              : "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# AWS S3 연동 (선택적)
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_upload_s3(task_id: str, csv_content: str) -> Optional[str]:
    """
    결과 CSV를 S3에 업로드하고 object key를 반환.
    s3_bucket_name이 기본값("dgudili-batch-results")이면 실제 연결 시도하지 않음.
    실패해도 예외를 전파하지 않고 None 반환 (in-memory 결과는 유지됨).
    """
    # 기본 더미 버킷명이면 S3 연동 비활성
    if not settings.s3_bucket_name:
        return None

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        client = boto3.client("s3", region_name=settings.aws_region)
        key = f"batch-results/{task_id}.csv"
        client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=key,
            Body=csv_content.encode("utf-8"),
            ContentType="text/csv; charset=utf-8",
            ContentDisposition=f'attachment; filename="dili_results_{task_id[:8]}.csv"',
        )
        logger.info("S3 upload OK: s3://%s/%s", settings.s3_bucket_name, key)
        return key
    except Exception as exc:
        logger.warning("S3 upload failed (task=%s): %s", task_id, exc)
        return None


def generate_presigned_url(s3_key: str) -> Optional[str]:
    """S3 object에 대한 presigned GET URL 생성. 실패 시 None."""
    try:
        import boto3

        client = boto3.client("s3", region_name=settings.aws_region)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": s3_key},
            ExpiresIn=settings.presigned_url_expiry,
        )
    except Exception as exc:
        logger.warning("Presigned URL generation failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 배치 워커 (FastAPI BackgroundTasks → ThreadPoolExecutor에서 실행)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(task_id: str, content: bytes) -> None:
    """
    CSV 배치 추론 백그라운드 워커.

    FastAPI BackgroundTasks는 sync 함수를 자동으로 스레드풀에서 실행합니다.
    내부적으로 _run_model_inference()를 직접 호출하며, 해당 함수는
    threading.Lock(get_inference_lock())으로 Forward pass를 직렬화합니다.

    실행 흐름:
      1. CSV 파싱 및 SMILES 컬럼 탐지
      2. 행 수 제한 검증 (max_batch_rows)
      3. 행 단위 추론 + 진행률 실시간 갱신
      4. 결과 DataFrame → CSV 문자열 저장
      5. S3 업로드 시도 (선택적)
      6. 상태를 DONE 또는 FAILED로 전환

    에러 정책:
      - 개별 행 에러: Error 컬럼에 기록, 배치 계속 진행
      - 전체 배치 에러 (CSV 파싱 불가, 컬럼 없음 등): FAILED 상태로 즉시 종료
    """
    state = task_store.get(task_id)
    if state is None:
        logger.error("run_batch: task_id=%s not found in task_store", task_id)
        return

    state.status = TaskStatus.RUNNING
    logger.info("Batch task %s STARTED", task_id)

    try:
        # ── 1. CSV 파싱 ──────────────────────────────────────────────────────
        df = _parse_csv(content)

        if df.empty:
            raise ValueError("업로드된 CSV 파일이 비어 있습니다.")

        # ── 2. SMILES 컬럼 탐지 ──────────────────────────────────────────────
        smiles_col = _detect_smiles_column(df)
        if smiles_col is None:
            raise ValueError(
                "SMILES 컬럼을 찾을 수 없습니다. "
                f"컬럼명을 다음 중 하나로 설정하세요: {', '.join(_SMILES_COL_CANDIDATES[:5])}"
            )

        # ── 3. 행 수 제한 검증 ───────────────────────────────────────────────
        total_rows = len(df)
        if total_rows > settings.max_batch_rows:
            raise ValueError(
                f"최대 허용 행 수({settings.max_batch_rows:,}행)를 초과합니다. "
                f"업로드 행 수: {total_rows:,}행"
            )

        state.total = total_rows
        logger.info(
            "Batch task %s: %d rows, SMILES col='%s', columns=%s",
            task_id, total_rows, smiles_col,
            list(df.columns[:8]),
        )

        # SMILES 외 원본 컬럼 목록 (ID, Name 등 보존용)
        extra_cols = [c for c in df.columns if c != smiles_col]

        # ── 4. 행 단위 추론 ──────────────────────────────────────────────────
        result_rows: list[dict] = []

        for row_idx in range(total_rows):
            row = df.iloc[row_idx]
            smiles_raw = str(row[smiles_col])

            # 빈 셀 건너뜀 (nan, None, 공백 포함)
            stripped = smiles_raw.strip()
            if not stripped or stripped.lower() in ("nan", "none", ""):
                result_row = {
                    "SMILES_Input": smiles_raw,
                    **_ERROR_ROW_DEFAULTS,
                    "Risk_Level": "SKIP",
                    "Error": "빈 SMILES",
                }
            else:
                row_result = _process_row(smiles_raw)
                result_row = {"SMILES_Input": smiles_raw, **row_result}

            # 원본 컬럼 값 보존 (ID, 화합물명 등)
            for col in extra_cols:
                if col not in result_row:
                    result_row[col] = row.get(col, "")

            result_rows.append(result_row)
            state.processed = row_idx + 1  # 실시간 진행률 갱신

            if (row_idx + 1) % 100 == 0:
                logger.info(
                    "Batch task %s: %d/%d rows processed",
                    task_id, state.processed, total_rows,
                )

        # ── 5. 결과 DataFrame 구성 (컬럼 순서 정렬) ─────────────────────────
        primary_cols = [
            "SMILES_Input", "Canonical_SMILES",
            "DILI_Probability", "Risk_Level",
            "Molecular_Weight", "LogP", "HBD", "HBA",
            "TPSA", "Rotatable_Bonds", "QED",
            "Lipinski_Violations", "Error",
        ]
        preserved_cols = [c for c in extra_cols if c not in primary_cols]
        ordered_cols   = primary_cols + preserved_cols

        result_df = pd.DataFrame(result_rows)
        # 존재하는 컬럼만 선택 (누락 방지)
        final_cols = [c for c in ordered_cols if c in result_df.columns]
        result_df  = result_df[final_cols]

        state.result_csv = result_df.to_csv(index=False, encoding="utf-8")
        logger.info("Batch task %s: result CSV ready (%d bytes)", task_id, len(state.result_csv))

        # ── 6. S3 업로드 (선택적) ─────────────────────────────────────────────
        s3_key = _maybe_upload_s3(task_id, state.result_csv)
        if s3_key:
            state.result_s3_key = s3_key

        # ── 7. 완료 ───────────────────────────────────────────────────────────
        state.status = TaskStatus.DONE
        logger.info(
            "Batch task %s DONE: %d/%d rows | result=%d bytes",
            task_id, state.processed, state.total, len(state.result_csv),
        )

    except Exception as exc:
        state.status = TaskStatus.FAILED
        state.error  = str(exc)
        logger.error("Batch task %s FAILED: %s", task_id, exc, exc_info=True)
