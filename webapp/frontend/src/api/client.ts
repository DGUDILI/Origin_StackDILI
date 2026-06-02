/**
 * api/client.ts — Axios 인스턴스 및 백엔드 API 함수 세트
 *
 * URL 전략:
 *   개발 환경: VITE_API_BASE_URL="" → Vite dev server가 /api/* 요청을 localhost:8000으로 프록시
 *   프로덕션 : VITE_API_BASE_URL="https://your-alb.elb.amazonaws.com" → 직접 연결
 *
 * 엔드포인트 매핑:
 *   predictSingle       → POST  /api/v1/predict/single
 *   uploadBatchCSV      → POST  /api/v1/predict/batch
 *   getBatchStatus      → GET   /api/v1/predict/batch/status/{taskId}
 *   getBatchDownloadUrl → (sync) /api/v1/predict/batch/download/{taskId}
 */

import axios, {
  type AxiosInstance,
  type AxiosError,
  type InternalAxiosRequestConfig,
} from 'axios';

import type {
  SinglePredictRequest,
  SinglePredictResponse,
  BatchUploadResponse,
  BatchStatusResponse,
} from '@/types/predict';

// ─────────────────────────────────────────────────────────────────────────────
// 환경 변수
// ─────────────────────────────────────────────────────────────────────────────

/** 백엔드 Origin. 개발 시 빈 문자열 → Vite proxy가 처리 */
const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

/** 모든 예측 엔드포인트 공통 prefix */
const PREDICT_PREFIX = '/api/v1/predict' as const;

// ─────────────────────────────────────────────────────────────────────────────
// 커스텀 에러 클래스
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 백엔드 HTTP 에러를 래핑하는 도메인 에러.
 *
 * 사용법:
 *   try { await predictSingle(smiles) }
 *   catch (e) {
 *     if (e instanceof ApiError) {
 *       if (e.status === 422) { // 유효하지 않은 SMILES }
 *       if (e.status === 500) { // 서버 추론 오류 }
 *     }
 *   }
 */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    /** FastAPI 상세 메시지 (배열일 경우 첫 번째 항목의 msg 추출) */
    public readonly detail?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** axios 에러에서 사람이 읽을 수 있는 메시지를 추출 */
function extractErrorMessage(error: AxiosError): string {
  const data = error.response?.data as Record<string, unknown> | undefined;
  if (!data) return '서버와 통신할 수 없습니다.';

  // FastAPI validation error: { detail: [{ msg: "...", ... }] }
  if (Array.isArray(data.detail)) {
    const first = data.detail[0] as Record<string, unknown>;
    return String(first?.msg ?? data.detail[0]);
  }

  // FastAPI HTTPException: { detail: "..." }
  if (typeof data.detail === 'string') return data.detail;

  return `오류가 발생했습니다 (HTTP ${error.response?.status ?? 0})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Axios 인스턴스
// ─────────────────────────────────────────────────────────────────────────────

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  /**
   * 기본 타임아웃: 120초
   * GraphMACCSEncoder CPU 추론은 분자 복잡도에 따라 10~60초 소요 가능.
   * 배치 업로드는 개별 함수에서 별도 타임아웃 지정.
   */
  timeout: 120_000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// ── Request 인터셉터: 요청 로깅 (개발 디버깅용) ───────────────────────────
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (import.meta.env.DEV) {
      console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
);

// ── Response 인터셉터: 에러 정규화 ────────────────────────────────────────
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK') {
      return Promise.reject(
        new ApiError(0, '서버에 연결할 수 없습니다. 백엔드 서버(port 8000)가 실행 중인지 확인하세요.'),
      );
    }

    if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
      return Promise.reject(
        new ApiError(408, '요청 시간이 초과되었습니다. 분자가 너무 크거나 서버가 과부하 상태일 수 있습니다.'),
      );
    }

    if (error.response) {
      const status  = error.response.status;
      const message = extractErrorMessage(error);
      return Promise.reject(new ApiError(status, message, error.response.data));
    }

    return Promise.reject(new ApiError(0, error.message));
  },
);

// ─────────────────────────────────────────────────────────────────────────────
// API 함수
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 단일 SMILES DILI 예측.
 *
 * @param smiles     - SMILES 문자열 (예: "CC(=O)Oc1ccccc1C(=O)O")
 * @param includeXai - true: XAI SVG + MACCS 패턴 포함 (기본값 true)
 * @returns SinglePredictResponse — 확률, 등급, 물성치, SVG, MACCS 패턴
 *
 * @throws ApiError status=422 — 유효하지 않은 SMILES
 * @throws ApiError status=500 — 모델 추론 실패
 */
export async function predictSingle(
  smiles: string,
  includeXai = true,
): Promise<SinglePredictResponse> {
  const body: SinglePredictRequest = {
    smiles,
    include_xai: includeXai,
  };
  const { data } = await apiClient.post<SinglePredictResponse>(
    `${PREDICT_PREFIX}/single`,
    body,
  );
  return data;
}

/**
 * SMILES 컬럼이 포함된 CSV 파일을 업로드하고 배치 분석을 시작합니다.
 * 비동기 처리: 즉시 task_id를 반환하며 실제 분석은 백엔드 백그라운드에서 수행됩니다.
 *
 * @param file - .csv 파일 (최대 10 MB, 최대 5,000행)
 * @returns BatchUploadResponse — { task_id, message }
 *
 * @throws ApiError status=413 — 파일 크기 초과
 * @throws ApiError status=422 — CSV가 아닌 파일 또는 빈 파일
 */
export async function uploadBatchCSV(file: File): Promise<BatchUploadResponse> {
  const form = new FormData();
  form.append('file', file);

  const { data } = await apiClient.post<BatchUploadResponse>(
    `${PREDICT_PREFIX}/batch`,
    form,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30_000,  // 파일 업로드는 30초로 충분
    },
  );
  return data;
}

/**
 * 배치 태스크의 현재 진행률과 상태를 조회합니다.
 * 프론트엔드에서 일정 간격(VITE_BATCH_POLL_INTERVAL_MS, 기본 3초)으로 폴링합니다.
 *
 * @param taskId - uploadBatchCSV()가 반환한 task_id
 * @returns BatchStatusResponse — 상태, 진행률, 완료 여부
 *
 * @throws ApiError status=404 — 존재하지 않거나 TTL 만료된 task_id
 */
export async function getBatchStatus(taskId: string): Promise<BatchStatusResponse> {
  const { data } = await apiClient.get<BatchStatusResponse>(
    `${PREDICT_PREFIX}/batch/status/${encodeURIComponent(taskId)}`,
    { timeout: 10_000 },
  );
  return data;
}

/**
 * 완료된 배치 결과 CSV 다운로드 URL을 반환합니다 (동기 함수).
 *
 * 내부 동작:
 *   - S3 presigned URL이 설정된 경우: 백엔드가 302 리디렉션 → S3에서 직접 다운로드
 *   - 인메모리 결과인 경우: 백엔드가 CSV를 직접 스트리밍
 *
 * 사용법:
 *   const url = getBatchDownloadUrl(taskId);
 *   window.location.href = url;          // 직접 탐색
 *   // 또는 <a href={url} download>        // anchor 방식
 *
 * @param taskId - 완료 상태(status === 'done')인 태스크 ID
 * @returns 다운로드 URL 문자열
 */
export function getBatchDownloadUrl(taskId: string): string {
  return `${API_BASE_URL}${PREDICT_PREFIX}/batch/download/${encodeURIComponent(taskId)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 유틸리티: TanStack Query와 함께 사용하는 쿼리 키 팩토리
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @tanstack/react-query의 queryKey를 일관되게 관리하는 팩토리 객체.
 *
 * 사용법:
 *   const { data } = useQuery({
 *     queryKey: queryKeys.batchStatus(taskId),
 *     queryFn:  () => getBatchStatus(taskId),
 *     refetchInterval: taskStatus === 'running' ? 3000 : false,
 *   });
 */
export const queryKeys = {
  /** 단일 예측 결과 캐시 키 (smiles + xai 포함 여부로 구분) */
  singlePredict: (smiles: string, includeXai = true) =>
    ['predict', 'single', smiles, includeXai] as const,

  /** 배치 상태 폴링 캐시 키 */
  batchStatus: (taskId: string) =>
    ['predict', 'batch', 'status', taskId] as const,
} as const;

/** 폴링 기본 간격 (ms). .env의 VITE_BATCH_POLL_INTERVAL_MS로 오버라이드 가능. */
export const BATCH_POLL_INTERVAL_MS: number =
  Number(import.meta.env.VITE_BATCH_POLL_INTERVAL_MS ?? 3000);
