/**
 * predict.ts — 백엔드 Pydantic 스키마와 1:1 대응하는 TypeScript 타입 정의
 *
 * 매핑 기준:
 *   backend schemas/predict.py  → SinglePredictRequest / Response, MaccsPattern, PhysChemProps
 *   backend schemas/batch.py    → TaskStatus, BatchStatusResponse, BatchUploadResponse
 *
 * 필드명 규칙: Python snake_case를 그대로 사용
 *   (axios response는 변환 없이 그대로 받으므로 camelCase 변환 없음)
 */

// ─────────────────────────────────────────────────────────────────────────────
// 공통 열거형
// ─────────────────────────────────────────────────────────────────────────────

/** DILI 위험 등급 (backend: "HIGH" | "LOW") */
export type RiskLevel = 'HIGH' | 'LOW';

/** 배치 태스크 처리 상태 (backend: TaskStatus enum) */
export type TaskStatus = 'pending' | 'running' | 'done' | 'failed';

// ─────────────────────────────────────────────────────────────────────────────
// 단일 예측 — Request
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/predict/single 요청 본문
 * backend: SinglePredictRequest
 */
export interface SinglePredictRequest {
  /** SMILES 문자열 (최대 2000자) */
  smiles: string;
  /**
   * true: XAI SVG + MACCS 패턴 포함 (기본값)
   * false: 확률/물성치만 반환 (빠른 응답)
   */
  include_xai?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// 단일 예측 — Response 서브타입
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SMARTS 기반 작용기 독성 기여 정보
 * backend: ToxicReason
 */
export interface ToxicReason {
  /** 기여도 순위 (1 = 가장 높음) */
  rank: number;
  /** 작용기 이름 (예: "Benzene Ring", "Carboxyl (-COOH)") */
  name: string;
  /**
   * 전체 원자 중요도 대비 기여율 (0.0 ~ 100.0 %)
   * UI 표시: `${contribution.toFixed(1)}%`
   */
  contribution: number;
}

/**
 * 상위 기여 MACCS 구조 키 하나
 * backend: MaccsPattern
 */
export interface MaccsPattern {
  /** MACCS bit 인덱스 (1 ~ 166) */
  bit_index: number;
  /** Durant et al. 2002 기반 인간 가독 이름 */
  name: string;
  /**
   * 전체 어텐션 대비 해당 패턴의 기여 비율 (0.0 ~ 1.0)
   * UI에서 %로 표시할 때: (importance * 100).toFixed(1)
   */
  importance: number;
}

/**
 * RDKit Descriptors 기반 물리화학 특성 9종 + Lipinski 위반 수
 * backend: PhysChemProps (computed_field: lipinski_violations)
 *
 * Lipinski Rule-of-Five 기준:
 *   MW ≤ 500 Da  |  LogP ≤ 5  |  HBD ≤ 5  |  HBA ≤ 10
 */
export interface PhysChemProps {
  /** 정확 분자량 ExactMolWt (Da) */
  molecular_weight: number;
  /** Wildman-Crippen 분배계수 LogP */
  logp: number;
  /** 수소 결합 공여체(HBD) 수 */
  hbd: number;
  /** 수소 결합 수용체(HBA) 수 */
  hba: number;
  /** 위상 극성 표면적 TPSA (Å²) */
  tpsa: number;
  /** 회전 가능 결합(Rotatable Bonds) 수 */
  rotatable_bonds: number;
  /** QED 약물 유사성 점수 (0.0 ~ 1.0) */
  qed: number;
  /** 총 링 수 */
  ring_count: number;
  /** 방향족 링 수 */
  aromatic_rings: number;
  /**
   * Lipinski Rule-of-Five 위반 항목 수 (0 ~ 4)
   * backend computed_field: MW>500, LogP>5, HBD>5, HBA>10 각 1점
   */
  lipinski_violations: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 단일 예측 — Response
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/predict/single 응답
 * backend: SinglePredictResponse
 */
export interface SinglePredictResponse {
  /** 사용자가 입력한 원본 SMILES */
  smiles: string;
  /** RDKit Canonical SMILES (원자 인덱스 고정, XAI와 동일 기준) */
  canonical_smiles: string;
  /**
   * DILI 발생 확률 (0.0 ~ 100.0 %)
   * UI 표시: `${probability.toFixed(1)}%`
   */
  probability: number;
  /** 위험 등급: 임계값(기본 45%) 기준 HIGH 또는 LOW */
  risk_level: RiskLevel;
  /**
   * 어텐션 기여도 상위 3개 MACCS 구조 패턴
   * include_xai=false 요청 시 빈 배열
   */
  top_maccs_patterns: MaccsPattern[];
  /** 물리화학 특성 세트 */
  physicochemical: PhysChemProps;
  /**
   * XAI 원자 하이라이트가 포함된 RDKit SVG 문자열
   * include_xai=false 요청 시 빈 문자열 ""
   * UI 렌더링: dangerouslySetInnerHTML={{ __html: molecule_svg }}
   */
  molecule_svg: string;
  /**
   * SMARTS 기반 작용기 독성 기여도 상위 3개 순위
   * include_xai=false 요청 시 빈 배열 []
   */
  toxic_reasons: ToxicReason[];
}

// ─────────────────────────────────────────────────────────────────────────────
// 배치 처리 — Upload Response
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/predict/batch 응답 (202 Accepted)
 * backend: BatchUploadResponse
 */
export interface BatchUploadResponse {
  /** 프론트엔드 폴링에 사용할 태스크 ID (UUID) */
  task_id: string;
  /** 사용자 친화적 시작 메시지 */
  message: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// 배치 처리 — Status Response
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GET /api/v1/predict/batch/status/{task_id} 응답
 * backend: BatchStatusResponse
 *
 * 프론트엔드 폴링 권장 주기: 3초
 * 폴링 종료 조건: status === 'done' || status === 'failed'
 */
export interface BatchStatusResponse {
  /** 배치 태스크 고유 ID */
  task_id: string;
  /** 현재 처리 상태 */
  status: TaskStatus;
  /** 처리 예정 총 행 수 (CSV 파싱 완료 후 확정, 그 전까지 0) */
  total: number;
  /** 완료된 행 수 (실시간 갱신) */
  processed: number;
  /**
   * 진행률 (0.0 ~ 100.0 %)
   * UI Progress Bar: `${progress_pct}%` width
   */
  progress_pct: number;
  /**
   * true이면 /batch/download/{task_id} 로 결과 CSV 다운로드 가능
   * 다운로드 버튼 활성화 조건: status === 'done' && has_result
   */
  has_result: boolean;
  /** 실패 시 오류 메시지, 그 외 null */
  error: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// 유틸리티 타입 (UI 컴포넌트에서 사용)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Lipinski Rule-of-Five 위반 항목 상세 정보 (PhysChemTable 컴포넌트용)
 * API에서 직접 오지 않으며, 프론트엔드에서 PhysChemProps를 파싱해 생성합니다.
 */
export interface LipinskiRule {
  label: string;
  value: number;
  threshold: number;
  unit: string;
  violated: boolean;
}

/**
 * PhysChemProps → LipinskiRule[] 변환 유틸리티
 * 사용: PhysChemTable, RuleOfFiveBadge 컴포넌트에서 호출
 */
export function getLipinskiRules(props: PhysChemProps): LipinskiRule[] {
  return [
    {
      label: 'Mol. Weight',
      value: props.molecular_weight,
      threshold: 500,
      unit: 'Da',
      violated: props.molecular_weight > 500,
    },
    {
      label: 'LogP',
      value: props.logp,
      threshold: 5,
      unit: '',
      violated: props.logp > 5,
    },
    {
      label: 'HBD',
      value: props.hbd,
      threshold: 5,
      unit: '',
      violated: props.hbd > 5,
    },
    {
      label: 'HBA',
      value: props.hba,
      threshold: 10,
      unit: '',
      violated: props.hba > 10,
    },
  ];
}

/**
 * 위험 등급에 따른 Tailwind 색상 클래스 매핑
 * 사용: RiskGauge, RiskBadge 컴포넌트
 */
export const RISK_COLORS = {
  HIGH: {
    text:   'text-red-600',
    bg:     'bg-red-50',
    ring:   'ring-red-200',
    fill:   '#ef4444',
    label:  '고위험 (HIGH)',
  },
  LOW: {
    text:   'text-green-600',
    bg:     'bg-green-50',
    ring:   'ring-green-200',
    fill:   '#22c55e',
    label:  '저위험 (LOW)',
  },
} as const satisfies Record<RiskLevel, {
  text: string; bg: string; ring: string; fill: string; label: string;
}>;

/**
 * 배치 상태별 UI 표시 텍스트 및 색상
 * 사용: BatchProgressBar, BatchStatusBadge 컴포넌트
 */
export const TASK_STATUS_META = {
  pending : { label: '대기 중',   color: 'text-slate-500', bg: 'bg-slate-100'  },
  running : { label: '분석 중',   color: 'text-blue-600',  bg: 'bg-blue-50'   },
  done    : { label: '완료',      color: 'text-green-600', bg: 'bg-green-50'  },
  failed  : { label: '실패',      color: 'text-red-600',   bg: 'bg-red-50'    },
} as const satisfies Record<TaskStatus, { label: string; color: string; bg: string }>;
