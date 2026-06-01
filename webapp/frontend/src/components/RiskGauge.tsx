/**
 * RiskGauge.tsx — SVG 반원 게이지 + 위험 등급 배지
 *
 * 시각화 설계:
 *  - 반원 호 (0% 좌측 → 100% 우측)를 stroke-dasharray / stroke-dashoffset으로 채움
 *  - 45% 위치에 임계값 눈금 표시
 *  - 확률값과 위험 등급 배지 출력
 *  - 색상 전환 + 채움 길이 변화에 CSS transition 0.8s 적용
 */

import { RISK_COLORS, type RiskLevel } from '@/types/predict'

// ─── SVG 게이지 치수 상수 ─────────────────────────────────────────────────────
const CX = 110          // 원 중심 X
const CY = 112          // 원 중심 Y (SVG 하단 부근 배치)
const R  = 90           // 반지름
const SW = 16           // 트랙 두께 (strokeWidth)
const TOTAL_LEN = Math.PI * R   // 반원 호 길이 ≈ 282.74

// ─── Props ────────────────────────────────────────────────────────────────────

interface RiskGaugeProps {
  /** DILI 발생 확률 0.0 ~ 100.0 */
  probability: number
  /** 위험 등급 HIGH | LOW */
  riskLevel: RiskLevel
}

// ─── 컴포넌트 ──────────────────────────────────────────────────────────────────

export default function RiskGauge({ probability, riskLevel }: RiskGaugeProps) {
  const prob       = Math.max(0, Math.min(100, probability))
  const dashOffset = TOTAL_LEN * (1 - prob / 100)
  const fillColor  = riskLevel === 'HIGH' ? '#ef4444' : '#22c55e'
  const colors     = RISK_COLORS[riskLevel]

  // 45% 임계값 눈금 좌표
  const threshold   = 45
  const tAngle      = Math.PI * (1 - threshold / 100)   // ~99°
  const outerR      = R + 10
  const innerR      = R - 10
  const tick1 = { x: CX + outerR * Math.cos(tAngle), y: CY - outerR * Math.sin(tAngle) }
  const tick2 = { x: CX + innerR * Math.cos(tAngle), y: CY - innerR * Math.sin(tAngle) }
  const tickLabel = { x: CX + (R + 24) * Math.cos(tAngle), y: CY - (R + 24) * Math.sin(tAngle) }

  // 반원 호 경로
  const arcPath = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`

  return (
    <div className="flex flex-col items-center gap-2">

      {/* SVG 게이지 */}
      <div className="relative w-[220px]">
        <svg
          width="220"
          height="128"
          viewBox="0 0 220 128"
          aria-label={`DILI 확률 ${prob.toFixed(1)}%, ${riskLevel === 'HIGH' ? '고위험' : '저위험'}`}
        >
          {/* 트랙 배경 */}
          <path
            d={arcPath}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={SW}
            strokeLinecap="round"
          />

          {/* 채움 호 — transition으로 부드럽게 변화 */}
          <path
            d={arcPath}
            fill="none"
            stroke={fillColor}
            strokeWidth={SW}
            strokeLinecap="round"
            strokeDasharray={TOTAL_LEN.toFixed(2)}
            strokeDashoffset={dashOffset.toFixed(2)}
            style={{
              transition:
                'stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.35s ease',
            }}
          />

          {/* 45% 임계값 눈금 */}
          <line
            x1={tick1.x} y1={tick1.y}
            x2={tick2.x} y2={tick2.y}
            stroke="#94a3b8"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <text
            x={tickLabel.x}
            y={tickLabel.y + 3}
            textAnchor="middle"
            fontSize="9"
            fill="#94a3b8"
            fontFamily="system-ui, sans-serif"
          >
            45%
          </text>

          {/* 축 레이블 */}
          <text x={CX - R + 4} y={CY + 16} textAnchor="middle" fontSize="10" fill="#cbd5e1" fontFamily="system-ui, sans-serif">0</text>
          <text x={CX + R - 4} y={CY + 16} textAnchor="middle" fontSize="10" fill="#cbd5e1" fontFamily="system-ui, sans-serif">100</text>

          {/* 확률 수치 */}
          <text
            x={CX}
            y={CY - 26}
            textAnchor="middle"
            fontSize="34"
            fontWeight="700"
            fill={fillColor}
            fontFamily="system-ui, -apple-system, sans-serif"
            style={{ transition: 'fill 0.35s ease' }}
          >
            {prob.toFixed(1)}
          </text>
          <text
            x={CX}
            y={CY - 7}
            textAnchor="middle"
            fontSize="12"
            fill="#94a3b8"
            fontFamily="system-ui, sans-serif"
          >
            % DILI 확률
          </text>
        </svg>
      </div>

      {/* 위험 등급 배지 */}
      <div
        className={[
          'inline-flex items-center gap-2 rounded-full px-5 py-1.5',
          'text-sm font-semibold ring-1 ring-inset select-none',
          colors.bg, colors.text, colors.ring,
        ].join(' ')}
      >
        <span
          className="h-2.5 w-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: colors.fill }}
          aria-hidden
        />
        {colors.label}
      </div>
    </div>
  )
}
