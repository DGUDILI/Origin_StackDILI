/**
 * PhysChemTable.tsx — 물리화학 특성 9종 + Lipinski Rule-of-Five 위반 테이블
 *
 * 레이아웃:
 *  섹션 1) Lipinski Rule-of-Five: MW / LogP / HBD / HBA + 위반 여부 배지
 *  섹션 2) 기타 디스크립터: TPSA / RotBonds / QED / Rings / AromaticRings
 *  요약 행: 총 위반 항목 수 (n/4)
 */

import { AlertTriangle, CheckCircle2 } from 'lucide-react'
import { getLipinskiRules, type PhysChemProps } from '@/types/predict'

// ─── 행 컴포넌트 ───────────────────────────────────────────────────────────────

interface PropertyRowProps {
  label: string
  value: number | null | undefined
  unit?: string
  threshold?: string
  violated?: boolean
}

function PropertyRow({ label, value, unit = '', threshold, violated }: PropertyRowProps) {
  const displayValue =
    value === null || value === undefined
      ? '—'
      : Number.isInteger(value)
      ? value.toString()
      : value.toFixed(value < 10 ? 3 : 1)

  return (
    <tr className={violated ? 'bg-red-50/60' : 'hover:bg-slate-50/70'}>
      <td className="py-2 pl-4 pr-2 text-sm text-slate-500 font-medium whitespace-nowrap">
        {label}
      </td>
      <td className="py-2 px-2 text-sm font-mono text-slate-800 text-right">
        {displayValue}
        {unit && <span className="ml-1 text-xs text-slate-400">{unit}</span>}
      </td>
      {threshold !== undefined && (
        <td className="py-2 pl-2 pr-4 text-xs text-slate-400 text-right whitespace-nowrap">
          ≤ {threshold}
        </td>
      )}
      {violated !== undefined && (
        <td className="py-2 pr-4 text-right">
          {violated ? (
            <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
              위반
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs font-medium text-green-600">
              <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" />
              통과
            </span>
          )}
        </td>
      )}
    </tr>
  )
}

function SectionHeader({ title }: { title: string }) {
  return (
    <tr>
      <td
        colSpan={4}
        className="pb-1 pt-4 pl-4 text-[11px] font-semibold uppercase tracking-widest text-slate-400"
      >
        {title}
      </td>
    </tr>
  )
}

// ─── 메인 컴포넌트 ─────────────────────────────────────────────────────────────

interface PhysChemTableProps {
  data: PhysChemProps
}

export default function PhysChemTable({ data }: PhysChemTableProps) {
  const rules = getLipinskiRules(data)
  const v     = data.lipinski_violations

  const violationColor =
    v === 0 ? 'text-green-600 bg-green-50 ring-green-200'
    : v === 1 ? 'text-amber-600 bg-amber-50 ring-amber-200'
    : 'text-red-600 bg-red-50 ring-red-200'

  return (
    <div className="overflow-hidden rounded-xl border border-slate-100 bg-white">
      <table className="w-full border-collapse text-sm">
        <tbody>
          {/* ── Lipinski Rule-of-Five ─────────────────────────────── */}
          <SectionHeader title="Lipinski Rule-of-Five" />

          {rules.map((rule) => (
            <PropertyRow
              key={rule.label}
              label={rule.label}
              value={rule.value}
              unit={rule.unit}
              threshold={`${rule.threshold}${rule.unit}`}
              violated={rule.violated}
            />
          ))}

          {/* 위반 요약 */}
          <tr className="border-t border-slate-100">
            <td colSpan={2} className="pb-3 pl-4 pt-3 text-sm font-semibold text-slate-700">
              Lipinski 위반
            </td>
            <td colSpan={2} className="pb-3 pr-4 pt-3 text-right">
              <span
                className={[
                  'inline-flex items-center rounded-full px-3 py-0.5',
                  'text-sm font-bold ring-1 ring-inset',
                  violationColor,
                ].join(' ')}
              >
                {v} / 4
              </span>
            </td>
          </tr>

          {/* ── 기타 디스크립터 ───────────────────────────────────── */}
          <tr className="border-t border-slate-100">
            <td
              colSpan={4}
              className="pb-1 pt-4 pl-4 text-[11px] font-semibold uppercase tracking-widest text-slate-400"
            >
              기타 디스크립터
            </td>
          </tr>

          <PropertyRow
            label="TPSA"
            value={data.tpsa}
            unit="Å²"
          />
          <PropertyRow
            label="Rotatable Bonds"
            value={data.rotatable_bonds}
          />
          <PropertyRow
            label="QED Drug-likeness"
            value={data.qed}
          />
          <PropertyRow
            label="Ring Count"
            value={data.ring_count}
          />
          <PropertyRow
            label="Aromatic Rings"
            value={data.aromatic_rings}
          />

          {/* 하단 여백 */}
          <tr>
            <td colSpan={4} className="h-2" />
          </tr>
        </tbody>
      </table>
    </div>
  )
}
