/**
 * SinglePredictView.tsx — 단일 SMILES DILI 예측 대시보드 뷰
 *
 * 레이아웃:
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  SMILES 입력창                           [분석하기 버튼]          │
 *  ├──────────────────────────────────────────────────────────────────┤
 *  │  좌측 패널 (1/3)              우측 패널 (2/3)                    │
 *  │  ─────────────────            ──────────────────────────────     │
 *  │  RiskGauge                    분자 구조 SVG (XAI 하이라이트)     │
 *  │  Top-3 MACCS 패턴 카드        물리화학 특성 테이블               │
 *  └──────────────────────────────────────────────────────────────────┘
 */

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FlaskConical, Loader2, AlertCircle, ChevronRight } from 'lucide-react'
import { clsx } from 'clsx'

import { predictSingle, ApiError } from '@/api/client'
import { type SinglePredictResponse, type MaccsPattern } from '@/types/predict'
import RiskGauge    from '@/components/RiskGauge'
import PhysChemTable from '@/components/PhysChemTable'

// ─── SVG 정제 (<?xml ...?> 헤더 제거 → 인라인 렌더링) ────────────────────────
function sanitizeSvg(svg: string): string {
  return svg.replace(/<\?xml[^?]*\?>\s*/g, '').trim()
}

// ─── MACCS 패턴 카드 ──────────────────────────────────────────────────────────

function MaccsPatternCard({ pattern, rank }: { pattern: MaccsPattern; rank: number }) {
  const impPct = (pattern.importance * 100).toFixed(1)
  return (
    <div className="rounded-lg border border-amber-100 bg-gradient-to-br from-amber-50 to-orange-50 p-3.5">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-mono font-semibold text-amber-700">
            bit&nbsp;{pattern.bit_index}
          </span>
          <span className="text-[10px] font-semibold text-amber-400">#{rank}</span>
        </div>
        <span className="flex-shrink-0 tabular-nums text-xs font-bold text-amber-700">
          {impPct}%
        </span>
      </div>
      <p className="mb-2.5 text-[11px] font-semibold leading-snug text-slate-700">
        {pattern.name}
      </p>
      <div className="h-1.5 overflow-hidden rounded-full bg-amber-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-amber-400 to-orange-400 transition-all duration-700"
          style={{ width: `${pattern.importance * 100}%` }}
          aria-valuenow={pattern.importance * 100}
          aria-valuemin={0}
          aria-valuemax={100}
          role="progressbar"
        />
      </div>
    </div>
  )
}

// ─── 예제 SMILES 목록 ─────────────────────────────────────────────────────────

const EXAMPLES = [
  { label: '아세트아미노펜', smiles: 'CC(=O)Nc1ccc(O)cc1' },
  { label: '아스피린',       smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { label: '이부프로펜',     smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' },
]

// ─── 메인 컴포넌트 ─────────────────────────────────────────────────────────────

export default function SinglePredictView() {
  const [smiles, setSmiles] = useState('')
  const [result, setResult] = useState<SinglePredictResponse | null>(null)

  const mutation = useMutation({
    mutationFn: (s: string) => predictSingle(s.trim(), true),
    onSuccess : (data) => setResult(data),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!smiles.trim()) return
    setResult(null)
    mutation.mutate(smiles.trim())
  }

  const handleExampleClick = (s: string) => {
    setSmiles(s)
    setResult(null)
    mutation.mutate(s)
  }

  const isPending = mutation.isPending
  const apiError  = mutation.error instanceof ApiError ? mutation.error : null

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── 입력 카드 ──────────────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-blue-500" />
          <h2 className="text-base font-semibold text-slate-800">SMILES 입력</h2>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="relative flex-1">
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="SMILES를 입력하세요 (예: CC(=O)Oc1ccccc1C(=O)O)"
              className={clsx(
                'w-full rounded-lg border px-4 py-2.5 font-mono text-sm',
                'placeholder-slate-300 outline-none transition-all',
                'focus:ring-2 focus:ring-blue-500/30 focus:border-blue-400',
                apiError
                  ? 'border-red-300 bg-red-50 text-red-800'
                  : 'border-slate-200 bg-slate-50 text-slate-900',
              )}
              disabled={isPending}
              aria-label="SMILES 입력"
            />
            {apiError && (
              <div className="mt-1.5 flex items-start gap-1.5 text-xs text-red-600">
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                <span>{apiError.message}</span>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={isPending || !smiles.trim()}
            className={clsx(
              'inline-flex flex-shrink-0 items-center gap-2 rounded-lg px-5 py-2.5',
              'text-sm font-semibold transition-all duration-150',
              isPending || !smiles.trim()
                ? 'cursor-not-allowed bg-slate-100 text-slate-400'
                : 'bg-blue-600 text-white shadow-sm hover:bg-blue-700 active:scale-[0.97]',
            )}
          >
            {isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                분석 중…
              </>
            ) : (
              <>
                <FlaskConical className="h-4 w-4" />
                분석하기
              </>
            )}
          </button>
        </form>

        {/* 예제 버튼 */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">예제:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.smiles}
              onClick={() => handleExampleClick(ex.smiles)}
              disabled={isPending}
              className="inline-flex items-center gap-1 rounded-md border border-slate-200
                         bg-white px-2.5 py-1 text-xs text-slate-600
                         hover:border-blue-300 hover:text-blue-600 transition-colors
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <ChevronRight className="h-3 w-3" />
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── 로딩 스켈레톤 ────────────────────────────────────────────────────── */}
      {isPending && (
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-1 space-y-4">
            <div className="h-56 animate-pulse rounded-2xl bg-slate-100" />
            <div className="h-24 animate-pulse rounded-2xl bg-slate-100" />
          </div>
          <div className="col-span-2 space-y-4">
            <div className="h-56 animate-pulse rounded-2xl bg-slate-100" />
            <div className="h-24 animate-pulse rounded-2xl bg-slate-100" />
          </div>
        </div>
      )}

      {/* ── 결과 패널 ────────────────────────────────────────────────────────── */}
      {result && !isPending && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 animate-fade-in">

          {/* 좌측: 위험도 + MACCS 패턴 */}
          <div className="space-y-4 lg:col-span-1">

            {/* 위험도 게이지 카드 */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="mb-4 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                DILI 위험도
              </h3>
              <RiskGauge
                probability={result.probability}
                riskLevel={result.risk_level}
              />
              <div className="mt-4 border-t border-slate-100 pt-3">
                <p className="text-[11px] text-slate-400 text-center leading-relaxed">
                  Canonical SMILES
                </p>
                <p className="mt-1 text-center font-mono text-[11px] text-slate-600 break-all leading-relaxed">
                  {result.canonical_smiles}
                </p>
              </div>
            </div>

            {/* Top-3 MACCS 독성 패턴 */}
            {result.top_maccs_patterns.length > 0 && (
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="mb-3 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                  주요 독성 기여 패턴
                </h3>
                <div className="space-y-2.5">
                  {result.top_maccs_patterns.map((p, i) => (
                    <MaccsPatternCard key={p.bit_index} pattern={p} rank={i + 1} />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 우측: 분자 구조 + 물성치 */}
          <div className="space-y-4 lg:col-span-2">

            {/* 분자 구조 SVG (XAI 하이라이트) */}
            {result.molecule_svg && (
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="mb-3 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                  분자 구조 (XAI 원자 기여도 오버레이)
                </h3>
                <div
                  className="molecule-svg flex justify-center overflow-hidden rounded-lg bg-slate-50/50"
                  /* SVG는 백엔드 RDKit 렌더링 결과이므로 안전. 외부 입력 SMILES는 이미 유효성 검증됨. */
                  dangerouslySetInnerHTML={{ __html: sanitizeSvg(result.molecule_svg) }}
                  aria-label="XAI 원자 기여도 분자 구조 이미지"
                />
                <p className="mt-2 text-center text-[11px] text-slate-400">
                  빨간색 → 독성 기여도 높은 원자 / 흰색 → 낮은 기여도
                </p>
              </div>
            )}

            {/* 물리화학 특성 테이블 */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="mb-3 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                물리화학 특성
              </h3>
              <PhysChemTable data={result.physicochemical} />
            </div>

          </div>
        </div>
      )}

      {/* ── 초기 안내 메시지 (결과·로딩·에러 없을 때) ───────────────────────── */}
      {!result && !isPending && !mutation.isError && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/50 py-16 text-center">
          <FlaskConical className="mb-3 h-10 w-10 text-slate-300" />
          <p className="text-sm font-medium text-slate-400">
            SMILES를 입력하고 분석하기를 눌러보세요.
          </p>
          <p className="mt-1 text-xs text-slate-300">
            GINEConv + ChemBERTa + Differential Cross-Attention 기반 예측
          </p>
        </div>
      )}
    </div>
  )
}
