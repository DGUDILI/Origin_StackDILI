/**
 * SinglePredictView.tsx — 단일 SMILES DILI 예측 대시보드 뷰
 *
 * 레이아웃:
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  SMILES 입력창                           [분석하기 버튼]          │
 *  ├──────────────────────────────────────────────────────────────────┤
 *  │                                      [PDF 리포트 다운로드 버튼]   │
 *  │  ──────────────────────────── id="dili-report-content" ────────  │
 *  │  좌측 패널 (1/3)              우측 패널 (2/3)                    │
 *  │  RiskGauge                    분자 구조 SVG (XAI 하이라이트)     │
 *  │  Top-3 MACCS 패턴 카드        물리화학 특성 테이블               │
 *  └──────────────────────────────────────────────────────────────────┘
 */

import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FlaskConical, Loader2, AlertCircle, ChevronRight, FileDown } from 'lucide-react'
import { clsx } from 'clsx'
import html2canvas from 'html2canvas'
import { jsPDF } from 'jspdf'

import { predictSingle, ApiError } from '@/api/client'
import { type SinglePredictResponse, type MaccsPattern, type ToxicReason } from '@/types/predict'
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

// ─── 독성 원인 작용기 카드 ────────────────────────────────────────────────────

const RANK_STYLES = [
  { border: 'border-red-200',    bg: 'from-red-50 to-orange-50',    badge: 'bg-red-100 text-red-700',    bar: 'from-red-400 to-orange-400',    medal: '🥇' },
  { border: 'border-orange-200', bg: 'from-orange-50 to-amber-50',  badge: 'bg-orange-100 text-orange-700', bar: 'from-orange-400 to-amber-400', medal: '🥈' },
  { border: 'border-amber-200',  bg: 'from-amber-50 to-yellow-50',  badge: 'bg-amber-100 text-amber-700',  bar: 'from-amber-400 to-yellow-400',  medal: '🥉' },
] as const

function ToxicReasonCard({ reason }: { reason: ToxicReason }) {
  const style = RANK_STYLES[Math.min(reason.rank - 1, RANK_STYLES.length - 1)]
  return (
    <div className={`rounded-lg border ${style.border} bg-gradient-to-br ${style.bg} p-3.5`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-base leading-none">{style.medal}</span>
          <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${style.badge}`}>
            {reason.rank}순위
          </span>
        </div>
        <span className="tabular-nums text-xs font-bold text-slate-600">
          {reason.contribution.toFixed(1)}%
        </span>
      </div>
      <p className="mb-2.5 text-[11px] font-semibold leading-snug text-slate-700">
        {reason.name}
      </p>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/60">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${style.bar} transition-all duration-700`}
          style={{ width: `${Math.min(reason.contribution, 100)}%` }}
          role="progressbar"
          aria-valuenow={reason.contribution}
          aria-valuemin={0}
          aria-valuemax={100}
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
  const [smiles, setSmiles]               = useState('')
  const [result, setResult]               = useState<SinglePredictResponse | null>(null)
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false)

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

  // ── PDF 리포트 생성 ─────────────────────────────────────────────────────────
  const generatePDF = useCallback(async () => {
    if (!result || isGeneratingPdf) return
    setIsGeneratingPdf(true)

    try {
      const reportEl = document.getElementById('dili-report-content')
      if (!reportEl) return

      // html2canvas의 onclone 콜백을 사용
      // ─ 수동 clone + 극단적 오프스크린 배치(top:-99999px) + zIndex:-1 조합은
      //   브라우저가 요소를 페인팅 대상에서 제외하거나 body 배경 뒤로 밀어내어
      //   전체 콘텐츠가 반투명·흐릿하게 캡처되는 원인이 됨.
      // ─ onclone을 쓰면 html2canvas가 자체 클론 파이프라인 안에서 처리하므로
      //   해당 렌더링 버그가 발생하지 않음.
      const canvas = await html2canvas(reportEl, {
        scale          : 2,       // 2× 해상도 → PDF 선명도 확보
        useCORS        : true,
        allowTaint     : true,    // SVG · inline 엘리먼트 렌더링 허용
        backgroundColor: '#ffffff',
        logging        : false,
        imageTimeout   : 15_000,
        onclone        : (clonedDoc) => {
          const el = clonedDoc.getElementById('dili-report-content')
          if (!el) return

          // animate-fade-in 등 CSS 애니메이션이 opacity 과도기 상태로
          // 캡처되지 않도록 모든 하위 요소의 animation/transition 제거
          el.style.animation = 'none'
          el.style.opacity   = '1'
          clonedDoc
            .querySelectorAll<HTMLElement>('#dili-report-content *')
            .forEach((child) => {
              child.style.animation  = 'none'
              child.style.transition = 'none'
            })

          // dangerouslySetInnerHTML SVG에 명시적 width/height/xmlns 부여
          // (html2canvas의 SVG 렌더러는 치수가 명시되어야 올바르게 그림)
          const origSvgs = Array.from(reportEl.querySelectorAll<SVGSVGElement>('svg'))
          el.querySelectorAll<SVGSVGElement>('svg').forEach((svg, i) => {
            const orig = origSvgs[i]
            if (orig) {
              const { width, height } = orig.getBoundingClientRect()
              if (width  > 0) svg.setAttribute('width',  `${Math.ceil(width)}`)
              if (height > 0) svg.setAttribute('height', `${Math.ceil(height)}`)
            }
            if (!svg.getAttribute('xmlns')) {
              svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
            }
          })
        },
      })

      // A4 PDF 빌드 (컨텐츠 높이에 따라 자동 페이지 분할)
      const pdf      = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
      const MARGIN   = 10
      const PAGE_W   = pdf.internal.pageSize.getWidth()  - MARGIN * 2  // ≈ 190mm
      const PAGE_H   = pdf.internal.pageSize.getHeight() - MARGIN * 2  // ≈ 277mm
      const mmPerPx  = PAGE_W / canvas.width                           // mm/canvas pixel
      const pxPerPage = PAGE_H / mmPerPx                               // 한 페이지 픽셀 행 수

      let srcY = 0; let first = true
      while (srcY < canvas.height) {
        if (!first) pdf.addPage()
        const sliceH = Math.min(pxPerPage, canvas.height - srcY)
        const slice  = document.createElement('canvas')
        slice.width  = canvas.width
        slice.height = Math.ceil(sliceH)
        slice.getContext('2d')!.drawImage(
          canvas, 0, srcY, canvas.width, sliceH,
          0, 0, canvas.width, sliceH,
        )
        pdf.addImage(
          slice.toDataURL('image/png'),
          'PNG', MARGIN, MARGIN, PAGE_W, sliceH * mmPerPx,
        )
        srcY += sliceH
        first = false
      }

      // 파일명: DILI_Analysis_Report_[canonical SMILES].pdf
      const safe = result.canonical_smiles.replace(/[^\w]/g, '_').slice(0, 40)
      pdf.save(`DILI_Analysis_Report_${safe}.pdf`)

    } catch (err) {
      console.error('PDF 생성 오류:', err)
    } finally {
      setIsGeneratingPdf(false)
    }
  }, [result, isGeneratingPdf])

  const isPending = mutation.isPending
  const apiError  = mutation.error instanceof ApiError ? mutation.error : null

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── 입력 카드 ──────────────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-blue-500" />
          <h2 className="text-base font-semibold text-slate-800">분자 입력</h2>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="relative flex-1">
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="SMILES 또는 영문 분자 이름 입력 (예: CC(=O)Oc1ccccc1C(=O)O 또는 Aspirin)"
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

      {/* ── 에러 배너 (유효하지 않은 SMILES 또는 분자 이름) ─────────────────── */}
      {!result && !isPending && mutation.isError && (
        <div className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-5">
          <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-500" />
          <div>
            <p className="text-sm font-semibold text-red-700">분석을 수행할 수 없습니다</p>
            <p className="mt-1 text-sm text-red-600">
              {apiError?.message ?? '알 수 없는 오류가 발생했습니다.'}
            </p>
          </div>
        </div>
      )}

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
        <>
          {/* PDF 다운로드 버튼 (캡처 영역 외부 — PDF에 포함되지 않음) */}
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-400">
              분석이 완료되었습니다.&nbsp;
              <span className="font-mono text-slate-500">{result.canonical_smiles}</span>
            </p>
            <button
              onClick={generatePDF}
              disabled={isGeneratingPdf}
              className={clsx(
                'inline-flex flex-shrink-0 items-center gap-2 rounded-lg px-4 py-2',
                'text-sm font-semibold transition-all duration-150 shadow-sm',
                isGeneratingPdf
                  ? 'cursor-not-allowed bg-slate-100 text-slate-400'
                  : 'bg-slate-800 text-white hover:bg-slate-700 active:scale-[0.97]',
              )}
              aria-label="PDF 리포트 다운로드"
            >
              {isGeneratingPdf ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  PDF 생성 중…
                </>
              ) : (
                <>
                  <FileDown className="h-4 w-4" />
                  PDF 리포트 다운로드
                </>
              )}
            </button>
          </div>

          {/* ── PDF 캡처 대상 영역 (id="dili-report-content") ────────────── */}
          <div
            id="dili-report-content"
            className="grid grid-cols-1 gap-6 lg:grid-cols-3 animate-fade-in"
          >

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

              {/* 독성 원인 작용기 TOP 3 */}
              {result.toxic_reasons.length > 0 && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <h3 className="mb-1 text-sm font-semibold text-slate-500 uppercase tracking-wide">
                    독성 원인 작용기 TOP {result.toxic_reasons.length}
                  </h3>
                  <p className="mb-3 text-[10px] text-slate-400">
                    AI 어텐션 기반 SMARTS 구조 매핑
                  </p>
                  <div className="space-y-2.5">
                    {result.toxic_reasons.map((r) => (
                      <ToxicReasonCard key={r.rank} reason={r} />
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
                    {result.risk_level === 'HIGH'
                      ? '🔴 빨간색 → 독성 기여 원자 (진할수록 강함)'
                      : '🔵 파란색 → 어텐션 집중 원자'
                    }
                    &nbsp;·&nbsp;흰색 → 낮은 기여도
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
        </>
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
