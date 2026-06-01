/**
 * BatchScreeningView.tsx — CSV 배치 스크리닝 대시보드 뷰
 *
 * 흐름:
 *  1) DropZone으로 .csv 파일 선택
 *  2) uploadBatchCSV() → task_id 수령 (202 Accepted)
 *  3) useQuery로 /batch/status 3초 폴링 → 실시간 Progress Bar
 *  4) status === 'done' → CSV 다운로드 → 파싱 → TanStack Table 렌더링
 *  5) '📥 결과 다운로드' 버튼 활성화
 */

import { useCallback, useMemo, useState } from 'react'
import { useDropzone }  from 'react-dropzone'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from '@tanstack/react-table'
import {
  Upload, FileText, CheckCircle2, XCircle,
  Loader2, Download, ChevronUp, ChevronDown,
  ChevronsLeft, ChevronsRight, ChevronLeft, ChevronRight,
} from 'lucide-react'
import { clsx } from 'clsx'

import {
  uploadBatchCSV, getBatchStatus, getBatchDownloadUrl,
  queryKeys, BATCH_POLL_INTERVAL_MS, ApiError,
} from '@/api/client'
import { TASK_STATUS_META, type TaskStatus } from '@/types/predict'

// ─── CSV 파서 (RFC 4180 부분 준수: 따옴표 처리 포함) ─────────────────────────

function parseCSVRow(line: string): string[] {
  const result: string[] = []
  let cur = '', inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { cur += '"'; i++ }
      else inQuotes = !inQuotes
    } else if (ch === ',' && !inQuotes) {
      result.push(cur); cur = ''
    } else {
      cur += ch
    }
  }
  result.push(cur)
  return result
}

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/)
  if (lines.length < 2) return []
  const headers = parseCSVRow(lines[0])
  return lines.slice(1)
    .filter(l => l.trim())
    .map(l => {
      const vals = parseCSVRow(l)
      return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? '']))
    })
}

// ─── TanStack Table 컬럼 정의 ─────────────────────────────────────────────────

type BatchRow = Record<string, string>
const ch = createColumnHelper<BatchRow>()

function makeColumns(headers: string[]) {
  // 우선 표시 컬럼 순서
  const priority = [
    'SMILES_Input', 'Canonical_SMILES', 'DILI_Probability',
    'Risk_Level', 'Molecular_Weight', 'LogP',
    'HBD', 'HBA', 'TPSA', 'Rotatable_Bonds', 'QED',
    'Lipinski_Violations', 'Error',
  ]
  const ordered = [
    ...priority.filter(c => headers.includes(c)),
    ...headers.filter(c => !priority.includes(c)),
  ]

  return ordered.map(key =>
    ch.accessor(key, {
      header: key.replace(/_/g, ' '),
      cell: info => {
        const val = info.getValue()
        if (key === 'Risk_Level') {
          if (val === 'HIGH') return <span className="font-semibold text-red-600">HIGH</span>
          if (val === 'LOW')  return <span className="font-semibold text-green-600">LOW</span>
          if (val === 'SKIP') return <span className="text-slate-400 italic">SKIP</span>
          if (val === 'ERROR') return <span className="text-red-400 italic">ERROR</span>
          return <span className="text-slate-400">{val}</span>
        }
        if (key === 'DILI_Probability') {
          const n = parseFloat(val)
          if (isNaN(n)) return <span className="text-slate-400">—</span>
          return <span className="font-mono tabular-nums">{n.toFixed(1)}%</span>
        }
        if (key === 'Error' && !val) return null
        if (key === 'Error') return <span className="text-xs text-red-500 break-words">{val}</span>
        if (key === 'SMILES_Input' || key === 'Canonical_SMILES') {
          return (
            <span className="font-mono text-xs text-slate-600 max-w-[160px] block truncate" title={val}>
              {val || '—'}
            </span>
          )
        }
        if (!val) return <span className="text-slate-400">—</span>
        const num = parseFloat(val)
        if (!isNaN(num)) return <span className="font-mono tabular-nums">{num}</span>
        return <span>{val}</span>
      },
    })
  )
}

// ─── 진행률 바 ────────────────────────────────────────────────────────────────

function ProgressBar({ pct, status }: { pct: number; status: TaskStatus }) {
  const isRunning  = status === 'running' || status === 'pending'
  const isDone     = status === 'done'
  const isFailed   = status === 'failed'

  const barColor = isDone ? 'bg-green-500' : isFailed ? 'bg-red-400' : 'bg-blue-500'

  return (
    <div className="w-full overflow-hidden rounded-full bg-slate-100 h-2.5">
      {isRunning && pct === 0 ? (
        /* 시작 직후 진행률 0%일 때 애니메이션 */
        <div className="relative h-full w-full overflow-hidden rounded-full">
          <div className="absolute h-full w-1/3 animate-progress-indeterminate rounded-full bg-blue-400 opacity-70" />
        </div>
      ) : (
        <div
          className={clsx('h-full rounded-full transition-all duration-500', barColor)}
          style={{ width: `${Math.max(2, pct)}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      )}
    </div>
  )
}

// ─── 상태 배지 ────────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: TaskStatus }) {
  const meta = TASK_STATUS_META[status]
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full px-3 py-1',
        'text-xs font-semibold',
        meta.bg, meta.color,
      )}
    >
      {status === 'running'  && <Loader2 className="h-3 w-3 animate-spin" />}
      {status === 'done'     && <CheckCircle2 className="h-3 w-3" />}
      {status === 'failed'   && <XCircle className="h-3 w-3" />}
      {meta.label}
    </span>
  )
}

// ─── 메인 컴포넌트 ─────────────────────────────────────────────────────────────

export default function BatchScreeningView() {
  const [taskId,      setTaskId]      = useState<string | null>(null)
  const [csvRows,     setCsvRows]     = useState<BatchRow[]>([])
  const [sorting,     setSorting]     = useState<SortingState>([])
  const [_parseError, setParseError]  = useState<string | null>(null)

  // ── 업로드 Mutation ────────────────────────────────────────────────────────
  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadBatchCSV(file),
    onSuccess: (data) => {
      setTaskId(data.task_id)
      setCsvRows([])
      setParseError(null)
    },
  })

  // ── 상태 폴링 ──────────────────────────────────────────────────────────────
  const statusQuery = useQuery({
    queryKey: queryKeys.batchStatus(taskId ?? ''),
    queryFn:  () => getBatchStatus(taskId!),
    enabled:  !!taskId,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      return s === 'running' || s === 'pending' ? BATCH_POLL_INTERVAL_MS : false
    },
  })

  const status  = statusQuery.data?.status
  const pct     = statusQuery.data?.progress_pct ?? 0
  const total   = statusQuery.data?.total ?? 0
  const processed = statusQuery.data?.processed ?? 0

  // ── 완료 시 CSV 자동 다운로드 + 파싱 ─────────────────────────────────────
  useQuery({
    queryKey: ['batch', 'csv', taskId],
    queryFn: async () => {
      const url  = getBatchDownloadUrl(taskId!)
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`다운로드 실패: HTTP ${resp.status}`)
      const text = await resp.text()
      const rows = parseCSV(text)
      setCsvRows(rows)
      return rows
    },
    enabled:   !!taskId && status === 'done' && statusQuery.data?.has_result === true,
    staleTime: Infinity,
    retry: 1,
  })

  // ── Dropzone ───────────────────────────────────────────────────────────────
  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length === 0) return
    setTaskId(null)
    setCsvRows([])
    setParseError(null)
    uploadMutation.mutate(accepted[0])
  }, [uploadMutation])

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'text/plain': ['.csv'] },
    maxFiles: 1,
    disabled: uploadMutation.isPending,
  })

  // ── TanStack Table ─────────────────────────────────────────────────────────
  const columns = useMemo(
    () => csvRows.length > 0 ? makeColumns(Object.keys(csvRows[0])) : [],
    [csvRows],
  )

  const table = useReactTable({
    data:           csvRows,
    columns,
    state:          { sorting },
    onSortingChange: setSorting,
    getCoreRowModel:       getCoreRowModel(),
    getSortedRowModel:     getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 20 } },
  })

  const uploadError = uploadMutation.error instanceof ApiError
    ? uploadMutation.error
    : null

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── 업로드 존 ──────────────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <FileText className="h-5 w-5 text-blue-500" />
          <h2 className="text-base font-semibold text-slate-800">CSV 파일 업로드</h2>
          <span className="ml-auto text-xs text-slate-400">
            최대 10 MB · 최대 5,000행 · SMILES 컬럼 필수
          </span>
        </div>

        <div
          {...getRootProps()}
          className={clsx(
            'flex cursor-pointer flex-col items-center justify-center gap-3',
            'rounded-xl border-2 border-dashed py-10 transition-all duration-150',
            isDragActive
              ? 'border-blue-400 bg-blue-50/60 scale-[1.01]'
              : uploadMutation.isPending
              ? 'border-slate-200 bg-slate-50 cursor-wait'
              : 'border-slate-200 bg-slate-50/50 hover:border-blue-300 hover:bg-blue-50/30',
          )}
        >
          <input {...getInputProps()} aria-label="CSV 파일 선택" />

          {uploadMutation.isPending ? (
            <>
              <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
              <p className="text-sm font-medium text-blue-600">업로드 중…</p>
            </>
          ) : acceptedFiles.length > 0 && taskId ? (
            <>
              <FileText className="h-8 w-8 text-green-500" />
              <div className="text-center">
                <p className="text-sm font-semibold text-slate-700">{acceptedFiles[0].name}</p>
                <p className="text-xs text-slate-400">
                  {(acceptedFiles[0].size / 1024).toFixed(0)} KB
                </p>
              </div>
              <p className="text-xs text-slate-400">다른 파일로 교체하려면 클릭하거나 드롭하세요.</p>
            </>
          ) : (
            <>
              <Upload className="h-8 w-8 text-slate-300" />
              <div className="text-center">
                <p className="text-sm font-semibold text-slate-600">
                  {isDragActive ? 'CSV 파일을 여기에 놓으세요' : 'CSV 파일을 드래그하거나 클릭하여 선택'}
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  컬럼명: smiles / SMILES / Smiles 중 하나 필수
                </p>
              </div>
            </>
          )}
        </div>

        {/* 업로드 에러 */}
        {uploadError && (
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3">
            <XCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500" />
            <p className="text-sm text-red-700">{uploadError.message}</p>
          </div>
        )}
      </div>

      {/* ── 진행 상태 섹션 ──────────────────────────────────────────────────── */}
      {taskId && statusQuery.data && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <h2 className="text-base font-semibold text-slate-800">분석 진행 상태</h2>
              <StatusBadge status={statusQuery.data.status} />
            </div>

            {status === 'done' && statusQuery.data.has_result && (
              <a
                href={getBatchDownloadUrl(taskId)}
                download={`dili_results_${taskId.slice(0, 8)}.csv`}
                className={clsx(
                  'inline-flex items-center gap-2 rounded-lg px-4 py-2',
                  'bg-green-600 text-sm font-semibold text-white shadow-sm',
                  'hover:bg-green-700 active:scale-[0.97] transition-all',
                )}
              >
                <Download className="h-4 w-4" />
                CSV 결과 다운로드
              </a>
            )}
          </div>

          {/* 진행률 바 */}
          <ProgressBar pct={pct} status={statusQuery.data.status} />

          <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
            <span>
              {processed.toLocaleString()} / {total > 0 ? total.toLocaleString() : '?'} 행 처리 완료
            </span>
            <span className="font-mono font-semibold tabular-nums">{pct.toFixed(1)}%</span>
          </div>

          {/* 실패 에러 메시지 */}
          {status === 'failed' && statusQuery.data.error && (
            <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3">
              <XCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500" />
              <p className="text-sm text-red-700">{statusQuery.data.error}</p>
            </div>
          )}

          {/* task_id 표시 */}
          <p className="mt-3 text-[11px] text-slate-300 font-mono">
            Task ID: {taskId}
          </p>
        </div>
      )}

      {/* ── 결과 테이블 ─────────────────────────────────────────────────────── */}
      {csvRows.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
            <div>
              <h2 className="text-base font-semibold text-slate-800">분석 결과</h2>
              <p className="mt-0.5 text-xs text-slate-400">
                {csvRows.length.toLocaleString()}행 · 열 헤더 클릭으로 정렬 가능
              </p>
            </div>
            <div className="flex items-center gap-2">
              {/* HIGH / LOW 통계 */}
              {['HIGH','LOW','ERROR'].map(level => {
                const count = csvRows.filter(r => r['Risk_Level'] === level).length
                if (count === 0) return null
                return (
                  <span
                    key={level}
                    className={clsx(
                      'rounded-full px-2.5 py-1 text-xs font-semibold',
                      level === 'HIGH'  ? 'bg-red-50 text-red-600'
                      : level === 'LOW' ? 'bg-green-50 text-green-600'
                      : 'bg-slate-100 text-slate-500',
                    )}
                  >
                    {level} {count}
                  </span>
                )
              })}
            </div>
          </div>

          {/* 스크롤 테이블 */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                {table.getHeaderGroups().map(hg => (
                  <tr key={hg.id} className="border-b border-slate-100 bg-slate-50/80">
                    {hg.headers.map(header => (
                      <th
                        key={header.id}
                        className={clsx(
                          'px-4 py-2.5 text-left text-[11px] font-semibold',
                          'uppercase tracking-wide text-slate-500 whitespace-nowrap',
                          header.column.getCanSort() && 'cursor-pointer select-none hover:text-slate-800',
                        )}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center gap-1">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {header.column.getIsSorted() === 'asc'  && <ChevronUp   className="h-3 w-3" />}
                          {header.column.getIsSorted() === 'desc' && <ChevronDown className="h-3 w-3" />}
                        </div>
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map(row => (
                  <tr
                    key={row.id}
                    className={clsx(
                      'border-b border-slate-50 transition-colors',
                      row.original['Risk_Level'] === 'HIGH'
                        ? 'hover:bg-red-50/40'
                        : 'hover:bg-slate-50/60',
                    )}
                  >
                    {row.getVisibleCells().map(cell => (
                      <td key={cell.id} className="px-4 py-2 text-slate-700">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* 페이지네이션 */}
          <div className="flex items-center justify-between border-t border-slate-100 px-6 py-3">
            <p className="text-xs text-slate-400">
              페이지 {table.getState().pagination.pageIndex + 1} / {table.getPageCount()}
            </p>
            <div className="flex items-center gap-1">
              <PaginationBtn onClick={() => table.setPageIndex(0)}            disabled={!table.getCanPreviousPage()} icon={<ChevronsLeft  className="h-3.5 w-3.5" />} />
              <PaginationBtn onClick={() => table.previousPage()}             disabled={!table.getCanPreviousPage()} icon={<ChevronLeft   className="h-3.5 w-3.5" />} />
              <PaginationBtn onClick={() => table.nextPage()}                 disabled={!table.getCanNextPage()}     icon={<ChevronRight  className="h-3.5 w-3.5" />} />
              <PaginationBtn onClick={() => table.setPageIndex(table.getPageCount() - 1)} disabled={!table.getCanNextPage()} icon={<ChevronsRight className="h-3.5 w-3.5" />} />
            </div>
            <select
              value={table.getState().pagination.pageSize}
              onChange={e => table.setPageSize(Number(e.target.value))}
              className="rounded border border-slate-200 px-2 py-1 text-xs text-slate-600"
            >
              {[10, 20, 50, 100].map(s => <option key={s} value={s}>{s}행</option>)}
            </select>
          </div>
        </div>
      )}

      {/* ── 초기 안내 ────────────────────────────────────────────────────────── */}
      {!taskId && !uploadMutation.isPending && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/50 py-16 text-center">
          <Upload className="mb-3 h-10 w-10 text-slate-300" />
          <p className="text-sm font-medium text-slate-400">
            SMILES 컬럼이 포함된 CSV를 업로드하면 대량 분석이 시작됩니다.
          </p>
          <p className="mt-1 text-xs text-slate-300">
            결과 CSV에는 DILI 확률, 위험 등급, 물성치가 포함됩니다.
          </p>
        </div>
      )}
    </div>
  )
}

// ─── 페이지네이션 버튼 헬퍼 ──────────────────────────────────────────────────

function PaginationBtn({
  onClick, disabled, icon,
}: {
  onClick: () => void
  disabled: boolean
  icon: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'rounded p-1.5 transition-colors',
        disabled
          ? 'cursor-not-allowed text-slate-200'
          : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800',
      )}
    >
      {icon}
    </button>
  )
}
