/**
 * App.tsx — 전체 대시보드 레이아웃 및 탭 라우팅
 *
 * 구조:
 *  ┌──────────────────────── Header ────────────────────────────────────┐
 *  │  [FlaskConical] DGUDILI   [단일 분석] [배치 스크리닝]   모델 배지  │
 *  └───────────────────────────────────────────────────────────────────┘
 *  ┌──────────────────────── Main ──────────────────────────────────────┐
 *  │  <SinglePredictView />  또는  <BatchScreeningView />               │
 *  └───────────────────────────────────────────────────────────────────┘
 *  ┌──────────────────────── Footer ────────────────────────────────────┐
 *  │  DGUDILI 2026 · GINEConv + ChemBERTa + DiffAttn                   │
 *  └───────────────────────────────────────────────────────────────────┘
 */

import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { FlaskConical, Layers } from 'lucide-react'
import { clsx } from 'clsx'

import SinglePredictView   from '@/components/SinglePredictView'
import BatchScreeningView  from '@/components/BatchScreeningView'

// ─── 탭 링크 스타일 ───────────────────────────────────────────────────────────

const tabCls = ({ isActive }: { isActive: boolean }) =>
  clsx(
    'relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/50',
    isActive
      ? 'bg-blue-600 text-white shadow-sm shadow-blue-200'
      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
  )

// ─── 메인 앱 ──────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50">

      {/* ── 헤더 ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 sm:px-6">

          {/* 브랜드 */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 shadow-sm">
              <FlaskConical className="h-4.5 w-4.5 text-white" aria-hidden />
            </div>
            <div>
              <span className="text-base font-bold tracking-tight text-slate-900">DGUDILI</span>
              <span className="ml-2 hidden text-xs font-normal text-slate-400 sm:inline">
                Drug-Induced Liver Injury Predictor
              </span>
            </div>
          </div>

          {/* 탭 내비게이션 */}
          <nav className="flex gap-1" role="tablist" aria-label="분석 모드">
            <NavLink to="/"      end className={tabCls} role="tab">단일 분석</NavLink>
            <NavLink to="/batch"     className={tabCls} role="tab">배치 스크리닝</NavLink>
          </nav>

          {/* 모델 정보 배지 (우측 밀기) */}
          <div className="ml-auto hidden items-center gap-1.5 sm:flex">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-medium text-slate-500">
              <Layers className="h-3 w-3 text-blue-500" aria-hidden />
              GINEConv · ChemBERTa · DiffAttn
            </span>
          </div>
        </div>
      </header>

      {/* ── 메인 콘텐츠 ──────────────────────────────────────────────────── */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6">
        <Routes>
          <Route path="/"      element={<SinglePredictView />} />
          <Route path="/batch" element={<BatchScreeningView />} />
          <Route path="*"      element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {/* ── 푸터 ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-100 bg-white py-4">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <p className="text-center text-[11px] text-slate-400">
            DGUDILI 2026 · GINEConv + ChemBERTa-77M + Differential Cross-Attention ·
            {' '}
            <span className="font-mono">val AUC 0.9558 ± 0.015</span>
            {' (10-fold CV)'}
          </p>
        </div>
      </footer>
    </div>
  )
}
