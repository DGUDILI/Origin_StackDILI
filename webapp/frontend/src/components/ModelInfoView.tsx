/**
 * ModelInfoView.tsx — MAGDA-DILI 모델 소개 (정적 정보 페이지)
 *
 * 백엔드 API 호출 없음. 논문 기반 순수 프론트엔드 UI.
 * 출처: "차분 교차 어텐션 기반 다중모달 분자 표현을 활용한 약물유발간손상 예측"
 *       안윤지, 김현정, 나기현, 백지은, 임상수 (동국대학교, 2026)
 */

import { Network, Key, BookOpen, Zap, Shield, TrendingUp, BarChart3, ArrowRight, CheckCircle2 } from 'lucide-react'

// ─────────────────────────────────────────────────────────────────────────────
// 정적 데이터
// ─────────────────────────────────────────────────────────────────────────────

const MODALITIES = [
  {
    icon: Network,
    label: 'GINEConv Graph',
    subtitle: '국소 원자·결합 구조 인코더',
    accentBorder: 'border-t-4 border-t-emerald-500',
    iconBg: 'bg-emerald-100',
    iconColor: 'text-emerald-700',
    badgeBg: 'bg-emerald-50',
    badgeText: 'text-emerald-700',
    badgeBorder: 'border-emerald-200',
    description:
      '결합 특성을 메시지 패싱에 명시적으로 반영하는 Graph Isomorphism Network with Edge features. 결합 종류 및 E/Z 입체화학 차이에서 비롯되는 미세한 구조 이성질체를 포착합니다.',
    specs: [
      { label: '원자 특성', value: '43차원', detail: '원자번호, 결합차수, 혼성화(6종), 방향족, 고리 크기' },
      { label: '결합 특성', value: '9차원',  detail: '결합종류, 공액, 고리 여부, E/Z 입체화학' },
      { label: 'GINEConv 층',  value: '2 layers', detail: 'hidden_dim = 64, MAX_ATOMS = 100' },
    ],
    highlight: 'GIN 대비 AUC +0.050, Specificity +0.136 향상',
  },
  {
    icon: Key,
    label: 'MACCS Keys',
    subtitle: '해석 가능 구조 키 임베딩',
    accentBorder: 'border-t-4 border-t-amber-500',
    iconBg: 'bg-amber-100',
    iconColor: 'text-amber-700',
    badgeBg: 'bg-amber-50',
    badgeText: 'text-amber-700',
    badgeBorder: 'border-amber-200',
    description:
      '167비트 이진 MACCS 구조 키(Durant et al. 2002)를 학습 가능한 64차원 임베딩 행렬로 변환합니다. 비활성 비트(값=0)는 이진 게이트로 영벡터 처리되어 불필요한 잡음을 제거합니다.',
    specs: [
      { label: 'MACCS bits',    value: '167 bits',  detail: 'Durant et al. 2002 표준 키' },
      { label: '임베딩 차원',   value: '64차원',    detail: 'Embedding(167, 64) 학습 가능' },
      { label: 'Binary Gate',   value: '적용',      detail: '비활성 비트 → 영벡터 (노이즈 차단)' },
    ],
    highlight: 'Key/Value 역할로 원자-구조 키 상호작용 직접 학습',
  },
  {
    icon: BookOpen,
    label: 'ChemBERTa-77M',
    subtitle: '전역 화학 언어 모델',
    accentBorder: 'border-t-4 border-t-blue-500',
    iconBg: 'bg-blue-100',
    iconColor: 'text-blue-700',
    badgeBg: 'bg-blue-50',
    badgeText: 'text-blue-700',
    badgeBorder: 'border-blue-200',
    description:
      'SMILES 문자열을 처리하는 대형 화학 사전학습 언어 모델입니다. 마지막 인코더 레이어만 낮은 학습률(1e-4)로 미세조정하여 사전학습 표현을 보존하면서 DILI 과제에 특화합니다.',
    specs: [
      { label: '사전학습 규모', value: '77M SMILES', detail: 'DeepChem/ChemBERTa-MLM' },
      { label: 'CLS 임베딩',   value: '384차원',    detail: '마지막 레이어만 미세조정' },
      { label: '미세조정 전략', value: '1-layer',   detail: '2-layer 미세조정 시 AUC 하락 확인' },
    ],
    highlight: 'SMILES 전역 문맥으로 GNN 국소 표현 상호 보완',
  },
] as const

const PERF_METRICS = [
  {
    label: 'AUC-ROC',
    value: '0.937',
    sub: '±0.007',
    delta: '+ 0.128 vs InterDILI',
    color: 'text-blue-600',
    bigBg: 'bg-blue-600',
    cardBg: 'bg-blue-50',
    border: 'border-blue-100',
  },
  {
    label: 'MCC',
    value: '0.760',
    sub: '±0.035',
    delta: '+ 0.269 vs InterDILI',
    color: 'text-indigo-600',
    bigBg: 'bg-indigo-600',
    cardBg: 'bg-indigo-50',
    border: 'border-indigo-100',
  },
  {
    label: 'Specificity',
    value: '0.906',
    sub: '±0.037',
    delta: '+ 0.231 vs CAMDA-DILI',
    color: 'text-emerald-600',
    bigBg: 'bg-emerald-600',
    cardBg: 'bg-emerald-50',
    border: 'border-emerald-100',
  },
  {
    label: 'F1 Score',
    value: '0.878',
    sub: '±0.018',
    delta: '+ 0.112 vs StackDILI',
    color: 'text-violet-600',
    bigBg: 'bg-violet-600',
    cardBg: 'bg-violet-50',
    border: 'border-violet-100',
  },
] as const

const ABLATION_ROWS = [
  { model: 'GINE + DiffAttn',     auc: '0.937', mcc: '0.757', f1: '0.882', sens: '0.872', spec: '0.859', proposed: true  },
  { model: 'GIN + DiffAttn',      auc: '0.887', mcc: '0.628', f1: '0.828', sens: '0.894', spec: '0.723', proposed: false },
  { model: 'GINE + SoftMaxAttn',  auc: '0.891', mcc: '0.623', f1: '0.816', sens: '0.834', spec: '0.784', proposed: false },
] as const

// ─────────────────────────────────────────────────────────────────────────────
// 서브 컴포넌트
// ─────────────────────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-6 flex items-center gap-3">
      <div className="h-px flex-1 bg-slate-200" />
      <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
        {children}
      </span>
      <div className="h-px flex-1 bg-slate-200" />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 메인 컴포넌트
// ─────────────────────────────────────────────────────────────────────────────

export default function ModelInfoView() {
  return (
    <div className="space-y-10 animate-fade-in">

      {/* ══════════════════════════════════════════════════════════════════
          HERO — 모델 정체성
      ══════════════════════════════════════════════════════════════════ */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-800 via-slate-800 to-blue-900 p-8 shadow-xl">
        {/* 배경 패턴 */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage:
              'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
            backgroundSize: '28px 28px',
          }}
          aria-hidden
        />

        <div className="relative flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          {/* 왼쪽: 타이틀 */}
          <div className="flex-1 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-blue-500/20 px-3 py-1 text-xs font-bold uppercase tracking-widest text-blue-300 ring-1 ring-blue-400/30">
                MAGDA-DILI
              </span>
              <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-slate-300 ring-1 ring-white/10">
                10-fold CV · Random-split
              </span>
            </div>

            <h1 className="text-2xl font-bold leading-snug text-white sm:text-3xl">
              차분 교차 어텐션 기반<br />
              <span className="text-blue-300">다중모달 분자 표현</span>을 활용한<br />
              약물유발간손상 예측
            </h1>

            <p className="max-w-xl text-sm leading-relaxed text-slate-400">
              Multimodal Attention-based Graph with Differential Attention for Drug-Induced Liver Injury Prediction
            </p>

            <p className="text-xs text-slate-500">
              안윤지 · 김현정 · 나기현 · 백지은 · 임상수† &nbsp;|&nbsp; 동국대학교 AI소프트웨어융합학부 · 컴퓨터·AI학부, 2026
            </p>
          </div>

          {/* 오른쪽: 핵심 지표 배지 그리드 */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-2 lg:w-56">
            {PERF_METRICS.map((m) => (
              <div
                key={m.label}
                className="flex flex-col items-center rounded-xl bg-white/10 py-3 px-2 ring-1 ring-white/10 backdrop-blur-sm"
              >
                <span className={`text-2xl font-black tabular-nums ${m.color.replace('600', '300')}`}>
                  {m.value}
                </span>
                <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  {m.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 하단: 기술 스택 칩 */}
        <div className="relative mt-6 flex flex-wrap gap-2 border-t border-white/10 pt-5">
          {[
            'GINEConv (edge-attr 9-dim)',
            'MACCS Keys 167-bit',
            'ChemBERTa-77M',
            'Differential Cross-Attention',
            '5-fold OOF Stacking',
            'Logistic Regression Meta',
          ].map((tag) => (
            <span
              key={tag}
              className="rounded-md bg-white/8 px-2.5 py-1 text-[11px] font-medium text-slate-300 ring-1 ring-white/10"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          2-PHASE PIPELINE 개요
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <SectionLabel>2단계 학습 전략 (Two-Phase Pipeline)</SectionLabel>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {/* Phase 1 */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-xs font-black text-white shadow-sm">
                1
              </div>
              <div>
                <p className="text-sm font-bold text-slate-800">Pre-Training Phase</p>
                <p className="text-xs text-slate-400">GraphMACCSEncoder 사전학습</p>
              </div>
            </div>
            <ul className="space-y-2 text-xs text-slate-600">
              {[
                'GINEConv × 2 레이어 → 원자 노드 표현 (Query)',
                'MACCS 임베딩 + Binary Gate → 구조 키 (Key/Value)',
                'Differential Cross-Attention → 원자-구조 키 상호작용',
                'ChemBERTa CLS → 전역 표현 concat',
                '32차원 융합 표현 encode_out 생성',
                '검증 AUC 기반 조기 종료 (patience=30)',
              ].map((item) => (
                <li key={item} className="flex items-start gap-1.5">
                  <ArrowRight className="mt-0.5 h-3 w-3 flex-shrink-0 text-blue-400" />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          {/* Phase 2 */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-violet-600 text-xs font-black text-white shadow-sm">
                2
              </div>
              <div>
                <p className="text-sm font-bold text-slate-800">Ensemble Phase</p>
                <p className="text-xs text-slate-400">5-fold OOF Stacking 분류기</p>
              </div>
            </div>
            <ul className="space-y-2 text-xs text-slate-600">
              {[
                'Phase 1 인코더 가중치 동결(freeze)',
                '32차원 융합 표현 추출 (전체 학습 데이터)',
                '기저 모델: RF · ET · HistGB · XGBoost 4종',
                '5-fold 계층화 OOF 예측 확률 생성',
                'Logistic Regression 메타 학습기 학습',
                'MCC 최대화 임계값 탐색 (0.10 ~ 0.90)',
              ].map((item) => (
                <li key={item} className="flex items-start gap-1.5">
                  <ArrowRight className="mt-0.5 h-3 w-3 flex-shrink-0 text-violet-400" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          THREE MODALITIES
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <SectionLabel>다중모달 분자 표현 (Three Modalities)</SectionLabel>
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          {MODALITIES.map((mod) => {
            const Icon = mod.icon
            return (
              <div
                key={mod.label}
                className={`flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm ${mod.accentBorder} overflow-hidden`}
              >
                {/* 카드 헤더 */}
                <div className="flex items-center gap-3 p-5 pb-4">
                  <div className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl ${mod.iconBg}`}>
                    <Icon className={`h-5 w-5 ${mod.iconColor}`} />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-slate-800">{mod.label}</p>
                    <p className="text-xs text-slate-400">{mod.subtitle}</p>
                  </div>
                </div>

                {/* 설명 */}
                <p className="flex-1 px-5 pb-4 text-xs leading-relaxed text-slate-600">
                  {mod.description}
                </p>

                {/* 스펙 리스트 */}
                <div className="mx-5 mb-4 divide-y divide-slate-100 rounded-xl border border-slate-100 bg-slate-50/60">
                  {mod.specs.map((spec) => (
                    <div key={spec.label} className="flex items-center justify-between px-3 py-2">
                      <span className="text-[11px] text-slate-500">{spec.label}</span>
                      <div className="text-right">
                        <span className="text-xs font-bold text-slate-800">{spec.value}</span>
                        <p className="text-[10px] text-slate-400">{spec.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* 하이라이트 배지 */}
                <div className={`mx-5 mb-5 rounded-lg border px-3 py-2 ${mod.badgeBg} ${mod.badgeBorder}`}>
                  <p className={`text-[11px] font-semibold ${mod.badgeText}`}>
                    <CheckCircle2 className="mr-1 inline h-3 w-3" />
                    {mod.highlight}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          DIFFERENTIAL CROSS-ATTENTION
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <SectionLabel>핵심 혁신 — 차분 교차 어텐션 메커니즘</SectionLabel>
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">

          {/* 설명 (3/5) */}
          <div className="lg:col-span-3 space-y-4">
            {/* 동기 */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <Zap className="h-4 w-4 text-orange-500" />
                <h3 className="text-sm font-bold text-slate-800">왜 차분(Differential) 연산인가?</h3>
              </div>
              <p className="text-xs leading-relaxed text-slate-600">
                표준 Softmax Attention은 <span className="font-semibold text-slate-800">비특이적 배경 활성화</span>를 억제하지 못해
                독성과 무관한 공통 구조 패턴에도 높은 어텐션이 할당됩니다.
                차분 연산은 두 어텐션 맵의 차이를 이용해 이런 배경 노이즈를 상쇄하고,
                <span className="font-semibold text-orange-600"> DILI와 직결되는 원자-구조 키 상호작용만을 선별적으로 강조</span>합니다.
              </p>
            </div>

            {/* 수식 */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-sm">
              <div className="mb-4 flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-emerald-400" />
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  Differential Cross-Attention 수식
                </p>
              </div>

              <div className="space-y-3 font-mono text-[11px] leading-relaxed">
                <div className="rounded-lg bg-slate-800/60 px-4 py-2.5">
                  <span className="text-slate-400">λ = </span>
                  <span className="text-emerald-300">exp(λ_q1 · λ_k1)</span>
                  <span className="text-slate-300"> − </span>
                  <span className="text-blue-300">exp(λ_q2 · λ_k2)</span>
                  <span className="text-slate-300"> + λ_init</span>
                </div>

                <div className="rounded-lg bg-slate-800/60 px-4 py-2.5">
                  <span className="text-slate-400">attn = </span>
                  <span className="text-emerald-300">softmax(Q₁K₁ᵀ / √d_h)</span>
                  <span className="text-slate-300"> − λ · </span>
                  <span className="text-blue-300">softmax(Q₂K₂ᵀ / √d_h)</span>
                </div>

                <div className="rounded-lg bg-slate-800/60 px-4 py-2.5">
                  <span className="text-slate-400">λ = </span>
                  <span className="text-amber-300">clamp(λ,  min = 1e-4,  max = 2.0)</span>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
                {[
                  { symbol: 'Q', desc: '원자 노드 표현 (GINEConv 출력)' },
                  { symbol: 'K/V', desc: 'MACCS 구조 키 임베딩' },
                  { symbol: 'λ', desc: '헤드별 차분 강도 (학습 가능)' },
                ].map((item) => (
                  <div key={item.symbol} className="rounded-md bg-white/5 px-2.5 py-2">
                    <span className="font-mono text-xs font-bold text-amber-300">{item.symbol}</span>
                    <p className="mt-0.5 text-[10px] text-slate-400">{item.desc}</p>
                  </div>
                ))}
              </div>

              <p className="mt-4 text-[10px] leading-relaxed text-slate-500">
                h=4 헤드, d_head=16. GroupNorm + 스케일 보정으로 학습 안정성 확보.
                Ye et al. (arXiv:2410.05258)의 Differential Transformer를 교차 어텐션 형태로 재구성.
              </p>
            </div>

            {/* 효과 요약 */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="mb-3 text-sm font-bold text-slate-800">차분 어텐션의 3가지 효과</h3>
              <div className="space-y-2">
                {[
                  { color: 'bg-orange-400', text: '배경 노이즈 억제 — 공통적·비특이적 활성화를 상쇄' },
                  { color: 'bg-blue-400',   text: '선별적 강조 — 독성 관련 원자-구조 키 상호작용 부각' },
                  { color: 'bg-emerald-400', text: '해석 가능성 — 헤드별 λ 수렴값으로 기여 구조 단서 시각화' },
                ].map((item) => (
                  <div key={item.text} className="flex items-start gap-2.5 text-xs text-slate-600">
                    <div className={`mt-1.5 h-2 w-2 flex-shrink-0 rounded-full ${item.color}`} />
                    {item.text}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 아키텍처 다이어그램 (2/5) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="border-b border-slate-100 px-5 py-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  Figure 1 · 모델 아키텍처 개요
                </p>
              </div>
              <img
                src="/architecture.png"
                alt="MAGDA-DILI 모델 아키텍처 — GraphMACCSEncoder + Stacking"
                className="w-full object-contain"
              />
            </div>

            {/* 어텐션 헤드 시각화 설명 카드 */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-violet-500" />
                <h3 className="text-sm font-bold text-slate-800">헤드별 역할 분화</h3>
              </div>
              <p className="text-xs leading-relaxed text-slate-600">
                학습 후 헤드별 <span className="font-mono font-semibold">λ</span> 파라미터가 서로 다른 값으로 수렴하여,
                각 헤드가 서로 다른 구조적 특성 공간을 담당함을 보여줍니다.
                복수의 헤드가 공통적으로 주목하는 원자-MACCS 쌍이
                예측에 핵심적으로 기여하는 구조 단서입니다.
              </p>
              <div className="mt-3 grid grid-cols-4 gap-1">
                {['Head 1', 'Head 2', 'Head 3', 'Head 4'].map((h, i) => (
                  <div
                    key={h}
                    className="rounded-md bg-violet-50 py-2 text-center"
                    style={{ opacity: 0.5 + i * 0.15 }}
                  >
                    <p className="text-[9px] font-semibold text-violet-600">{h}</p>
                    <p className="text-[9px] text-violet-400">λ → {(0.3 + i * 0.25).toFixed(2)}</p>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[10px] text-slate-400">
                * λ 수렴값은 학습 데이터에 따라 달라집니다 (예시 수치)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          PERFORMANCE METRICS
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <SectionLabel>실험 성과 (10-fold 교차검증, Random-split)</SectionLabel>

        {/* 4대 지표 카드 */}
        <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {PERF_METRICS.map((m) => (
            <div
              key={m.label}
              className={`relative overflow-hidden rounded-2xl border ${m.border} ${m.cardBg} p-5 shadow-sm`}
            >
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                {m.label}
              </p>
              <div className="mt-2 flex items-end gap-1">
                <span className={`text-4xl font-black tabular-nums ${m.color}`}>
                  {m.value}
                </span>
                <span className={`mb-1 text-sm font-semibold ${m.color} opacity-60`}>
                  {m.sub}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] font-medium text-slate-500">
                {m.delta}
              </p>
              {/* 장식용 원 */}
              <div
                className={`pointer-events-none absolute -right-4 -top-4 h-16 w-16 rounded-full ${m.bigBg} opacity-10`}
                aria-hidden
              />
            </div>
          ))}
        </div>

        {/* FP 감소 효과 배너 */}
        <div className="overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 p-6 shadow-md">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-white/20">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <p className="text-base font-bold text-white">
                  위양성(False Positive) 획기적 감소 — 신약 개발 비용 절감
                </p>
                <p className="mt-1 text-sm leading-relaxed text-emerald-100">
                  Specificity <span className="font-black text-white">0.906</span> 달성 — 비독성 화합물을 독성으로 오진하는 비율을
                  기존 최고 모델(0.675) 대비 <span className="font-black text-white">+23.1%p</span> 감소시켰습니다.
                  초기 신약 개발 단계에서 유망한 후보 물질의 불필요한 탈락을 방지합니다.
                </p>
              </div>
            </div>
            <div className="flex flex-shrink-0 flex-col items-center rounded-xl bg-white/20 px-6 py-4">
              <span className="text-4xl font-black text-white">+23.1</span>
              <span className="mt-0.5 text-xs font-semibold uppercase tracking-wider text-emerald-200">
                Specificity ↑ (%p)
              </span>
            </div>
          </div>
        </div>

        {/* 비교 맥락 카드 */}
        <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-blue-500" />
            <h3 className="text-sm font-bold text-slate-800">기존 DILI 예측 모델 대비 성능 우위</h3>
            <span className="ml-auto text-[10px] text-slate-400">Random-split, 10-fold CV</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-100">
                  {['Model', 'AUC', 'MCC', 'F1', 'Sensitivity', 'Specificity'].map((h) => (
                    <th key={h} className="pb-2 pr-4 text-left font-semibold uppercase tracking-wider text-slate-400 last:pr-0">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {[
                  { model: 'InterDILI',   auc: '0.809', mcc: '0.491', f1: '0.765', sens: '0.820', spec: '0.663', proposed: false },
                  { model: 'CAMDA-DILI',  auc: '0.805', mcc: '0.470', f1: '0.752', sens: '0.791', spec: '0.675', proposed: false },
                  { model: 'StackDILI',   auc: '0.798', mcc: '0.476', f1: '0.765', sens: '0.864', spec: '0.592', proposed: false },
                  { model: 'MAGDA-DILI',  auc: '0.937', mcc: '0.760', f1: '0.878', sens: '0.854', spec: '0.906', proposed: true  },
                ].map((row) => (
                  <tr
                    key={row.model}
                    className={row.proposed ? 'bg-blue-50/60 font-semibold' : ''}
                  >
                    <td className="py-2.5 pr-4 text-slate-700">
                      {row.proposed
                        ? <span className="inline-flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-blue-500" />{row.model}</span>
                        : row.model
                      }
                    </td>
                    {[row.auc, row.mcc, row.f1, row.sens, row.spec].map((v, i) => (
                      <td key={i} className={`py-2.5 pr-4 tabular-nums last:pr-0 ${row.proposed ? 'text-blue-700' : 'text-slate-600'}`}>
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          ABLATION STUDY
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <SectionLabel>구성 요소 분석 (Ablation Study)</SectionLabel>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">

          {/* 테이블 */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-slate-500" />
              <h3 className="text-sm font-bold text-slate-800">노드 인코더 × 어텐션 방식 비교</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-100">
                    {['구성', 'AUC', 'MCC', 'Spec.'].map((h) => (
                      <th key={h} className="pb-2 pr-3 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-400 last:pr-0">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {ABLATION_ROWS.map((row) => (
                    <tr key={row.model} className={row.proposed ? 'bg-blue-50/60' : ''}>
                      <td className={`py-2.5 pr-3 ${row.proposed ? 'font-semibold text-blue-800' : 'text-slate-600'}`}>
                        {row.proposed
                          ? <span className="inline-flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-blue-500" />{row.model} ★</span>
                          : row.model
                        }
                      </td>
                      {[row.auc, row.mcc, row.spec].map((v, i) => (
                        <td key={i} className={`py-2.5 pr-3 tabular-nums last:pr-0 ${row.proposed ? 'font-semibold text-blue-700' : 'text-slate-600'}`}>
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 해석 */}
          <div className="space-y-3">
            {[
              {
                title: 'GINEConv vs GIN',
                diff: 'AUC +0.050 · Specificity +0.136',
                body: '결합 특성(9차원)을 명시적으로 메시지 패싱에 반영함으로써, GIN이 놓치는 결합 종류·입체화학 차이를 포착합니다.',
                color: 'border-l-emerald-400',
              },
              {
                title: 'Differential vs SoftMax Attention',
                diff: 'AUC +0.046 · MCC +0.133',
                body: '배경 활성화를 차분 연산으로 상쇄함으로써 독성 관련 원자-구조 키 상호작용을 선별적으로 강조합니다.',
                color: 'border-l-orange-400',
              },
              {
                title: '두 구성 요소의 상호 보완',
                diff: 'Full Model 최고 성능',
                body: '결합 속성 반영 노드 인코딩과 차분 어텐션이 상호 보완적으로 작용하여 두 구성 요소 모두가 핵심적으로 기여합니다.',
                color: 'border-l-blue-400',
              },
            ].map((item) => (
              <div key={item.title} className={`rounded-xl border border-slate-100 border-l-4 bg-white p-4 shadow-sm ${item.color}`}>
                <div className="flex items-start justify-between gap-2">
                  <p className="text-xs font-bold text-slate-800">{item.title}</p>
                  <span className="flex-shrink-0 rounded bg-slate-100 px-2 py-0.5 font-mono text-[10px] font-semibold text-slate-600">
                    {item.diff}
                  </span>
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{item.body}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          FOOTER NOTE
      ══════════════════════════════════════════════════════════════════ */}
      <div className="rounded-xl border border-slate-100 bg-white px-6 py-4 text-center shadow-sm">
        <p className="text-[11px] text-slate-400">
          MAGDA-DILI · 동국대학교 AI소프트웨어융합학부 · 2026 &nbsp;|&nbsp;
          GINEConv + MACCS Keys + ChemBERTa + Differential Cross-Attention + 5-fold OOF Stacking &nbsp;|&nbsp;
          <span className="font-mono">10-fold CV AUC 0.937 ± 0.007</span>
        </p>
      </div>

    </div>
  )
}
