# DGUDILI Webapp — 구조 문서

> GINEConv + ChemBERTa + Differential Cross-Attention 기반 약물 유발 간독성(DILI) 예측 웹 애플리케이션

---

## 전체 디렉터리 구조

```
webapp/
├── backend/                    # FastAPI 백엔드
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 진입점 (CORS, lifespan, 전역 예외 핸들러)
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── router.py           # /api/v1 라우터 집합
│   │   │       └── endpoints/
│   │   │           └── predict.py      # 단일 예측·배치 스크리닝 HTTP 엔드포인트
│   │   ├── core/
│   │   │   ├── config.py       # pydantic-settings 기반 전역 설정 (환경 변수)
│   │   │   └── model_loader.py # GraphMACCSEncoder + AutoTokenizer 싱글턴 관리
│   │   ├── ml/                 # DGUDILI_2026/src/ 에서 복사한 ML 코어 파일
│   │   │   ├── model.py                # GraphMACCSEncoder 모델 클래스
│   │   │   ├── differential_attention.py  # DifferentialCrossAttention
│   │   │   ├── graph_utils.py          # smiles_to_pyg, get_maccs, MACCS_NAMES
│   │   │   └── config.py               # 하이퍼파라미터 상수 (학습 코드와 공유)
│   │   ├── schemas/
│   │   │   ├── predict.py      # SinglePredictRequest / Response Pydantic 모델
│   │   │   └── batch.py        # BatchStatusResponse, TaskStatus enum
│   │   └── services/
│   │       ├── inference.py    # 단일 추론 오케스트레이터 (Lock, asyncio.to_thread)
│   │       ├── batch_runner.py # CSV 배치 백그라운드 워커 (진행률 실시간 갱신)
│   │       ├── chemistry.py    # RDKit: SMILES 검증, 물성치 계산, 기본 SVG
│   │       └── xai.py          # Differential Attention XAI (원자 하이라이트 SVG)
│   ├── weights/
│   │   └── pretrained_graph_encoder.pt  # 학습된 모델 가중치
│   ├── requirements.txt        # Python 의존성 (torch/torch-geometric은 별도 설치)
│   ├── Dockerfile              # ECS Fargate용 CPU-only 컨테이너
│   ├── .env                    # 로컬 개발 환경 변수 (git 제외 권장)
│   └── .env.example            # 환경 변수 템플릿
│
├── frontend/                   # React + Vite 프론트엔드
│   ├── src/
│   │   ├── main.tsx            # 앱 진입점 (QueryClient, BrowserRouter)
│   │   ├── App.tsx             # 헤더·탭 라우팅·푸터 레이아웃
│   │   ├── index.css           # Tailwind 디렉티브 + SVG 유틸리티
│   │   ├── api/
│   │   │   └── client.ts       # Axios 인스턴스 + API 함수 (predictSingle, uploadBatchCSV 등)
│   │   ├── types/
│   │   │   └── predict.ts      # TypeScript 타입 (백엔드 Pydantic 스키마와 1:1 대응)
│   │   └── components/
│   │       ├── SinglePredictView.tsx   # 단일 SMILES 예측 뷰 (SMILES 입력 → 결과 패널)
│   │       ├── BatchScreeningView.tsx  # CSV 배치 스크리닝 뷰 (Dropzone → 폴링 → 테이블)
│   │       ├── RiskGauge.tsx           # SVG 반원 게이지 (DILI 확률 시각화)
│   │       └── PhysChemTable.tsx       # 물리화학 특성 + Lipinski Rule-of-Five 테이블
│   ├── index.html
│   ├── vite.config.ts          # Vite 설정 (개발 프록시, 청크 분리)
│   ├── tailwind.config.js      # Tailwind 커스텀 색상·애니메이션
│   ├── tsconfig.json
│   ├── package.json
│   ├── Dockerfile              # nginx 2-stage 빌드
│   ├── nginx.conf              # SPA fallback + 캐시 설정
│   ├── .env.local              # 로컬 개발 환경 변수 (git 제외 권장)
│   └── .env.example            # 환경 변수 템플릿
│
└── docker-compose.yml          # 로컬 Docker Compose (백엔드 + 프론트엔드)
```

---

## 백엔드 아키텍처

### 요청 흐름 (단일 예측)

```
POST /api/v1/predict/single
        │
        ▼
predict.py:predict_single_endpoint()
        │
        ▼
inference.py:predict_single()          ← 비동기 오케스트레이터
  │
  ├─ [1] validate_and_canonicalize()   ← asyncio.to_thread (RDKit)
  │
  ├─ [2] _run_model_inference()        ← asyncio.to_thread
  │     │
  │     ├─ threading.Lock 획득
  │     │   ├─ smiles_to_pyg()        ← PyG 그래프 생성
  │     │   ├─ get_maccs()            ← MACCS 167-dim 벡터
  │     │   ├─ tokenizer()            ← ChemBERTa 토크나이저
  │     │   └─ model.forward()        ← GINEConv + ChemBERTa + DiffAttn
  │     │       → logit → sigmoid → P(DILI)
  │     │       → multihead_scores.clone()  (XAI용)
  │     └─ Lock 해제
  │         └─ build_xai_outputs()    ← 원자 중요도 + SVG 렌더링
  │
  └─ [3] get_physicochemical_props()  ← asyncio.to_thread (RDKit)
        │
        ▼
  SinglePredictResponse (JSON)
```

### 배치 처리 흐름

```
POST /api/v1/predict/batch (CSV 업로드)
  → task_id 즉시 반환 (202 Accepted)
  → BackgroundTasks: run_batch() 실행

GET /api/v1/predict/batch/status/{task_id}  ← 3초 폴링
  → { status, progress_pct, processed/total }

GET /api/v1/predict/batch/download/{task_id}
  → StreamingResponse (CSV)  또는  302 → S3 presigned URL
```

### 스레드 안전성

- `model.diff_attn.multihead_scores`는 Forward pass마다 덮어씌워지는 가변 속성
- `threading.Lock`으로 Forward pass + scores 캡처 구간을 직렬화
- numpy/RDKit 후처리는 Lock 외부 → 처리량 최대화

---

## 프론트엔드 아키텍처

### 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 프레임워크 | React 18 + TypeScript |
| 빌드 도구 | Vite 5 |
| 라우팅 | react-router-dom v6 |
| 서버 상태 | @tanstack/react-query v5 |
| 테이블 | @tanstack/react-table v8 |
| HTTP 클라이언트 | Axios |
| 스타일링 | Tailwind CSS v3 |
| 아이콘 | lucide-react |
| 파일 업로드 | react-dropzone |

### 탭별 컴포넌트

**단일 분석 (`/`)**
```
SinglePredictView
  ├─ SMILES 입력 폼 + 예제 버튼
  ├─ useMutation (predictSingle)
  └─ 결과 패널 (lg: 3-column grid)
      ├─ 좌측 (1/3): RiskGauge + MACCS 패턴 카드
      └─ 우측 (2/3): 분자 SVG (XAI) + PhysChemTable
```

**배치 스크리닝 (`/batch`)**
```
BatchScreeningView
  ├─ Dropzone (CSV 업로드)
  ├─ useMutation (uploadBatchCSV)
  ├─ useQuery (getBatchStatus, 3초 폴링)
  ├─ useQuery (getBatchDownloadUrl, done 시 CSV 파싱)
  └─ TanStack Table (정렬·페이지네이션)
```

### API 연결 설정

| 환경 | `VITE_API_BASE_URL` | 동작 |
|------|---------------------|------|
| 로컬 개발 | `""` (빈 문자열) | Vite proxy `/api → localhost:8000` |
| Docker Compose | `http://localhost:8000` | 직접 연결 |
| AWS 프로덕션 | ALB 도메인 또는 CloudFront 도메인 | 직접 연결 |

---

## 로컬 실행 방법

### 방법 1: 직접 실행 (권장 — 개발 시)

**백엔드**

```bash
cd webapp/backend

# 가상환경 생성 (Python 3.10)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# PyTorch CPU 설치 (GPU 없는 환경)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3

# 나머지 의존성
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> Swagger UI: http://localhost:8000/docs

**프론트엔드**

```bash
cd webapp/frontend
npm install
npm run dev
```

> 브라우저: http://localhost:3000

### 방법 2: Docker Compose

```bash
cd webapp
docker compose up --build
```

> - 프론트엔드: http://localhost:3000
> - 백엔드 API: http://localhost:8000
> - Swagger: http://localhost:8000/docs

---

## 환경 변수 요약

### 백엔드 (`webapp/backend/.env`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_WEIGHTS_PATH` | `weights/pretrained_graph_encoder.pt` | 모델 가중치 경로 |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `DILI_THRESHOLD` | `0.45` | DILI HIGH/LOW 분류 임계값 |
| `ALLOWED_ORIGINS` | `*` | CORS 허용 Origin (프로덕션 시 제한) |
| `S3_ENABLED` | `false` | `true`로 설정 시 배치 결과 S3 저장 |
| `S3_BUCKET_NAME` | `dgudili-batch-results` | S3 버킷명 |
| `AWS_REGION` | `ap-northeast-2` | AWS 리전 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

### 프론트엔드 (`webapp/frontend/.env.local`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VITE_API_BASE_URL` | `""` | 백엔드 API Origin. 빈 값이면 Vite proxy 사용 |
| `VITE_BATCH_POLL_INTERVAL_MS` | `3000` | 배치 상태 폴링 간격 (ms) |

---

## 모델 파이프라인 요약

```
SMILES 입력
    ├─ ChemBERTa-77M-MLM (last layer unfreeze)
    │       → CLS (B, 384) → LayerNorm → Linear → chem_feat (B, 64)
    │
    ├─ RDKit mol → 43-dim atom features
    │       → atom_proj → GINEConv×2 (edge_attr 9-dim)
    │       → to_dense_batch → node_proj → node_q (B, 100, 64)   [Query]
    │
    └─ MACCSkeys (B, 167)
            → Embedding(167, 64) × binary_gate → maccs_kv (B, 167, 64) [Key/Value]

DifferentialCrossAttention (Q=node_q, KV=maccs_kv)
    → attn_out (B, 100, 64) + multihead_scores (B, 4, 100, 167) [XAI용]

masked_mean_pool → graph_feat (B, 64)
concat([chem_feat, graph_feat]) → fuse_proj → LayerNorm → MLP → encode_out (B, 32)
head Linear(32, 1) → logit → sigmoid → P(DILI)
```

**성능 (10-Fold CV, N=1,850)**

| AUC | MCC | F1 | Sensitivity | Specificity |
|-----|-----|----|-------------|-------------|
| 0.9558 ±0.015 | 0.8080 ±0.051 | 0.9068 ±0.024 | 0.908 ±0.025 | 0.900 ±0.041 |
