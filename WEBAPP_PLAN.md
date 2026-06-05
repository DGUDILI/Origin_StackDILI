# DGUDILI_2026 웹 애플리케이션 전환 구현 계획서

> 작성일: 2026-05-29  
> 목표: PyTorch + RDKit 기반 DILI 예측 파이프라인을 엔터프라이즈급 React/FastAPI/AWS 풀스택 웹 앱으로 전환

---

## 전체 아키텍처 개요

```
[사용자 브라우저]
       │
       ▼
[CloudFront CDN]──── S3 (React 정적 빌드)
       │
       ▼
[ALB: Application Load Balancer]
       │
       ▼
[ECS Fargate Cluster]
  ├── FastAPI 컨테이너 (추론 + XAI)
  │     ├── GraphMACCSEncoder (GINEConv + DiffAttn + ChemBERTa)
  │     ├── RDKit (SVG 렌더링, 물리화학 특성)
  │     └── BackgroundTasks (배치 큐 처리)
  └── Auto Scaling (CPU 70% 기준)
       │
       ▼
[S3 Bucket: 배치 결과 CSV 저장]

[ECR: Docker 이미지 레지스트리]
[VPC: Public/Private 서브넷 격리]
```

---

## 디렉토리 구조

```
Origin_StackDILI/          ← 현재 레포 루트
├── DGUDILI_2026/           ← 기존 ML 파이프라인 (변경 없음)
│   └── src/                ← model.py, graph_utils.py 등 재사용
│
├── webapp/                 ← 신규 생성 (전체 웹앱 루트)
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── api/v1/
│   │   │   │   ├── router.py
│   │   │   │   └── endpoints/
│   │   │   │       ├── predict.py   # /predict/single, /predict/batch
│   │   │   │       └── health.py    # /health
│   │   │   ├── core/
│   │   │   │   ├── config.py        # pydantic-settings 환경 변수
│   │   │   │   └── model_loader.py  # 모델 싱글턴 (앱 시작 시 로드)
│   │   │   ├── services/
│   │   │   │   ├── inference.py     # 단일 SMILES 추론 로직
│   │   │   │   ├── batch_runner.py  # CSV 배치 + 태스크 상태 관리
│   │   │   │   ├── xai.py           # attn_weights → SVG 오버레이
│   │   │   │   └── chemistry.py     # RDKit 물리화학 특성, 분자 SVG
│   │   │   ├── schemas/
│   │   │   │   ├── predict.py       # Request/Response Pydantic 모델
│   │   │   │   └── batch.py
│   │   │   └── ml/                  # src/ 복사본 (임포트용)
│   │   │       ├── model.py
│   │   │       ├── graph_utils.py
│   │   │       ├── differential_attention.py
│   │   │       └── config.py
│   │   ├── tests/
│   │   │   ├── test_predict.py
│   │   │   └── test_batch.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── main.tsx
│   │   │   ├── App.tsx
│   │   │   ├── api/
│   │   │   │   └── client.ts        # Axios 인스턴스 + API 함수
│   │   │   ├── types/
│   │   │   │   └── predict.ts       # TypeScript 타입 (백엔드 schemas/predict.py 1:1 매핑)
│   │   │   ├── components/
│   │   │   │   ├── SinglePredictView.tsx  # 단일 분석 뷰 (입력+결과 통합)
│   │   │   │   │   # 내장 컴포넌트:
│   │   │   │   │   # - ToxicReasonCard   : SMARTS 작용기 기여도 게이지 바 카드
│   │   │   │   │   # - MaccsPatternCard  : MACCS 비트 기여도 카드 (제거됨)
│   │   │   │   ├── BatchScreeningView.tsx # 배치 CSV 업로드 + 진행률 + 결과 테이블
│   │   │   │   ├── ModelInfoView.tsx      # 모델 아키텍처 소개 탭
│   │   │   │   ├── RiskGauge.tsx          # 반원 게이지 (확률 → 색상)
│   │   │   │   └── PhysChemTable.tsx      # 물성치 테이블 + Lipinski 위반 표시
│   │   ├── index.html
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   ├── tailwind.config.js
│   │   └── tsconfig.json
│   │
│   └── infra/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── terraform.tfvars.example
│       └── modules/
│           ├── vpc/
│           │   ├── main.tf
│           │   └── variables.tf
│           ├── ecr/
│           │   ├── main.tf
│           │   └── variables.tf
│           ├── ecs/
│           │   ├── main.tf
│           │   ├── variables.tf
│           │   └── task_definition.json.tpl
│           ├── alb/
│           │   ├── main.tf
│           │   └── variables.tf
│           └── s3_cloudfront/
│               ├── main.tf
│               └── variables.tf
│
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## Phase 0: 프로젝트 세팅 (½일)

### 0-1. 디렉토리 초기화
```bash
mkdir -p webapp/backend/app/{api/v1/endpoints,core,services,schemas,ml}
mkdir -p webapp/backend/tests
mkdir -p webapp/frontend/src/{api,types,components/{layout,single,batch},pages}
mkdir -p webapp/infra/modules/{vpc,ecr,ecs,alb,s3_cloudfront}
mkdir -p .github/workflows
```

### 0-2. ML 모듈 복사
기존 `DGUDILI_2026/src/` → `webapp/backend/app/ml/` 로 복사 (model.py, graph_utils.py, differential_attention.py, config.py). 원본은 수정하지 않음.

### 0-3. 환경 변수 파일
```
webapp/backend/.env.example:
  MODEL_WEIGHTS_PATH=./weights/pretrained_graph_encoder.pt
  CHEMBERTA_MODEL_NAME=DeepChem/ChemBERTa-77M-MLM
  S3_BUCKET_NAME=dgudili-batch-results
  AWS_REGION=ap-northeast-2
  BATCH_RESULT_TTL_SECONDS=3600
  MAX_BATCH_SIZE=5000
```

---

## Phase 1: 백엔드 구현 (3일)

### 1-1. `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic-settings==2.5.0
python-multipart==0.0.12
httpx==0.27.0
torch==2.3.0
torch-geometric==2.5.3
transformers==4.44.0
rdkit==2024.03.5
numpy==1.26.4
scikit-learn==1.7.1
xgboost==2.1.0
einops==0.8.0
boto3==1.35.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

### 1-2. `app/core/config.py`
pydantic-settings로 환경 변수를 타입 안전하게 로드. 개발/프로덕션 공통 설정.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_weights_path: str = "./weights/pretrained_graph_encoder.pt"
    chemberta_model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    s3_bucket_name: str = "dgudili-batch-results"
    aws_region: str = "ap-northeast-2"
    batch_result_ttl: int = 3600
    max_batch_size: int = 5000
    device: str = "cpu"  # Fargate는 CPU 전용
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 1-3. `app/core/model_loader.py`
FastAPI 앱 시작 시 **한 번만** 모델을 로드하는 싱글턴. `lifespan` 이벤트 사용.

```python
# 핵심 구조
from contextlib import asynccontextmanager
from fastapi import FastAPI

_model: GraphMACCSEncoder | None = None
_tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    # 앱 시작: 모델 + 토크나이저 로드
    _tokenizer = AutoTokenizer.from_pretrained(settings.chemberta_model_name)
    _model = GraphMACCSEncoder(...)
    _model.load_state_dict(torch.load(settings.model_weights_path, map_location="cpu"))
    _model.eval()
    yield
    # 앱 종료: 정리 (필요시)
    _model = None

def get_model() -> GraphMACCSEncoder:
    return _model

def get_tokenizer():
    return _tokenizer
```

### 1-4. `app/services/chemistry.py`
RDKit 기반 유틸리티.

**기능:**
- `smiles_to_svg(smiles, atom_weights=None) → str`: 원자 중요도로 색상 오버레이된 SVG 반환.  
  atom_weights가 있으면 `rdkit.Chem.Draw.rdMolDraw2D` + `DrawMoleculeWithHighlights` 사용.
- `get_physicochemical_props(smiles) → dict`: RDKit Descriptors로 MW, LogP, HBD, HBA, TPSA, RotBonds, QED 계산.
- `validate_smiles(smiles) → bool`: `Chem.MolFromSmiles(smiles) is not None`

**주의:** RDKit은 thread-safe하지 않으므로 `asyncio.to_thread` 또는 `ThreadPoolExecutor`로 래핑.

### 1-5. `app/services/xai.py`
Differential Cross-Attention 멀티헤드 어텐션 → XAI 산출물 전체 파이프라인.

**주요 함수:**
- `extract_mh_scores_numpy(mh_raw, n_atoms) → np.ndarray`:  
  `(B, h, MAX_ATOMS, 167)` → ReLU → bit 0 제외 → `(h, n_atoms, 166)` numpy
- `compute_atom_importance(mh_scores) → np.ndarray`:  
  4헤드 × 166비트 합산 → Min-Max 정규화 → `(n_atoms,)` [0, 1]
- `extract_top_maccs_patterns(mh_scores, smiles, top_k) → list[MaccsPatternScore]`:  
  MACCS 축 합산 → **실제 분자 활성 비트(MACCSkeys) 마스킹** → 상위 k개 반환  
  이름은 `smartsPatts` 실제 SMARTS 기반 파생 (`MACCS_NAMES` 불사용 — 비트 번호 불일치)
- `compute_fg_toxic_reasons(smiles, atom_importance, top_k=5) → list[ToxicReasonScore]`:  
  18가지 SMARTS 작용기 패턴 매칭 → 원자 기여도 합산 → 부족분 원자 레벨 폴백 →  
  **통합 정규화 (TOP-K 합계 = 정확히 100%)** → 기여도 내림차순 정렬
- `render_xai_svg(smiles, atom_importance, prob) → str`:  
  prob > 45% → 빨강 그라디언트, ≤ 45% → 파랑 그라디언트, 원자+결합 하이라이트
- `build_xai_outputs(mh_raw, smiles, n_atoms, prob, top_k) → tuple[list, list, str]`:  
  퍼사드 — `(top_maccs, toxic_reasons, svg_string)` 반환

**SMARTS_FG_LIST (18가지):** Nitro Group, Aromatic Amine, Aliphatic Amine, Secondary Amine, Amide, Phenol, Hydroxyl, Carboxylic Acid, Ester, Aldehyde, Ketone, Epoxide, Thiol, Thioether, Sulfoxide, Sulfonamide, Halide, Imine

### 1-6. `app/services/inference.py`
단일 SMILES 추론 서비스 (스레드 안전성 설계 포함).

```python
async def predict_single(smiles: str, include_xai: bool = True) -> SinglePredictResponse:
    # 1. SMILES 유효성 검사 + Canonical 변환  [asyncio.to_thread]
    # 2. (실패 시) PubChem 분자 이름 조회      [endpoint 레벨]
    # ── threading.Lock 구간 (Forward pass 직렬화) ──
    # 3. PyG 그래프 생성 (smiles_to_pyg)
    # 4. MACCS 167-dim 벡터 (get_maccs)
    # 5. ChemBERTa 토크나이저 인코딩
    # 6. model.forward() → logit → sigmoid → probability (0~100%)
    # 7. model.diff_attn.multihead_scores.clone() 즉시 복사   ← 핵심
    # ── Lock 해제 후 (numpy/RDKit, thread-safe) ──
    # 8. extract_mh_scores_numpy() → ReLU → (h, n_atoms, 166)
    # 9. compute_atom_importance() → Min-Max [0,1]
    # 10. extract_top_maccs_patterns() → 활성 비트 마스킹 → TOP-K
    # 11. compute_fg_toxic_reasons() → SMARTS 18종 → TOP-5 → 정규화
    # 12. render_xai_svg() → 빨강/파랑 그라디언트 SVG
    # 13. get_physicochemical_props()  [asyncio.to_thread]
    # 14. SinglePredictResponse 조립 및 반환
```

**스레드 안전성:** `model.diff_attn.multihead_scores`는 Forward pass마다 덮어씌워지므로 `threading.Lock` 구간 내에서 `.clone()`으로 즉시 복사. numpy/RDKit 처리는 Lock 외부에서 수행해 처리량 극대화.

### 1-7. `app/services/batch_runner.py`
배치 처리 + 태스크 상태 관리.

**태스크 저장소:** 인메모리 `dict[str, TaskState]` (소규모 프로토타입). 프로덕션 확장 시 Redis로 교체.

```python
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    total: int = 0
    processed: int = 0
    error: str | None = None
    result_s3_key: str | None = None
    result_rows: list[dict] = field(default_factory=list)  # 소규모 시 메모리 저장

# BackgroundTask 흐름:
# 1. CSV 파싱 (pandas)
# 2. SMILES 컬럼 자동 감지 (컬럼명 'smiles', 'SMILES', 'Smiles' 우선)
# 3. 배치 크기 32씩 분할 처리
# 4. 각 SMILES → predict_single (동기 버전 호출)
# 5. 진행률 업데이트: task_state.processed += 1
# 6. 완료 후 결과 DataFrame → CSV → S3 업로드
# 7. 에러 시 TaskStatus.FAILED + error 메시지 저장
```

**MAX_BATCH_SIZE 초과 시 422 에러 즉시 반환.**

### 1-8. `app/schemas/predict.py` (Pydantic 모델)

```python
# Request
class SinglePredictRequest(BaseModel):
    smiles: str                  # SMILES 또는 영문 분자 이름 (PubChem 조회 지원)
    include_xai: bool = True

# Response 서브모델
class MaccsPattern(BaseModel):
    bit_index: int               # MACCS bit 인덱스 (1~166)
    name: str                    # smartsPatts 실제 SMARTS 기반 이름
    importance: float            # 활성 비트 내 기여 비율 (0~1)

class ToxicReason(BaseModel):   # ← 신규 (SMARTS 작용기 기반 XAI)
    name: str                    # 작용기명 또는 "Atom #N (X)" 폴백
    importance: float            # TOP-K 합산 정규화 기여도 (0~1, 합계=1.0)
    rank: int                    # 기여도 순위 (1=최고)

class PhysChemProps(BaseModel):
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    qed: float
    ring_count: int              # ← 신규
    aromatic_rings: int          # ← 신규
    lipinski_violations: int     # computed_field: MW>500, LogP>5, HBD>5, HBA>10

class SinglePredictResponse(BaseModel):
    smiles: str                  # 사용자 입력 원본
    canonical_smiles: str        # ← 신규: RDKit Canonical SMILES
    probability: float           # 0.0 ~ 100.0 (%)
    risk_level: str              # "HIGH" | "LOW"
    top_maccs_patterns: list[MaccsPattern]
    toxic_reasons: list[ToxicReason]  # ← 신규: SMARTS 작용기 TOP 5
    physicochemical: PhysChemProps
    molecule_svg: str            # XAI 원자+결합 하이라이트 SVG

# Batch
class BatchStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    total: int
    processed: int
    progress_pct: float
    has_result: bool             # ← 신규: 다운로드 가능 여부
    error: str | None = None
```

### 1-9. `app/api/v1/endpoints/predict.py`

```python
@router.post("/single", response_model=SinglePredictResponse)
async def predict_single_endpoint(req: SinglePredictRequest):
    if not validate_smiles(req.smiles):
        raise HTTPException(422, "Invalid SMILES string")
    return await inference.predict_single(req.smiles, req.include_xai)

@router.post("/batch", response_model=dict)
async def predict_batch_endpoint(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    content = await file.read()
    # CSV 크기 사전 검증 (pandas로 행 수 확인)
    task_id = str(uuid.uuid4())
    task_store[task_id] = TaskState(task_id=task_id)
    background_tasks.add_task(batch_runner.run_batch, task_id, content)
    return {"task_id": task_id}

@router.get("/batch/status/{task_id}", response_model=BatchStatusResponse)
async def get_batch_status(task_id: str):
    state = task_store.get(task_id)
    if not state:
        raise HTTPException(404, "Task not found")
    return BatchStatusResponse(
        task_id=state.task_id,
        status=state.status,
        total=state.total,
        processed=state.processed,
        progress_pct=state.processed / max(state.total, 1) * 100,
        result_download_url=_presign_url(state.result_s3_key) if state.result_s3_key else None,
        error=state.error,
    )
```

### 1-10. `app/main.py`

```python
app = FastAPI(title="DGUDILI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])  # 프로덕션: CloudFront 도메인으로 제한
app.include_router(v1_router, prefix="/api/v1")
```

### 1-11. `Dockerfile` (backend)

```dockerfile
FROM python:3.10-slim

# RDKit SVG 렌더링 필수 시스템 라이브러리
# libexpat1    : RDKit rdMolDraw2D SVG XML 파싱
# libfreetype6 : 원자 레이블 폰트 렌더링
# libfontconfig1: 폰트 설정
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 curl \
    libexpat1 libfreetype6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY weights/ ./weights/

EXPOSE 8000
# workers=1 고정: 모델 싱글턴이 프로세스 간 공유 안 됨
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> **주의:** `python:3.10-slim`은 XML/폰트 관련 시스템 라이브러리를 제거한 이미지이므로 `libexpat1` 등을 명시적으로 설치해야 합니다. 누락 시 RDKit SVG 렌더링이 `libexpat.so.1: cannot open shared object file` 오류로 전면 실패합니다.

> **모델 가중치 전략:** `pretrained_graph_encoder.pt`(~수십MB)는 ECR 이미지에 포함하거나 S3에서 `lifespan` 시작 시 `boto3.download_file`로 받는 방식 중 선택. 빠른 cold start 위해 이미지 포함 권장.

---

## Phase 2: 프론트엔드 구현 (4일)

### 2-1. 기술 스택

| 항목 | 선택 | 이유 |
|---|---|---|
| 프레임워크 | React 18 + Vite | 빠른 빌드, HMR |
| UI 스타일링 | Tailwind CSS v3 | 빠른 프로토타이핑 |
| 상태 관리 | React Query (TanStack) | API 캐싱 + 폴링 내장 |
| HTTP 클라이언트 | Axios | 인터셉터 설정 용이 |
| 차트/게이지 | Recharts | React 네이티브 |
| 테이블 | TanStack Table v8 | 정렬/필터 기본 제공 |
| 파일 업로드 | react-dropzone | 드래그앤드롭 |
| 타입 | TypeScript | 안전성 |

### 2-2. `package.json` 핵심 의존성

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "axios": "^1.7.0",
    "@tanstack/react-query": "^5.56.0",
    "@tanstack/react-table": "^8.20.0",
    "recharts": "^2.12.0",
    "react-dropzone": "^14.2.0",
    "clsx": "^2.1.0"
  },
  "devDependencies": {
    "vite": "^5.4.0",
    "@vitejs/plugin-react": "^4.3.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.5.0"
  }
}
```

### 2-3. `api/client.ts`

```typescript
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const apiClient = axios.create({ baseURL: BASE_URL });

export const predictSingle = (smiles: string) =>
  apiClient.post<SinglePredictResponse>("/api/v1/predict/single", { smiles });

export const uploadBatch = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return apiClient.post<{ task_id: string }>("/api/v1/predict/batch", form);
};

export const getBatchStatus = (taskId: string) =>
  apiClient.get<BatchStatusResponse>(`/api/v1/predict/batch/status/${taskId}`);
```

### 2-4. 주요 컴포넌트 상세 설계

#### `single/RiskGauge.tsx`
- Recharts `RadialBarChart` 또는 CSS 기반 반원 게이지
- probability 0~1 → 0~100%
- **45% 미만:** 초록(`text-green-500`, `bg-green-100`) + "LOW RISK"
- **45% 이상:** 빨강(`text-red-500`, `bg-red-100`) + "HIGH RISK"
- 중앙에 확률 퍼센트 표시 (bold, 32px)
- Tailwind `transition-colors duration-500`으로 색상 변환 애니메이션

#### `ToxicReasonCard` (SinglePredictView.tsx 내장)
- SMARTS 기반 독성 원인 작용기 TOP 5 카드 (세로 목록)
- 각 카드: 순위 배지, 작용기명, 기여도 % (Progress Bar), 폴백 시 "(원자 레벨)" 뱃지
- 색상: 1순위 빨강(`border-red-200 bg-red-50`), 2~5위 주황(`border-orange-100 bg-orange-50`), 원자 폴백 회색
- 기여도 합계 반드시 100%: TOP-K raw score 통합 정규화 적용

#### `single/MoleculeViewer.tsx`
- `<div dangerouslySetInnerHTML={{ __html: svg }}>`로 SVG 직접 렌더링
- 원자 중요도 색상: 빨강(높음) ~ 흰색(낮음) 그라디언트
- 이미지 로딩 중 skeleton placeholder 표시

#### `single/PhysChemTable.tsx`
- 7개 항목 테이블: MW, LogP, HBD, HBA, TPSA, RotBonds, QED
- 각 항목 옆 Lipinski 기준 초과 시 ⚠️ 아이콘 표시
  - MW > 500, LogP > 5, HBD > 5, HBA > 10

#### `batch/DropZone.tsx`
- react-dropzone으로 `.csv` 파일만 허용
- 드래그 중: `border-blue-400 bg-blue-50` 스타일 변경
- 파일 선택 후: 파일명, 크기 표시 + "분석 시작" 버튼 활성화

#### `batch/ProgressBar.tsx`
- React Query `useQuery`로 `/batch/status/{taskId}` **3초 폴링**
  - `refetchInterval: 3000`
  - `enabled: !!taskId && status !== "done" && status !== "failed"`
- Tailwind animated progress: `w-[${pct}%] transition-all duration-500`
- 완료 시 폴링 자동 중단, 결과 테이블 표시

#### `batch/ResultTable.tsx`
- TanStack Table v8로 정렬/필터 구현
- 컬럼: SMILES, Probability(%), Risk Level, MW, LogP, 상태
- `ColumnFiltersState`로 Risk Level 필터 (HIGH/LOW/ALL)
- 행 클릭 → 해당 SMILES 단일 분석 탭으로 이동 (router state 전달)

#### `batch/DownloadButton.tsx`
- 백엔드에서 받은 `result_download_url` (S3 presigned URL)로 직접 다운로드
- `window.location.href = url` 방식

### 2-5. 페이지 라우팅
- React Router v6 사용 (`/` → SinglePage, `/batch` → BatchPage)
- 탭 내비게이션은 `TabNav.tsx`에서 링크 처리

### 2-6. `vite.config.ts`

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000"  // 개발 시 CORS 우회
    }
  }
});
```

---

## Phase 3: AWS 인프라 (Terraform) (2일)

### 3-1. 모듈 구조 및 리소스

#### `modules/vpc/`
```hcl
# 리소스: VPC, Public Subnet ×2, Private Subnet ×2
# IGW, NAT Gateway ×1 (비용 절감: 단일 AZ), Route Tables
# AZ: ap-northeast-2a, ap-northeast-2c
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
}
# Public: 10.0.1.0/24, 10.0.2.0/24 (ALB용)
# Private: 10.0.11.0/24, 10.0.12.0/24 (ECS Fargate용)
```

#### `modules/ecr/`
```hcl
resource "aws_ecr_repository" "backend" {
  name                 = "dgudili-backend"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}
# lifecycle policy: untagged 이미지 7일 후 삭제
```

#### `modules/alb/`
```hcl
# ALB: Public 서브넷에 위치
resource "aws_lb" "backend" {
  internal           = false
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [aws_security_group.alb.id]
}
# Target Group: /health 헬스체크, 포트 8000
# Listener: HTTP 80 (프로덕션: HTTPS 443 + ACM 인증서)
```

#### `modules/ecs/`
```hcl
resource "aws_ecs_cluster" "main" { name = "dgudili-cluster" }

resource "aws_ecs_task_definition" "backend" {
  family                   = "dgudili-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"   # 2 vCPU
  memory                   = "8192"   # 8GB (PyTorch + RDKit 로드)
  container_definitions    = templatefile("task_definition.json.tpl", {
    image       = var.ecr_image_uri
    region      = var.aws_region
    bucket_name = var.s3_bucket_name
  })
}

resource "aws_ecs_service" "backend" {
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = var.private_subnet_ids   # Private 서브넷
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "dgudili-backend"
    container_port   = 8000
  }
}

# Auto Scaling: CPU 70% → 스케일 아웃, 최소 1 / 최대 3
resource "aws_appautoscaling_policy" "cpu" {
  policy_type = "TargetTrackingScaling"
  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

**비용 추정 (서울 리전, 최소 1 task 24/7):**
| 항목 | 스펙 | 월 비용(USD) |
|---|---|---|
| ECS Fargate | 2vCPU, 8GB × 730h | ~$60 |
| ALB | 1 LCU avg | ~$22 |
| NAT Gateway | 1개 | ~$35 |
| S3 + CloudFront | 소규모 트래픽 | ~$5 |
| **합계** | | **~$122/월** |

#### `modules/s3_cloudfront/`
```hcl
resource "aws_s3_bucket" "frontend" {
  bucket = "dgudili-frontend-${random_id.suffix.hex}"
}
resource "aws_s3_bucket_public_access_block" "frontend" {
  block_public_acls = true  # OAI 통해서만 접근
}

resource "aws_cloudfront_distribution" "frontend" {
  origin {
    domain_name = aws_s3_bucket.frontend.bucket_regional_domain_name
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.oai.cloudfront_access_identity_path
    }
  }
  default_cache_behavior {
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
  }
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"  # SPA 라우팅 처리
  }
  price_class = "PriceClass_200"  # 한국 + 아시아 + 북미
}

# 배치 결과 저장 버킷 (private)
resource "aws_s3_bucket" "batch_results" {
  bucket = "dgudili-batch-results-${random_id.suffix.hex}"
}
resource "aws_s3_bucket_lifecycle_configuration" "batch_results" {
  rule {
    id     = "expire-results"
    status = "Enabled"
    expiration { days = 7 }  # 7일 후 자동 삭제
  }
}
```

### 3-2. `main.tf` (루트)
```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "dgudili-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "ap-northeast-2"
  }
}

module "vpc"           { source = "./modules/vpc" }
module "ecr"           { source = "./modules/ecr" }
module "alb"           { source = "./modules/alb" ... }
module "ecs"           { source = "./modules/ecs" ... }
module "s3_cloudfront" { source = "./modules/s3_cloudfront" }
```

### 3-3. IAM 역할
- **ECS Task Role:** `s3:PutObject`, `s3:GetObject` (배치 결과 버킷 전용)
- **ECS Execution Role:** ECR 풀, CloudWatch Logs 쓰기 (AWS 관리형 정책 사용)
- **GitHub Actions:** `ecr:*`, `ecs:UpdateService`, `s3:Sync`, `cloudfront:CreateInvalidation`

---

## Phase 4: CI/CD 파이프라인 (1일)

### `.github/workflows/deploy.yml`

```yaml
name: Deploy DGUDILI Web App

on:
  push:
    branches: [main]

env:
  AWS_REGION: ap-northeast-2
  ECR_REPOSITORY: dgudili-backend
  ECS_SERVICE: dgudili-backend-service
  ECS_CLUSTER: dgudili-cluster
  CONTAINER_NAME: dgudili-backend

jobs:
  # ────────────────────────────────────────────
  # JOB 1: Backend — Test → Build → Push → Deploy
  # ────────────────────────────────────────────
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: webapp/backend
    
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with: { python-version: "3.11" }

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v --tb=short
        env:
          MODEL_WEIGHTS_PATH: ./tests/fixtures/mock_weights.pt  # 테스트용 더미 가중치

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push Docker image to ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.ecr-login.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

      - name: Fill in the new image ID in ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: webapp/infra/task_definition.json
          container-name: ${{ env.CONTAINER_NAME }}
          image: ${{ steps.build-image.outputs.image }}

      - name: Deploy Amazon ECS service (Rolling Update)
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true

  # ────────────────────────────────────────────
  # JOB 2: Frontend — Build → S3 Sync → CF Invalidation
  # ────────────────────────────────────────────
  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: webapp/frontend
    
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js 20
        uses: actions/setup-node@v4
        with: { node-version: "20", cache: "npm", cache-dependency-path: webapp/frontend/package-lock.json }

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build
        env:
          VITE_API_BASE_URL: https://${{ secrets.ALB_DNS_NAME }}  # ALB 도메인

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Sync to S3
        run: aws s3 sync dist/ s3://${{ secrets.S3_FRONTEND_BUCKET }} --delete

      - name: Invalidate CloudFront cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.CLOUDFRONT_DISTRIBUTION_ID }} \
            --paths "/*"
```

**GitHub Secrets 목록:**
| Secret | 설명 |
|---|---|
| `AWS_ACCESS_KEY_ID` | GitHub Actions IAM 유저 키 |
| `AWS_SECRET_ACCESS_KEY` | GitHub Actions IAM 유저 시크릿 |
| `ALB_DNS_NAME` | ALB DNS (`xxx.elb.amazonaws.com`) |
| `S3_FRONTEND_BUCKET` | 프론트엔드 S3 버킷 이름 |
| `CLOUDFRONT_DISTRIBUTION_ID` | CloudFront 배포 ID |

---

## Phase 5: 테스트 작성 (½일)

### `tests/test_predict.py`
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

# 단일 예측 엔드포인트 테스트
def test_predict_single_valid_smiles():
    with patch("app.services.inference.predict_single") as mock:
        mock.return_value = {...}  # 더미 응답
        resp = client.post("/api/v1/predict/single", json={"smiles": "CCO"})
        assert resp.status_code == 200
        assert "probability" in resp.json()

def test_predict_single_invalid_smiles():
    resp = client.post("/api/v1/predict/single", json={"smiles": "NOT_A_SMILES"})
    assert resp.status_code == 422

# 배치 업로드 테스트
def test_batch_upload_csv():
    csv_content = b"smiles,name\nCCO,ethanol\nCC(=O)O,acetic acid\n"
    resp = client.post("/api/v1/predict/batch", files={"file": ("test.csv", csv_content, "text/csv")})
    assert resp.status_code == 200
    assert "task_id" in resp.json()

# 태스크 상태 조회 테스트
def test_batch_status_not_found():
    resp = client.get("/api/v1/predict/batch/status/nonexistent-id")
    assert resp.status_code == 404
```

---

## 구현 순서 (추천 실행 순서)

```
Week 1 (5일):
  Day 1: Phase 0 (세팅) + Phase 1-1~1-5 (BE 기반 구조)
  Day 2: Phase 1-6~1-10 (BE 추론 서비스 + 엔드포인트)
  Day 3: Phase 2-1~2-3 (FE 세팅 + API 클라이언트)
  Day 4: Phase 2-4 (FE 컴포넌트 — Single 탭)
  Day 5: Phase 2-4 (FE 컴포넌트 — Batch 탭)

Week 2 (3일):
  Day 6: Phase 3 (Terraform 인프라)
  Day 7: Phase 4 (CI/CD) + Phase 5 (테스트)
  Day 8: 통합 테스트, 버그 수정, 문서 정리
```

---

## 주요 기술적 결정사항 및 근거

| 결정 | 선택지 | 근거 |
|---|---|---|
| CPU vs GPU | CPU (Fargate) | GPU Fargate 미지원; 단일 SMILES 추론은 CPU로 충분 (<2초) |
| 배치 큐 | BackgroundTasks (인메모리) | 프로토타입 단계; 프로덕션 확장 시 Celery+SQS로 교체 |
| 상태 저장 | 인메모리 dict | 단일 컨테이너 가정; 멀티 태스크 시 ElastiCache Redis 전환 |
| 가중치 배포 | Docker 이미지 포함 | S3 다운로드 대비 cold start 시간 절감 (~30초 → ~5초) |
| 모델 로드 | `lifespan` 이벤트 | 요청마다 로드 방지; 싱글턴 패턴으로 메모리 효율 |
| FE 프레임워크 | Vite+React (not Next.js) | 정적 파일 S3 배포, SSR 불필요 |
| Tailwind | v3 (not v4) | v4 alpha 단계; 생태계 안정성 우선 |

---

## 개발 시작 전 체크리스트

- [ ] `DGUDILI_2026/outputs/pretrained_graph_encoder.pt` 파일 존재 확인 (Step2 완료 필요)
- [ ] AWS 계정 및 리전(ap-northeast-2) 설정 완료
- [ ] Terraform 상태 저장용 S3 버킷 수동 생성 (`dgudili-terraform-state`)
- [ ] GitHub Secrets 5개 사전 등록
- [ ] Node.js 20, Python 3.11, Terraform 1.9+ 로컬 설치 확인
- [ ] Docker Desktop 실행 확인 (백엔드 로컬 테스트용)
