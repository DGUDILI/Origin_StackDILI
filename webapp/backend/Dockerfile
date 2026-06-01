# ──────────────────────────────────────────────────────────────────────────────
# DGUDILI Backend — Dockerfile
# 루트 Dockerfile과 동일한 Python 환경 기반
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 루트 Dockerfile과 동일한 torch CPU 설치
RUN pip install --no-cache-dir "torch>=2.6.0" --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    scikit-learn \
    transformers \
    rdkit \
    einops

# torch-geometric PyPI에서 직접 설치
# (torch-scatter/sparse 제외: pyg 서버에 aarch64 wheel 없음, torch-geometric 2.5+에서 필수 아님)
RUN pip install --no-cache-dir torch-geometric

# 웹 서버 전용 의존성
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    python-multipart==0.0.12 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2 \
    boto3 \
    python-dotenv

# ChemBERTa 이미지 레이어에 캐싱 (루트 Dockerfile과 동일)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM'); \
    AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')" || true

COPY app/ ./app/
COPY weights/ ./weights/

ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 8000

# workers=1 고정: 모델 싱글턴이 프로세스 간 공유 안 됨
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
