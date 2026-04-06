FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CPU 먼저 설치 (transformers safety check >= 2.6 필요)
RUN pip install --no-cache-dir "torch>=2.6.0" --extra-index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치 (버전 고정으로 재현성 보장)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    scikit-learn==1.7.1 \
    xgboost \
    transformers \
    matplotlib \
    seaborn \
    rdkit

# ChemBERTa 모델 이미지 레이어에 캐싱 (첫 실행 속도 개선)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM'); \
    AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')" || true

WORKDIR /workspace/DGUDILI_2026

ENV STACKDILI_ROOT=/workspace
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD ["bash"]
