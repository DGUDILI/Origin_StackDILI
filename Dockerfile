FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install torch (CPU) first — requires >=2.6 for transformers safety check
RUN pip install --no-cache-dir "torch>=2.6.0" --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas \
    scikit-learn==1.7.1 \
    xgboost \
    transformers \
    matplotlib \
    seaborn \
    rdkit

# Pre-download ChemBERTa model into the image layer (optional but speeds up first run)
# Comment this out if you want a smaller image
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1'); \
    AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')" || true

# Default working directory inside the project
WORKDIR /workspace/DGUDILI_2026

ENV STACKDILI_ROOT=/workspace
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD ["bash"]
