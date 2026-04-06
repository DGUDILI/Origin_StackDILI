import os

# ── 모델 하이퍼파라미터 ──────────────────────────────────────────────────────────
K          = 16       # encode() 출력 차원 (Stacking 입력)
D_MODEL    = 64       # MHA 내부 차원 (실험 결과: d=64가 최적, d=128 이상은 FP proj 과적합)
NUM_HEADS  = 4        # MHA 헤드 수 (d_model % num_heads == 0 조건)
DROPOUT    = 0.3
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

# ── 구버전 호환 (E2E_FTV6StyleEncoder) ──────────────────────────────────────────
D_K        = 32
D_V        = 1

# ── 학습 설정 ────────────────────────────────────────────────────────────────────
BATCH_SIZE     = 16
MAX_LENGTH     = 256
SEED           = 42
EPOCHS         = 200
PATIENCE       = 30
LR_CHEM        = 1e-4   # ChemBERTa 마지막 레이어
LR_OTHER       = 3e-4   # CrossAttn + Projection
WEIGHT_DECAY   = 1e-4
SCHED_PATIENCE = 8
SCHED_FACTOR   = 0.5
SCHED_MIN_LR   = 1e-5

# ── 경로 ─────────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
OUT_DIR   = os.path.join(ROOT, "outputs")
DATA_PATH = r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv"
FEAT_PATH = r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv"
