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
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN       = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix          = "_clean" if _USE_CLEAN else ""
DATA_DIR         = os.path.join(ROOT, f"data{_suffix}")
OUT_DIR          = os.path.join(ROOT, f"outputs{_suffix}")
_STACKDILI_ROOT  = os.environ.get("STACKDILI_ROOT", os.path.dirname(ROOT))
DATA_PATH        = os.path.join(_STACKDILI_ROOT, "Data", f"Dataset{_suffix}.csv")
FEAT_PATH        = os.path.join(_STACKDILI_ROOT, "Code", f"Dataset_feature{_suffix}.csv")
