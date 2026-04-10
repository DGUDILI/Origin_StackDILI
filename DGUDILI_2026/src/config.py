import os

# ── 모델 하이퍼파라미터 ──────────────────────────────────────────────────────────
K          = 32       # encode() 출력 차원 (Stacking 입력, 16→32: 표현력 확장)
D_MODEL    = 64       # MHA 내부 차원 (실험 결과: d=64가 최적, d=128 이상은 FP proj 과적합)
NUM_HEADS  = 4        # MHA 헤드 수 (d_model % num_heads == 0 조건)
DROPOUT    = 0.3
MODEL_NAME = os.environ.get(
    "CHEMBERTA_PATH",
    r"C:\Users\samsung\.cache\huggingface\hub\models--DeepChem--ChemBERTa-77M-MLM\snapshots\ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
)

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

# ── 데이터 증강 ──────────────────────────────────────────────────────────────
N_AUG = 2   # Train subset 당 랜덤 SMILES 증강 개수 (0 = 비활성화)
             # N_AUG=2 → 원본 1 + 증강 2 = 샘플 최대 3배 (train만, val/test 불변)
             # epoch 소요 시간 참고 (CPU 환경): N_AUG=0 → ~3분, N_AUG=2 → ~7분, N_AUG=5 → ~20분

# ── GraphMACCSEncoder 전용 하이퍼파라미터 ────────────────────────────────────
MACCS_DIM      = 167    # RDKit MACCSkeys 벡터 길이 (bit 0 미사용, bits 1~166 유효)
MAX_ATOMS      = 100    # to_dense_batch 패딩 기준 (분자당 최대 원자 수)
GINE_LAYERS    = 2      # GINEConv layer 수
GINE_HIDDEN    = 64     # GINEConv hidden dim
ATOM_FEAT_DIM  = 43     # get_atom_features() 출력 차원 (graph_utils.py 기준)
BOND_FEAT_DIM  = 9      # get_bond_features() 출력 차원 (graph_utils.py 기준)

# ── 경로 ─────────────────────────────────────────────────────────────────────────
ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN       = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix          = "_clean" if _USE_CLEAN else ""
DATA_DIR         = os.path.join(ROOT, f"data{_suffix}")
OUT_DIR          = os.path.join(ROOT, f"outputs{_suffix}")
_STACKDILI_ROOT  = os.environ.get("STACKDILI_ROOT", os.path.dirname(ROOT))
DATA_PATH        = os.path.join(_STACKDILI_ROOT, "Data", f"Dataset{_suffix}.csv")
FEAT_PATH        = os.path.join(_STACKDILI_ROOT, "Code", f"Dataset_feature{_suffix}.csv")
