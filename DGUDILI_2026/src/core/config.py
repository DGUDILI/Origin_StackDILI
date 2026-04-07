"""
config.py — 실험 설정 (dataclass + YAML 로더)

사용법:
    from config import load_cfg
    cfg = load_cfg("config.yaml")   # YAML 파일에서 로드
    cfg = load_cfg()                # 기본값 사용

YAML 파일 예시 (config.yaml):
    data_dir: data_clean
    output_dir: outputs/exp_002
    seed: 0
    k: 64
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cfg:
    # ── 모델 ──────────────────────────────────────────────────────────────────
    k: int = 32                           # encode() 출력 차원
    d_model: int = 64                     # attention/SAGE hidden dim
    num_heads: int = 4                    # differential attention 헤드 수
    dropout: float = 0.3
    model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    maccs_dim: int = 167                  # RDKit MACCSkeys 벡터 길이
    max_atoms: int = 100                  # to_dense_batch 패딩 기준
    sage_layers: int = 2                  # GraphSAGE layer 수
    sage_hidden: int = 64                 # SAGEConv hidden dim
    atom_feat_dim: int = 43               # get_atom_features() 출력 차원

    # ── 학습 ──────────────────────────────────────────────────────────────────
    batch_size: int = 16
    max_length: int = 256                 # SMILES tokenizer max_length
    seed: int = 42
    epochs: int = 200
    patience: int = 30                    # early stopping (val_AUC)
    lr_chem: float = 1e-4                 # ChemBERTa 마지막 레이어 lr
    lr_other: float = 3e-4               # 그래프/MACCS/Fusion 컴포넌트 lr
    weight_decay: float = 1e-4
    sched_patience: int = 8
    sched_factor: float = 0.5
    sched_min_lr: float = 1e-5

    # ── 경로 (절대경로 또는 project root 기준 상대경로) ─────────────────────
    data_dir: str = "data"
    output_dir: str = "outputs"
    data_csv: str = "Data/Dataset.csv"         # Origin_StackDILI/ 기준
    feat_csv: str = "Code/Dataset_feature.csv" # Origin_StackDILI/ 기준

    # ── 런타임 (load_cfg에서 채워짐, YAML 미사용) ────────────────────────────
    _resolved: bool = field(default=False, repr=False, compare=False)


def load_cfg(yaml_path: Optional[str] = None) -> Cfg:
    """
    YAML 파일에서 설정 로드. yaml_path가 없으면 기본값 사용.
    경로(data_dir, output_dir, data_csv, feat_csv)는 project root 기준으로 절대경로 변환.
    """
    cfg = Cfg()

    if yaml_path and os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            # PyYAML 미설치 시 경고 후 기본값 사용
            import warnings
            warnings.warn("PyYAML not found. Using default config values.", stacklevel=2)
            data = {}

        for key, val in data.items():
            if hasattr(cfg, key) and not key.startswith("_"):
                setattr(cfg, key, val)

    # project root = DGUDILI_2026/ (src/core/ 의 2단계 위)
    core_dir = os.path.dirname(os.path.abspath(__file__))  # src/core/
    src_dir  = os.path.dirname(core_dir)                   # src/
    root     = os.path.dirname(src_dir)                    # DGUDILI_2026/
    stackdili_root = os.path.dirname(root)                 # Origin_StackDILI/

    if not os.path.isabs(cfg.data_dir):
        cfg.data_dir = os.path.join(root, cfg.data_dir)
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(root, cfg.output_dir)
    if not os.path.isabs(cfg.data_csv):
        cfg.data_csv = os.path.join(stackdili_root, cfg.data_csv)
    if not os.path.isabs(cfg.feat_csv):
        cfg.feat_csv = os.path.join(stackdili_root, cfg.feat_csv)

    cfg._resolved = True
    return cfg
