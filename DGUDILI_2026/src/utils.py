import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(data_path: str, feat_path: str):
    """
    CSV 두 개를 로드하여 raw 배열로 반환.

    Returns:
        smiles_all  : (N,) str array
        X_fp_all    : (N, fp_dim) float32 array
        y_all       : (N,) float32 array
        ref_all     : (N,) str array  ("DILIrank" = test)
        feat_cols   : list[str]
    """
    import pandas as pd
    df_meta = pd.read_csv(data_path)
    df_feat = pd.read_csv(feat_path)
    assert list(df_meta["SMILES"]) == list(df_feat["SMILES"]), \
        "SMILES order mismatch between Dataset.csv and Dataset_feature.csv"

    feat_cols  = [c for c in df_feat.columns if c not in ["SMILES", "Label", "ref"]]
    X_fp_all   = df_feat[feat_cols].values.astype(np.float32)
    y_all      = df_feat["Label"].values.astype(np.float32)
    ref_all    = df_feat["ref"].values
    smiles_all = df_meta["SMILES"].values

    return smiles_all, X_fp_all, y_all, ref_all, feat_cols


