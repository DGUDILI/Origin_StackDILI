import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


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


class SMILESDataset(Dataset):
    """
    SMILES + FP 데이터셋. labels=None이면 추론 전용 (label 반환 없음).
    tokenizer는 외부에서 주입 (전역 변수 의존 제거).
    cache_path를 지정하면 토크나이징 결과를 디스크에 캐싱.
    """

    def __init__(self, smiles_list, fp_array, tokenizer, max_length: int,
                 labels=None, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"  [Cache] 토큰 로딩 중: {cache_path}")
            enc = torch.load(cache_path, weights_only=False)
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
        else:
            print(f"  [Tokenize] {len(smiles_list)}개 토크나이징 중...")
            enc = tokenizer(
                list(smiles_list),
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            if cache_path:
                torch.save(
                    {"input_ids": self.input_ids, "attention_mask": self.attention_mask},
                    cache_path,
                )
                print(f"  [Cache] 토큰 저장 완료: {cache_path}")

        self.fp         = torch.tensor(fp_array, dtype=torch.float32)
        self.has_labels = labels is not None
        self.labels     = torch.tensor(labels, dtype=torch.float32) if self.has_labels else None

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, idx):
        if self.has_labels:
            return (self.input_ids[idx], self.attention_mask[idx],
                    self.fp[idx], self.labels[idx])
        return (self.input_ids[idx], self.attention_mask[idx], self.fp[idx])
