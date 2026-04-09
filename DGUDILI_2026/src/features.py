"""
features.py — encoder feature 추출 공유 모듈

Step3_stacking.py 와 Step_CV.py 에서 공통으로 사용하는
collate_fn / extract_features 를 한 곳에서 관리.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as PyGBatch
from transformers import PreTrainedTokenizerBase

from config import BATCH_SIZE, MAX_LENGTH


def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attn_mask": torch.stack([b["attn_mask"] for b in batch]),
        "maccs":     torch.stack([b["maccs"]     for b in batch]),
        "graph":     PyGBatch.from_data_list([b["pyg_data"] for b in batch]),
    }


def extract_features(
    model,
    data_list: list,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> np.ndarray:
    """
    GraphMACCSEncoder.encode() 로 (N, k) feature 행렬 반환.
    model 은 eval 모드로 진입된 상태여야 함.
    """

    enc = tokenizer(
        [d["smiles"] for d in data_list],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class _DS(Dataset):
        def __init__(self, data, enc):
            self.data = data
            self.ids  = enc["input_ids"]
            self.mask = enc["attention_mask"]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return {
                "input_ids": self.ids[i],
                "attn_mask": self.mask[i],
                "maccs":     self.data[i]["maccs"],
                "pyg_data":  self.data[i]["pyg"],
            }

    dl = DataLoader(
        _DS(data_list, enc),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    feats = []
    with torch.no_grad():
        for batch in dl:
            feats.append(
                model.encode(
                    batch["input_ids"].to(device),
                    batch["attn_mask"].to(device),
                    batch["maccs"].to(device),
                    batch["graph"].to(device),
                ).cpu().numpy()
            )
    return np.concatenate(feats, axis=0)
