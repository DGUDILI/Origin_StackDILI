"""
dataset.py — 공통 Dataset / DataLoader 유틸리티

MolGraphDataset  : train_graphs.pt / test_graphs.pt 를 로드하고 SMILES를 토크나이징.
MolBatch         : collate_fn 반환 타입 (dataclass).
collate_fn       : 모든 학습 스테이지에서 공유하는 단일 collate 함수.

사용법 (train.py / Step4_xai.py 등에서):
    from dataset import MolGraphDataset, collate_fn, MolBatch
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch


@dataclass
class MolBatch:
    """collate_fn 반환 타입."""
    input_ids: torch.Tensor        # (B, MAX_LENGTH)
    attn_mask: torch.Tensor        # (B, MAX_LENGTH)
    maccs: torch.Tensor            # (B, 167)
    graph: PyGBatch                # PyG heterogeneous batch
    labels: Optional[torch.Tensor] = None  # (B,) — pretrain 시에만 존재


class MolGraphDataset(Dataset):
    """
    {split}_graphs.pt 로드 + SMILES 토크나이징.
    각 항목: {"pyg": Data, "maccs": Tensor(167,), "label": Tensor, "smiles": str}

    __getitem__ 반환 dict:
        input_ids, attn_mask, maccs, pyg_data, [label]
    """

    def __init__(self, data_list: list, tokenizer, max_length: int,
                 cache_path: Optional[str] = None):
        self.data_list = data_list

        smiles_list = [d["smiles"] for d in data_list]

        if cache_path and os.path.exists(cache_path):
            print(f"  [Cache] 토큰 로딩: {cache_path}")
            enc = torch.load(cache_path, weights_only=False)
        else:
            print(f"  [Tokenize] {len(smiles_list)}개 토크나이징 중...")
            enc = tokenizer(
                smiles_list,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if cache_path:
                torch.save(
                    {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                    cache_path,
                )
                print(f"  [Cache] 저장: {cache_path}")

        self.input_ids = enc["input_ids"]        # (N, MAX_LENGTH)
        self.attn_mask = enc["attention_mask"]   # (N, MAX_LENGTH)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        d = self.data_list[idx]
        item = {
            "input_ids": self.input_ids[idx],
            "attn_mask": self.attn_mask[idx],
            "maccs":     d["maccs"],   # (167,)
            "pyg_data":  d["pyg"],
        }
        if "label" in d:
            item["label"] = d["label"]
        return item


def collate_fn(batch: list[dict]) -> MolBatch:
    """모든 스테이지에서 공유하는 단일 collate 함수."""
    result = MolBatch(
        input_ids=torch.stack([b["input_ids"] for b in batch]),
        attn_mask=torch.stack([b["attn_mask"] for b in batch]),
        maccs=torch.stack([b["maccs"] for b in batch]),
        graph=PyGBatch.from_data_list([b["pyg_data"] for b in batch]),
    )
    if "label" in batch[0]:
        result.labels = torch.stack([b["label"] for b in batch])
    return result


def make_loader(data_list: list, tokenizer, max_length: int, batch_size: int,
                shuffle: bool = False, cache_path: Optional[str] = None) -> DataLoader:
    """MolGraphDataset + DataLoader 한번에 생성하는 헬퍼."""
    ds = MolGraphDataset(data_list, tokenizer, max_length, cache_path=cache_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_fn)
