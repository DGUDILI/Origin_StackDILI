import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class CrossAttentionEncoder(nn.Module):
    """
    FP(지문)와 ChemBERTa 피처 각 k=16개를 Cross-Attention으로 융합.
    d_v=1 고정 → 출력: (B, k, 1) → squeeze → (B, k) Feature Space
    Q ← ChemBERTa (query), K/V ← FP (key/value)
    """

    def __init__(self, k: int = 16, d_k: int = 32):
        super().__init__()
        self.k = k
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        # 각 피처(스칼라)를 d_k 차원으로 투영
        self.W_Q = nn.Linear(1, d_k)   # ChemBERTa → Query
        self.W_K = nn.Linear(1, d_k)   # FP → Key
        self.W_V = nn.Linear(1, 1)     # FP → Value  (d_v=1 고정)

        # Stage 1 사전학습용 임시 분류 head
        self.head = nn.Linear(k, 1)

    def encode(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cham: (B, k) — 선택된 ChemBERTa 피처 (스케일링됨)
            x_fp:   (B, k) — 선택된 FP 피처 (스케일링됨)
        Returns:
            (B, k) — Feature Space (k=16)
        """
        # (B, k) → (B, k, 1): 각 피처를 1-dim 토큰으로 취급
        q_in = x_cham.unsqueeze(-1)   # (B, k, 1)
        k_in = x_fp.unsqueeze(-1)     # (B, k, 1)

        Q = self.W_Q(q_in)            # (B, k, d_k)
        K = self.W_K(k_in)            # (B, k, d_k)
        V = self.W_V(k_in)            # (B, k, 1)

        # Scaled dot-product attention
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, k, k)
        weights = torch.softmax(scores, dim=-1)                   # (B, k, k)
        attn    = torch.bmm(weights, V)                           # (B, k, 1)

        return attn.squeeze(-1)        # (B, k)  ← Feature Space

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        """Stage 1 사전학습용: encode → head → logit (B, 1)"""
        return self.head(self.encode(x_cham, x_fp))
