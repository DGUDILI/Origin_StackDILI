import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class CrossAttentionEncoder(nn.Module):
    """
    Experiment 12: Experiment 8 structure + ChemBERTa-2 backbone

    ChemBERTa input: (B, 384)
    FP input:        (B, 16)

    Projection:
        LayerNorm -> Linear(384,128) -> GELU -> Dropout(0.3) -> Linear(128,16)
    """

    def __init__(
        self,
        chem_in_dim: int = 384,
        chem_hidden_dim: int = 128,
        k: int = 16,
        d_k: int = 32,
        d_v: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.chem_in_dim = chem_in_dim
        self.chem_hidden_dim = chem_hidden_dim
        self.k = k
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(d_k)

        self.chem_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, chem_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chem_hidden_dim, k),
        )

        self.W_Q = nn.Linear(1, d_k)
        self.W_K = nn.Linear(1, d_k)
        self.W_V = nn.Linear(1, d_v)
        self.W_O = nn.Linear(d_v, 1)

        self.head = nn.Linear(k, 1)

    def encode(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        x_cham_proj = self.chem_proj(x_cham)

        q_in = x_cham_proj.unsqueeze(-1)
        k_in = x_fp.unsqueeze(-1)

        Q = self.W_Q(q_in)
        K = self.W_K(k_in)
        V = self.W_V(k_in)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        attn = torch.bmm(weights, V)   # (B, k, d_v)
        attn = self.W_O(attn)           # (B, k, 1)

        return attn.squeeze(-1)         # (B, k)

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_cham, x_fp))