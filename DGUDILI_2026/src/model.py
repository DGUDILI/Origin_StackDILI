import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class CrossAttentionEncoder(nn.Module):
    """
    Experiment 4: MLP projection version

    - ChemBERTa input: (B, 768)
    - FP input:        (B, 16)

    ChemBERTa:
        LayerNorm -> Linear(768,128) -> GELU -> Dropout -> Linear(128,16)

    Then:
        projected ChemBERTa (16) and FP (16) are treated as scalar tokens
        and fused by cross-attention
    """

    def __init__(
        self,
        chem_in_dim: int = 768,
        chem_hidden_dim: int = 128,
        k: int = 16,
        d_k: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chem_in_dim = chem_in_dim
        self.chem_hidden_dim = chem_hidden_dim
        self.k = k
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        # MLP projection: 768 -> 128 -> 16
        self.chem_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, chem_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chem_hidden_dim, k),
        )

        # scalar-token attention
        self.W_Q = nn.Linear(1, d_k)   # projected ChemBERTa -> Query
        self.W_K = nn.Linear(1, d_k)   # FP -> Key
        self.W_V = nn.Linear(1, 1)     # FP -> Value

        # Stage 1 head
        self.head = nn.Linear(k, 1)

    def encode(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        """
        x_cham: (B, 768)
        x_fp:   (B, 16)
        returns: (B, 16)
        """
        x_cham_proj = self.chem_proj(x_cham)   # (B, 16)

        q_in = x_cham_proj.unsqueeze(-1)       # (B, 16, 1)
        k_in = x_fp.unsqueeze(-1)              # (B, 16, 1)

        Q = self.W_Q(q_in)                     # (B, 16, d_k)
        K = self.W_K(k_in)                     # (B, 16, d_k)
        V = self.W_V(k_in)                     # (B, 16, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        attn = torch.bmm(weights, V)

        return attn.squeeze(-1)                # (B, 16)

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_cham, x_fp))