import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class E2E_FTV6StyleEncoder(nn.Module):
    """
    End-to-End FTV6 Encoder: ChemBERTa-77M-MLM 내장.

    Pipeline:
      SMILES → ChemBERTa (마지막 레이어만 unfreeze) → CLS (B, 384)
      CLS  → chem_proj: LN→Linear→GELU→Drop→Linear → (B, k)  [Query]
      FP   → fp_proj:   Linear(fp_in_dim, k) + LN          → (B, k)  [Key/Value]
      Cross-Attention: Q·K^T/√d_k → softmax → ·V            → (B, k, d_v)
      flatten → (B, k*d_v)

    Stage 1: head Linear(k*d_v, 1) + BCEWithLogitsLoss
    Stage 2: encode() → (B, k*d_v) fused feature → Stacking OOF
    """

    def __init__(
        self,
        fp_in_dim: int = 425,
        k: int = 16,
        d_k: int = 32,
        d_v: int = 1,
        dropout: float = 0.3,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    ):
        super().__init__()
        from transformers import AutoModel

        self.chemberta = AutoModel.from_pretrained(model_name)
        chem_in_dim: int = self.chemberta.config.hidden_size  # 384

        # 전체 frozen 후 마지막 레이어만 unfreeze
        for p in self.chemberta.parameters():
            p.requires_grad = False
        n_layers = len(self.chemberta.encoder.layer)
        for p in self.chemberta.encoder.layer[n_layers - 1].parameters():
            p.requires_grad = True

        self.k = k
        self.d_v = d_v
        self.scale = math.sqrt(d_k)

        self.chem_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, k),
        )

        self.fp_proj = nn.Sequential(
            nn.Linear(fp_in_dim, k),
            nn.LayerNorm(k),
        )

        self.W_Q = nn.Linear(1, d_k)
        self.W_K = nn.Linear(1, d_k)
        self.W_V = nn.Linear(1, d_v)

        self.head = nn.Linear(k * d_v, 1)

    def _cls_embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, 384)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_fp: torch.Tensor,
    ) -> torch.Tensor:
        """returns (B, k*d_v) fused feature vector"""
        x_chem = self._cls_embed(input_ids, attention_mask)

        q  = self.chem_proj(x_chem)      # (B, k)
        kv = self.fp_proj(x_fp)          # (B, k)

        Q = self.W_Q(q.unsqueeze(-1))    # (B, k, d_k)
        K = self.W_K(kv.unsqueeze(-1))   # (B, k, d_k)
        V = self.W_V(kv.unsqueeze(-1))   # (B, k, d_v)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, k, k)
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.bmm(weights, V)  # (B, k, d_v)

        return attn.reshape(attn.size(0), -1)   # (B, k*d_v)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_fp: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask, x_fp))
