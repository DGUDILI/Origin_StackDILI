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

        # Positional embedding: Q/K 각각 분리하여 비대칭 attention 패턴 학습 가능
        # rank-1 collapse 방지 (W_Q/W_K가 스칼라 투영이므로 위치 구별 불가 문제 해결)
        self.pos_emb_q = nn.Parameter(torch.empty(1, k, d_k))
        self.pos_emb_k = nn.Parameter(torch.empty(1, k, d_k))
        nn.init.normal_(self.pos_emb_q, std=0.02)
        nn.init.normal_(self.pos_emb_k, std=0.02)

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

        # 위치 정보 주입: Q/K 독립 pos_emb → 비대칭 attention 가능
        Q = Q + self.pos_emb_q           # (B, k, d_k)
        K = K + self.pos_emb_k           # (B, k, d_k)

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


class E2E_MHAResidualEncoder(nn.Module):
    """
    MHA Residual Encoder: ChemBERTa CLS + FP Latent Tokens + Residual Cross-Attention.

    Pipeline:
      SMILES → ChemBERTa (마지막 레이어만 unfreeze) → CLS (B, 384)
      CLS  → q_proj: LN → Linear(384, d_model)      → (B, 1, d_model)  [Query]
      FP   → fp_proj: Linear(fp_in, k*d_model)
           → reshape (B, k, d_model) → LayerNorm     → (B, k, d_model)  [Key/Value]
      MHA(d_model, num_heads): Q × K/V               → Attn_Out (B, 1, d_model)
      Residual: fused = Q + Attn_Out → squeeze       → (B, d_model)
      MLP: LN → Linear(d_model, k) → GELU → Dropout → (B, k)

    Stage 1: head Linear(k, 1) + BCEWithLogitsLoss
    Stage 2: encode() → (B, k) fused feature → Stacking OOF
    """

    def __init__(
        self,
        fp_in_dim: int = 425,
        k: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    ):
        super().__init__()
        from transformers import AutoModel

        self.chemberta = AutoModel.from_pretrained(model_name)
        chem_in_dim: int = self.chemberta.config.hidden_size  # 384

        for p in self.chemberta.parameters():
            p.requires_grad = False
        n_layers = len(self.chemberta.encoder.layer)
        # 마지막 2개 레이어 unfreeze: 더 풍부한 화학 표현 학습
        for p in self.chemberta.encoder.layer[n_layers - 2].parameters():
            p.requires_grad = True
        for p in self.chemberta.encoder.layer[n_layers - 1].parameters():
            p.requires_grad = True

        self.k = k
        self.d_model = d_model

        # ChemBERTa CLS → d_model (Query)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, d_model),
        )

        # FP → k개의 d_model-dim 잠재 토큰 (Key/Value)
        self.fp_proj = nn.Linear(fp_in_dim, k * d_model)
        self.fp_drop = nn.Dropout(0.1)
        self.fp_ln   = nn.LayerNorm(d_model)

        # Multi-Head Cross-Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )

        # MLP bottleneck: d_model → k
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, k),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(k, 1)

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
        """returns (B, k) fused feature vector"""
        x_chem = self._cls_embed(input_ids, attention_mask)   # (B, 384)

        # Query: (B, 1, d_model)
        q = self.q_proj(x_chem).unsqueeze(1)

        # Key/Value: (B, k, d_model)
        kv = self.fp_drop(self.fp_proj(x_fp))                 # (B, k*d_model)
        kv = self.fp_ln(kv.view(-1, self.k, self.d_model))    # (B, k, d_model)

        # Cross-Attention + Residual
        attn_out, _ = self.mha(q, kv, kv)                     # (B, 1, d_model)
        fused = (q + attn_out).squeeze(1)                     # (B, d_model)

        return self.mlp(fused)                                 # (B, k)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_fp: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask, x_fp))
