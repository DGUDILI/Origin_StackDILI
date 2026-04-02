import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class GroupCrossAttentionModB(nn.Module):
    """
    Mode B Cross-Attention DILI Predictor

    FP 그룹 재분할 (PC1-6 독립 토큰):
      Const+CalcCATS : dims [0:23] + [29:179]  (173-dim, 비연속)
      PC1-6          : dims [23:29]              (  6-dim)
      MACCS          : dims [179:346]            (167-dim)
      E-state        : dims [346:425]            ( 79-dim)

    FP branch  : Linear(in, 64) + LayerNorm(64) + ReLU + Dropout
                 4 tokens → FP_emb (B, 4, 64)   [K, V]

    CB branch  : Linear(768, 64) + LayerNorm(64) + ReLU + Dropout
                 → CB_emb (B, 1, 64)             [Q]

    Attention  : Q=CB, K=V=FP
                 scores = QKᵀ / √64  →  (B, 1, 4)
                 output             →  (B, 1, 64)
                 Residual+LayerNorm →  (B, 1, 64)
                 squeeze + Linear(64, 16) → (B, 16)

    최종 출력  : Linear(16, 1) → logit
    """

    GROUP_NAMES = ["Const+CalcCATS", "PC1-6", "MACCS", "E-state"]

    def _proj_block(self, in_dim: int, d: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def __init__(self, d_model: int = 64, d_out: int = 16, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.scale   = math.sqrt(d_model)

        # FP branch projectors
        self.proj_const  = self._proj_block(173, d_model, dropout)
        self.proj_pc     = self._proj_block(6,   d_model, dropout)
        self.proj_maccs  = self._proj_block(167, d_model, dropout)
        self.proj_estate = self._proj_block(79,  d_model, dropout)

        # ChemBERTa branch projector
        self.proj_cb = self._proj_block(768, d_model, dropout)

        # Cross-attention QKV 투영 (bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Post-attention Residual + LayerNorm
        self.attn_norm = nn.LayerNorm(d_model)

        # 64 → 16 압축
        self.compress = nn.Linear(d_model, d_out)

        # 분류 헤드
        self.classifier = nn.Linear(d_out, 1)

    def _split_fp(self, x_fp: torch.Tensor):
        """FP 425-dim을 4개 그룹으로 분리"""
        x_const  = torch.cat([x_fp[:, 0:23], x_fp[:, 29:179]], dim=1)  # (B, 173)
        x_pc     = x_fp[:, 23:29]    # (B,   6)
        x_maccs  = x_fp[:, 179:346]  # (B, 167)
        x_estate = x_fp[:, 346:425]  # (B,  79)
        return x_const, x_pc, x_maccs, x_estate

    def encode(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_fp : (B, 425) — 전체 FP (StandardScaler 적용)
            x_lm : (B, 768) — ChemBERTa CLS 임베딩
        Returns:
            (B, 16) — 압축된 16-dim 표현
        """
        # FP branch: 4 tokens → (B, 4, 64)
        x_const, x_pc, x_maccs, x_estate = self._split_fp(x_fp)
        FP_emb = torch.stack([
            self.proj_const(x_const),    # (B, 64)
            self.proj_pc(x_pc),          # (B, 64)
            self.proj_maccs(x_maccs),    # (B, 64)
            self.proj_estate(x_estate),  # (B, 64)
        ], dim=1)                        # (B, 4, 64)

        # ChemBERTa branch: Q token → (B, 1, 64)
        Q_tok = self.proj_cb(x_lm).unsqueeze(1)  # (B, 1, 64)

        # Cross-attention: Q=CB, K=V=FP
        Q = self.W_Q(Q_tok)              # (B, 1, 64)
        K = self.W_K(FP_emb)             # (B, 4, 64)
        V = self.W_V(FP_emb)             # (B, 4, 64)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, 4)
        weights = torch.softmax(scores, dim=-1)                   # (B, 1, 4)
        attn    = torch.bmm(weights, V)                           # (B, 1, 64)

        # Residual (CB_emb) + LayerNorm
        fused = self.attn_norm(attn + Q_tok)  # (B, 1, 64)
        fused = fused.squeeze(1)              # (B, 64)

        # 64 → 16 압축
        return self.compress(fused)           # (B, 16)

    def forward(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 1) raw logit"""
        return self.classifier(self.encode(x_fp, x_lm))

    @torch.no_grad()
    def get_attention_weights(
        self, x_fp: torch.Tensor, x_lm: torch.Tensor
    ) -> torch.Tensor:
        """
        Group Attention Weight (해석 가능성용)
        Returns: (B, 4) — 각 FP 그룹의 attention score
        """
        self.eval()
        x_const, x_pc, x_maccs, x_estate = self._split_fp(x_fp)
        FP_emb = torch.stack([
            self.proj_const(x_const),
            self.proj_pc(x_pc),
            self.proj_maccs(x_maccs),
            self.proj_estate(x_estate),
        ], dim=1)
        Q_tok = self.proj_cb(x_lm).unsqueeze(1)
        Q     = self.W_Q(Q_tok)
        K     = self.W_K(FP_emb)
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        return torch.softmax(scores, dim=-1).squeeze(1)  # (B, 4)


class GroupCrossAttentionModB_32(nn.Module):
    """
    Mode B-32 Cross-Attention DILI Predictor

    ModeB와 동일한 아키텍처, 변경 사항:
      - d_out : 16 → 32  (더 풍부한 분류 경계)
      - dropout : 0.20 → 0.30  (과적합 억제 강화)
      - dropout_pre : Dropout(0.15) — compress 이후 classifier 이전에 추가

    FP branch  : Linear(in, 64) + LayerNorm(64) + ReLU + Dropout(0.30)
                 4 tokens → FP_emb (B, 4, 64)   [K, V]

    CB branch  : Linear(768, 64) + LayerNorm(64) + ReLU + Dropout(0.30)
                 → CB_emb (B, 1, 64)             [Q]

    Attention  : Q=CB, K=V=FP
                 scores = QKt / sqrt(64)  ->  (B, 1, 4)
                 output                   ->  (B, 1, 64)
                 Residual+LayerNorm       ->  (B, 1, 64)
                 squeeze + Linear(64, 32) ->  (B, 32)

    Pre-classifier : Dropout(0.15)

    최종 출력  : Linear(32, 1) -> logit
    """

    GROUP_NAMES = ["Const+CalcCATS", "PC1-6", "MACCS", "E-state"]

    def _proj_block(self, in_dim: int, d: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def __init__(self, d_model: int = 64, d_out: int = 32,
                 dropout: float = 0.30, dropout_pre: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.scale   = math.sqrt(d_model)

        # FP branch projectors
        self.proj_const  = self._proj_block(173, d_model, dropout)
        self.proj_pc     = self._proj_block(6,   d_model, dropout)
        self.proj_maccs  = self._proj_block(167, d_model, dropout)
        self.proj_estate = self._proj_block(79,  d_model, dropout)

        # ChemBERTa branch projector
        self.proj_cb = self._proj_block(768, d_model, dropout)

        # Cross-attention QKV
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Post-attention Residual + LayerNorm
        self.attn_norm = nn.LayerNorm(d_model)

        # 64 -> 32 압축
        self.compress = nn.Linear(d_model, d_out)

        # Pre-classifier dropout
        self.dropout_pre = nn.Dropout(dropout_pre)

        # 분류 헤드
        self.classifier = nn.Linear(d_out, 1)

    def _split_fp(self, x_fp: torch.Tensor):
        x_const  = torch.cat([x_fp[:, 0:23], x_fp[:, 29:179]], dim=1)  # (B, 173)
        x_pc     = x_fp[:, 23:29]    # (B,   6)
        x_maccs  = x_fp[:, 179:346]  # (B, 167)
        x_estate = x_fp[:, 346:425]  # (B,  79)
        return x_const, x_pc, x_maccs, x_estate

    def encode(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, 32)
        """
        x_const, x_pc, x_maccs, x_estate = self._split_fp(x_fp)
        FP_emb = torch.stack([
            self.proj_const(x_const),
            self.proj_pc(x_pc),
            self.proj_maccs(x_maccs),
            self.proj_estate(x_estate),
        ], dim=1)  # (B, 4, 64)

        Q_tok = self.proj_cb(x_lm).unsqueeze(1)  # (B, 1, 64)

        Q = self.W_Q(Q_tok)
        K = self.W_K(FP_emb)
        V = self.W_V(FP_emb)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, 4)
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.bmm(weights, V)                          # (B, 1, 64)

        fused = self.attn_norm(attn + Q_tok).squeeze(1)          # (B, 64)
        return self.compress(fused)                               # (B, 32)

    def forward(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 1) raw logit"""
        return self.classifier(self.dropout_pre(self.encode(x_fp, x_lm)))

    @torch.no_grad()
    def get_attention_weights(
        self, x_fp: torch.Tensor, x_lm: torch.Tensor
    ) -> torch.Tensor:
        """Returns: (B, 4)"""
        self.eval()
        x_const, x_pc, x_maccs, x_estate = self._split_fp(x_fp)
        FP_emb = torch.stack([
            self.proj_const(x_const),
            self.proj_pc(x_pc),
            self.proj_maccs(x_maccs),
            self.proj_estate(x_estate),
        ], dim=1)
        Q_tok = self.proj_cb(x_lm).unsqueeze(1)
        Q     = self.W_Q(Q_tok)
        K     = self.W_K(FP_emb)
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        return torch.softmax(scores, dim=-1).squeeze(1)  # (B, 4)
