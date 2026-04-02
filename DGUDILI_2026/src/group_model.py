import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class GroupCrossAttentionDILI(nn.Module):
    """
    [Module 1] Group-wise Data Pipeline
      FP 425-dim을 화학적 의미 단위로 4개 그룹 분리 (고정 인덱스, 정보 손실 없음):
        G0 Constitution : dims [  0: 29]  (29-dim)
        G1 CalcCATS     : dims [ 29:179] (150-dim)
        G2 MACCS        : dims [179:346] (167-dim)
        G3 E-state      : dims [346:425]  (79-dim)

    [Module 2] Semantic Tokenization (Linear Projection)
      FP Tokens (K, V source): 각 그룹 → Linear(group_dim, d) → token (B, d)
                                4개 stack → FP_tokens (B, 4, d)
      LM Token (Q source):     ChemBERTa 768-dim → Linear(768, d) → (B, 1, d)

    [Module 3] Cross-Attention Fusion
      Q = W_Q(LM_token)   (B, 1, d)  ← ChemBERTa가 "어느 FP 그룹이 독성과 관련?"
      K = W_K(FP_tokens)  (B, 4, d)
      V = W_V(FP_tokens)  (B, 4, d)
      scores  = Q @ Kᵀ / √d   → (B, 1, 4)
      weights = softmax(scores) → (B, 1, 4)  ← Group Attention Score
      fused   = weights @ V    → (B, 1, d) → squeeze → (B, d)
      → LayerNorm + Dropout → Linear(d, 1)
    """

    # 고정 그룹 인덱스 (Feature.py 생성 순서 기준)
    GROUP_SLICES = [
        (0,   29),   # G0: Constitution  (29-dim)
        (29,  179),  # G1: CalcCATS     (150-dim)
        (179, 346),  # G2: MACCS        (167-dim)
        (346, 425),  # G3: E-state       (79-dim)
    ]
    GROUP_NAMES = ["Constitution", "CalcCATS", "MACCS", "E-state"]

    def __init__(self, d: int = 16, dropout: float = 0.3):
        super().__init__()
        self.d     = d
        self.scale = math.sqrt(d)

        # Module 2: FP 그룹별 투영 (K, V 소스)
        self.fp_projectors = nn.ModuleList([
            nn.Linear(end - start, d)
            for start, end in self.GROUP_SLICES
        ])

        # Module 2: ChemBERTa 투영 (Q 소스)
        self.lm_projector = nn.Linear(768, d)

        # Module 3: Cross-Attention 투영 행렬
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

        # Post-attention
        self.norm    = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

        # 분류 헤드
        self.classifier = nn.Linear(d, 1)

    def encode(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_fp : (B, 425) — 전체 FP (StandardScaler 적용)
            x_lm : (B, 768) — ChemBERTa CLS 임베딩 (StandardScaler 적용)
        Returns:
            fused: (B, d)   — 융합된 d=16 차원 표현
        """
        # [Module 2] FP 그룹 → 토큰 생성
        fp_tokens = []
        for proj, (start, end) in zip(self.fp_projectors, self.GROUP_SLICES):
            group = x_fp[:, start:end]  # (B, group_dim)
            fp_tokens.append(proj(group))  # (B, d)
        FP_tokens = torch.stack(fp_tokens, dim=1)  # (B, 4, d)

        # [Module 2] ChemBERTa → LM 토큰 생성
        lm_token = self.lm_projector(x_lm).unsqueeze(1)  # (B, 1, d)

        # [Module 3] Cross-Attention: Q(LM) ← K,V(FP)
        Q = self.W_Q(lm_token)    # (B, 1, d)
        K = self.W_K(FP_tokens)   # (B, 4, d)
        V = self.W_V(FP_tokens)   # (B, 4, d)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, 4)
        weights = torch.softmax(scores, dim=-1)                   # (B, 1, 4)
        fused   = torch.bmm(weights, V).squeeze(1)               # (B, d)

        fused = self.norm(fused)
        fused = self.dropout(fused)
        return fused

    def forward(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 1) raw logit"""
        return self.classifier(self.encode(x_fp, x_lm))

    @torch.no_grad()
    def get_attention_weights(
        self, x_fp: torch.Tensor, x_lm: torch.Tensor
    ) -> torch.Tensor:
        """
        그룹별 Attention Weight 반환 (해석 가능성용)
        Returns: (B, 4) — 각 FP 그룹의 Attention Score
        """
        self.eval()
        fp_tokens = []
        for proj, (start, end) in zip(self.fp_projectors, self.GROUP_SLICES):
            group = x_fp[:, start:end]
            fp_tokens.append(proj(group))
        FP_tokens = torch.stack(fp_tokens, dim=1)  # (B, 4, d)

        lm_token = self.lm_projector(x_lm).unsqueeze(1)  # (B, 1, d)
        Q = self.W_Q(lm_token)
        K = self.W_K(FP_tokens)
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, 4)
        weights = torch.softmax(scores, dim=-1).squeeze(1)       # (B, 4)
        return weights
