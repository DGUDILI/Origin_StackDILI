import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


class GroupCrossAttentionDILIv2(nn.Module):
    """
    GroupCrossAttentionDILI v2 개선 사항:
      1. Multi-head Cross-Attention (h=4 heads, d_head=4, scale=sqrt(4))
      2. Residual Connection + FFN (Transformer 블록 패턴)
      3. Dropout 3곳 적용 (attention weights / FFN 내부 / pre-classifier)

    데이터 흐름 (v1과 동일하게 유지):
      FP 4 그룹 (K, V) <- ChemBERTa Query (Q) -> 16-dim fused

    [Module 1] 고정 인덱스 그룹 분리 (정보 손실 없음):
      G0 Constitution : dims [  0: 29]  (29-dim)
      G1 CalcCATS     : dims [ 29:179] (150-dim)
      G2 MACCS        : dims [179:346] (167-dim)
      G3 E-state      : dims [346:425]  (79-dim)

    [Module 2] Semantic Tokenization:
      FP Tokens (K, V): 4 groups -> Linear(group_dim, d) -> (B, 4, d)
      LM Token  (Q)  : ChemBERTa  -> Linear(768, d)      -> (B, 1, d)

    [Module 3] Multi-head Cross-Attention + Residual + FFN:
      Q (B,1,d) -> h heads: each (B,h,1,d_head)
      K,V (B,4,d)-> h heads: each (B,h,4,d_head)
      scores = Q @ K^T / sqrt(d_head) -> (B,h,1,4)
      fused  = W_O(concat heads) + lm_raw -> norm1 -> FFN -> norm2 -> (B,d)
    """

    GROUP_SLICES = [
        (0,   29),   # G0: Constitution  (29-dim)
        (29,  179),  # G1: CalcCATS     (150-dim)
        (179, 346),  # G2: MACCS        (167-dim)
        (346, 425),  # G3: E-state       (79-dim)
    ]
    GROUP_NAMES = ["Constitution", "CalcCATS", "MACCS", "E-state"]

    def __init__(self, d: int = 16, h: int = 4, dropout: float = 0.35):
        super().__init__()
        assert d % h == 0, f"d={d} must be divisible by h={h}"
        self.d      = d
        self.h      = h
        self.d_head = d // h        # 4
        self.scale  = math.sqrt(self.d_head)  # sqrt(4)=2.0 (per-head scale)

        # [Module 2] FP 그룹 투영 (v1과 동일)
        self.fp_projectors = nn.ModuleList([
            nn.Linear(end - start, d)
            for start, end in self.GROUP_SLICES
        ])

        # [Module 2] ChemBERTa 투영 (v1과 동일)
        self.lm_projector = nn.Linear(768, d)

        # [Module 3] Cross-Attention QKV 투영 (v1과 동일, bias=False)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

        # [신규] 멀티헤드 출력 투영
        self.W_O = nn.Linear(d, d)

        # [신규] Residual + FFN
        self.norm1 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * 2),    # 16 -> 32
            nn.GELU(),
            nn.Dropout(dropout),    # FFN 내부 dropout (별도 인스턴스)
            nn.Linear(d * 2, d),    # 32 -> 16
        )
        self.norm2 = nn.LayerNorm(d)

        # 공유 dropout (attention weights + pre-classifier)
        self.dropout = nn.Dropout(dropout)

        # 분류 헤드 (v1과 동일)
        self.classifier = nn.Linear(d, 1)

    def _multihead_attn(
        self,
        lm_token: torch.Tensor,   # (B, 1, d)
        FP_tokens: torch.Tensor,  # (B, 4, d)
    ):
        """
        Multi-head cross-attention.
        Returns:
            attn_out    : (B, d)    — W_O 투영 후 결과
            weights_mean: (B, 4)   — 헤드 평균 attention weight (해석용)
        """
        B = lm_token.size(0)

        Q = self.W_Q(lm_token)    # (B, 1, d)
        K = self.W_K(FP_tokens)   # (B, 4, d)
        V = self.W_V(FP_tokens)   # (B, 4, d)

        # 헤드 분리: (B, seq, d) -> (B, h, seq, d_head)
        Q = Q.view(B, 1, self.h, self.d_head).transpose(1, 2)  # (B, h, 1, d_head)
        K = K.view(B, 4, self.h, self.d_head).transpose(1, 2)  # (B, h, 4, d_head)
        V = V.view(B, 4, self.h, self.d_head).transpose(1, 2)  # (B, h, 4, d_head)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, 1, 4)
        weights = torch.softmax(scores, dim=-1)                        # (B, h, 1, 4)
        # attention dropout 제거 — 4개 그룹에 dropout은 과도함

        attn = torch.matmul(weights, V)              # (B, h, 1, d_head)
        attn = attn.transpose(1, 2)                  # (B, 1, h, d_head)
        attn = attn.contiguous().view(B, 1, self.d)  # (B, 1, d)
        attn = self.W_O(attn).squeeze(1)             # (B, d)

        # 해석용: 헤드 평균 weight (B, h, 1, 4) -> (B, 4)
        weights_mean = weights.squeeze(2).mean(dim=1)  # (B, 4)

        return attn, weights_mean

    def encode(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_fp : (B, 425) — 전체 FP (StandardScaler 적용)
            x_lm : (B, 768) — ChemBERTa CLS 임베딩 (StandardScaler 적용)
        Returns:
            out  : (B, 16)  — 16-dim 융합 표현
        """
        # [Module 2] FP 그룹 -> 토큰
        fp_tokens = [
            proj(x_fp[:, s:e])
            for proj, (s, e) in zip(self.fp_projectors, self.GROUP_SLICES)
        ]
        FP_tokens = torch.stack(fp_tokens, dim=1)  # (B, 4, d)

        # [Module 2] ChemBERTa -> LM 토큰
        lm_raw   = self.lm_projector(x_lm)         # (B, d) <- residual용 저장
        lm_token = lm_raw.unsqueeze(1)             # (B, 1, d)

        # [Module 3] Multi-head cross-attention
        attn_out, _ = self._multihead_attn(lm_token, FP_tokens)  # (B, d)

        # Residual 1: norm1(attn_out + lm_raw)
        fused = self.norm1(attn_out + lm_raw)      # (B, d)

        # FFN + Residual 2: norm2(fused + ffn_out)
        ffn_out = self.ffn(fused)                  # (B, d)
        out     = self.norm2(fused + ffn_out)      # (B, d)

        # pre-classifier dropout
        out = self.dropout(out)
        return out  # (B, 16)

    def forward(self, x_fp: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
        """Returns: (B, 1) raw logit"""
        return self.classifier(self.encode(x_fp, x_lm))

    @torch.no_grad()
    def get_attention_weights(
        self, x_fp: torch.Tensor, x_lm: torch.Tensor
    ) -> torch.Tensor:
        """
        헤드 평균 Group Attention Weight (v1 호환 인터페이스)
        Returns: (B, 4)
        """
        self.eval()
        fp_tokens = [
            proj(x_fp[:, s:e])
            for proj, (s, e) in zip(self.fp_projectors, self.GROUP_SLICES)
        ]
        FP_tokens = torch.stack(fp_tokens, dim=1)
        lm_token  = self.lm_projector(x_lm).unsqueeze(1)
        _, weights_mean = self._multihead_attn(lm_token, FP_tokens)
        return weights_mean  # (B, 4)

    @torch.no_grad()
    def get_attention_weights_per_head(
        self, x_fp: torch.Tensor, x_lm: torch.Tensor
    ) -> torch.Tensor:
        """
        헤드별 Group Attention Weight (히트맵 시각화용)
        Returns: (B, h, 4)
        """
        self.eval()
        B = x_fp.size(0)
        fp_tokens = [
            proj(x_fp[:, s:e])
            for proj, (s, e) in zip(self.fp_projectors, self.GROUP_SLICES)
        ]
        FP_tokens = torch.stack(fp_tokens, dim=1)
        lm_token  = self.lm_projector(x_lm).unsqueeze(1)

        Q = self.W_Q(lm_token)
        K = self.W_K(FP_tokens)
        Q = Q.view(B, 1, self.h, self.d_head).transpose(1, 2)
        K = K.view(B, 4, self.h, self.d_head).transpose(1, 2)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, 1, 4)
        weights = torch.softmax(scores, dim=-1).squeeze(2)            # (B, h, 4)
        return weights
