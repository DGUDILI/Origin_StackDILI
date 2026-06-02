"""
differential_attention.py — Differential Cross-Attention

원본 Differential Transformer (arxiv 2410.05258) 메커니즘을 cross-attention으로 재구성.
- self-attention 전용인 원본 repo와 달리 Query와 Key/Value가 서로 다른 소스에서 옴
  (Query: atom node embeddings, Key/Value: MACCS tokens)
- head-wise learnable lambda (논문 방식)
- einops.rearrange로 Q1/Q2 분리 (shape 혼용 방지)
- GroupNorm으로 출력 안정화
- padding_mask 지원 (가변 원자 수)
- attn_weights 반환 (XAI용 — head 평균, relu는 Step4_xai에서 적용)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DifferentialCrossAttention(nn.Module):
    """
    Differential Cross-Attention.

    Args:
        d_model   : 임베딩 차원 (default 64)
        num_heads : 어텐션 헤드 수 (d_model % num_heads == 0)
        dropout   : dropout 비율
        lambda_init: 초기 lambda 스케일 상수 (non-trainable, 논문 권장 0.8)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        lambda_init: float = 0.8,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.scale     = math.sqrt(self.d_head)
        self.lambda_init = lambda_init

        # W_q: Q1, Q2 동시 생성 (2 * d_model output)
        self.W_q = nn.Linear(d_model, 2 * d_model, bias=False)
        # W_k: K1, K2 동시 생성
        self.W_k = nn.Linear(d_model, 2 * d_model, bias=False)
        # W_v: 단일 Value
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Head-wise lambda 파라미터 (논문 방식: dot-product 기반 reparametrization)
        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads, self.d_head))

        # GroupNorm for output stabilization
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

        self.dropout = nn.Dropout(dropout)

    def _compute_lambda(self) -> torch.Tensor:
        """
        Head-wise lambda: exp(q1·k1) - exp(q2·k2) + lambda_init
        논문 eq. (4). shape: (1, num_heads, 1, 1) for broadcasting.
        """
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1))
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1))
            + self.lambda_init
        ).clamp(min=1e-4, max=2.0)  # (num_heads,) — 음수화 방지 + 포화 방지
        return lam.view(1, self.num_heads, 1, 1)

    def forward(
        self,
        query: torch.Tensor,                      # (B, N_atoms, d_model)
        key_value: torch.Tensor,                  # (B, N_kv, d_model)  ← MACCS tokens
        key_padding_mask: torch.Tensor = None,    # (B, N_atoms) bool, True = PAD atom
        kv_padding_mask: torch.Tensor = None,     # (B, N_kv) bool, True = inactive key
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output      : (B, N_atoms, d_model)
            attn_weights: (B, N_atoms, N_kv)  — head-average differential scores
                          (음수 포함; XAI 시각화에서 relu 적용)

        Args:
            kv_padding_mask: inactive MACCS bit 위치에 True. softmax 전 scores1/scores2에
                             -1e9를 적용하여 해당 key의 attention weight를 0으로 억제.
                             shape (B, N_kv), dtype=bool.
        """
        B, N_q, _ = query.shape
        N_kv = key_value.shape[1]

        # ── Q1, Q2 분리 (einops) ──────────────────────────────────────────────
        # W_q: (B, N_q, 2*d_model) → rearrange → (2, B, h, N_q, d_head)
        q = rearrange(
            self.W_q(query),
            'b n (two h d) -> two b h n d',
            two=2, h=self.num_heads
        )
        q1, q2 = q[0], q[1]  # each: (B, h, N_q, d_head)

        # ── K1, K2 분리 ──────────────────────────────────────────────────────
        k = rearrange(
            self.W_k(key_value),
            'b n (two h d) -> two b h n d',
            two=2, h=self.num_heads
        )
        k1, k2 = k[0], k[1]  # each: (B, h, N_kv, d_head)

        # ── V ─────────────────────────────────────────────────────────────────
        v = rearrange(
            self.W_v(key_value),
            'b n (h d) -> b h n d',
            h=self.num_heads
        )  # (B, h, N_kv, d_head)

        # ── Differential Scores ───────────────────────────────────────────────
        # scores shape: (B, h, N_q, N_kv)
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / self.scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / self.scale

        # key_padding_mask: (B, N_atoms) True=PAD atom
        # → query-side 패딩은 pooling 단계에서 pad_mask로 처리되므로 여기서 별도 불필요
        if key_padding_mask is not None:
            pass  # 현재 미사용 (atom query padding은 masked_mean_pool에서 처리)

        # kv_padding_mask: (B, N_kv) True=inactive MACCS bit
        # → softmax 분모 오염 방지: inactive key를 -1e9로 마스킹하여 attention weight ≈ 0
        if kv_padding_mask is not None:
            # (B, N_kv) → (B, 1, 1, N_kv) for broadcasting over (B, h, N_q, N_kv)
            kv_mask = kv_padding_mask.unsqueeze(1).unsqueeze(2)
            scores1 = scores1.masked_fill(kv_mask, -1e9)
            scores2 = scores2.masked_fill(kv_mask, -1e9)

        lam = self._compute_lambda()  # (1, h, 1, 1)

        # softmax over N_kv dimension
        a1 = F.softmax(scores1, dim=-1)  # (B, h, N_q, N_kv)
        a2 = F.softmax(scores2, dim=-1)
        scores = a1 - lam * a2           # Differential attention scores

        scores = self.dropout(scores)

        # ── Weighted sum ──────────────────────────────────────────────────────
        out = torch.matmul(scores, v)    # (B, h, N_q, d_head)

        # ── GroupNorm + scale ─────────────────────────────────────────────────
        # GroupNorm 적용을 위해 (B, d_model, N_q) 형태로 변환
        out = rearrange(out, 'b h n d -> b n (h d)')       # (B, N_q, d_model)
        out = out.transpose(1, 2)                           # (B, d_model, N_q)
        out = self.group_norm(out)
        out = out.transpose(1, 2)                           # (B, N_q, d_model)
        out = out * (1.0 - self.lambda_init)               # 논문 스케일 보정

        # ── Output projection ─────────────────────────────────────────────────
        out = self.W_o(out)  # (B, N_q, d_model)

        # ── attn_weights for XAI (head average) ──────────────────────────────
        with torch.no_grad():
            self.multihead_scores = scores.detach()  # (B, h, N_q, N_kv) — Step5_mHeadHeatmap용
            attn_weights = scores.mean(dim=1)  # (B, N_q, N_kv)

        return out, attn_weights
