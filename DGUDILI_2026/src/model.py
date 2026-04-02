import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn


FP_GROUP_INDICES = [
    [0, 13, 14, 15],   # G0 Continuous: nhyd, Estate8, Estate12, Estate32
    [1,  2,  3,  4],   # G1 MACCS Simple: MACCS30, 80, 81, 94
    [5,  6,  7,  8],   # G2 MACCS Mid: MACCS115, 116, 118, 124
    [9, 10, 11, 12],   # G3 MACCS Late: MACCS128, 133, 135, 153
]


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
        dropout: float = 0.3,
    ):
        super().__init__()
        self.chem_in_dim = chem_in_dim
        self.chem_hidden_dim = chem_hidden_dim
        self.k = k
        self.d_k = d_k
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
        self.W_V = nn.Linear(1, 1)

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
        attn = torch.bmm(weights, V)

        return attn.squeeze(-1)

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_cham, x_fp))


class GroupedCrossAttentionEncoder(nn.Module):

    """
    Semantic FP Group Tokenization + d_v expansion.

    FP 16개를 의미론적 4그룹(4-4-4-4)으로 묶어 독립 K/V 토큰으로 처리.
    Q: ChemBERTa 16 토큰 유지, K/V: 4 그룹 토큰, d_v=8.
    encode() → (B, feat_dim) = (B, 32)
    """

    def __init__(
        self,
        chem_in_dim: int = 384,
        chem_hidden_dim: int = 128,
        k: int = 16,
        d_k: int = 32,
        d_v: int = 8,
        feat_dim: int = 32,
        n_groups: int = 4,
        group_size: int = 4,
        dropout: float = 0.3,
        group_indices: list = None,
    ):
        super().__init__()
        self.k = k
        self.d_k = d_k
        self.d_v = d_v
        self.feat_dim = feat_dim
        self.n_groups = n_groups
        self.scale = math.sqrt(d_k)

        if group_indices is None:
            group_indices = FP_GROUP_INDICES
        self.register_buffer(
            "group_idx", torch.tensor(group_indices, dtype=torch.long)
        )

        self.chem_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, chem_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chem_hidden_dim, k),
        )

        self.W_Q = nn.Linear(1, d_k)

        self.group_norms = nn.ModuleList(
            [nn.LayerNorm(group_size) for _ in range(n_groups)]
        )
        self.W_K_list = nn.ModuleList(
            [nn.Linear(group_size, d_k) for _ in range(n_groups)]
        )
        self.W_V_list = nn.ModuleList(
            [nn.Linear(group_size, d_v) for _ in range(n_groups)]
        )

        self.compress = nn.Sequential(
            nn.Linear(k * d_v, feat_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(feat_dim, 1)

    def encode(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        B = x_fp.size(0)
        proj = self.chem_proj(x_cham)
        Q = self.W_Q(proj.unsqueeze(-1))          # (B, k, d_k)

        K_list, V_list = [], []
        for g in range(self.n_groups):
            fp_g = x_fp[:, self.group_idx[g]]     # (B, group_size)
            fp_g = self.group_norms[g](fp_g)
            K_list.append(self.W_K_list[g](fp_g))  # (B, d_k)
            V_list.append(self.W_V_list[g](fp_g))  # (B, d_v)

        K = torch.stack(K_list, dim=1)            # (B, n_groups, d_k)
        V = torch.stack(V_list, dim=1)            # (B, n_groups, d_v)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, k, n_groups)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.bmm(weights, V)             # (B, k, d_v)

        flat = attn.reshape(B, self.k * self.d_v)   # (B, k*d_v=128)
        return self.compress(flat)                    # (B, feat_dim=32)

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_cham, x_fp))


class FTV6StyleEncoder(nn.Module):
    """
    ft_v6-inspired: Learnable FP Linear Projection + Cross-Attention → k-dim fused.

    핵심 차이 (vs CrossAttentionEncoder):
      FP side: SelectKBest(16개 하드컷) → Linear(fp_in_dim, k) 학습 가능 Soft Projection
      → 전체 FP 정보를 손실 없이 Attention K/V에 반영

    Pipeline:
      FP  fp_in_dim → fp_proj: Linear(fp_in_dim, k) + LayerNorm → (B, k) [K,V 토큰]
      Chem chem_in_dim → chem_proj: LN→Linear→GELU→Drop→Linear → (B, k) [Q 토큰]
      CrossAttention: Q·K^T/√d_k → softmax → ·V → (B, k) fused
      head: Linear(k, 1)   [Stage 1 학습용]
      encode() → (B, k)    [Stage 2: Stacking 입력용]
    """

    def __init__(
        self,
        fp_in_dim: int = 425,
        chem_in_dim: int = 384,
        chem_hidden_dim: int = 128,
        k: int = 16,
        d_k: int = 32,
        d_v: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.k = k
        self.d_v = d_v
        self.scale = math.sqrt(d_k)

        # ChemBERTa projection (기존 CrossAttentionEncoder와 동일 구조)
        self.chem_proj = nn.Sequential(
            nn.LayerNorm(chem_in_dim),
            nn.Linear(chem_in_dim, chem_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chem_hidden_dim, k),
        )

        # FP: 학습 가능한 Soft Projection (SelectKBest 대체)
        # fp_in_dim개 피처 전체를 k개 토큰으로 압축 (가중치가 피처 중요도를 학습)
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_in_dim, k),
            nn.LayerNorm(k),
        )

        # Cross-Attention weight matrices
        self.W_Q = nn.Linear(1, d_k)
        self.W_K = nn.Linear(1, d_k)
        self.W_V = nn.Linear(1, d_v)   # d_v=4: value 표현력 확장

        # Stage 1 학습용 head (k * d_v → 1)
        self.head = nn.Linear(k * d_v, 1)

    def encode(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        """
        x_cham : (B, chem_in_dim)    ChemBERTa 전체 임베딩
        x_fp   : (B, fp_in_dim)      FP 전체 피처 (e.g., 425-dim)
        returns: (B, k * d_v)        fused feature vector
        """
        q  = self.chem_proj(x_cham)    # (B, k) — Query tokens
        kv = self.fp_proj(x_fp)        # (B, k) — Key/Value tokens

        Q = self.W_Q(q.unsqueeze(-1))   # (B, k, d_k)
        K = self.W_K(kv.unsqueeze(-1))  # (B, k, d_k)
        V = self.W_V(kv.unsqueeze(-1))  # (B, k, d_v)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, k, k)
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.bmm(weights, V)   # (B, k, d_v)

        return attn.reshape(attn.size(0), -1)   # (B, k * d_v)

    def forward(self, x_cham: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_cham, x_fp))


class E2E_FTV6StyleEncoder(nn.Module):
    """
    End-to-End FTV6 Encoder: ChemBERTa-77M-MLM 내장.

    Freeze:   embeddings + encoder.layer[0 .. N-2]  (마지막 제외 전체)
    Unfreeze: encoder.layer[N-1]                    (실제 마지막 레이어)

    encode(input_ids, attention_mask, x_fp) → (B, k)
    forward(...)                             → (B, 1)
    """

    def __init__(
        self,
        fp_in_dim: int = 425,
        k: int = 16,
        d_k: int = 32,
        d_v: int = 4,
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
        self.W_V = nn.Linear(1, d_v)   # d_v=4

        self.head = nn.Linear(k * d_v, 1)

    def _cls_embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, hidden_size)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_fp: torch.Tensor,
    ) -> torch.Tensor:
        x_cham = self._cls_embed(input_ids, attention_mask)  # (B, hidden_size)

        q  = self.chem_proj(x_cham)      # (B, k)
        kv = self.fp_proj(x_fp)          # (B, k)

        Q = self.W_Q(q.unsqueeze(-1))    # (B, k, d_k)
        K = self.W_K(kv.unsqueeze(-1))   # (B, k, d_k)
        V = self.W_V(kv.unsqueeze(-1))   # (B, k, d_v)

        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, k, k)
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.bmm(weights, V)  # (B, k, d_v)

        return attn.reshape(attn.size(0), -1)   # (B, k * d_v)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        x_fp: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask, x_fp))