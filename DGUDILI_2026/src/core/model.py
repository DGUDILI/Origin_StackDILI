import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_batch


class GraphMACCSEncoder(nn.Module):
    """
    GraphMACS Differential Cross-Attention Encoder.

    Pipeline:
      SMILES → ChemBERTa (마지막 레이어 unfreeze) → CLS (B, 384)
             → LayerNorm → Linear(384, d_model) → chem_feat (B, d_model)  [residual]

      SMILES → RDKit mol → atom_features (N_total, atom_feat_dim)
             → atom_proj Linear → SAGEConv × sage_layers
             → to_dense_batch → (B, MAX_ATOMS, sage_hidden) + pad_mask
             → node_proj Linear → node_q (B, MAX_ATOMS, d_model)          [Query]

      SMILES → MACCSkeys (B, 167) binary
             → Embedding(167, d_model)[idx]  (binary mask 제거, 임베딩 그대로)
             → maccs_kv (B, 167, d_model)                                 [Key/Value]
             → inactive bit mask → DiffAttn kv_padding_mask으로 전달 (score -1e9)

      DifferentialCrossAttention → attn_out (B, MAX_ATOMS, d_model)
                                   attn_weights (B, MAX_ATOMS, 167)  ← XAI

      masked_mean_pool (pad_mask 기반) → graph_feat (B, d_model)
      fused = fuse_proj(cat([chem_feat, graph_feat])) → LayerNorm
      MLP: Linear(d_model, k) → GELU → Dropout → encode_out (B, k)
      head: Linear(k, 1) → logits (B, 1)
    """

    def __init__(
        self,
        atom_feat_dim: int = 43,
        maccs_dim: int = 167,
        sage_hidden: int = 16,
        sage_layers: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        k: int = 16,
        max_atoms: int = 100,
        dropout: float = 0.3,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    ):
        super().__init__()
        from transformers import AutoModel
        from core.differential_attention import DifferentialCrossAttention

        self.max_atoms   = max_atoms
        self.sage_hidden = sage_hidden
        self.d_model     = d_model
        self.k           = k
        self._last_attn_weights: torch.Tensor | None = None

        # ── ChemBERTa ─────────────────────────────────────────────────────────
        self.chemberta = AutoModel.from_pretrained(model_name)
        chem_in_dim = self.chemberta.config.hidden_size  # 384
        for p in self.chemberta.parameters():
            p.requires_grad = False
        n_layers = len(self.chemberta.encoder.layer)
        for p in self.chemberta.encoder.layer[n_layers - 1].parameters():
            p.requires_grad = True

        self.chem_norm = nn.LayerNorm(chem_in_dim)
        self.chem_proj = nn.Linear(chem_in_dim, d_model)

        # ── GraphSAGE ─────────────────────────────────────────────────────────
        self.atom_proj  = nn.Linear(atom_feat_dim, sage_hidden)
        self.sage_convs = nn.ModuleList()
        self.sage_bns   = nn.ModuleList()
        for _ in range(sage_layers):
            self.sage_convs.append(SAGEConv(sage_hidden, sage_hidden))
            self.sage_bns.append(nn.BatchNorm1d(sage_hidden))
        self.node_proj = nn.Linear(sage_hidden, d_model)

        # ── MACCS Identity Embedding ───────────────────────────────────────────
        # bit 인덱스(0~166) → d_model-dim 학습 벡터
        # binary mask 곱으로 inactive key(=0) 소거
        self.maccs_emb = nn.Embedding(maccs_dim, d_model)
        nn.init.normal_(self.maccs_emb.weight, std=0.02)

        # ── Differential Cross-Attention ───────────────────────────────────────
        self.diff_attn = DifferentialCrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=0.1
        )

        # ── Fusion + MLP ───────────────────────────────────────────────────────
        # concat fusion: chem_feat || graph_feat → project → d_model (정보 손실 방지)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, k),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(k, 1)

    def _cls_embed(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, 384)

    def encode(
        self,
        input_ids: torch.Tensor,       # (B, MAX_LEN)
        attention_mask: torch.Tensor,  # (B, MAX_LEN)
        maccs: torch.Tensor,           # (B, 167) float32 binary
        graph_batch,                   # torch_geometric.data.Batch
    ) -> torch.Tensor:
        """
        Returns (B, k) fused feature vector.
        Saves self._last_attn_weights: (B, MAX_ATOMS, 167) for XAI.
        """
        B = input_ids.size(0)
        device = input_ids.device

        # ── ChemBERTa CLS ──────────────────────────────────────────────────────
        cls = self._cls_embed(input_ids, attention_mask)       # (B, 384)
        chem_feat = self.chem_proj(self.chem_norm(cls))        # (B, d_model)

        # ── GraphSAGE ──────────────────────────────────────────────────────────
        x = self.atom_proj(graph_batch.x)                      # (N_total, sage_hidden)
        edge_index = graph_batch.edge_index
        for conv, bn in zip(self.sage_convs, self.sage_bns):
            x = torch.relu(bn(conv(x, edge_index)))            # (N_total, sage_hidden)

        # to_dense_batch: variable nodes → padded dense tensor
        node_dense, pad_mask = to_dense_batch(
            x,
            graph_batch.batch,
            max_num_nodes=self.max_atoms,
        )  # node_dense: (B, MAX_ATOMS, sage_hidden), pad_mask: (B, MAX_ATOMS) True=valid

        node_q = self.node_proj(node_dense)                    # (B, MAX_ATOMS, d_model)

        # ── MACCS Identity Embedding ────────────────────────────────────────────
        idx = torch.arange(maccs.size(1), device=device).unsqueeze(0).expand(B, -1)
        maccs_kv = self.maccs_emb(idx)  # (B, 167, d_model) — binary mask 곱 제거
        # inactive bit는 embedding을 0으로 만드는 대신,
        # attention score 단계에서 -1e9 마스킹으로 처리 (softmax 분모 오염 방지)
        maccs_inactive = (maccs == 0)   # (B, 167) bool, True = inactive bit

        # ── Differential Cross-Attention ────────────────────────────────────────
        attn_out, attn_weights = self.diff_attn(
            query=node_q,                    # (B, MAX_ATOMS, d_model)
            key_value=maccs_kv,              # (B, 167, d_model)
            kv_padding_mask=maccs_inactive,  # (B, 167) bool, True = inactive
        )  # out: (B, MAX_ATOMS, d_model), weights: (B, MAX_ATOMS, 167)

        self._last_attn_weights = attn_weights.detach()

        # ── Masked Mean Pooling (pad 노드 제외) ─────────────────────────────────
        valid = pad_mask.unsqueeze(-1).float()                 # (B, MAX_ATOMS, 1)
        graph_feat = (attn_out * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        # (B, d_model)

        # ── Concat Fusion ───────────────────────────────────────────────────────
        # add fusion 대신 concat → linear로 두 표현을 독립적으로 보존
        fused = self.fuse_norm(
            self.fuse_proj(torch.cat([chem_feat, graph_feat], dim=-1))
        )  # (B, 2*d_model) → fuse_proj → (B, d_model) → LayerNorm

        return self.mlp(fused)                                 # (B, k)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        maccs: torch.Tensor,
        graph_batch,
    ) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask, maccs, graph_batch))

    def get_attn_weights(self) -> torch.Tensor | None:
        """마지막 encode() 호출에서 저장된 attention weights. (B, MAX_ATOMS, 167)"""
        return self._last_attn_weights
