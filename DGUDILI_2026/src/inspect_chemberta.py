from transformers import AutoModel, AutoConfig
import torch

config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
m = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

total = sum(p.numel() for p in m.parameters())
print(f"Total params: {total:,}")
print(f"Num transformer layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num attention heads: {config.num_attention_heads}")

# 레이어별 파라미터 수
for i in range(config.num_hidden_layers):
    layer = m.encoder.layer[i]
    n = sum(p.numel() for p in layer.parameters())
    print(f"  encoder.layer[{i}]: {n:,} params")

pooler = sum(p.numel() for p in m.pooler.parameters()) if hasattr(m, 'pooler') else 0
print(f"  pooler: {pooler:,} params")

embed = sum(p.numel() for p in m.embeddings.parameters())
print(f"  embeddings: {embed:,} params")
