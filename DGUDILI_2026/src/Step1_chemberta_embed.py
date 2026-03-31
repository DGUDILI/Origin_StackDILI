import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_CLEAN = os.environ.get("USE_CLEAN_DATA", "0") == "1"
_suffix    = "_clean" if _USE_CLEAN else ""
DATA_DIR   = os.path.join(ROOT, f"data{_suffix}")
_STACKDILI_ROOT = os.environ.get("STACKDILI_ROOT", os.path.dirname(ROOT))
DATA_PATH  = os.path.join(_STACKDILI_ROOT, "Data", f"Dataset{_suffix}.csv")
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 32
MAX_LENGTH = 512

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: ChemBERTa Embedding Extraction")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
smiles_list = df["SMILES"].tolist()
print(f"Total SMILES: {len(smiles_list)}")

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

all_embeddings = []
n_batches = math.ceil(len(smiles_list) / BATCH_SIZE)

for i in range(0, len(smiles_list), BATCH_SIZE):
    batch = smiles_list[i : i + BATCH_SIZE]
    enc = tokenizer(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].cpu().numpy()
    all_embeddings.append(cls)
    print(f"  Batch {i//BATCH_SIZE+1}/{n_batches} done - shape: {cls.shape}")

embeddings = np.vstack(all_embeddings).astype(np.float32)
print(f"\nEmbedding shape: {embeddings.shape}")

assert embeddings.shape == (len(smiles_list), 768)
assert not np.isnan(embeddings).any(), "NaN found"

emb_path   = os.path.join(DATA_DIR, "chemberta_embeddings.npy")
order_path = os.path.join(DATA_DIR, "smiles_order.npy")
np.save(emb_path,   embeddings)
np.save(order_path, np.array(smiles_list))
print(f"Saved: {emb_path}")
print(f"Saved: {order_path}")
print("Step 1 OK")
