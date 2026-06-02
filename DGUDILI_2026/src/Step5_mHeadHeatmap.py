"""
Step5_mHeadHeatmap.py — 멀티헤드 어텐션 개별 히트맵 (올바른 딕셔너리 연동 버전)
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    K, D_MODEL, NUM_HEADS, DROPOUT, MODEL_NAME,
    MAX_LENGTH, MACCS_DIM, MAX_ATOMS,
    GINE_LAYERS, GINE_HIDDEN, BOND_FEAT_DIM, ATOM_FEAT_DIM,
    OUT_DIR, DATA_DIR,
)
from model import GraphMACCSEncoder
from graph_utils import smiles_to_pyg, get_maccs, MACCS_NAMES
from transformers import AutoTokenizer
from rdkit import Chem


def load_model(ckpt_path: str, device: torch.device) -> GraphMACCSEncoder:
    encoder = GraphMACCSEncoder(
        atom_feat_dim=ATOM_FEAT_DIM, bond_feat_dim=BOND_FEAT_DIM,
        maccs_dim=MACCS_DIM, gine_hidden=GINE_HIDDEN, gine_layers=GINE_LAYERS,
        d_model=D_MODEL, num_heads=NUM_HEADS,
        k=K, max_atoms=MAX_ATOMS, dropout=DROPOUT, model_name=MODEL_NAME,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    return encoder


def extract_attn(encoder, smiles: str, tokenizer, device) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    enc = tokenizer([smiles], max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    pyg = smiles_to_pyg(smiles)
    mac = get_maccs(smiles).unsqueeze(0).to(device)

    from torch_geometric.data import Batch as PyGBatch
    graph_batch = PyGBatch.from_data_list([pyg]).to(device)

    with torch.no_grad():
        encoder.encode(input_ids, attn_mask, mac, graph_batch)

    for module in encoder.modules():
        if hasattr(module, 'multihead_scores'):
            attn_raw = module.multihead_scores
            break

    attn_raw = attn_raw[0, :, :n_atoms, :]
    attn_relu = torch.relu(attn_raw)[:, :, 1:].cpu().numpy()

    # 아스피린에 없는 특징은 여기서 무조건 0으로 삭제됨
    active_maccs = mac[0, 1:].cpu().numpy()
    attn_relu = attn_relu * active_maccs

    atom_syms = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    return attn_relu, atom_syms


def plot_independent_head_heatmap(attn: np.ndarray, atom_syms: list[str], out_path: str, top_k: int):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))
    axes = axes.flatten()

    for head_idx in range(4):
        head_attn = attn[head_idx] 
        key_importance = head_attn.sum(axis=0)
        
        top_idx = np.argsort(key_importance)[::-1][:top_k]
        top_idx = [idx for idx in top_idx if key_importance[idx] > 0]
        
        if not top_idx:
            axes[head_idx].axis('off')
            continue
            
        sub_attn = head_attn[:, top_idx]
        
        x_labels = [MACCS_NAMES.get(idx + 1, f"Key{idx+1}") or f"Key{idx+1}" for idx in top_idx]

        sns.heatmap(sub_attn, ax=axes[head_idx], cmap="Reds", annot=False,
                    xticklabels=x_labels, yticklabels=atom_syms)
        
        axes[head_idx].set_title(f"Head {head_idx+1} Focus (Top {len(top_idx)} Keys)", fontsize=14, fontweight='bold')
        axes[head_idx].set_xlabel("MACCS Keys")
        axes[head_idx].set_ylabel("Atoms")
        
        axes[head_idx].set_xticklabels(axes[head_idx].get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, default="CC(=O)Oc1ccccc1C(=O)O")
    parser.add_argument("--ckpt", type=str, default="outputs/pretrained_graph_encoder.pt")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = load_model(args.ckpt, device)
    
    attn, atom_syms = extract_attn(encoder, args.smiles, tokenizer, device)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "mhead_independent_heatmap.png")
    plot_independent_head_heatmap(attn, atom_syms, out_path, args.top_k)