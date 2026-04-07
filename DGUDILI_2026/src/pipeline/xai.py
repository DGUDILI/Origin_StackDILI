"""
xai.py — GraphMACCSEncoder Differential Attention XAI 시각화

사용법:
    python src/pipeline/xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python src/pipeline/xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --top_k 20
    python src/pipeline/xai.py --smiles "..." --config config_clean.yaml

출력:
    - 콘솔: 활성 MACCS keys, 원자별 중요도 상위 N
    - 이미지: outputs/original/xai_{분자이름}_heatmap.png
              outputs/original/xai_{분자이름}_mol.svg (--mol 시)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import torch
import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.config import load_cfg
from core.model import GraphMACCSEncoder
from core.graph_utils import smiles_to_pyg, get_maccs, MACCS_NAMES, verify_maccs_mapping
from transformers import AutoTokenizer
from rdkit import Chem


def load_model(cfg, ckpt_path: str, device: torch.device) -> GraphMACCSEncoder:
    encoder = GraphMACCSEncoder(
        atom_feat_dim=cfg.atom_feat_dim,
        maccs_dim=cfg.maccs_dim,
        sage_hidden=cfg.sage_hidden,
        sage_layers=cfg.sage_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        k=cfg.k,
        max_atoms=cfg.max_atoms,
        dropout=cfg.dropout,
        model_name=cfg.model_name,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    print(f"Model loaded: {ckpt_path}")
    return encoder


def extract_attn(encoder, smiles: str, tokenizer, cfg, device) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    n_atoms = mol.GetNumAtoms()

    enc = tokenizer(
        [smiles], max_length=cfg.max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    pyg = smiles_to_pyg(smiles)
    mac = get_maccs(smiles).unsqueeze(0).to(device)

    from torch_geometric.data import Batch as PyGBatch
    graph_batch = PyGBatch.from_data_list([pyg]).to(device)

    with torch.no_grad():
        encoder.encode(input_ids, attn_mask, mac, graph_batch)

    attn_raw  = encoder.get_attn_weights()      # (1, MAX_ATOMS, 167)
    attn_raw  = attn_raw[0, :n_atoms, :]        # (n_atoms, 167)
    attn_relu = torch.relu(attn_raw)[:, 1:].cpu().numpy()  # (n_atoms, 166) bit0 제외

    atom_syms = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    return attn_relu, atom_syms, n_atoms


def plot_heatmap(attn: np.ndarray, atom_syms: list, out_path: str,
                 top_k: int | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_labels = [MACCS_NAMES.get(i, f"Key{i}") or f"Key{i}" for i in range(1, 167)]

    if top_k is not None and top_k < 166:
        key_importance = attn.sum(axis=0)
        top_idx  = sorted(np.argsort(key_importance)[::-1][:top_k])
        attn     = attn[:, top_idx]
        x_labels = [x_labels[i] for i in top_idx]

    fig_w = max(14, len(x_labels) * 0.18)
    fig_h = max(4,  len(atom_syms) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        attn, xticklabels=x_labels, yticklabels=atom_syms,
        cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="lightgray",
    )
    ax.set_xlabel("MACCS Key (bit 1~166)", fontsize=9)
    ax.set_ylabel("Atom", fontsize=9)
    ax.set_title("Differential Cross-Attention XAI\nrelu(attn_weights)", fontsize=10)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved: {out_path}")


def plot_mol_importance(smiles: str, atom_importance: np.ndarray, out_path: str):
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        imp = atom_importance.copy()
        if imp.max() > 0:
            imp = imp / imp.max()
        atom_colors = {i: (1.0, 1.0 - v * 0.8, 1.0 - v * 0.8)
                       for i, v in enumerate(imp) if v > 0.05}
        highlight_atoms = list(atom_colors.keys())
        drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                            highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        svg_path = out_path.replace(".png", ".svg")
        with open(svg_path, "w") as f:
            f.write(drawer.GetDrawingText())
        print(f"Mol SVG saved: {svg_path}")
    except Exception as e:
        print(f"[WARN] 분자 시각화 실패: {e}")


def main():
    _ROOT = os.path.dirname(_SRC)
    default_cfg = os.path.join(_ROOT, "config.yaml")

    parser = argparse.ArgumentParser(description="GraphMACCSEncoder XAI 시각화")
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--config", default=default_cfg)
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="체크포인트 경로 (기본: output_dir/pretrained_graph_encoder.pt)")
    parser.add_argument("--top_k",  type=int, default=None)
    parser.add_argument("--mol",    action="store_true")
    parser.add_argument("--name",   type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    verify_maccs_mapping()

    ckpt      = args.ckpt or os.path.join(cfg.output_dir, "pretrained_graph_encoder.pt")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    encoder   = load_model(cfg, ckpt, device)

    attn_relu, atom_syms, n_atoms = extract_attn(encoder, args.smiles, tokenizer, cfg, device)

    maccs = get_maccs(args.smiles)
    print(f"\n분자: {args.smiles[:50]}")
    print(f"원자 수: {n_atoms}")
    print(f"활성 MACCS keys ({int(maccs[1:].sum())}개):")
    for i in range(1, 167):
        if maccs[i] == 1:
            print(f"  bit {i:3d}: {MACCS_NAMES.get(i, 'Unknown')}")

    atom_importance = attn_relu.sum(axis=1)
    print(f"\n원자별 총 어텐션 (상위 5):")
    for rank, idx in enumerate(np.argsort(atom_importance)[::-1][:5]):
        print(f"  #{rank+1} {atom_syms[idx]}: {atom_importance[idx]:.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    prefix       = args.name or args.smiles[:10].replace("/", "_")
    heatmap_path = os.path.join(cfg.output_dir, f"xai_{prefix}_heatmap.png")
    plot_heatmap(attn_relu, atom_syms, heatmap_path, top_k=args.top_k)

    if args.mol:
        mol_path = os.path.join(cfg.output_dir, f"xai_{prefix}_mol.png")
        plot_mol_importance(args.smiles, atom_importance, mol_path)


if __name__ == "__main__":
    main()
