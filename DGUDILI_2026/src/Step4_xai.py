"""
Step4_xai.py — GraphMACCSEncoder Differential Attention XAI 시각화

사용법:
    python src/Step4_xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python src/Step4_xai.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --top_k 20

출력:
    - 콘솔: 활성 MACCS keys, 원자별 중요도 상위 N
    - 이미지: outputs/xai_{분자이름}_heatmap.png
              outputs/xai_{분자이름}_mol.png (원자 중요도 오버레이, 선택)

핵심:
    - attn_weights: (1, MAX_ATOMS, 167) — 학습된 모델에서 추출
    - torch.relu(attn_weights) → 음수 노이즈 제거, 양(+)의 상관관계만 시각화
    - bit 0 (Unused) 제외, bits 1~166만 히트맵 표시
"""

import os
import sys
import argparse
import torch
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    K, D_MODEL, NUM_HEADS, DROPOUT, MODEL_NAME,
    MAX_LENGTH, MACCS_DIM, MAX_ATOMS, SAGE_LAYERS, SAGE_HIDDEN, ATOM_FEAT_DIM,
    OUT_DIR, DATA_DIR,
)
from model import GraphMACCSEncoder
from graph_utils import smiles_to_pyg, get_maccs, MACCS_NAMES, verify_maccs_mapping
from transformers import AutoTokenizer
from rdkit import Chem


def load_model(ckpt_path: str, device: torch.device) -> GraphMACCSEncoder:
    encoder = GraphMACCSEncoder(
        atom_feat_dim=ATOM_FEAT_DIM,
        maccs_dim=MACCS_DIM,
        sage_hidden=SAGE_HIDDEN,
        sage_layers=SAGE_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        k=K,
        max_atoms=MAX_ATOMS,
        dropout=DROPOUT,
        model_name=MODEL_NAME,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    print(f"Model loaded: {ckpt_path}")
    return encoder


def extract_attn(encoder, smiles: str, tokenizer, device) -> tuple:
    """
    단일 SMILES에 대한 attn_weights 추출.

    Returns:
        attn_relu : (n_atoms, 166) float32 — relu + bit 0 제외
        atom_syms : list[str] — 원자 기호 + 인덱스 (e.g. "C0", "N3")
        n_atoms   : int
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    n_atoms = mol.GetNumAtoms()

    # Tokenize
    enc = tokenizer(
        [smiles], max_length=MAX_LENGTH,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    input_ids   = enc["input_ids"].to(device)
    attn_mask   = enc["attention_mask"].to(device)

    # Graph + MACCS
    pyg  = smiles_to_pyg(smiles)
    mac  = get_maccs(smiles).unsqueeze(0).to(device)  # (1, 167)

    from torch_geometric.data import Batch as PyGBatch
    graph_batch = PyGBatch.from_data_list([pyg]).to(device)

    with torch.no_grad():
        encoder.encode(input_ids, attn_mask, mac, graph_batch)

    attn_raw = encoder.get_attn_weights()  # (1, MAX_ATOMS, 167)
    attn_raw = attn_raw[0, :n_atoms, :]    # (n_atoms, 167)

    # Differential ReLU XAI: 음수 제거, bit 0 제외
    attn_relu = torch.relu(attn_raw)[:, 1:].cpu().numpy()  # (n_atoms, 166)

    atom_syms = [
        f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"
        for i in range(n_atoms)
    ]

    return attn_relu, atom_syms, n_atoms


def plot_heatmap(attn: np.ndarray, atom_syms: list[str], out_path: str, top_k: int | None = None):
    """
    seaborn 히트맵 저장.

    Args:
        attn     : (n_atoms, 166) — relu 적용된 양의 상관관계
        atom_syms: y축 레이블
        out_path : 저장 경로 (.png)
        top_k    : 중요도 상위 k개 MACCS key만 표시 (None이면 전체)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # x축 레이블: bits 1~166
    x_labels = [MACCS_NAMES.get(i, f"Key{i}") or f"Key{i}" for i in range(1, 167)]

    if top_k is not None and top_k < 166:
        key_importance = attn.sum(axis=0)                       # (166,)
        top_idx = np.argsort(key_importance)[::-1][:top_k]
        top_idx = sorted(top_idx)
        attn     = attn[:, top_idx]
        x_labels = [x_labels[i] for i in top_idx]

    fig_w = max(14, len(x_labels) * 0.18)
    fig_h = max(4,  len(atom_syms) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        attn,
        xticklabels=x_labels,
        yticklabels=atom_syms,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="lightgray",
    )
    ax.set_xlabel("MACCS Key (bit index 1~166, bit 0=Unused 제외)", fontsize=9)
    ax.set_ylabel("Atom", fontsize=9)
    ax.set_title(
        "Differential Cross-Attention XAI\n"
        "relu(attn_weights) — 양(+)의 원자-MACCS 상관관계",
        fontsize=10,
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved: {out_path}")


def plot_mol_importance(smiles: str, atom_importance: np.ndarray, out_path: str):
    """
    RDKit Draw로 원자 중요도 오버레이 저장.

    atom_importance: (n_atoms,) — 각 원자의 총 어텐션 합
    """
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image
        import io

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        # 정규화
        imp = atom_importance.copy()
        if imp.max() > 0:
            imp = imp / imp.max()

        # 원자별 색상 (히트맵: 빨간색 계열)
        atom_colors = {}
        highlight_atoms = []
        for i, v in enumerate(imp):
            if v > 0.05:
                atom_colors[i] = (1.0, 1.0 - v * 0.8, 1.0 - v * 0.8)  # RGB
                highlight_atoms.append(i)

        drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText()
        with open(out_path.replace(".png", ".svg"), "w") as f:
            f.write(svg)
        print(f"Mol SVG saved: {out_path.replace('.png', '.svg')}")
    except Exception as e:
        print(f"[WARN] 분자 시각화 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="GraphMACCSEncoder XAI 시각화")
    parser.add_argument("--smiles", type=str, required=True, help="분석할 SMILES")
    parser.add_argument("--ckpt",   type=str,
                        default=os.path.join(OUT_DIR, "pretrained_graph_encoder.pt"),
                        help="학습된 모델 체크포인트 경로")
    parser.add_argument("--top_k",  type=int, default=None,
                        help="히트맵에 표시할 상위 MACCS key 수 (None=전체 166)")
    parser.add_argument("--mol",    action="store_true",
                        help="원자 중요도 분자 이미지 생성")
    parser.add_argument("--name",   type=str, default=None,
                        help="출력 파일 이름 prefix (기본: SMILES 앞 10자)")
    args = parser.parse_args()

    # 검증
    verify_maccs_mapping()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder   = load_model(args.ckpt, device)

    # 어텐션 추출
    attn_relu, atom_syms, n_atoms = extract_attn(encoder, args.smiles, tokenizer, device)

    # 콘솔 요약
    mol = Chem.MolFromSmiles(args.smiles)
    maccs = get_maccs(args.smiles)
    print(f"\n분자: {args.smiles[:50]}")
    print(f"원자 수: {n_atoms}")
    print(f"활성 MACCS keys ({int(maccs[1:].sum())}개):")
    for i in range(1, 167):
        if maccs[i] == 1:
            print(f"  bit {i:3d}: {MACCS_NAMES.get(i, 'Unknown')}")

    atom_importance = attn_relu.sum(axis=1)  # (n_atoms,)
    print(f"\n원자별 총 어텐션 (상위 5):")
    for rank, idx in enumerate(np.argsort(atom_importance)[::-1][:5]):
        print(f"  #{rank+1} {atom_syms[idx]}: {atom_importance[idx]:.4f}")

    # 히트맵 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = args.name or args.smiles[:10].replace("/", "_")
    heatmap_path = os.path.join(OUT_DIR, f"xai_{prefix}_heatmap.png")
    plot_heatmap(attn_relu, atom_syms, heatmap_path, top_k=args.top_k)

    # 분자 시각화 (선택)
    if args.mol:
        mol_path = os.path.join(OUT_DIR, f"xai_{prefix}_mol.png")
        plot_mol_importance(args.smiles, atom_importance, mol_path)


if __name__ == "__main__":
    main()
