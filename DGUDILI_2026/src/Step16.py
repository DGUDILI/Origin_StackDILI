import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from rdkit import Chem

# 경로 설정
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import (
    K, D_MODEL, NUM_HEADS, DROPOUT, MODEL_NAME,
    MAX_LENGTH, MACCS_DIM, MAX_ATOMS,
    GINE_LAYERS, GINE_HIDDEN, BOND_FEAT_DIM, ATOM_FEAT_DIM,
    OUT_DIR, DATA_PATH
)
from model import GraphMACCSEncoder
from graph_utils import smiles_to_pyg, get_maccs, MACCS_NAMES
from torch_geometric.data import Batch as PyGBatch

def load_trained_model(ckpt_path, device):
    encoder = GraphMACCSEncoder(
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        maccs_dim=MACCS_DIM,
        gine_hidden=GINE_HIDDEN,
        gine_layers=GINE_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        k=K,
        max_atoms=MAX_ATOMS,
        dropout=DROPOUT,
        model_name=MODEL_NAME,
    )
    # weights_only=False는 DifferentialCrossAttention 등 커스텀 클래스 로드를 위함
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    return encoder

def plot_multihead_heatmap(smiles, multihead_weights, atom_syms, threshold, pred_val, output_dir):
    """
    4개 헤드의 어텐션 맵을 하나의 Figure에 시각화 (ReLU 적용)
    multihead_weights: (num_heads, n_atoms, 167)
    """
    num_heads = multihead_weights.shape[0]
    # bit 0 제외 및 ReLU 적용 (양의 상관관계만)
    weights = np.maximum(0, multihead_weights[:, :, 1:]) 
    
    # 활성 MACCS bit만 추출 (모든 헤드 합산 기준 값이 있는 것들만)
    active_bits = np.where(weights.sum(axis=(0, 1)) > 1e-5)[0]
    if len(active_bits) == 0: return # 표시할 내용 없음
    
    weights = weights[:, :, active_bits]
    bit_names = [MACCS_NAMES.get(b + 1, f"Bit{b+1}") for b in active_bits]

    fig, axes = plt.subplots(num_heads, 1, figsize=(max(12, len(active_bits)*0.4), num_heads * 4))
    fig.suptitle(f"SMILES: {smiles[:50]}...\nPred Score: {pred_val:.4f} (Threshold: {threshold})", fontsize=15)

    for h in range(num_heads):
        sns.heatmap(weights[h], xticklabels=bit_names, yticklabels=atom_syms, 
                    ax=axes[h], cmap="YlOrRd", cbar_kws={'label': f'Head {h} Weight'})
        axes[h].set_title(f"Attention Head {h}")
        axes[h].tick_params(axis='x', rotation=90, labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    safe_smiles = smiles[:15].replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(output_dir, f"MH_XAI_{safe_smiles}.png"), dpi=150)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. 모델 로드
    ckpt_path = os.path.join(OUT_DIR, "pretrained_graph_encoder.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
    model = load_trained_model(ckpt_path, device)
    
    # 2. 데이터 로드 (Test set - 여기서는 예시로 전체 데이터 로드 후 필터링하거나 별도 리스트 활용)
    # 실무에서는 split 정보를 담은 csv를 로드하는 것을 권장합니다.
    df = pd.read_csv(DATA_PATH)
    if 'Split' in df.columns:
        test_df = df[df['Split'] == 'test'].reset_index(drop=True)
    else:
        test_df = df # Split 정보 없으면 전체 수행
    
    threshold = 0.55
    output_xai_dir = os.path.join(OUT_DIR, "multihead_xai")
    os.makedirs(output_xai_dir, exist_ok=True)
    
    print(f"Analyzing test samples with prediction > {threshold}...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        n_atoms = mol.GetNumAtoms()
        atom_syms = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]

        # 추론 준비
        enc = tokenizer([smiles], max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        mac = get_maccs(smiles).unsqueeze(0).to(device)
        pyg = smiles_to_pyg(smiles)
        graph_batch = PyGBatch.from_data_list([pyg]).to(device)

        with torch.no_grad():
            logit = model(input_ids, attn_mask, mac, graph_batch)
            prob = torch.sigmoid(logit).item()

        # 3. 독성 판별 (Threshold 0.55)
        if prob >= threshold:
            # DifferentialCrossAttention에 저장된 multihead_scores 추출
            # shape: (1, num_heads, MAX_ATOMS, 167)
            mh_scores = model.diff_attn.multihead_scores.cpu().numpy()[0] 
            # 실제 원자 개수만큼 슬라이싱: (num_heads, n_atoms, 167)
            mh_scores = mh_scores[:, :n_atoms, :]
            
            # 히트맵 그리기
            plot_multihead_heatmap(smiles, mh_scores, atom_syms, threshold, prob, output_xai_dir)

    print(f"XAI Plots saved to: {output_xai_dir}")

if __name__ == "__main__":
    main()