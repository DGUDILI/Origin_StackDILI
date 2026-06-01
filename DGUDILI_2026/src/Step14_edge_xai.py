import os
import sys
import torch
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# 기존 프로젝트 경로 추가
sys.path.append('DGUDILI_2026/src')
from Step5_mHeadHeatmap import load_model
from config import MODEL_NAME, OUT_DIR
from graph_utils import smiles_to_pyg, get_maccs

def get_edge_importance(model, smi, device, tokenizer):
    model.eval()
    inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    input_ids = inputs["input_ids"].long()
    attn_mask = inputs["attention_mask"].long()
    
    pyg_data = smiles_to_pyg(smi)
    if pyg_data is None: return None, None
    
    pyg_data = pyg_data.to(device)
    maccs = torch.FloatTensor(get_maccs(smi)).unsqueeze(0).to(device)
    
    edge_mask = torch.ones(pyg_data.edge_attr.shape[0], 1, device=device, requires_grad=True)
    
    try:
        original_edge_attr = pyg_data.edge_attr.clone()
        pyg_data.edge_attr = original_edge_attr * edge_mask
        logits = model(input_ids, attn_mask, maccs, pyg_data)
        prob = torch.sigmoid(logits)
        prob.backward()
        
        importance = edge_mask.grad.abs().squeeze().detach().cpu().numpy()
        if importance.max() > 0:
            importance = importance / (importance.max() + 1e-8)
        return importance, pyg_data
    except Exception:
        return None, None

def plot_edge_shaper_stable(smi, importance, pyg_data, rank, category, out_dir, is_toxic):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return
    
    edge_index = pyg_data.edge_index.cpu().numpy()
    bond_highlights = {}
    
    for i in range(len(importance)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        bond = mol.GetBondBetweenAtoms(u, v)
        if bond:
            idx = bond.GetIdx()
            bond_highlights[idx] = max(bond_highlights.get(idx, 0), importance[i])

    # 중요도 0.2 이상인 결합만 강조
    highlight_bonds = [idx for idx, sc in bond_highlights.items() if sc > 0.2]
    # 독성은 빨간색(1, 0.4, 0.4), 비독성은 파란색(0.4, 0.4, 1)
    color = (1.0, 0.4, 0.4) if is_toxic else (0.4, 0.6, 1.0)
    highlight_colors = {idx: color for idx in highlight_bonds}

    # RDKit DrawMolecule 표준 인자만 사용 (에러 방지)
    d2d = rdMolDraw2D.MolDraw2DCairo(600, 600)
    d2d.drawOptions().bondLineWidth = 5
    d2d.drawOptions().useBWAtomPalette()
    
    d2d.DrawMolecule(mol, 
                     highlightAtoms=[], 
                     highlightAtomColors={}, 
                     highlightBonds=highlight_bonds, 
                     highlightBondColors=highlight_colors)
    d2d.FinishDrawing()
    
    # 파일 이름에 SMILES 포함 (최대 50자)
    safe_smi = re.sub(r'[\\/*?:"<>|]', "_", smi)[:50]
    filename = f"{category}_Rank{rank:02d}_{safe_smi}.png"
    
    with open(os.path.join(out_dir, filename), "wb") as f:
        f.write(d2d.GetDrawingText())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model("DGUDILI_2026/outputs/pretrained_graph_encoder.pt", device)
    
    edge_out_dir = os.path.join(OUT_DIR, "EdgeSHAPer_Results")
    os.makedirs(edge_out_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"))
    df["Prediction"] = (df["Probability"] >= 0.5).astype(float)
    correct_df = df[df["True_Label"] == df["Prediction"]]
    
    toxic_correct = correct_df[correct_df["True_Label"] == 1].sort_values(by="Probability", ascending=False).head(50)
    safe_correct = correct_df[correct_df["True_Label"] == 0].sort_values(by="Probability", ascending=True).head(50)
    
    print(f"🚀 분석 시작: 정답을 맞춘 독성 50개 + 비독성 50개 (총 100개)")

    for category, targets, is_tox in [("Toxic", toxic_correct, True), ("Safe", safe_correct, False)]:
        for i, (idx, row) in enumerate(targets.iterrows()):
            smi = row["SMILES"]
            rank = i + 1
            
            # [요청사항] 터미널에 전체 SMILES 출력
            print(f"\n[{category} Rank {rank:02d}] --------------------------------")
            print(f"SMILES: {smi}")
            
            importance, pyg_data = get_edge_importance(model, smi, device, tokenizer)
            if importance is not None:
                plot_edge_shaper_stable(smi, importance, pyg_data, rank, category, edge_out_dir, is_tox)
                print(f"✅ 시각화 완료")

    print(f"\n✨ 모든 작업 완료! {edge_out_dir} 폴더를 확인하세요.")

if __name__ == "__main__":
    main()