import os
import sys
import torch
import numpy as np
import re
import io
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Draw

# 기존 모듈 임포트
sys.path.append('DGUDILI_2026/src')
from Step5_mHeadHeatmap import load_model, extract_attn, plot_independent_head_heatmap
from config import MODEL_NAME, OUT_DIR
from graph_utils import MACCS_NAMES  

# 터미널에서 뽑으신 Top 10 독성/비독성 약물들
toxic_smiles = [
    r"CS(=O)(=O)NC1=C(C=C(C=C1)[N+](=O)[O-])OC2=CC=CC=C2",
    r"C[C@H]1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O",
    r"CN1C=NC(=C1SC2=NC=NC3=C2NC=N3)[N+](=O)[O-]",
    r"CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C(N2)C=C(C=C4)Cl",
    r"CN1C2=C(C3=CC=CC=C3S1(=O)=O)OC(=O)N(C2=O)C4=CC=CC=N4",
    r"CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl",
    r"C[C@H]1/C=C/C=C(\C(=O)NC\2=C(C3=C(C(=C4C(=C3C(=O)/C2=C/NN5CCN(CC5)C)C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C",
    r"C1=CC(=C(C(=C1)Cl)Cl)C2=C(N=C(N=N2)N)N",
    r"CCCC1=CC(=O)NC(=S)N1",
    r"CC(C)(C)C1=CC=C(C=C1)S(=O)(=O)NC2=C(C(=NC(=N2)C3=NC=CC=N3)OCCO)OC4=CC=CC=C4OC"
]

safe_smiles = [
    r"C[N+]1(C2CC(CC1C3C2O3)OC(=O)C(CO)C4=CC=CC=C4)C",
    r"C[N+]1(CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O)CC6CC6.[Br-]",
    r"CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O",
    r"C=CCN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O",
    r"CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O",
    r"C[N+]1(CCCC(C1)OC(=O)C(C2=CC=CC=C2)(C3=CC=CC=C3)O)C",
    r"CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
    r"C[N+]1(CCC(C1)OC(=O)C(C2CCCC2)(C3=CC=CC=C3)O)C",
    r"CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O",
    r"C[N+](C)(C)C[C@@H](CC(=O)[O-])O"
]

GLOBAL_MAX_ATTN = 0.25  
MIN_ATTN_CUTOFF = 0.05  

def print_top_reasons(attn: np.ndarray, atom_syms: list):
    total_attn = attn.sum(axis=0) 
    flat_indices = np.argsort(total_attn.flatten())[::-1]
    
    print("    [🔍 모델의 결정적 판단 근거 TOP 3]")
    printed_count = 0
    for idx in flat_indices:
        atom_idx = idx // 166
        maccs_idx = idx % 166
        score = total_attn[atom_idx, maccs_idx]
        
        if score < 0.1: break
        atom_name = atom_syms[atom_idx]
        maccs_name = MACCS_NAMES.get(maccs_idx + 1, f"구조 규칙 {maccs_idx+1}")
        
        print(f"      👉 [{atom_name} 원자 주변 구조]가 '{maccs_name}' 특징과 강하게 매칭됨! (매칭 점수: {score:.4f})")
        printed_count += 1
        if printed_count >= 3: break
    if printed_count == 0:
        print("      👉 특별히 튀는 특정 구조 없이, 분자 전체적인 형태를 보고 판단함.")


def plot_mhead_mol_grid(smiles: str, attn: np.ndarray, prefix: str, is_toxic: bool):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return

    mols = [mol, mol, mol, mol]
    legends = ["Head 1", "Head 2", "Head 3", "Head 4"]
    hl_atoms_list, hl_colors_list = [], []

    for head_idx in range(4):
        imp = attn[head_idx].sum(axis=1) 
        atom_colors = {}
        highlight_atoms = []
        for i, v in enumerate(imp):
            if v >= MIN_ATTN_CUTOFF:  
                highlight_atoms.append(i)
                v_norm = min(max(v / GLOBAL_MAX_ATTN, 0.0), 1.0) 
                color_val = 1.0 - v_norm 
                
                if is_toxic:
                    atom_colors[i] = (1.0, color_val, color_val)
                else:
                    atom_colors[i] = (color_val, color_val, 1.0)
        hl_atoms_list.append(highlight_atoms)
        hl_colors_list.append(atom_colors)

    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.clearBackground = True
    drawOptions.bondLineWidth = 2
    drawOptions.highlightRadius = 0.4
    try: drawOptions.useBWAtomPalette()
    except Exception: pass 

    # 1. 일단 그리드 이미지를 바이트(bytes) 데이터로 생성
    img_data = Draw.MolsToGridImage(
        mols, molsPerRow=2, subImgSize=(400, 400), legends=legends,
        highlightAtomLists=hl_atoms_list, highlightAtomColors=hl_colors_list,
        useSVG=False, drawOptions=drawOptions, returnPNG=True
    )

    # 2. PIL 라이브러리를 사용해 사진 윗부분에 흰 여백을 만들고 SMILES 글씨 쓰기
    img = Image.open(io.BytesIO(img_data))
    w, h = img.size
    
    # 폰트 로드 (아까 도커에 설치한 글꼴 사용, 실패하면 기본 글꼴)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except:
        font = ImageFont.load_default()
        
    new_img = Image.new("RGB", (w, h + 40), "white") # 위로 40픽셀 공간 확보
    new_img.paste(img, (0, 40)) # 기존 그림은 40픽셀 밑으로 밀어서 붙임
    
    draw = ImageDraw.Draw(new_img)
    draw.text((15, 12), f"SMILES: {smiles}", fill="black", font=font) # 여백에 글씨 쓱싹

    # 3. 완성된 이미지 저장
    out_path = os.path.join(OUT_DIR, f"xai_MolGrid_{prefix}.png")
    new_img.save(out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    ckpt_path = "DGUDILI_2026/outputs/pretrained_graph_encoder.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "DGUDILI_2026/outputs/pretrained_encoder.pt"

    print("="*70)
    print("Step 12: XAI 후보군 Top 20 일괄 시각화 (SMILES 라벨링 포함)")
    print("="*70)
    encoder = load_model(ckpt_path, device)
    os.makedirs(OUT_DIR, exist_ok=True)

    def process_list(smiles_list, category_prefix, is_toxic):
        for i, smi in enumerate(smiles_list):
            rank = i + 1
            
            # 💡 파일명에 사용할 안전한 SMILES 문자열 생성 (특수문자 제거, 길이 제한)
            safe_smi = re.sub(r'[\\/*?:"<>|]', "_", smi)[:40] 
            safe_prefix = f"{category_prefix}_Rank{rank}_{safe_smi}"
            
            print(f"\n[{category_prefix} Rank {rank}] 진행 중...")
            
            try:
                attn, atom_syms = extract_attn(encoder, smi, tokenizer, device)
                print_top_reasons(attn, atom_syms)
                
                # 히트맵 이름에도 SMILES 반영
                heatmap_out = os.path.join(OUT_DIR, f"{safe_prefix}_heatmap.png")
                plot_independent_head_heatmap(attn, atom_syms, heatmap_out, top_k=10)
                plot_mhead_mol_grid(smi, attn, safe_prefix, is_toxic=is_toxic)
            except Exception as e:
                print(f"  ❌ 오류 발생 ({safe_prefix}): {e}")

    print("\n[1/2] 🔴 간독성(Toxic) 약물 Top 10 시각화 중...")
    process_list(toxic_smiles, "Toxic", is_toxic=True)

    print("\n[2/2] 🟢 비독성(Safe) 약물 Top 10 시각화 중...")
    process_list(safe_smiles, "Safe", is_toxic=False)

if __name__ == "__main__":
    main()