import os
import sys
import torch
import numpy as np
import pandas as pd
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

# 가중치 튜닝 수치
GLOBAL_MAX_ATTN = 0.25  
MIN_ATTN_CUTOFF = 0.05  

def plot_mhead_mol_grid(smiles: str, attn: np.ndarray, prefix: str, out_dir: str, is_toxic: bool):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return

    mols = [mol, mol, mol, mol]
    legends = ["Head 1", "Head 2", "Head 3", "Head 4"]
    hl_atoms_list, hl_colors_list = [], []

    for head_idx in range(4):
        # 해당 헤드의 원자별 중요도 합산
        imp = attn[head_idx].sum(axis=1) # (n_atoms,)
        
        # 1. 상대적 스케일링을 위해 현재 헤드의 최대/최소값 파악
        max_v = imp.max()
        min_v = imp.min()
        v_range = max_v - min_v if max_v > min_v else 1.0
        
        atom_colors = {}
        highlight_atoms = []
        
        for i, v in enumerate(imp):
            # 2. 선형 정규화 (0.0 ~ 1.0)
            # 이 분자 내에서 가장 높은 놈은 1.0, 가장 낮은 놈은 0.0이 됨
            v_norm = (v - min_v) / v_range
            
            # 3. 시각적 강조를 위해 비선형 가중치 적용 (선택 사항)
            # v_norm = v_norm ** 2  # 이 줄을 활성화하면 상위권만 더 진하게 보임
            
            highlight_atoms.append(i)
            
            # 4. 색상 계산 (v_norm이 1일수록 color_val이 낮아져서 진한 색이 됨)
            # 기본 배경을 연하게 깔고 싶으면 0.8 등을 곱해 조절
            color_val = 1.0 - (v_norm * 0.9) 
            
            if is_toxic:
                # 중요할수록 진한 빨강 (R=1, G/B는 낮게)
                atom_colors[i] = (1.0, color_val, color_val)
            else:
                # 중요할수록 진한 파랑 (B=1, R/G는 낮게)
                atom_colors[i] = (color_val, color_val, 1.0)
        
        hl_atoms_list.append(highlight_atoms)
        hl_colors_list.append(atom_colors)

    # RDKit 그리기 옵션 설정
    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = True
    drawOptions.fixedFontSize = 14
    
    # 이미지 생성
    img_data = Draw.MolsToGridImage(
        mols, molsPerRow=2, subImgSize=(500, 500), legends=legends,
        highlightAtomLists=hl_atoms_list, highlightAtomColors=hl_colors_list,
        useSVG=False, drawOptions=drawOptions, returnPNG=True
    )
    # ... (이하 PIL 저장 로직 동일)

    img = Image.open(io.BytesIO(img_data))
    w, h = img.size
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except: font = ImageFont.load_default()
        
    new_img = Image.new("RGB", (w, h + 40), "white")
    new_img.paste(img, (0, 40)) 
    
    draw = ImageDraw.Draw(new_img)
    draw.text((15, 12), f"SMILES: {smiles}", fill="black", font=font) 

    out_path = os.path.join(out_dir, f"{prefix}_structure.png")
    new_img.save(out_path)


def process_list(smiles_list, category_prefix, is_toxic, encoder, tokenizer, device, mass_out_dir):
    """리스트를 순회하며 터미널 출력 및 이미지 저장"""
    for i, smi in enumerate(smiles_list):
        rank = i + 1
        
        # 파일명용: 특수문자 제거 및 길이 제한
        safe_smi = re.sub(r'[\\/*?:"<>|]', "_", smi)[:40] 
        safe_prefix = f"{category_prefix}_Rank{rank:03d}_{safe_smi}" # 100단위가 될 수 있으므로 03d로 수정
        
        # 🟢 [터미널 출력] 여기서 전체 SMILES가 찍힙니다!
        print(f"\n" + "="*60)
        print(f">>> [{category_prefix} Rank {rank:03d}] 원본 데이터 정보")
        print(f"FULL_SMILES: {smi}")
        print(f"SAVING_AS  : {safe_prefix}...")
        print("="*60)
        
        try:
            # Attention 데이터 추출
            attn, atom_syms = extract_attn(encoder, smi, tokenizer, device)
            
            # 💡 [핵심 추가] 독성 여부에 따라 히트맵 색상 맵 결정
            heatmap_cmap = "Reds" if is_toxic else "Blues"
            
            # 1. 히트맵 저장 (색상 파라미터 전달)
            heatmap_out = os.path.join(mass_out_dir, f"{safe_prefix}_heatmap.png")
            plot_independent_head_heatmap(attn, atom_syms, heatmap_out, top_k=10, cmap=heatmap_cmap)
            
            # 2. 분자 구조 오버레이 저장
            plot_mhead_mol_grid(smi, attn, safe_prefix, mass_out_dir, is_toxic)
            
            print(f"✅ Rank {rank:03d} 처리 및 이미지 저장 완료.")
            
        except Exception as e:
            print(f"      [❌ Error] Rank {rank:03d} 처리 중 오류 발생: {e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    ckpt_path = "DGUDILI_2026/outputs/pretrained_graph_encoder.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "outputs/pretrained_graph_encoder.pt"

    print("="*70)
    print("Step 13: 대규모 XAI 시각화 (Top 100 Toxic & Safe)")
    print("="*70)
    
    csv_path = os.path.join(OUT_DIR, "test_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} 파일이 없습니다! Step3를 먼저 다시 실행해주세요.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 데이터 필터링 (True Positive & True Negative)
    correct_toxic = df[(df["True_Label"] == 1) & (df["Probability"] >= 0.5)]
    correct_safe = df[(df["True_Label"] == 0) & (df["Probability"] < 0.5)]

    # 💡 50 -> 100 으로 변경
    toxic_100 = correct_toxic.sort_values(by="Probability", ascending=False).head(100)["SMILES"].tolist()
    safe_100 = correct_safe.sort_values(by="Probability", ascending=True).head(100)["SMILES"].tolist()

    encoder = load_model(ckpt_path, device)
    
    mass_out_dir = os.path.join(OUT_DIR, "Mass_XAI_Results")
    os.makedirs(mass_out_dir, exist_ok=True)

    # 🟢 [실행부] main 함수 안에서 호출해야 모든 변수를 인식합니다!
    print(f"\n[1/2] 🔴 간독성(Toxic) 약물 Top {len(toxic_100)} 시각화 시작...")
    process_list(toxic_100, "Toxic", True, encoder, tokenizer, device, mass_out_dir)

    print(f"\n[2/2] 🟢 비독성(Safe) 약물 Top {len(safe_100)} 시각화 시작...")
    process_list(safe_100, "Safe", False, encoder, tokenizer, device, mass_out_dir)
    
    print(f"\n✨ 모든 작업 완료! 총 이미지가 {mass_out_dir} 에 저장되었습니다.")

if __name__ == "__main__":
    main()