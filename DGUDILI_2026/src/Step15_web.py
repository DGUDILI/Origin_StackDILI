import os
import sys
import torch
import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from transformers import AutoTokenizer

# 현재 경로를 Python path에 추가
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 업로드된 파일 모듈 임포트
from config import (
    MODEL_NAME, MAX_LENGTH, D_MODEL, NUM_HEADS, K,
    GINE_LAYERS, GINE_HIDDEN, ATOM_FEAT_DIM, BOND_FEAT_DIM
)
from model import GraphMACCSEncoder
from graph_utils import smiles_to_pyg, get_maccs, MACCS_NAMES
from features import collate_fn

# ==========================================
# 1. 모델 로드 함수 (캐싱 적용)
# ==========================================
@st.cache_resource
def load_full_stack_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Encoder 로드 (Pretrained weights)
    encoder = GraphMACCSEncoder(
        model_name=MODEL_NAME,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        gine_layers=GINE_LAYERS,
        gine_hidden=GINE_HIDDEN,
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        k=K
    ).to(device)
    
    # 실제 환경에선 저장된 .pt 경로를 지정하세요.
    ckpt_path = os.path.join(os.path.dirname(SRC_DIR), "outputs", "pretrained_graph_encoder.pt")
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.eval()
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    return encoder, tokenizer, device

# ==========================================
# 2. 분석 보조 함수 (시각화 & ADME)
# ==========================================
def get_mol_svg(smiles, weights, prob):
    """원자별 가중치 시각화 (Step13 로직 적용)"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # 모든 원자 기호가 표시되도록 강제 설정
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", atom.GetSymbol())

    num_atoms = mol.GetNumAtoms()
    weights = weights[:num_atoms]
    
    # 가중치 정규화 (0~1)
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-9:
        norm_w = np.zeros(num_atoms)
    else:
        norm_w = (weights - w_min) / (w_max - w_min)
    
    is_toxic = prob > 45.0 # 기준치
    highlight_atoms = []
    atom_colors = {}
    atom_radii = {}
    
    for i in range(num_atoms):
        v_norm = float(norm_w[i])
        if v_norm > 0.15: # Cutoff
            highlight_atoms.append(i)
            color_val = 1.0 - v_norm
            
            if is_toxic: # 독성: 빨간색 계열
                atom_colors[i] = (1.0, color_val, color_val)
            else: # 안전: 파란색 계열
                atom_colors[i] = (color_val, color_val, 1.0)
            
            atom_radii[i] = 0.35
            
    # 이미지 사이즈를 키우고 퀄리티 높임
    d2d = rdMolDraw2D.MolDraw2DSVG(600, 500)
    options = d2d.drawOptions()
    options.useBWAtomPalette() # 기호 가독성 향상
    options.bondLineWidth = 2.5
    options.clearBackground = False # 배경 투명하게
    
    rdMolDraw2D.PrepareAndDrawMolecule(
        d2d, mol, 
        highlightAtoms=highlight_atoms, 
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def get_adme_properties(smiles):
    """RDKit 기반 화학적 특성 추출"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return {}
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP (Solubility)": round(Descriptors.MolLogP(mol), 2),
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol)
    }

# ==========================================
# 3. 예측 실행 함수 (판단 근거 추출 포함)
# ==========================================
def run_prediction(smiles):
    encoder, tokenizer, device = load_full_stack_model()
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, None, None
    
    # 데이터 전처리
    pyg_data = smiles_to_pyg(smiles)
    maccs_vec = get_maccs(smiles)
    
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    mac = maccs_vec.unsqueeze(0).to(device)
    
    from torch_geometric.data import Batch
    grph = Batch.from_data_list([pyg_data]).to(device)
    
    with torch.no_grad():
        features = encoder(ids, mask, mac, grph)
        prob = torch.sigmoid(features.sum()).item() * 100 
        
        # [핵심] 판단 근거 추출 (MACCS Attention)
        attn = encoder._last_attn_weights[0] # (num_atoms, 167)
        maccs_importance = attn.sum(dim=0).cpu().numpy()
        top_indices = np.argsort(maccs_importance)[-3:][::-1]
        
        reasons = []
        for idx in top_indices:
            if maccs_importance[idx] > 0.01:
                name = MACCS_NAMES.get(idx, f"Structure bit {idx}")
                reasons.append(name)
                
        atom_weights = attn.sum(dim=1).cpu().numpy()
        
    return prob, atom_weights, reasons

# ==========================================
# 4. Streamlit UI 구성 (대시보드 스타일)
# ==========================================
st.set_page_config(page_title="MACDADILI 2026 | Tox Analyzer", page_icon="🧬", layout="wide")

# CSS 커스텀 스타일링 (배경, 카드, 텍스트 정렬)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .css-1v0mbdj {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 헤더 섹션
st.markdown('<p class="main-header">🧬 MACDADILI 2026: DILI Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Drug-Induced Liver Injury (DILI) Prediction using Graph & MACCS Stacking Model</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Single Molecule Analysis", "📂 Massive Batch Screening"])

with tab1:
    # 상단: 입력부
    with st.container():
        st.markdown("### Input Compound")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            smi_input = st.text_input("Enter SMILES String:", "CC(=O)Nc1ccc(O)cc1", label_visibility="collapsed")
        with col_btn:
            analyze_btn = st.button("🚀 Analyze Molecule", use_container_width=True, type="primary")
            
    st.divider()

    # 하단: 결과창 (버튼을 눌렀을 때만 표시)
    if analyze_btn:
        with st.spinner("Analyzing molecular structure & extracting MACCS features..."):
            prob, weights, reasons = run_prediction(smi_input)
            
            if prob is None:
                st.error("❌ Invalid SMILES format. Please check the structure.")
            else:
                is_toxic = prob > 45.0
                color_hex = "#EF4444" if is_toxic else "#10B981" # Red or Green
                bg_color = "rgba(239, 68, 68, 0.1)" if is_toxic else "rgba(16, 185, 129, 0.1)"
                status_text = "🚨 TOXIC (High Risk)" if is_toxic else "✅ SAFE (Low Risk)"
                
                # 좌우 영역 분리 (비율 4.5 : 5.5)
                col_left, col_right = st.columns([4.5, 5.5], gap="large")
                
                with col_left:
                    st.markdown("### Prediction Result")
                    # 확률 박스 디자인
                    st.markdown(f"""
                    <div style="padding:25px; border-radius:15px; border:2px solid {color_hex}; background-color:{bg_color}; text-align:center;">
                        <h4 style="color:#374151; margin:0; padding-bottom:10px;">DILI Probability</h4>
                        <h1 style="color:{color_hex}; font-size:3.5rem; margin:0;">{prob:.1f}%</h1>
                        <p style="font-size:1.2rem; font-weight:600; color:{color_hex}; margin-top:10px;">{status_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(prob / 100.0) # 프로그레스 바 추가
                    
                    # 판단 근거 박스
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 💡 Structural Tox Reasons")
                    if reasons:
                        for r in reasons:
                            # Streamlit의 기본 info 박스를 사용해 깔끔하게 정렬
                            st.info(f"**Detected Motif:** {r}", icon="🔬")
                    else:
                        st.success("No specific high-risk toxic motifs detected.", icon="✨")

                with col_right:
                    st.markdown("### Attention Weight Map")
                    # 모델이 집중한 부위 (SVG)
                    svg = get_mol_svg(smi_input, weights, prob)
                    # SVG를 중앙 정렬하여 렌더링
                    st.markdown(f'<div style="display: flex; justify-content: center; background-color: white; border-radius: 15px; border: 1px solid #E5E7EB; padding: 10px;">{svg}</div>', unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 ADME Properties")
                # ADME 속성을 4개의 예쁜 메트릭 타일로 분할
                adme = get_adme_properties(smi_input)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(label="⚖️ Molecular Weight", value=f"{adme['Molecular Weight']} g/mol")
                m2.metric(label="💧 LogP (Solubility)", value=adme["LogP (Solubility)"])
                m3.metric(label="🔼 H-Bond Donors", value=adme["H-Bond Donors"])
                m4.metric(label="🔽 H-Bond Acceptors", value=adme["H-Bond Acceptors"])

with tab2:
    st.markdown("### 📂 Massive Batch Screening")
    st.write("Upload a CSV file containing multiple compounds for rapid batch DILI prediction. **The file must contain a column named `SMILES`.**")
    
    uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            st.success(f"File loaded successfully! Found {len(df)} compounds.")
            if st.button("Start Batch Prediction", type="primary"):
                with st.spinner("Processing massive screening..."):
                    probs = []
                    results = []
                    
                    # 진행률 바
                    progress_bar = st.progress(0)
                    total = len(df['SMILES'])
                    
                    for i, s in enumerate(df['SMILES']):
                        p, _, _ = run_prediction(str(s))
                        if p is not None:
                            probs.append(round(p, 2))
                            results.append("High Risk" if p > 45.0 else "Safe")
                        else:
                            probs.append("Error")
                            results.append("Error")
                        # 진행률 업데이트
                        progress_bar.progress((i + 1) / total)
                            
                    df['DILI_Prob(%)'] = probs
                    df['Prediction (Threshold=45%)'] = results
                    
                    st.markdown("### 📈 Screening Results")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Results as CSV",
                        data=csv,
                        file_name='DILI_batch_results.csv',
                        mime='text/csv',
                    )
        else:
            st.error("❌ The uploaded CSV must contain a column named 'SMILES'.")