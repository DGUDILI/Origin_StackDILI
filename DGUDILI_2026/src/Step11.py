import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch as PyGBatch

# 커스텀 모듈 임포트
sys.path.append('DGUDILI_2026/src')
from config import MODEL_NAME, ATOM_FEAT_DIM, BOND_FEAT_DIM, MACCS_DIM, GINE_HIDDEN, GINE_LAYERS, D_MODEL, NUM_HEADS, MAX_LENGTH, DATA_PATH, FEAT_PATH
from model import GraphMACCSEncoder
from graph_utils import smiles_to_pyg, get_maccs
from utils import load_dataset

def main():
    print("="*65)
    print("Step 11: XAI 체리피킹을 위한 '완벽한 샘플' 자동 검색기")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. 모델 로드
    ckpt_path = "DGUDILI_2026/outputs/pretrained_graph_encoder.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "DGUDILI_2026/outputs/pretrained_encoder.pt"

    encoder = GraphMACCSEncoder(
        atom_feat_dim=ATOM_FEAT_DIM, bond_feat_dim=BOND_FEAT_DIM,
        maccs_dim=MACCS_DIM, gine_hidden=GINE_HIDDEN, gine_layers=GINE_LAYERS,
        d_model=D_MODEL, num_heads=NUM_HEADS,
    ).to(device)
    
    try:
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        print(f"가중치 로드 실패: {e}")
        return
        
    encoder.eval()

    # 2. 데이터 로드 (오직 테스트용 DILIrank 데이터만 필터링)
    print(f"\n데이터셋 로딩 중...")
    
    # 💡 도커 환경 경로 문제 해결 (/workspace를 현재 마운트된 /app으로 자동 변환)
    safe_data_path = DATA_PATH.replace("/workspace", "/app")
    safe_feat_path = FEAT_PATH.replace("/workspace", "/app") if FEAT_PATH else FEAT_PATH
    
    try:
        smiles_all, _, y_all, ref_all, _ = load_dataset(safe_data_path, safe_feat_path)
    except Exception as e:
        print(f"데이터를 불러오는 중 에러가 발생했습니다: {e}")
        return
        
    test_mask = (ref_all == "DILIrank")
    test_smiles = smiles_all[test_mask]
    test_y = y_all[test_mask]

    print(f"DILIrank 테스트 셋 {len(test_smiles)}개 약물에 대해 '확신도 검사'를 시작합니다...")

    results = []
    for smi, label in tqdm(zip(test_smiles, test_y), total=len(test_smiles)):
        try:
            # 입력 데이터 변환
            enc = tokenizer([smi], padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            maccs_t = torch.as_tensor(get_maccs(smi), dtype=torch.float32).unsqueeze(0).to(device)
            pyg_data = smiles_to_pyg(smi)
            if pyg_data is None: continue
            
            # 결합(Edge)이 없는 단일 원자 예외 처리 (수정된 로직 반영)
            if pyg_data.edge_index.size(1) == 0:
                pyg_data.edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)
                
            graph_batch = PyGBatch.from_data_list([pyg_data]).to(device)

            # 추론 (확률 계산)
            with torch.no_grad():
                logit = encoder(ids, mask, maccs_t, graph_batch).squeeze()
                # logit을 0~1 사이의 확률값으로 변환
                prob = torch.sigmoid(logit).item()

            # 교수님 조건: 정답을 맞힌 것 (Threshold 0.5 기준)
            is_correct = (prob >= 0.5 and label == 1) or (prob < 0.5 and label == 0)
            
            if is_correct:
                results.append({
                    "smiles": smi,
                    "label": label,
                    "prob": prob,
                    "confidence": prob if label == 1 else (1.0 - prob) # 확신도 (1.0에 가까울수록 강함)
                })
        except Exception as e:
            continue

    # 3. 확신도 기준으로 정렬
    df = pd.DataFrame(results)
    if df.empty:
        print("조건을 만족하는 샘플이 없습니다.")
        return

    df = df.sort_values(by="confidence", ascending=False)

    # 독성(1) Top 10, 비독성(0) Top 10 추출
    top_toxic = df[df["label"] == 1].head(10)
    top_nontoxic = df[df["label"] == 0].head(10)

    print("\n" + "="*70)
    print("💡 [교수님 지시사항 완료] XAI 체리피킹 후보 Top 20 추출 성공!")
    print("="*70)

    print("\n[🔴 간독성(DILI) 강하게 확신하여 정답을 맞춘 샘플 Top 10]")
    for i, row in top_toxic.iterrows():
        print(f"확률: {row['prob']:.4f} (독성 확실) | SMILES: {row['smiles']}")
        print(f"  👉 시각화 실행: docker run --rm -v \"/$(pwd):/app\" -w \"//app\" dgudili:latest python DGUDILI_2026/src/Step4_xai.py --smiles \"{row['smiles']}\"")
        print("-" * 70)

    print("\n[🟢 비독성(Non-DILI) 강하게 확신하여 정답을 맞춘 샘플 Top 10]")
    for i, row in top_nontoxic.iterrows():
        print(f"확률: {row['prob']:.4f} (안전 확실) | SMILES: {row['smiles']}")
        print(f"  👉 시각화 실행: docker run --rm -v \"/$(pwd):/app\" -w \"//app\" dgudili:latest python DGUDILI_2026/src/Step4_xai.py --smiles \"{row['smiles']}\"")
        print("-" * 70)

if __name__ == "__main__":
    main()