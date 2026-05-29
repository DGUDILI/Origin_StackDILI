import torch
import os

# 검사할 파일 목록 (가중치 파일들이 있는 경로로 수정하세요)
files_to_check = [
    "weights/pretrained_graph_encoder.pt",
    "weights/pretrained_encoder.pt",
    "weights/pretrained_encoder_e2e_cls.pt",
    "weights/pretrained_encoder_tune_cls.pt"
]

for file_path in files_to_check:
    if not os.path.exists(file_path):
        print(f"⏩ {file_path} 파일이 존재하지 않아 건너뜁니다.")
        continue
        
    print(f"\n==================================================")
    print(f"🔍 검사 중: {file_path}")
    print(f"==================================================")
    try:
        sd = torch.load(file_path, map_location="cpu")
        keys = list(sd.keys())
        
        # 1. GNN 종류 판별
        has_gine = any("gine" in k for k in keys)
        has_sage = any("sage" in k for k in keys)
        gnn_type = "GINE" if has_gine else ("SAGE" if has_sage else "없음(MLP/Language 단독)")
        
        # 2. 최종 분류 헤드(head) 및 차원 확인
        head_shape = None
        for k in keys:
            if "head.weight" in k:
                head_shape = sd[k].shape
                break
                
        print(f"📈 그래프 레이어 타입: {gnn_type}")
        print(f"🎯 최종 분류 헤드(head.weight) 크기: {head_shape}")
        print(f"📝 포함된 주요 레이어 예시 (Top 5):")
        for i, k in enumerate(keys[:5]):
            print(f"   - {k}")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")