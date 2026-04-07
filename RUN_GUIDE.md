# DGUDILI 2026 — 실행 가이드

## Docker로 실행

```bash
cd Origin_StackDILI

# 최초 1회 빌드 (torch-geometric, einops 포함)
bash run.sh build
```

### GraphMACCSEncoder (현재 파이프라인)

| 명령어 | 설명 |
|---|---|
| `bash run.sh run` | env1 — 원본 데이터, fixed split (Step1→2→3) |
| `bash run.sh run-clean` | env1 — clean 데이터, fixed split |
| `bash run.sh run env2` | env2 — 원본 데이터, 10-fold CV |
| `bash run.sh run-clean env2` | env2 — clean 데이터, 10-fold CV |
| `bash run.sh run-graph` | env1 — 원본 데이터, fixed split (run과 동일) |

> `run`과 `run-graph`는 동일한 GraphMACCSEncoder 파이프라인을 실행합니다.  
> env2는 env1 실행 후 사용 가능 (`pretrained_graph_encoder.pt` 필요)

### 개별 스텝 / XAI

| 명령어 | 설명 |
|---|---|
| `bash run.sh step1` | Step1: FP/Graph/MACCS 전처리 |
| `bash run.sh step2` | Step2: GraphMACCSEncoder 학습만 |
| `bash run.sh step3` | Step3: Feature 추출 + Stacking + 평가만 |
| `bash run.sh step4` | XAI 히트맵 — 아스피린 (기본값) |
| `bash run.sh step4 "CC(=O)O"` | XAI 히트맵 — 지정 SMILES |
| `bash run.sh shell` | 컨테이너 bash 진입 |

> WSL에서 `./run.sh` 또는 Git Bash에서 `bash run.sh`로 실행 가능

---

## 로컬에서 직접 실행

패키지 설치:
```bash
pip install torch torch-geometric einops --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 scikit-learn==1.7.1 xgboost transformers pandas rdkit
```

실행:
```bash
cd Origin_StackDILI
export STACKDILI_ROOT=$(pwd)   # Windows: $env:STACKDILI_ROOT = (Get-Location).Path

# env1 (순서대로 실행)
python DGUDILI_2026/src/Step1_preprocess.py
python DGUDILI_2026/src/Step2_pretrain.py
python DGUDILI_2026/src/Step3_stacking.py

# env2 (env1 실행 후)
python DGUDILI_2026/src/Step_CV.py
```

clean 데이터로 실행하려면:
```bash
export USE_CLEAN_DATA=1   # Windows: $env:USE_CLEAN_DATA = "1"
```

Step 1은 `train_graphs.pt` / `test_graphs.pt` 캐시가 존재하면 자동으로 스킵됩니다.
강제 재실행이 필요한 경우:
```bash
python DGUDILI_2026/src/Step1_preprocess.py --force
```
