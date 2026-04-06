# DGUDILI 2026 — 실행 가이드

## Docker로 실행

```bash
cd Origin_StackDILI

# 최초 1회 빌드
bash run.sh build
```

| 명령어 | 설명 |
|---|---|
| `bash run.sh run` | env1 — 원본 데이터, fixed split |
| `bash run.sh run-clean` | env1 — clean 데이터, fixed split |
| `bash run.sh run env2` | env2 — 원본 데이터, 10-fold CV |
| `bash run.sh run-clean env2` | env2 — clean 데이터, 10-fold CV |

> env2는 env1 실행 후 사용 가능 (encoder 파일 필요)

---

## 로컬에서 직접 실행

패키지 설치:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
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
