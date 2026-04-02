"""
DGUDILI 2026 - Model Runner

사용법:
  python run.py prev        DGUDILI_2026_prev 학습 + 평가
  python run.py v1          GroupCrossAttn v1 학습 + 평가
  python run.py ensemble    Seed Ensemble (x5) 학습 + 평가
  python run.py modeB       Mode B 학습 + 평가
  python run.py modeB32     Mode B-32 학습 (d_out=32, dropout=0.30, label_smooth)
  python run.py ensemble32  Mode B-32 Seed Ensemble x5 학습 + 평가
  python run.py swa32       Mode B-32 + SWA 학습 + 평가
  python run.py compare     저장된 모든 모델 비교 (재학습 없음)
"""

import os, sys, subprocess, argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")

MODELS = {
    "prev":       ["Step3_pretrain.py", "Step4_extract_and_lr.py"],
    "v1":         ["Step5_group_train.py", "Step6_group_eval.py"],
    "ensemble":   ["Step7_seed_ensemble.py"],
    "modeB":      ["Step8_train_modeB.py", "Step9_eval_modeB.py"],
    "modeB32":    ["Step10_train_modeB32.py"],
    "ensemble32": ["Step11_ensemble_modeB32.py"],
    "swa32":      ["Step12_swa_modeB32.py"],
    "compare":    ["compare_all.py"],
}


def run(script: str):
    path = os.path.join(SRC_DIR, script)
    print(f"\n>>> {script}")
    print("-" * 50)
    ret = subprocess.run([sys.executable, path], cwd=ROOT)
    if ret.returncode != 0:
        print(f"[ERROR] {script} failed (exit code {ret.returncode})")
        sys.exit(ret.returncode)


parser = argparse.ArgumentParser(
    description="DGUDILI 2026 Model Runner",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="\n".join(f"  python run.py {k:<10} {MODELS[k]}" for k in MODELS),
)
parser.add_argument(
    "model",
    choices=list(MODELS.keys()),
    help="실행할 모델 선택",
)
args = parser.parse_args()

# prev는 Step2(feature select)가 먼저 필요
if args.model == "prev":
    fp_k16 = os.path.join(ROOT, "data", "fp_k16_train.npy")
    if not os.path.exists(fp_k16):
        run("Step2_feature_select.py")

for script in MODELS[args.model]:
    run(script)
