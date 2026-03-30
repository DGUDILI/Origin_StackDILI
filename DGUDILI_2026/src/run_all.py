import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(ROOT, "src")
DATA_DIR   = os.path.join(ROOT, "data")
PYTHON     = sys.executable


def run(script, desc):
    print(f"\n{'='*60}")
    print(f">> {desc}")
    print(f"{'='*60}")
    r = subprocess.run(
        [PYTHON, os.path.join(SRC_DIR, script)],
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    if r.returncode != 0:
        print(f"[ERROR] {script} failed (exit code {r.returncode})")
        sys.exit(r.returncode)


for req in [
    r"C:\DGUDILI\Origin_StackDILI\Data\Dataset.csv",
    r"C:\DGUDILI\Origin_StackDILI\Code\Dataset_feature.csv",
]:
    if not os.path.exists(req):
        print(f"[ERROR] Missing prerequisite: {req}")
        sys.exit(1)

emb_path = os.path.join(DATA_DIR, "chemberta_embeddings.npy")
if os.path.exists(emb_path):
    print(f"\n[SKIP] Step 1 -- {emb_path} already exists")
else:
    run("Step1_chemberta_embed.py", "Step 1: ChemBERTa Embedding Extraction")

run("Step2_feature_select.py", "Step 2: Feature Selection (k=16)")
run("Step3_pretrain.py",       "Step 3: Cross-Attention Pre-training")
run("Step4_extract_and_lr.py", "Step 4: Feature Extraction + LR + Evaluation")

print(f"\n{'='*60}")
print("Pipeline complete.")
print(f"Results: {os.path.join(ROOT, 'outputs', 'results_comparison.csv')}")
print(f"Model:   {os.path.join(ROOT, 'outputs', 'pretrained_encoder.pt')}")
print(f"{'='*60}")
