#!/usr/bin/env bash
# run.sh — DGUDILI_2026 pipeline launcher (Mac / Linux / Windows Git Bash)
set -e

IMAGE="dgudili:latest"
CMD="${1:-}"
ENV="${2:-env1}"   # env1 (fixed split) | env2 (10-fold CV)

# ── Resolve project root (the directory containing this script) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On Windows Git Bash, convert C:/... to Docker-compatible path
case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*)
        HOST_DIR="$(echo "$SCRIPT_DIR" | sed 's|^\([A-Za-z]\):|/\L\1|; s|\\|/|g')"
        ;;
    *)
        HOST_DIR="$SCRIPT_DIR"
        ;;
esac

# ── Helper: run a DGUDILI src script ────────────────────────────────────────
# $1 = script name, $2 = USE_CLEAN_DATA (0 or 1, default 0)
run_step() {
    local script="$1"
    local clean="${2:-0}"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "[run.sh] $script${clean:+  [CLEAN]}"
    echo "──────────────────────────────────────────────────────────"
    docker run --rm \
        -v "${HOST_DIR}:/workspace" \
        -e STACKDILI_ROOT=/workspace \
        -e USE_CLEAN_DATA="$clean" \
        -e KMP_DUPLICATE_LIB_OK=TRUE \
        -e PYTHONUNBUFFERED=1 \
        -w /workspace/DGUDILI_2026 \
        "$IMAGE" \
        python "src/$script"
}

# ── Helper: run a root-level script ─────────────────────────────────────────
# $1 = script name, $2 = USE_CLEAN_DATA (0 or 1, default 0)
run_root() {
    local script="$1"
    local clean="${2:-0}"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "[run.sh] $script${clean:+  [CLEAN]}"
    echo "──────────────────────────────────────────────────────────"
    docker run --rm \
        -v "${HOST_DIR}:/workspace" \
        -e USE_CLEAN_DATA="$clean" \
        -e PYTHONUNBUFFERED=1 \
        -w /workspace \
        "$IMAGE" \
        python "$script"
}

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$CMD" in
    build)
        echo "[run.sh] Building Docker image: $IMAGE"
        docker build -t "$IMAGE" "$SCRIPT_DIR"
        echo "[run.sh] Build complete."
        ;;

    run)
        if [ "$ENV" = "env2" ]; then
            echo "[run.sh] env2: 10-Fold CV, original data"
            run_step "Step1_chemberta_embed.py"
            run_root "compute_stackdili_cv_baseline.py"
            run_step "Step_CV.py"
        else
            echo "[run.sh] env1: fixed split, original data"
            run_step "Step1_chemberta_embed.py"
            run_step "Step2_feature_select.py"
            run_step "Step3_pretrain.py"
            run_step "Step4_extract_and_lr.py"
        fi
        echo ""
        echo "[run.sh] Done."
        ;;

    run-clean)
        run_root "make_clean_data.py"
        if [ "$ENV" = "env2" ]; then
            echo "[run.sh] env2: 10-Fold CV, clean data"
            run_step "Step1_chemberta_embed.py"               1
            run_root "compute_stackdili_cv_baseline.py"       1
            run_step "Step_CV.py"                             1
        else
            echo "[run.sh] env1: fixed split, clean data"
            run_root "compute_stackdili_clean_baseline.py"
            run_step "Step1_chemberta_embed.py"               1
            run_step "Step2_feature_select.py"                1
            run_step "Step3_pretrain.py"                      1
            run_step "Step4_extract_and_lr.py"                1
        fi
        echo ""
        echo "[run.sh] Done."
        ;;

    step1) run_step "Step1_chemberta_embed.py" ;;
    step2) run_step "Step2_feature_select.py"  ;;
    step3) run_step "Step3_pretrain.py"        ;;
    step4) run_step "Step4_extract_and_lr.py"  ;;

    shell)
        echo "[run.sh] Opening interactive shell..."
        docker run --rm -it \
            -v "${HOST_DIR}:/workspace" \
            -e STACKDILI_ROOT=/workspace \
            -e KMP_DUPLICATE_LIB_OK=TRUE \
            -e PYTHONUNBUFFERED=1 \
            -w /workspace/DGUDILI_2026 \
            "$IMAGE" bash
        ;;

    *)
        echo "Usage: $0 <command> [env1|env2]"
        echo ""
        echo "  build            — Docker 이미지 빌드 (최초/재빌드)"
        echo ""
        echo "  run              — env1: fixed split, 원본 데이터"
        echo "  run env2         — env2: 10-Fold CV, 원본 데이터"
        echo "  run-clean        — env1: fixed split, 중복 제거 데이터"
        echo "  run-clean env2   — env2: 10-Fold CV, 중복 제거 데이터"
        echo ""
        echo "  * env1: train(non-DILIrank) / test(DILIrank) 고정 분할"
        echo "  * env2: 전체 데이터 10-Fold CV (StackDILI도 동일 방식으로 재학습)"
        echo ""
        echo "  step1~4          — 개별 스텝 실행 (env1, 원본)"
        echo "  shell            — 컨테이너 bash 진입"
        exit 1
        ;;
esac
