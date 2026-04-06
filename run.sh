#!/usr/bin/env bash
# run.sh — DGUDILI_2026 (Residual/MHA 버전) 파이프라인 실행
set -e

IMAGE="dgudili:latest"
CMD="${1:-}"
ENV="${2:-env1}"   # env1 (fixed split) | env2 (10-Fold CV)

# ── 프로젝트 루트 경로 확인 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Windows Git Bash 경로 변환
case "$(uname -s)" in
    MINGW*|CYGWIN*|MSYS*)
        HOST_DIR="$(echo "$SCRIPT_DIR" | sed 's|^\([A-Za-z]\):|/\L\1|; s|\\|/|g')"
        ;;
    *)
        HOST_DIR="$SCRIPT_DIR"
        ;;
esac

# ── DGUDILI src 스크립트 실행 헬퍼 ───────────────────────────────────────────────
# $1 = 스크립트명, $2 = USE_CLEAN_DATA (0 or 1)
run_step() {
    local script="$1"
    local clean="${2:-0}"
    local tag="${clean:+  [CLEAN]}"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "[run.sh] $script${tag}"
    echo "──────────────────────────────────────────────────────────"
    docker run --rm \
        -v "${HOST_DIR}:/workspace" \
        -e STACKDILI_ROOT=/workspace \
        -e USE_CLEAN_DATA="$clean" \
        -e PYTHONUNBUFFERED=1 \
        -e KMP_DUPLICATE_LIB_OK=TRUE \
        -w /workspace/DGUDILI_2026 \
        "$IMAGE" \
        python "src/$script"
}

# ── 루트 스크립트 실행 헬퍼 ──────────────────────────────────────────────────────
run_root() {
    local script="$1"
    local clean="${2:-0}"
    local tag="${clean:+  [CLEAN]}"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "[run.sh] $script${tag}"
    echo "──────────────────────────────────────────────────────────"
    docker run --rm \
        -v "${HOST_DIR}:/workspace" \
        -e USE_CLEAN_DATA="$clean" \
        -e PYTHONUNBUFFERED=1 \
        -w /workspace \
        "$IMAGE" \
        python "$script"
}

# ── 명령어 분기 ──────────────────────────────────────────────────────────────────
case "$CMD" in
    build)
        echo "[run.sh] Docker 이미지 빌드: $IMAGE"
        docker build -t "$IMAGE" "$SCRIPT_DIR"
        echo "[run.sh] 빌드 완료."
        ;;

    run)
        if [ "$ENV" = "env2" ]; then
            echo "[run.sh] env2: 10-Fold CV, original data"
            run_step "Step_CV.py"
        else
            echo "[run.sh] env1: fixed split, original data"
            run_step "Step1_preprocess.py"
            run_step "Step2_pretrain.py"
            run_step "Step3_stacking.py"
        fi
        echo ""
        echo "[run.sh] Done."
        ;;

    run-clean)
        run_root "make_clean_data.py"
        if [ "$ENV" = "env2" ]; then
            echo "[run.sh] env2: 10-Fold CV, clean data"
            run_step "Step_CV.py" 1
        else
            echo "[run.sh] env1: fixed split, clean data"
            run_step "Step1_preprocess.py"  1
            run_step "Step2_pretrain.py"    1
            run_step "Step3_stacking.py"    1
        fi
        echo ""
        echo "[run.sh] Done."
        ;;

    step1) run_step "Step1_preprocess.py" ;;
    step2) run_step "Step2_pretrain.py"   ;;
    step3) run_step "Step3_stacking.py"   ;;

    shell)
        echo "[run.sh] 컨테이너 bash 진입..."
        docker run --rm -it \
            -v "${HOST_DIR}:/workspace" \
            -e STACKDILI_ROOT=/workspace \
            -e PYTHONUNBUFFERED=1 \
            -e KMP_DUPLICATE_LIB_OK=TRUE \
            -w /workspace/DGUDILI_2026 \
            "$IMAGE" bash
        ;;

    *)
        echo "Usage: $0 <command> [env1|env2]"
        echo ""
        echo "  build              — Docker 이미지 빌드 (최초/재빌드)"
        echo ""
        echo "  run                — env1: fixed split, original data"
        echo "  run env2           — env2: 10-Fold CV, original data"
        echo "  run-clean          — env1: fixed split, clean data"
        echo "  run-clean env2     — env2: 10-Fold CV, clean data"
        echo ""
        echo "  step1              — Step1: FP 전처리 (original)"
        echo "  step2              — Step2: E2E_MHAResidualEncoder 학습"
        echo "  step3              — Step3: Feature 추출 + Stacking + 평가"
        echo "  shell              — 컨테이너 bash 진입"
        exit 1
        ;;
esac
