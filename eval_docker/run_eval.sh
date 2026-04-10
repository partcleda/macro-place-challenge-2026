#!/usr/bin/env bash
#
# Air-gapped Docker evaluation of a macro placement submission.
#
# Usage:
#   ./eval_docker/run_eval.sh <team_name> <placer_path> [extra_mount...]
#
# Examples:
#   ./eval_docker/run_eval.sh convex_opt submissions_eval/convex_opt/submissions/dccp_placer.py
#   ./eval_docker/run_eval.sh mtk submissions_eval/mtk_dreamplace_pp/submissions/placer.py \
#       submissions_eval/mtk_dreamplace_pp/submissions/dreamplace
#
# The container runs with:
#   --network none    (air-gapped, no internet)
#   --gpus all        (GPU access for DREAMPlace etc.)
#   --memory 64g      (prevent OOM from killing host)
#   --cpus 16         (resource limit)
#   timeout 7200      (2 hour wall-clock limit)

set -euo pipefail

TEAM="${1:?Usage: $0 <team_name> <placer_path> [extra_mount...]}"
PLACER_PATH="${2:?Usage: $0 <team_name> <placer_path> [extra_mount...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="macro-place-eval"
RESULTS_DIR="$REPO_ROOT/eval_docker/results"

mkdir -p "$RESULTS_DIR"

# Build image if needed
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "=== Building evaluation Docker image ==="
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
fi

# Resolve placer path
ABS_PLACER="$(cd "$REPO_ROOT" && realpath "$PLACER_PATH")"
PLACER_DIR="$(dirname "$ABS_PLACER")"
PLACER_FILE="$(basename "$ABS_PLACER")"

# Build mount arguments
# Always mount the placer's directory
MOUNT_ARGS="-v $PLACER_DIR:/submission:ro"

# Mount any extra directories (e.g., bundled DREAMPlace package)
for extra in "$@"; do
    ABS_EXTRA="$(cd "$REPO_ROOT" && realpath "$extra")"
    BASE_EXTRA="$(basename "$ABS_EXTRA")"
    MOUNT_ARGS="$MOUNT_ARGS -v $ABS_EXTRA:/submission/$BASE_EXTRA:ro"
done

echo "=== Evaluating: $TEAM ==="
echo "    Placer: $PLACER_PATH"
echo "    Image:  $IMAGE_NAME"
echo "    Network: NONE (air-gapped)"
echo ""

# Run with air-gapping and resource limits
timeout 7200 docker run --rm \
    --network none \
    --gpus all \
    --memory 64g \
    --cpus 16 \
    $MOUNT_ARGS \
    "$IMAGE_NAME" \
    "/submission/$PLACER_FILE" --all \
    2>&1 | tee "$RESULTS_DIR/${TEAM}.log"

echo ""
echo "=== Results saved to: $RESULTS_DIR/${TEAM}.log ==="
