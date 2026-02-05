#!/bin/bash
# Build the kernelbot Docker image locally on a GPU node
# Usage: ./build-image.sh [--push]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="${KERNELBOT_IMAGE:-kernelbot:latest}"
BRANCH="${KERNELBOT_BRANCH:-buildkite-infrastructure}"

echo "=== Building Kernelbot Image ==="
echo "Image: $IMAGE_NAME"
echo "Branch: $BRANCH"
echo ""

# Update Dockerfile to use correct branch
sed -i "s|--branch [a-zA-Z0-9_-]*|--branch $BRANCH|g" "$SCRIPT_DIR/Dockerfile" 2>/dev/null || \
    sed -i '' "s|--branch [a-zA-Z0-9_-]*|--branch $BRANCH|g" "$SCRIPT_DIR/Dockerfile"

echo "Building image..."
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo ""
echo "=== Build Complete ==="
echo "Image: $IMAGE_NAME"
docker images "$IMAGE_NAME"

# Optional: push to registry
if [[ "${1:-}" == "--push" ]]; then
    REGISTRY="${KERNELBOT_REGISTRY:-ghcr.io/gpu-mode}"
    REMOTE_IMAGE="$REGISTRY/kernelbot:latest"
    echo ""
    echo "Pushing to $REMOTE_IMAGE..."
    docker tag "$IMAGE_NAME" "$REMOTE_IMAGE"
    docker push "$REMOTE_IMAGE"
    echo "Pushed: $REMOTE_IMAGE"
fi

echo ""
echo "To use this image, update your pipeline config:"
echo "  image: \"$IMAGE_NAME\""
