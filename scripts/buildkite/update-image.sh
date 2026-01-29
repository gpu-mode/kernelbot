#!/bin/bash
#
# Update Kernelbot Runner Image
#
# This script pulls the latest runner image and restarts agents to use it.
# Run this when notified of a new image release.
#
# Usage:
#   sudo ./update-image.sh
#

set -euo pipefail

IMAGE="ghcr.io/gpu-mode/kernelbot-runner:latest"

echo "Pulling latest kernelbot runner image..."
docker pull "$IMAGE"

echo ""
echo "Image updated. Checking for running agents..."

# Find all buildkite-agent-gpu* services
AGENTS=$(systemctl list-units --type=service --state=running --no-legend | grep 'buildkite-agent-gpu' | awk '{print $1}' || true)

if [[ -z "$AGENTS" ]]; then
    echo "No running Buildkite GPU agents found."
    echo "Image will be used on next job run."
else
    echo "Found running agents:"
    echo "$AGENTS"
    echo ""
    read -p "Restart agents to use new image? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for agent in $AGENTS; do
            echo "Restarting $agent..."
            systemctl restart "$agent"
        done
        echo ""
        echo "✅ All agents restarted with new image."
    else
        echo "Agents will use new image on next job run."
    fi
fi

echo ""
echo "Current image info:"
docker inspect "$IMAGE" --format='ID: {{.Id}}'
docker inspect "$IMAGE" --format='Created: {{.Created}}'
docker inspect "$IMAGE" --format='Labels: {{json .Config.Labels}}'
