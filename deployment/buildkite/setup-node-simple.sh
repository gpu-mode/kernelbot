#!/bin/bash
# Buildkite GPU Node Setup - Simplified version
# Usage: sudo BUILDKITE_AGENT_TOKEN=xxx GPU_TYPE=test ./setup-node-simple.sh

set -euo pipefail

# === CONFIGURATION ===
BUILDKITE_TOKEN="${BUILDKITE_AGENT_TOKEN:?Must set BUILDKITE_AGENT_TOKEN}"
GPU_TYPE="${GPU_TYPE:?Must set GPU_TYPE (e.g., b200, mi300, h100, test)}"
NODE_NAME="${NODE_NAME:-$(hostname)}"

# Auto-detect GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
else
    GPU_COUNT="${GPU_COUNT:-1}"
fi

echo "=== Buildkite GPU Node Setup ==="
echo "Node: ${NODE_NAME}"
echo "GPU Type: ${GPU_TYPE}"
echo "GPU Count: ${GPU_COUNT}"
echo ""

# === INSTALL BUILDKITE AGENT ===
if ! command -v buildkite-agent &> /dev/null; then
    echo "Installing Buildkite Agent..."
    apt-get update
    apt-get install -y apt-transport-https gnupg
    curl -fsSL https://keys.openpgp.org/vks/v1/by-fingerprint/32A37959C2FA5C3C99EFBC32A79206696452D198 | \
        gpg --dearmor -o /usr/share/keyrings/buildkite-agent-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/buildkite-agent-archive-keyring.gpg] https://apt.buildkite.com/buildkite-agent stable main" | \
        tee /etc/apt/sources.list.d/buildkite-agent.list
    apt-get update
    apt-get install -y buildkite-agent
fi

# === STOP EXISTING AGENTS ===
echo "Stopping existing agents..."
systemctl stop buildkite-agent 2>/dev/null || true
for i in $(seq 0 15); do
    systemctl stop "buildkite-agent-gpu${i}" 2>/dev/null || true
    systemctl disable "buildkite-agent-gpu${i}" 2>/dev/null || true
done

# === CREATE DIRECTORIES ===
mkdir -p /var/lib/buildkite-agent/builds
chown -R buildkite-agent:buildkite-agent /var/lib/buildkite-agent

# === CONFIGURE GIT TO USE HTTPS ===
sudo -u buildkite-agent git config --global url."https://github.com/".insteadOf "git@github.com:"

# === CREATE AGENT FOR EACH GPU ===
echo "Creating ${GPU_COUNT} agents..."

for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
    agent_name="${NODE_NAME}-gpu${gpu_idx}"
    config_file="/etc/buildkite-agent/buildkite-agent-gpu${gpu_idx}.cfg"
    build_dir="/var/lib/buildkite-agent/builds/gpu${gpu_idx}"

    mkdir -p "${build_dir}"
    chown buildkite-agent:buildkite-agent "${build_dir}"

    # Write config
    cat > "${config_file}" << EOF
token="${BUILDKITE_TOKEN}"
name="${agent_name}"
tags="queue=${GPU_TYPE},gpu=${GPU_TYPE},gpu-index=${gpu_idx},node=${NODE_NAME}"
build-path="${build_dir}"
hooks-path="/etc/buildkite-agent/hooks"
EOF

    # Write systemd service
    cat > "/etc/systemd/system/buildkite-agent-gpu${gpu_idx}.service" << EOF
[Unit]
Description=Buildkite Agent (GPU ${gpu_idx})
Documentation=https://buildkite.com/docs/agent/v3
After=network.target

[Service]
Type=simple
User=buildkite-agent
Environment=NVIDIA_VISIBLE_DEVICES=${gpu_idx}
Environment=CUDA_VISIBLE_DEVICES=${gpu_idx}
ExecStart=/usr/bin/buildkite-agent start --config ${config_file}
RestartSec=5
Restart=on-failure
TimeoutStartSec=10
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

    echo "  Created agent ${gpu_idx}: GPU=${gpu_idx}"
done

# === START AGENTS ===
echo "Starting agents..."
systemctl daemon-reload

for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
    systemctl enable "buildkite-agent-gpu${gpu_idx}"
    systemctl start "buildkite-agent-gpu${gpu_idx}"
done

sleep 3

echo ""
echo "=== Agent Status ==="
for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
    status=$(systemctl is-active "buildkite-agent-gpu${gpu_idx}" 2>/dev/null || echo "unknown")
    echo "  GPU ${gpu_idx}: ${status}"
done

echo ""
echo "=== Setup Complete ==="
echo "Created ${GPU_COUNT} agents for queue: ${GPU_TYPE}"
echo "Each agent sees only its assigned GPU via NVIDIA_VISIBLE_DEVICES"
echo ""
echo "Check agents at: https://buildkite.com/organizations/YOUR_ORG/agents"
