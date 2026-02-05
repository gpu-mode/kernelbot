#!/bin/bash
# Buildkite GPU Node Setup
# Usage: sudo BUILDKITE_AGENT_TOKEN=xxx GPU_TYPE=test ./setup-node-simple.sh
#
# PREREQUISITES:
#   1. Create a Buildkite account and pipeline named 'kernelbot'
#   2. Generate an Agent Token from: Agents > Agent Tokens
#   3. Create a queue in: Agents > Default cluster > Queues > New Queue
#      - Enter your GPU_TYPE as the key (e.g., 'test', 'b200', 'h100')
#      - Select 'Self hosted'
#   4. Run this script with the token and GPU type

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

# === CHECK ROOT ===
if [[ $EUID -ne 0 ]]; then
   echo "ERROR: This script must be run as root (use sudo)"
   exit 1
fi

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
    echo "Buildkite Agent installed."
else
    echo "Buildkite Agent already installed."
fi

# === STOP EXISTING AGENTS ===
echo "Stopping existing agents..."
systemctl stop buildkite-agent 2>/dev/null || true
systemctl disable buildkite-agent 2>/dev/null || true
for i in $(seq 0 15); do
    systemctl stop "buildkite-agent-gpu${i}" 2>/dev/null || true
    systemctl disable "buildkite-agent-gpu${i}" 2>/dev/null || true
done

# === CREATE DIRECTORIES ===
echo "Creating directories..."
mkdir -p /var/lib/buildkite-agent/builds
mkdir -p /var/lib/buildkite-agent/plugins
mkdir -p /etc/buildkite-agent/hooks
chown -R buildkite-agent:buildkite-agent /var/lib/buildkite-agent
chown -R buildkite-agent:buildkite-agent /etc/buildkite-agent

# === CONFIGURE GIT TO USE HTTPS (avoids SSH key issues) ===
echo "Configuring git to use HTTPS..."
cd /tmp
sudo -u buildkite-agent git config --global url."https://github.com/".insteadOf "git@github.com:"

# === CREATE ENVIRONMENT HOOK FOR GPU ISOLATION ===
echo "Creating environment hook for GPU/CPU/RAM isolation..."
cat > /etc/buildkite-agent/hooks/environment << 'HOOKEOF'
#!/bin/bash
# Resource isolation hook - auto-detects and divides resources by GPU count

GPU_INDEX="${BUILDKITE_AGENT_META_DATA_GPU_INDEX:-0}"

# Auto-detect total resources
TOTAL_CPUS=$(nproc)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))

# Auto-detect GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
else
    GPU_COUNT=1
fi

# Calculate per-GPU allocation
CPUS_PER_GPU=$((TOTAL_CPUS / GPU_COUNT))
RAM_PER_GPU=$((TOTAL_RAM_GB / GPU_COUNT))

# GPU isolation
export NVIDIA_VISIBLE_DEVICES="${GPU_INDEX}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

# CPU isolation (assign a range of CPUs to each GPU)
CPU_START=$((GPU_INDEX * CPUS_PER_GPU))
CPU_END=$((CPU_START + CPUS_PER_GPU - 1))
export KERNELBOT_CPUSET="${CPU_START}-${CPU_END}"
export KERNELBOT_CPUS="${CPUS_PER_GPU}"

# Memory isolation
export KERNELBOT_MEMORY="${RAM_PER_GPU}g"

# GPU index for the runner
export KERNELBOT_GPU_INDEX="${GPU_INDEX}"

echo "=== Resource Isolation ==="
echo "Machine: ${TOTAL_CPUS} CPUs, ${TOTAL_RAM_GB}GB RAM, ${GPU_COUNT} GPUs"
echo "This job: GPU ${NVIDIA_VISIBLE_DEVICES}, CPUs ${KERNELBOT_CPUSET}, RAM ${KERNELBOT_MEMORY}"
HOOKEOF
chmod +x /etc/buildkite-agent/hooks/environment
chown buildkite-agent:buildkite-agent /etc/buildkite-agent/hooks/environment

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
plugins-path="/var/lib/buildkite-agent/plugins"
EOF
    chown buildkite-agent:buildkite-agent "${config_file}"

    # Write systemd service
    cat > "/etc/systemd/system/buildkite-agent-gpu${gpu_idx}.service" << EOF
[Unit]
Description=Buildkite Agent (GPU ${gpu_idx})
Documentation=https://buildkite.com/docs/agent/v3
After=network.target

[Service]
Type=simple
User=buildkite-agent
ExecStart=/usr/bin/buildkite-agent start --config ${config_file}
RestartSec=5
Restart=on-failure
TimeoutStartSec=10
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF

    echo "  Created agent ${gpu_idx}: ${agent_name}"
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
echo ""
echo "Created ${GPU_COUNT} agents for queue: ${GPU_TYPE}"
echo "GPU isolation is handled via environment hook (NVIDIA_VISIBLE_DEVICES)"
echo ""
echo "IMPORTANT: Make sure you created the '${GPU_TYPE}' queue in Buildkite:"
echo "  1. Go to: https://buildkite.com/organizations/YOUR_ORG/clusters"
echo "  2. Click 'Default cluster' > 'Queues' > 'New Queue'"
echo "  3. Enter '${GPU_TYPE}' as the key, select 'Self hosted'"
echo ""
echo "Your agents should appear at: https://buildkite.com/organizations/YOUR_ORG/agents"
echo ""
echo "Test with this pipeline step:"
echo '  steps:'
echo '    - label: "GPU Test"'
echo '      command: "echo NVIDIA_VISIBLE_DEVICES=$$NVIDIA_VISIBLE_DEVICES && nvidia-smi -L"'
echo '      agents:'
echo "        queue: \"${GPU_TYPE}\""
