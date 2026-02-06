#!/bin/bash
# Buildkite GPU Node Setup
# Usage: BUILDKITE_AGENT_TOKEN=xxx GPU_TYPE=b200 ./setup-node.sh

set -euo pipefail

# === CONFIGURATION ===
BUILDKITE_TOKEN="${BUILDKITE_AGENT_TOKEN:?Must set BUILDKITE_AGENT_TOKEN}"
GPU_TYPE="${GPU_TYPE:?Must set GPU_TYPE (e.g., b200, mi300, h100)}"
NODE_NAME="${NODE_NAME:-$(hostname)}"

# Auto-detect GPU count
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=count --format=csv,noheader | head -1
    elif command -v rocm-smi &> /dev/null; then
        rocm-smi --showid | grep -c "GPU"
    else
        echo "8"  # Default
    fi
}

GPU_COUNT="${GPU_COUNT:-$(detect_gpu_count)}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
RAM_PER_GPU="${RAM_PER_GPU:-64g}"

# Queue name - same for all agents on this node
QUEUE_NAME="${GPU_TYPE}"

echo "=== Buildkite GPU Node Setup ==="
echo "Node: ${NODE_NAME}"
echo "GPU Type: ${GPU_TYPE}"
echo "GPU Count: ${GPU_COUNT}"
echo "Queue: ${QUEUE_NAME}"
echo "CPUs per GPU: ${CPUS_PER_GPU}"
echo "RAM per GPU: ${RAM_PER_GPU}"
echo ""

# === INSTALL DEPENDENCIES ===

install_docker_nvidia() {
    echo "Installing Docker and NVIDIA Container Toolkit..."

    # Docker
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com | sh
        usermod -aG docker ubuntu 2>/dev/null || true
    fi

    # NVIDIA Container Toolkit
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
        apt-get install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
    fi

    echo "Docker + NVIDIA toolkit installed."
}

install_buildkite_agent() {
    echo "Installing Buildkite Agent..."

    if ! command -v buildkite-agent &> /dev/null; then
        apt-get install -y apt-transport-https gnupg
        curl -fsSL https://keys.openpgp.org/vks/v1/by-fingerprint/32A37959C2FA5C3C99EFBC32A79206696452D198 | \
            gpg --dearmor -o /usr/share/keyrings/buildkite-agent-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/buildkite-agent-archive-keyring.gpg] https://apt.buildkite.com/buildkite-agent stable main" | \
            tee /etc/apt/sources.list.d/buildkite-agent.list
        apt-get update
        apt-get install -y buildkite-agent
    fi

    echo "Buildkite Agent installed."
}

# === CREATE PER-GPU AGENTS ===

setup_agents() {
    echo "Configuring ${GPU_COUNT} agents..."

    # Create base directories
    mkdir -p /etc/buildkite-agent/hooks
    mkdir -p /var/lib/buildkite-agent

    # Create shared hooks
    cat > /etc/buildkite-agent/hooks/environment << 'HOOKEOF'
#!/bin/bash
# GPU isolation hook - runs before each job
set -euo pipefail

# GPU index is set per-agent via environment
echo "GPU ${BUILDKITE_AGENT_META_DATA_GPU_INDEX} allocated for this job"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
HOOKEOF
    chmod +x /etc/buildkite-agent/hooks/environment

    # Create pre-exit hook for cleanup
    cat > /etc/buildkite-agent/hooks/pre-exit << 'HOOKEOF'
#!/bin/bash
# Cleanup after job
docker system prune -f --filter "until=1h" 2>/dev/null || true
HOOKEOF
    chmod +x /etc/buildkite-agent/hooks/pre-exit

    # Stop any existing agents
    systemctl stop 'buildkite-agent-gpu*' 2>/dev/null || true

    # Create agent for each GPU
    for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
        local cpu_start=$((gpu_idx * CPUS_PER_GPU))
        local cpu_end=$((cpu_start + CPUS_PER_GPU - 1))
        local agent_name="${NODE_NAME}-gpu${gpu_idx}"
        local config_dir="/etc/buildkite-agent/agent-${gpu_idx}"
        local build_dir="/var/lib/buildkite-agent/gpu-${gpu_idx}/builds"

        mkdir -p "${config_dir}"
        mkdir -p "${build_dir}"

        # Agent configuration
        cat > "${config_dir}/buildkite-agent.cfg" << CFGEOF
# Buildkite Agent Configuration - GPU ${gpu_idx}
token="${BUILDKITE_TOKEN}"
name="${agent_name}"
tags="queue=${QUEUE_NAME},gpu=${GPU_TYPE},gpu-index=${gpu_idx},node=${NODE_NAME}"
build-path="${build_dir}"
hooks-path="/etc/buildkite-agent/hooks"
plugins-path="/var/lib/buildkite-agent/plugins"
disconnect-after-job=false
disconnect-after-idle-timeout=0
CFGEOF

        # Agent environment file (for GPU isolation)
        cat > "${config_dir}/environment" << ENVEOF
NVIDIA_VISIBLE_DEVICES=${gpu_idx}
CUDA_VISIBLE_DEVICES=${gpu_idx}
KERNELBOT_GPU_INDEX=${gpu_idx}
KERNELBOT_CPU_START=${cpu_start}
KERNELBOT_CPU_END=${cpu_end}
KERNELBOT_CPUSET=${cpu_start}-${cpu_end}
KERNELBOT_MEMORY=${RAM_PER_GPU}
ENVEOF

        # Systemd service
        cat > "/etc/systemd/system/buildkite-agent-gpu${gpu_idx}.service" << SVCEOF
[Unit]
Description=Buildkite Agent (GPU ${gpu_idx})
Documentation=https://buildkite.com/docs/agent/v3
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=buildkite-agent
EnvironmentFile=${config_dir}/environment
ExecStart=/usr/bin/buildkite-agent start --config ${config_dir}/buildkite-agent.cfg
RestartSec=5
Restart=on-failure
RestartForceExitStatus=SIGPIPE
TimeoutStartSec=10
TimeoutStopSec=60
KillMode=process

[Install]
WantedBy=multi-user.target
SVCEOF

        echo "  Agent ${gpu_idx}: GPU=${gpu_idx}, CPUs=${cpu_start}-${cpu_end}"
    done

    # Fix permissions
    chown -R buildkite-agent:buildkite-agent /var/lib/buildkite-agent
    chown -R buildkite-agent:buildkite-agent /etc/buildkite-agent

    # Add buildkite-agent to docker group
    usermod -aG docker buildkite-agent
}

# === START AGENTS ===

start_agents() {
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
}

# === MAIN ===

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

install_docker_nvidia
install_buildkite_agent
setup_agents
start_agents

echo ""
echo "=== Setup Complete ==="
echo "Agents should appear at: https://buildkite.com/organizations/YOUR_ORG/agents"
echo "Queue: ${QUEUE_NAME}"
echo ""
echo "Test with: buildkite-agent start --help"
