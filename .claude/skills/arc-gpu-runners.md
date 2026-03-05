# ARC GPU Runner Setup

How to set up Actions Runner Controller (ARC) on a bare-metal k3s node with AMD GPUs so that multiple GitHub Actions jobs run concurrently, each isolated to its own GPU, CPU, and RAM slice.

## Prerequisites

- k3s cluster running (check with `sudo k3s kubectl get nodes`)
- AMD GPU device plugin daemonset deployed (`amdgpu-device-plugin-daemonset` in `kube-system`)
- Docker installed on the node
- A GitHub PAT with `repo` scope (classic) or "Administration" read/write (fine-grained) for the target repo

## Setup Steps

### 1. Install Helm

```bash
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | sudo bash
```

### 2. Add the ARC Helm repo

```bash
sudo helm repo add actions-runner-controller https://actions-runner-controller.github.io/actions-runner-controller
sudo helm repo update
```

### 3. Install the ARC controller

```bash
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm install arc \
  --namespace arc-systems \
  --create-namespace \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version 0.10.1
```

Verify the controller is running:

```bash
sudo k3s kubectl get pods -n arc-systems
```

### 4. Deploy the runner scale set

Create a values file (`arc-runner-values.yaml`):

```yaml
githubConfigUrl: "https://github.com/gpu-mode/kernelbot"
githubConfigSecret:
  github_token: "<YOUR_GITHUB_PAT>"

maxRunners: 8
minRunners: 0

template:
  spec:
    containers:
      - name: runner
        image: ghcr.io/gpu-mode/amd-runner:mi355
        command: ["/home/runner/run.sh"]
        resources:
          requests:
            cpu: "14"
            memory: "340Gi"
            amd.com/gpu: "1"
          limits:
            cpu: "14"
            memory: "340Gi"
            amd.com/gpu: "1"
        volumeMounts:
          - name: kfd
            mountPath: /dev/kfd
    volumes:
      - name: kfd
        hostPath:
          path: /dev/kfd
          type: CharDevice
    nodeSelector:
      kubernetes.io/os: linux
```

Install:

```bash
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm install arc-runner-set \
  --namespace arc-runners \
  --create-namespace \
  -f arc-runner-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.10.1
```

### 5. Verify

```bash
# Controller + listener running
sudo k3s kubectl get pods -n arc-systems

# Scale set registered
sudo k3s kubectl get autoscalingrunnerset -n arc-runners

# Listener connected to GitHub
sudo k3s kubectl logs -n arc-systems -l actions.github.com/scale-set-name=arc-runner-set --tail=10
```

## How It Works

- **GPU isolation**: The AMD device plugin exposes `amd.com/gpu` as a k8s resource. Each runner pod requests exactly 1 GPU. Kubernetes guarantees no two pods share a GPU — each gets a unique `/dev/dri/renderD*` device.
- **CPU isolation**: Each pod gets 14 dedicated cores via cgroup limits (`nproc` reports 14 inside the container).
- **RAM isolation**: Each pod gets a 340Gi memory limit enforced by cgroups. Exceeding it triggers OOM kill.
- **Autoscaling**: With `minRunners: 0` and `maxRunners: 40`, runners spin up on demand when GitHub queues jobs and are destroyed after completion (ephemeral runners). The scheduler spreads pods across all 5 nodes.

## Resource Budget (per MI355X node)

The MI355X node has 126 allocatable CPUs, ~3TB RAM, and 8 GPUs.

| Per runner | Value |
|------------|-------|
| CPU | 14 cores |
| RAM | 340 Gi |
| GPU | 1x MI355X |

At max capacity (40 runners across 5 nodes): 8 runners per node, each using 14 cores / 340 Gi / 1 GPU.

## Using in Workflows

Workflows target ARC runners with `runs-on: arc-runner-set`. Since the runner pod already uses the `ghcr.io/gpu-mode/amd-runner:mi355` image (with ROCm, Python, etc.), there is no need for a separate `container:` block.

```yaml
jobs:
  my-job:
    runs-on: arc-runner-set
    steps:
      - uses: actions/checkout@v4
      - run: rocm-smi  # GPU is available
```

## Updating the Configuration

To change resource limits, max runners, or the runner image:

```bash
# Edit values, then:
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm upgrade arc-runner-set \
  --namespace arc-runners \
  -f arc-runner-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.10.1
```

## Adding New Nodes

ARC is cluster-wide — no per-node setup is needed. When a new node joins the k3s cluster:

1. The AMD GPU device plugin (DaemonSet) auto-deploys to the new node
2. The k8s scheduler can immediately place runner pods on it
3. No changes needed to workflows or the GitHub launcher

The only thing to update is `maxRunners` to reflect the new total GPU count:

```bash
# Example: 3 nodes × 8 GPUs = 24 max runners
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm upgrade arc-runner-set \
  --namespace arc-runners \
  --set maxRunners=24 \
  --reuse-values \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.10.1
```

Verify the new node has GPUs registered:

```bash
sudo k3s kubectl describe node <new-node-name> | grep amd.com/gpu
```

## Troubleshooting

- **403 on token**: PAT needs `repo` scope or fine-grained "Administration" read/write permission
- **Pods stuck Pending**: Check GPU availability with `kubectl describe node <name> | grep amd.com/gpu`
- **Listener not starting**: Check controller logs: `kubectl logs -n arc-systems -l app.kubernetes.io/name=gha-rs-controller`
- **Runner image issues**: The image must have `/home/runner/run.sh` (GitHub Actions runner binary)

## Current Cluster Info

- **Nodes**: mia1-p02-g29, mia1-p02-g52, mia1-p02-g53, mia1-p02-g55, mia1-p02-g56 (5-node k3s cluster)
- **GPUs**: 8x AMD Instinct MI355X per node
- **CPU**: AMD EPYC 9575F 64-Core (128 threads, 2 sockets)
- **RAM**: ~3 TB per node
- **SSH**: `ssh -J marksaroufim@meta.com@64.139.223.122 marksaroufim@meta.com@mia1-p02-g29`
