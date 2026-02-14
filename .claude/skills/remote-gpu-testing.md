# Remote GPU Testing for GitHub Actions

Local machine is macOS with no GPUs. To test GitHub Action workflows and GPU code, run them on remote machines via SSH.

## Machines

- **l-bgx-01 (B200 x8)**: `ssh -i /Users/marksaroufim/Dev/kernelbot/.ssh_key_tmp -o IdentitiesOnly=yes ubuntu@154.57.34.106`
  - 8x NVIDIA B200 (183GB each), sm_100, CUDA 13.0, Driver 580.95.05
  - GPUs 0-3 may be occupied — use `CUDA_VISIBLE_DEVICES=4,5,6,7`
  - GH Actions runner label: `nvidia-docker-b200-8-x86-64`
  - Persistent vLLM + model weights at `/models/meta-llama/Llama-3.1-8B`
  - Working dir: `/home/ubuntu/kernelbot`

## How to run remote commands

Never run GPU code locally. Always use this pattern:

1. **Set up a tmux session** on the remote machine (idempotent):
   ```bash
   ssh host "tmux new-session -d -s work || true"
   ```

2. **Run commands** via send-keys, always tee to a log file:
   ```bash
   ssh host "tmux send-keys -t work 'command 2>&1 | tee /tmp/jobname.log' Enter"
   ```

3. **Check output** by tailing the log file directly:
   ```bash
   ssh host "tail -100 /tmp/jobname.log"
   ```

4. **Check if a command is still running**:
   ```bash
   ssh host "pgrep -f command_name"
   ```

## Testing GitHub Actions locally on a remote GPU machine

To replicate what a GitHub Action workflow does on a remote GPU box:

1. **Push code to the remote machine**:
   ```bash
   rsync -avz --exclude '.git' --exclude '__pycache__' ./ host:/home/user/kernelbot/
   ```

2. **Set up the environment** (mirrors the GH Action):
   ```bash
   ssh host "tmux send-keys -t work 'cd /home/user/kernelbot && pip install -r requirements-dev.txt && pip install -e . 2>&1 | tee /tmp/setup.log' Enter"
   ```

3. **Run the tests**:
   ```bash
   ssh host "tmux send-keys -t work 'cd /home/user/kernelbot && pytest 2>&1 | tee /tmp/tests.log' Enter"
   ```

4. **Run GPU kernel submissions** (what the bot dispatches to GitHub Actions):
   ```bash
   ssh host "tmux send-keys -t work 'cd /home/user/kernelbot && python src/kernelbot/main.py --debug 2>&1 | tee /tmp/bot.log' Enter"
   ```

## Rules

- Assume all remote commands are long-running.
- Always log output to a file with `tee`. Never rely on tmux scrollback.
- Use direct `ssh host "tail ..."` to read logs, not tmux capture-pane.
- For file transfers use `scp` or `rsync`.
- Multiple parallel jobs: use separate tmux windows (`tmux new-window -t work -n jobname`).
- Keep tmux sessions alive across interactions. Always `|| true` on session creation to avoid errors if it already exists.
- To run a command in a specific directory, include `cd` in the send-keys.
- After making local code changes, always `rsync` to the remote machine before re-running tests.
