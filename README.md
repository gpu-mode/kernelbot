# KernelBot

Backend service for the GPU Mode kernel competition platform.

**For users:** Submit kernels via the [popcorn-cli](https://github.com/gpu-mode/popcorn-cli).

**For problem authors:** See [reference-kernels](https://github.com/gpu-mode/reference-kernels) for problem configuration and examples.

## Local Development

### Prerequisites

- Python 3.11+
- PostgreSQL
- A Discord bot application (optional, for Discord integration - see [docs/discord.md](docs/discord.md))

### Clone and Install

```bash
git clone https://github.com/gpu-mode/discord-cluster-manager.git
cd discord-cluster-manager
pip install -e .
```

### Database Setup

Create a local Postgres database and apply migrations:

```bash
psql -U postgres -c "CREATE DATABASE clusterdev;"
yoyo apply src/migrations -d postgresql://user:password@localhost/clusterdev
```

See [docs/database.md](docs/database.md) for migration patterns and creating new migrations.

### Environment Variables

Create a `.env` file:

```bash
# Required
GITHUB_TOKEN=           # GitHub PAT with repo and workflow scopes
GITHUB_REPO=gpu-mode/discord-cluster-manager
PROBLEMS_REPO=gpu-mode/reference-kernels
DATABASE_URL=postgresql://user:password@localhost/clusterdev

# Optional - defaults shown
GITHUB_WORKFLOW_BRANCH=main
PROBLEM_DEV_DIR=examples
DISABLE_SSL=1           # Set for local development
GITHUB_TOKEN_BACKUP=    # Fallback token for rate limiting
ADMIN_TOKEN=            # Token for admin API endpoints

# Discord bot (only needed if testing Discord integration)
# See docs/discord.md for setup instructions
DISCORD_TOKEN=
DISCORD_DEBUG_TOKEN=
DISCORD_CLUSTER_STAGING_ID=
DISCORD_DEBUG_CLUSTER_STAGING_ID=

# CLI OAuth (only needed if running CLI against local instance)
CLI_DISCORD_CLIENT_ID=
CLI_DISCORD_CLIENT_SECRET=
CLI_TOKEN_URL=
CLI_GITHUB_CLIENT_ID=
CLI_GITHUB_CLIENT_SECRET=
```

### Run the Bot

```bash
python src/kernelbot/main.py --debug
```

Use `/verifyruns` to test GitHub Actions integration and `/verifydb` to check database connectivity.

## Adding GPUs to the Cluster

To donate a GPU, contact us to become a CI admin and add an org-level runner at https://github.com/organizations/gpu-mode/settings/actions/runners

## Acknowledgements

- Modal for credits
- Northflank for hosting the kernelbot service
- AMD for sponsoring an MI250 node
- NVIDIA for sponsoring an H100 node
- Nebius for credits and an H100 node

## Citation

```bibtex
@inproceedings{
  kernelbot2025,
  title={KernelBot: A Competition Platform for Writing Heterogeneous {GPU} Code},
  author={Alex L Zhang and Matej Sirovatka and Erik Schultheis and Benjamin Horowitz and Mark Saroufim},
  note={Equal Contribution},
  booktitle={Championing Open-source Development in ML Workshop @ ICML25},
  year={2025},
  url={https://openreview.net/forum?id=bq9U4dmuyJ}
}
```
