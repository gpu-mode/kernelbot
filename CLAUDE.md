# Kernelbot Development Guide

## Linting

Always run ruff before committing:

```bash
uv run ruff check . --exclude examples/ --line-length 120 --fix
```

To check without auto-fixing:

```bash
uv run ruff check . --exclude examples/ --line-length 120
```

## Testing

Run tests with pytest:

```bash
uv run pytest tests/ -v
```

## Local Development

See `SKILLS/test_bot.md` for local testing setup instructions.

## Architecture

### Problem Configuration

Problems are defined in the [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) repository. See that repo for examples of problem structure and `task.yml` format.

### Leaderboard Creation

- **Dev leaderboards** (via API): Created from a single problem directory. GPUs must be specified in the problem's `task.yml`. The leaderboard name is auto-derived as `{directory}-dev`.

- **Competition leaderboards** (via Discord admin_cog): Created from a competition YAML file that references multiple problems with their deadlines and GPU configurations.
