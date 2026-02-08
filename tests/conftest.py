import subprocess
import time
from pathlib import Path

import pytest

DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/clusterdev"


@pytest.fixture(scope="module")
def docker_compose(project_root: Path):
    """Start a test database and run migrations"""
    subprocess.check_call(
        ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d"], cwd=project_root
    )

    try:
        # Wait for migrations to finish
        while True:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.test.yml", "ps", "-q", "migrate-test"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            if not result.stdout.strip():  # Container no longer exists
                break
            time.sleep(1)

        # Check if migrations succeeded
        logs = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.test.yml", "logs", "migrate-test"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if "error" in logs.stdout.lower():
            raise Exception(f"Migrations failed: {logs.stdout}")

        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"], cwd=project_root
        )


def _nuke_contents(db):
    db.cursor.execute(
        "TRUNCATE leaderboard.code_files, leaderboard.submission, leaderboard.runs, "
        "leaderboard.leaderboard, leaderboard.user_info, leaderboard.templates, "
        "leaderboard.gpu_type, leaderboard.user_rate_limits RESTART IDENTITY CASCADE"
    )
    db.connection.commit()


@pytest.fixture()
def database(docker_compose):
    from libkernelbot import leaderboard_db

    db = leaderboard_db.LeaderboardDB(
        url=DATABASE_URL,
        ssl_mode="disable",
    )

    with db:
        _nuke_contents(db)
    yield db
    with db:
        _nuke_contents(db)


@pytest.fixture()
def bot(docker_compose, database):
    from types import SimpleNamespace

    from libkernelbot import backend

    env = SimpleNamespace()
    env.DATABASE_URL = DATABASE_URL
    env.DISABLE_SSL = "1"
    yield backend.KernelBackend(env, False)


TASK_YAML = """
lang: py
description: "Test task description"
ranking_by: geom
test_timeout: 120
files:
  - name: "kernel.py"
    source: "kernel.py"
  - name: "submission.py"
    source: "@SUBMISSION@"
config:
  main: "kernel.py"
tests:
  - input_size: 1000
    dtype: "float32"
benchmarks:
  - input_size: 10000
    dtype: "float32"
templates:
  Python: "template.py"
  CUDA: "template.cu"
"""

MULTi_GPU_TASK_YAML = """
lang: py
description: "Test task description"
ranking_by: geom
multi_gpu: true
test_timeout: 120
files:
  - name: "kernel.py"
    source: "kernel.py"
  - name: "submission.py"
    source: "@SUBMISSION@"
config:
  main: "kernel.py"
tests:
  - input_size: 1000
    world_size: 4
    dtype: "float32"
benchmarks:
  - input_size: 10000
    world_size: 4
    dtype: "float32"
templates:
  Python: "template.py"
  CUDA: "template.cu"
"""


@pytest.fixture
def task_directory(tmp_path):
    """Create a temporary directory structure for task definition testing"""
    # Create source files
    Path.write_text(tmp_path / "kernel.py", "def kernel(): pass")
    Path.write_text(tmp_path / "template.py", "# Python template")
    Path.write_text(tmp_path / "template.cu", "// CUDA template")

    # Create task.yml
    Path.write_text(tmp_path / "task.yml", TASK_YAML)
    Path.write_text(tmp_path / "multi-task.yml", MULTi_GPU_TASK_YAML)
    return tmp_path


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent
