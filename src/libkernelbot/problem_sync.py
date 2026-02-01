"""Shared logic for syncing problems from a repository.

This module provides the core functionality for downloading problem sets from GitHub
and creating/updating leaderboards. Used by both the API and Discord bot.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, TypedDict

import yaml

from .task import LeaderboardDefinition, make_task_definition
from .utils import parse_deadline, setup_logging

logger = setup_logging(__name__)


class ProblemData(TypedDict):
    name: str
    directory: str
    deadline: str
    gpus: list[str]


class CompetitionData(TypedDict):
    name: str
    description: str
    deadline: str
    problems: list[ProblemData]


@dataclass
class SyncResult:
    """Result of a problem sync operation."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


@dataclass
class ProblemPlan:
    """Plan for creating or updating a problem."""

    name: str
    directory: str
    definition: LeaderboardDefinition
    deadline: datetime
    gpus: list[str]
    action: str  # "create" or "update"


def download_problem_repo(repository: str, branch: str, temp_dir: str) -> Path:
    """Download and extract a problem repository from GitHub.

    Args:
        repository: Repository in "owner/repo" format
        branch: Branch name to download
        temp_dir: Temporary directory to extract to

    Returns:
        Path to the problems directory

    Raises:
        RuntimeError: If download or extraction fails
    """
    url = f"https://github.com/{repository}/archive/{branch}.zip"
    folder_name = repository.split("/")[-1] + "-" + branch

    # Download
    try:
        subprocess.check_call(
            ["curl", "-sL", "-o", f"{temp_dir}/problems.zip", url],
            encoding="utf-8",
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Could not download repository from {url}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Timeout downloading repository") from e

    # Extract
    try:
        subprocess.check_call(
            ["unzip", "-q", f"{temp_dir}/problems.zip", "-d", temp_dir],
            encoding="utf-8",
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Could not unzip repository: {e}") from e

    problem_dir = Path(temp_dir) / folder_name / "problems"
    if not problem_dir.exists():
        raise RuntimeError("No 'problems' directory found in repository")

    return problem_dir


def create_update_plan(  # noqa: C901
    competition: CompetitionData,
    problem_dir: Path,
    existing_leaderboards: dict,
    force: bool = False,
) -> tuple[list[ProblemPlan], list[dict]]:
    """Determine which problems to create or update.

    Args:
        competition: Parsed competition YAML data
        problem_dir: Path to the problems directory
        existing_leaderboards: Dict mapping leaderboard names to their data
        force: If True, allow significant task changes

    Returns:
        Tuple of (list of ProblemPlan objects, list of skip/error dicts)
    """
    plans = []
    skipped = []

    for problem in competition.get("problems", []):
        name = problem.get("name")
        directory = problem.get("directory")
        deadline_str = problem.get("deadline")
        gpus = problem.get("gpus", [])

        if not name or not directory:
            skipped.append({"name": name or "unknown", "reason": "Missing name or directory"})
            continue

        source_path = problem_dir / directory
        if not source_path.exists():
            skipped.append({"name": name, "reason": f"Directory {directory} not found"})
            continue

        try:
            definition = make_task_definition(source_path)
        except Exception as e:
            skipped.append({"name": name, "reason": f"Failed to parse task.yml: {e}"})
            continue

        deadline = parse_deadline(deadline_str) if deadline_str else None
        if deadline is None:
            deadline = datetime.now(timezone.utc) + timedelta(days=365)
        elif deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)

        # Use GPUs from YAML or task definition
        if not gpus:
            gpus = definition.gpus if definition.gpus else []

        if name in existing_leaderboards:
            old_lb = existing_leaderboards[name]
            old_deadline = old_lb["deadline"]
            if hasattr(old_deadline, "tzinfo") and old_deadline.tzinfo is None:
                old_deadline = old_deadline.replace(tzinfo=timezone.utc)

            deadline_changed = old_deadline != deadline
            task_changed = old_lb["task"] != definition.task

            if not deadline_changed and not task_changed:
                skipped.append({"name": name, "reason": "no changes"})
                continue

            if task_changed and not force:
                old_task = old_lb["task"]
                new_task = definition.task
                if (
                    old_task.files != new_task.files
                    or old_task.config != new_task.config
                    or old_task.lang != new_task.lang
                    or old_task.benchmarks != new_task.benchmarks
                ):
                    skipped.append({"name": name, "reason": "significant task changes require --force"})
                    continue

            plans.append(
                ProblemPlan(
                    name=name,
                    directory=directory,
                    definition=definition,
                    deadline=deadline,
                    gpus=gpus,
                    action="update",
                )
            )
        else:
            if not gpus:
                skipped.append({"name": name, "reason": "No GPUs specified in task.yml or YAML"})
                continue

            plans.append(
                ProblemPlan(
                    name=name,
                    directory=directory,
                    definition=definition,
                    deadline=deadline,
                    gpus=gpus,
                    action="create",
                )
            )

    return plans, skipped


def sync_problems(  # noqa: C901
    db_context,
    repository: str = "gpu-mode/reference-kernels",
    problem_set: Optional[str] = None,
    branch: str = "main",
    force: bool = False,
    creator_id: int = 0,
    forum_id: int = -1,
) -> SyncResult:
    """Sync problems from a GitHub repository.

    Downloads the repository, parses competition YAML files, and creates/updates leaderboards.

    Args:
        db_context: Database context manager
        repository: Repository in "owner/repo" format
        problem_set: Specific problem set to sync, or None for all
        branch: Branch to download
        force: If True, allow significant task changes
        creator_id: ID of the creator (0 for API)
        forum_id: Discord forum ID (-1 for API)

    Returns:
        SyncResult with created, updated, skipped, and errors lists
    """
    if "/" in branch:
        raise ValueError("Branch names with slashes are not supported")

    result = SyncResult()

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            problem_dir = download_problem_repo(repository, branch, temp_dir)
        except RuntimeError as e:
            result.errors.append({"name": "download", "error": str(e)})
            return result

        # Find YAML files
        if problem_set is None:
            yaml_files = list(problem_dir.glob("*.yaml"))
        else:
            yaml_file = problem_dir / f"{problem_set}.yaml"
            if not yaml_file.exists():
                available = [f.stem for f in problem_dir.glob("*.yaml")]
                result.errors.append({
                    "name": problem_set,
                    "error": f"Problem set not found. Available: {available}"
                })
                return result
            yaml_files = [yaml_file]

        # Get existing leaderboards
        with db_context as db:
            existing_leaderboards = {lb["name"]: lb for lb in db.get_leaderboards()}

        # Process each YAML file
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    competition = yaml.safe_load(f)

                plans, skipped = create_update_plan(
                    competition, problem_dir, existing_leaderboards, force
                )
                result.skipped.extend(skipped)

                for plan in plans:
                    try:
                        if plan.action == "create":
                            with db_context as db:
                                db.create_leaderboard(
                                    name=plan.name,
                                    deadline=plan.deadline,
                                    definition=plan.definition,
                                    creator_id=creator_id,
                                    forum_id=forum_id,
                                    gpu_types=plan.gpus,
                                )
                            result.created.append(plan.name)
                        else:  # update
                            with db_context as db:
                                db.update_leaderboard(plan.name, plan.deadline, plan.definition)
                            result.updated.append(plan.name)
                    except Exception as e:
                        result.errors.append({"name": plan.name, "error": f"{plan.action} failed: {e}"})

            except yaml.YAMLError as e:
                result.errors.append({"name": yaml_file.stem, "error": f"Invalid YAML: {e}"})
            except Exception as e:
                result.errors.append({"name": yaml_file.stem, "error": str(e)})

    return result
