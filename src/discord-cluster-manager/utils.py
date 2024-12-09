import logging
import subprocess
import datetime
import re
import importlib.util
from typing import TypedDict


def setup_logging():
    """Configure and setup logging for the application"""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def get_github_branch_name():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("/", 1)[1]
    except subprocess.CalledProcessError:
        return "main"


def extract_score(score_str: str) -> float:
    """
    Extract score from output logs and push to DB (kind of hacky).
    """
    match = re.search(r"score:\s*(-?\d+\.\d+)", score_str)
    if match:
        return float(match.group(1))
    else:
        return None


def strip_imports(code: str, global_vars: dict) -> str:
    """
    Removes import statements from the code that import functions
    already defined in global_vars.
    """
    lines = code.split("\n")
    modified_code = []

    # Extract the names of the functions already defined in global_vars
    existing_functions = {name for name, obj in global_vars.items() if callable(obj)}

    for line in lines:
        # If the line is an import statement, check if it's importing one of the existing functions
        if line.startswith("from") or line.startswith("import"):
            # Extract the function names from the import statements
            import_parts = line.split(" ")
            if len(import_parts) > 1:
                imported_name = import_parts[-1].strip()
                if imported_name in existing_functions:
                    continue  # Skip this import if the function is already defined
        modified_code.append(line)

    return "\n".join(modified_code)


async def get_user_from_id(id, interaction, bot):
    # This currently doesn't work.
    if interaction.guild:
        # In a guild, try to get the member by ID
        member = await interaction.guild.fetch_member(id)
        if member:
            username = member.global_name if member.nick is None else member.nick
            return username
        else:
            return id
    else:
        # If the interaction is in DMs, we can get the user directly
        user = await bot.fetch_user(id)
        if user:
            username = user.global_name if member.nick is None else member.nick
            return username
        else:
            return id


class LeaderboardItem(TypedDict):
    name: str
    deadline: datetime.datetime
    reference_code: str


class SubmissionItem(TypedDict):
    submission_name: str
    submission_time: datetime.datetime
    submission_score: float
    leaderboard_name: str
    code: str
    user_id: int


class ProfilingItem(TypedDict):
    submission_name: str
    ncu_output: str
    stdout: str
