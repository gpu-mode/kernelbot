import re

from openai import AsyncOpenAI

from libkernelbot.consts import Language
from libkernelbot.task import LeaderboardTask
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)


async def generate_kernel(
    prompt: str,
    task: LeaderboardTask,
    description: str,
    templates: dict[str, str],
) -> tuple[str, str]:
    """Generate kernel code from a natural language prompt using OpenAI.

    Args:
        prompt: The user's natural language description of the kernel to generate.
        task: The LeaderboardTask containing file signatures, tests, and config.
        description: The leaderboard's problem description.
        templates: Template/starter code files keyed by language name.

    Returns:
        A tuple of (generated_code, file_name).
    """
    # Build context from the task
    system_parts = [
        "You are an expert GPU kernel programmer. Generate code that solves the given problem.",
        "Return ONLY the code inside a single code block. No explanation outside the code block.",
    ]

    if description:
        system_parts.append(f"## Problem Description\n{description}")

    # Include template code so the AI knows the expected function signatures
    if templates:
        for lang, code in templates.items():
            system_parts.append(f"## Template ({lang})\n```\n{code}\n```")

    # Include reference/test files for additional context (skip submission placeholder)
    for name, content in task.files.items():
        if content != "@SUBMISSION@":
            system_parts.append(f"## Reference file: {name}\n```\n{content}\n```")

    # Include test specs so the AI knows input sizes / shapes
    if task.tests:
        system_parts.append(f"## Test cases\n{task.tests}")

    system_prompt = "\n\n".join(system_parts)

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="o3",
        max_completion_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content

    # Extract code from a fenced code block if present
    match = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()

    file_name = "submission.py" if task.lang == Language.Python else "submission.cu"
    return code, file_name
